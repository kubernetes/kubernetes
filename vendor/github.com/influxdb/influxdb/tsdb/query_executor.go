package tsdb

import (
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
)

// QueryExecutor executes every statement in an influxdb Query. It is responsible for
// coordinating between the local tsdb.Store, the meta.Store, and the other nodes in
// the cluster to run the query against their local tsdb.Stores. There should be one executor
// in a running process
type QueryExecutor struct {
	// The meta store for accessing and updating cluster and schema data.
	MetaStore interface {
		Database(name string) (*meta.DatabaseInfo, error)
		Databases() ([]meta.DatabaseInfo, error)
		User(name string) (*meta.UserInfo, error)
		AdminUserExists() (bool, error)
		Authenticate(username, password string) (*meta.UserInfo, error)
		RetentionPolicy(database, name string) (rpi *meta.RetentionPolicyInfo, err error)
		UserCount() (int, error)
		ShardGroupsByTimeRange(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error)
		NodeID() uint64
	}

	// Executes statements relating to meta data.
	MetaStatementExecutor interface {
		ExecuteStatement(stmt influxql.Statement) *influxql.Result
	}

	// Maps shards for queries.
	ShardMapper interface {
		CreateMapper(shard meta.ShardInfo, stmt string, chunkSize int) (Mapper, error)
	}

	Logger *log.Logger

	// the local data store
	store *Store
}

// NewQueryExecutor returns an initialized QueryExecutor
func NewQueryExecutor(store *Store) *QueryExecutor {
	return &QueryExecutor{
		store:  store,
		Logger: log.New(os.Stderr, "[query] ", log.LstdFlags),
	}
}

// Authorize user u to execute query q on database.
// database can be "" for queries that do not require a database.
// If no user is provided it will return an error unless the query's first statement is to create
// a root user.
func (q *QueryExecutor) Authorize(u *meta.UserInfo, query *influxql.Query, database string) error {
	// Special case if no users exist.
	if count, err := q.MetaStore.UserCount(); count == 0 && err == nil {
		// Ensure there is at least one statement.
		if len(query.Statements) > 0 {
			// First statement in the query must create a user with admin privilege.
			cu, ok := query.Statements[0].(*influxql.CreateUserStatement)
			if ok && cu.Admin == true {
				return nil
			}
		}
		return NewErrAuthorize(q, query, "", database, "create admin user first or disable authentication")
	}

	if u == nil {
		return NewErrAuthorize(q, query, "", database, "no user provided")
	}

	// Admin privilege allows the user to execute all statements.
	if u.Admin {
		return nil
	}

	// Check each statement in the query.
	for _, stmt := range query.Statements {
		// Get the privileges required to execute the statement.
		privs := stmt.RequiredPrivileges()

		// Make sure the user has the privileges required to execute
		// each statement.
		for _, p := range privs {
			if p.Admin {
				// Admin privilege already checked so statement requiring admin
				// privilege cannot be run.
				msg := fmt.Sprintf("statement '%s', requires admin privilege", stmt)
				return NewErrAuthorize(q, query, u.Name, database, msg)
			}

			// Use the db name specified by the statement or the db
			// name passed by the caller if one wasn't specified by
			// the statement.
			db := p.Name
			if db == "" {
				db = database
			}
			if !u.Authorize(p.Privilege, db) {
				msg := fmt.Sprintf("statement '%s', requires %s on %s", stmt, p.Privilege.String(), db)
				return NewErrAuthorize(q, query, u.Name, database, msg)
			}
		}
	}
	return nil
}

// ExecuteQuery executes an InfluxQL query against the server.
// It sends results down the passed in chan and closes it when done. It will close the chan
// on the first statement that throws an error.
func (q *QueryExecutor) ExecuteQuery(query *influxql.Query, database string, chunkSize int) (<-chan *influxql.Result, error) {
	// Execute each statement. Keep the iterator external so we can
	// track how many of the statements were executed
	results := make(chan *influxql.Result)
	go func() {
		var i int
		var stmt influxql.Statement
		for i, stmt = range query.Statements {
			// If a default database wasn't passed in by the caller, check the statement.
			// Some types of statements have an associated default database, even if it
			// is not explicitly included.
			defaultDB := database
			if defaultDB == "" {
				if s, ok := stmt.(influxql.HasDefaultDatabase); ok {
					defaultDB = s.DefaultDatabase()
				}
			}

			// Normalize each statement.
			if err := q.normalizeStatement(stmt, defaultDB); err != nil {
				results <- &influxql.Result{Err: err}
				break
			}

			var res *influxql.Result
			switch stmt := stmt.(type) {
			case *influxql.SelectStatement:
				if err := q.executeSelectStatement(i, stmt, results, chunkSize); err != nil {
					results <- &influxql.Result{Err: err}
					break
				}
			case *influxql.DropSeriesStatement:
				// TODO: handle this in a cluster
				res = q.executeDropSeriesStatement(stmt, database)
			case *influxql.ShowSeriesStatement:
				res = q.executeShowSeriesStatement(stmt, database)
			case *influxql.DropMeasurementStatement:
				// TODO: handle this in a cluster
				res = q.executeDropMeasurementStatement(stmt, database)
			case *influxql.ShowMeasurementsStatement:
				res = q.executeShowMeasurementsStatement(stmt, database)
			case *influxql.ShowTagKeysStatement:
				res = q.executeShowTagKeysStatement(stmt, database)
			case *influxql.ShowTagValuesStatement:
				res = q.executeShowTagValuesStatement(stmt, database)
			case *influxql.ShowFieldKeysStatement:
				res = q.executeShowFieldKeysStatement(stmt, database)
			case *influxql.ShowDiagnosticsStatement:
				res = q.executeShowDiagnosticsStatement(stmt)
			case *influxql.DeleteStatement:
				res = &influxql.Result{Err: ErrInvalidQuery}
			case *influxql.DropDatabaseStatement:
				// TODO: handle this in a cluster
				res = q.executeDropDatabaseStatement(stmt)
			default:
				// Delegate all other meta statements to a separate executor. They don't hit tsdb storage.
				res = q.MetaStatementExecutor.ExecuteStatement(stmt)
			}

			if res != nil {
				// set the StatementID for the handler on the other side to combine results
				res.StatementID = i

				// If an error occurs then stop processing remaining statements.
				results <- res
				if res.Err != nil {
					break
				}
			}
		}

		// if there was an error send results that the remaining statements weren't executed
		for ; i < len(query.Statements)-1; i++ {
			results <- &influxql.Result{Err: ErrNotExecuted}
		}

		close(results)
	}()

	return results, nil
}

// Plan creates an execution plan for the given SelectStatement and returns an Executor.
func (q *QueryExecutor) plan(stmt *influxql.SelectStatement, chunkSize int) (Executor, error) {
	shards := map[uint64]meta.ShardInfo{} // Shards requiring mappers.

	// Replace instances of "now()" with the current time, and check the resultant times.
	stmt.Condition = influxql.Reduce(stmt.Condition, &influxql.NowValuer{Now: time.Now().UTC()})
	tmin, tmax := influxql.TimeRange(stmt.Condition)
	if tmax.IsZero() {
		tmax = time.Now()
	}
	if tmin.IsZero() {
		tmin = time.Unix(0, 0)
	}

	for _, src := range stmt.Sources {
		mm, ok := src.(*influxql.Measurement)
		if !ok {
			return nil, fmt.Errorf("invalid source type: %#v", src)
		}

		// Build the set of target shards. Using shard IDs as keys ensures each shard ID
		// occurs only once.
		shardGroups, err := q.MetaStore.ShardGroupsByTimeRange(mm.Database, mm.RetentionPolicy, tmin, tmax)
		if err != nil {
			return nil, err
		}
		for _, g := range shardGroups {
			for _, sh := range g.Shards {
				shards[sh.ID] = sh
			}
		}
	}

	// Build the Mappers, one per shard.
	mappers := []Mapper{}
	for _, sh := range shards {
		m, err := q.ShardMapper.CreateMapper(sh, stmt.String(), chunkSize)
		if err != nil {
			return nil, err
		}
		if m == nil {
			// No data for this shard, skip it.
			continue
		}
		mappers = append(mappers, m)
	}

	var executor Executor
	if len(mappers) > 0 {
		// All Mapper are of same type, so check first to determine correct Executor type.
		if _, ok := mappers[0].(*RawMapper); ok {
			executor = NewRawExecutor(stmt, mappers, chunkSize)
		} else {
			executor = NewAggregateExecutor(stmt, mappers)
		}
	} else {
		// With no mappers, the Executor type doesn't matter.
		executor = NewRawExecutor(stmt, nil, chunkSize)
	}
	return executor, nil
}

// executeSelectStatement plans and executes a select statement against a database.
func (q *QueryExecutor) executeSelectStatement(statementID int, stmt *influxql.SelectStatement, results chan *influxql.Result, chunkSize int) error {
	// Perform any necessary query re-writing.
	stmt, err := q.rewriteSelectStatement(stmt)
	if err != nil {
		return err
	}

	// Plan statement execution.
	e, err := q.plan(stmt, chunkSize)
	if err != nil {
		return err
	}

	// Execute plan.
	ch := e.Execute()

	// Stream results from the channel. We should send an empty result if nothing comes through.
	resultSent := false
	for row := range ch {
		if row.Err != nil {
			return row.Err
		} else {
			resultSent = true
			results <- &influxql.Result{StatementID: statementID, Series: []*influxql.Row{row}}
		}
	}

	if !resultSent {
		results <- &influxql.Result{StatementID: statementID, Series: make([]*influxql.Row, 0)}
	}

	return nil
}

// rewriteSelectStatement performs any necessary query re-writing.
func (q *QueryExecutor) rewriteSelectStatement(stmt *influxql.SelectStatement) (*influxql.SelectStatement, error) {
	var err error

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return nil, err
	}
	stmt.Sources = sources

	// Expand wildcards in the fields or GROUP BY.
	if stmt.HasWildcard() {
		stmt, err = q.expandWildcards(stmt)
		if err != nil {
			return nil, err
		}
	}

	stmt.RewriteDistinct()

	return stmt, nil
}

// expandWildcards returns a new SelectStatement with wildcards in the fields
// and/or GROUP BY expanded with actual field names.
func (q *QueryExecutor) expandWildcards(stmt *influxql.SelectStatement) (*influxql.SelectStatement, error) {
	// If there are no wildcards in the statement, return it as-is.
	if !stmt.HasWildcard() {
		return stmt, nil
	}

	// Use sets to avoid duplicate field names.
	fieldSet := map[string]struct{}{}
	dimensionSet := map[string]struct{}{}

	var fields influxql.Fields
	var dimensions influxql.Dimensions

	// Iterate measurements in the FROM clause getting the fields & dimensions for each.
	for _, src := range stmt.Sources {
		if m, ok := src.(*influxql.Measurement); ok {
			// Lookup the database. The database may not exist if no data for this database
			// was ever written to the shard.
			db := q.store.DatabaseIndex(m.Database)
			if db == nil {
				return stmt, nil
			}

			// Lookup the measurement in the database.
			mm := db.measurements[m.Name]
			if mm == nil {
				return nil, ErrMeasurementNotFound(m.String())
			}

			// Get the fields for this measurement.
			for _, name := range mm.FieldNames() {
				if _, ok := fieldSet[name]; ok {
					continue
				}
				fieldSet[name] = struct{}{}
				fields = append(fields, &influxql.Field{Expr: &influxql.VarRef{Val: name}})
			}

			// Get the dimensions for this measurement.
			for _, t := range mm.TagKeys() {
				if _, ok := dimensionSet[t]; ok {
					continue
				}
				dimensionSet[t] = struct{}{}
				dimensions = append(dimensions, &influxql.Dimension{Expr: &influxql.VarRef{Val: t}})
			}
		}
	}

	// Return a new SelectStatement with the wild cards rewritten.
	return stmt.RewriteWildcards(fields, dimensions), nil
}

// expandSources expands regex sources and removes duplicates.
// NOTE: sources must be normalized (db and rp set) before calling this function.
func (q *QueryExecutor) expandSources(sources influxql.Sources) (influxql.Sources, error) {
	// Use a map as a set to prevent duplicates. Two regexes might produce
	// duplicates when expanded.
	set := map[string]influxql.Source{}
	names := []string{}

	// Iterate all sources, expanding regexes when they're found.
	for _, source := range sources {
		switch src := source.(type) {
		case *influxql.Measurement:
			if src.Regex == nil {
				name := src.String()
				set[name] = src
				names = append(names, name)
				continue
			}

			// Lookup the database.
			db := q.store.DatabaseIndex(src.Database)
			if db == nil {
				return nil, nil
			}

			// Get measurements from the database that match the regex.
			measurements := db.measurementsByRegex(src.Regex.Val)

			// Add those measurements to the set.
			for _, m := range measurements {
				m2 := &influxql.Measurement{
					Database:        src.Database,
					RetentionPolicy: src.RetentionPolicy,
					Name:            m.Name,
				}

				name := m2.String()
				if _, ok := set[name]; !ok {
					set[name] = m2
					names = append(names, name)
				}
			}

		default:
			return nil, fmt.Errorf("expandSources: unsuported source type: %T", source)
		}
	}

	// Sort the list of source names.
	sort.Strings(names)

	// Convert set to a list of Sources.
	expanded := make(influxql.Sources, 0, len(set))
	for _, name := range names {
		expanded = append(expanded, set[name])
	}

	return expanded, nil
}

// executeDropDatabaseStatement closes all local shards for the database and removes the directory. It then calls to the metastore to remove the database from there.
// TODO: make this work in a cluster/distributed
func (q *QueryExecutor) executeDropDatabaseStatement(stmt *influxql.DropDatabaseStatement) *influxql.Result {
	dbi, err := q.MetaStore.Database(stmt.Name)
	if err != nil {
		return &influxql.Result{Err: err}
	} else if dbi == nil {
		return &influxql.Result{Err: ErrDatabaseNotFound(stmt.Name)}
	}

	var shardIDs []uint64
	for _, rp := range dbi.RetentionPolicies {
		for _, sg := range rp.ShardGroups {
			for _, s := range sg.Shards {
				shardIDs = append(shardIDs, s.ID)
			}
		}
	}

	err = q.store.DeleteDatabase(stmt.Name, shardIDs)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	return q.MetaStatementExecutor.ExecuteStatement(stmt)
}

// executeDropMeasurementStatement removes the measurement and all series data from the local store for the given measurement
func (q *QueryExecutor) executeDropMeasurementStatement(stmt *influxql.DropMeasurementStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	m := db.Measurement(stmt.Name)
	if m == nil {
		return &influxql.Result{Err: ErrMeasurementNotFound(stmt.Name)}
	}

	// first remove from the index
	db.DropMeasurement(m.Name)

	// now drop the raw data
	if err := q.store.deleteMeasurement(m.Name, m.SeriesKeys()); err != nil {
		return &influxql.Result{Err: err}
	}

	return &influxql.Result{}
}

// executeDropSeriesStatement removes all series from the local store that match the drop query
func (q *QueryExecutor) executeDropSeriesStatement(stmt *influxql.DropSeriesStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	var seriesKeys []string
	for _, m := range measurements {
		var ids seriesIDs
		if stmt.Condition != nil {
			// Get series IDs that match the WHERE clause.
			ids, _, err = m.walkWhereForSeriesIds(stmt.Condition)
			if err != nil {
				return &influxql.Result{Err: err}
			}
		} else {
			// No WHERE clause so get all series IDs for this measurement.
			ids = m.seriesIDs
		}

		for _, id := range ids {
			seriesKeys = append(seriesKeys, m.seriesByID[id].Key)
		}
	}

	// delete the raw series data
	if err := q.store.deleteSeries(seriesKeys); err != nil {
		return &influxql.Result{Err: err}
	}
	// remove them from the index
	db.DropSeries(seriesKeys)

	return &influxql.Result{}
}

func (q *QueryExecutor) executeShowSeriesStatement(stmt *influxql.ShowSeriesStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Get the list of measurements we're interested in.
	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Create result struct that will be populated and returned.
	result := &influxql.Result{
		Series: make(influxql.Rows, 0, len(measurements)),
	}

	// Loop through measurements to build result. One result row / measurement.
	for _, m := range measurements {
		var ids seriesIDs

		if stmt.Condition != nil {
			// Get series IDs that match the WHERE clause.
			ids, _, err = m.walkWhereForSeriesIds(stmt.Condition)
			if err != nil {
				return &influxql.Result{Err: err}
			}

			// If no series matched, then go to the next measurement.
			if len(ids) == 0 {
				continue
			}

			// TODO: check return of walkWhereForSeriesIds for fields
		} else {
			// No WHERE clause so get all series IDs for this measurement.
			ids = m.seriesIDs
		}

		// Make a new row for this measurement.
		r := &influxql.Row{
			Name:    m.Name,
			Columns: m.TagKeys(),
		}

		// Loop through series IDs getting matching tag sets.
		for _, id := range ids {
			if s, ok := m.seriesByID[id]; ok {
				values := make([]interface{}, 0, len(r.Columns))

				// make the series key the first value
				values = append(values, s.Key)

				for _, column := range r.Columns {
					values = append(values, s.Tags[column])
				}

				// Add the tag values to the row.
				r.Values = append(r.Values, values)
			}
		}
		// make the id the first column
		r.Columns = append([]string{"_key"}, r.Columns...)

		// Append the row to the result.
		result.Series = append(result.Series, r)
	}

	if stmt.Limit > 0 || stmt.Offset > 0 {
		result.Series = q.filterShowSeriesResult(stmt.Limit, stmt.Offset, result.Series)
	}

	return result
}

// filterShowSeriesResult will limit the number of series returned based on the limit and the offset.
// Unlike limit and offset on SELECT statements, the limit and offset don't apply to the number of Rows, but
// to the number of total Values returned, since each Value represents a unique series.
func (q *QueryExecutor) filterShowSeriesResult(limit, offset int, rows influxql.Rows) influxql.Rows {
	var filteredSeries influxql.Rows
	seriesCount := 0
	for _, r := range rows {
		var currentSeries [][]interface{}

		// filter the values
		for _, v := range r.Values {
			if seriesCount >= offset && seriesCount-offset < limit {
				currentSeries = append(currentSeries, v)
			}
			seriesCount++
		}

		// only add the row back in if there are some values in it
		if len(currentSeries) > 0 {
			r.Values = currentSeries
			filteredSeries = append(filteredSeries, r)
			if seriesCount > limit+offset {
				return filteredSeries
			}
		}
	}
	return filteredSeries
}

func (q *QueryExecutor) executeShowMeasurementsStatement(stmt *influxql.ShowMeasurementsStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	var measurements Measurements

	// If a WHERE clause was specified, filter the measurements.
	if stmt.Condition != nil {
		var err error
		measurements, err = db.measurementsByExpr(stmt.Condition)
		if err != nil {
			return &influxql.Result{Err: err}
		}
	} else {
		// Otherwise, get all measurements from the database.
		measurements = db.Measurements()
	}
	sort.Sort(measurements)

	offset := stmt.Offset
	limit := stmt.Limit

	// If OFFSET is past the end of the array, return empty results.
	if offset > len(measurements)-1 {
		return &influxql.Result{}
	}

	// Calculate last index based on LIMIT.
	end := len(measurements)
	if limit > 0 && offset+limit < end {
		limit = offset + limit
	} else {
		limit = end
	}

	// Make a result row to hold all measurement names.
	row := &influxql.Row{
		Name:    "measurements",
		Columns: []string{"name"},
	}

	// Add one value to the row for each measurement name.
	for i := offset; i < limit; i++ {
		m := measurements[i]
		v := interface{}(m.Name)
		row.Values = append(row.Values, []interface{}{v})
	}

	// Make a result.
	result := &influxql.Result{
		Series: []*influxql.Row{row},
	}

	return result
}

func (q *QueryExecutor) executeShowTagKeysStatement(stmt *influxql.ShowTagKeysStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Get the list of measurements we're interested in.
	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Make result.
	result := &influxql.Result{
		Series: make(influxql.Rows, 0, len(measurements)),
	}

	// Add one row per measurement to the result.
	for _, m := range measurements {
		// TODO: filter tag keys by stmt.Condition

		// Get the tag keys in sorted order.
		keys := m.TagKeys()

		// Convert keys to an [][]interface{}.
		values := make([][]interface{}, 0, len(m.seriesByTagKeyValue))
		for _, k := range keys {
			v := interface{}(k)
			values = append(values, []interface{}{v})
		}

		// Make a result row for the measurement.
		r := &influxql.Row{
			Name:    m.Name,
			Columns: []string{"tagKey"},
			Values:  values,
		}

		result.Series = append(result.Series, r)
	}

	// TODO: LIMIT & OFFSET

	return result
}

func (q *QueryExecutor) executeShowTagValuesStatement(stmt *influxql.ShowTagValuesStatement, database string) *influxql.Result {
	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Get the list of measurements we're interested in.
	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Make result.
	result := &influxql.Result{
		Series: make(influxql.Rows, 0),
	}

	tagValues := make(map[string]stringSet)
	for _, m := range measurements {
		var ids seriesIDs

		if stmt.Condition != nil {
			// Get series IDs that match the WHERE clause.
			ids, _, err = m.walkWhereForSeriesIds(stmt.Condition)
			if err != nil {
				return &influxql.Result{Err: err}
			}

			// If no series matched, then go to the next measurement.
			if len(ids) == 0 {
				continue
			}

			// TODO: check return of walkWhereForSeriesIds for fields
		} else {
			// No WHERE clause so get all series IDs for this measurement.
			ids = m.seriesIDs
		}

		for k, v := range m.tagValuesByKeyAndSeriesID(stmt.TagKeys, ids) {
			_, ok := tagValues[k]
			if !ok {
				tagValues[k] = v
			}
			tagValues[k] = tagValues[k].union(v)
		}
	}

	for k, v := range tagValues {
		r := &influxql.Row{
			Name:    k + "TagValues",
			Columns: []string{k},
		}

		vals := v.list()
		sort.Strings(vals)

		for _, val := range vals {
			v := interface{}(val)
			r.Values = append(r.Values, []interface{}{v})
		}

		result.Series = append(result.Series, r)
	}

	sort.Sort(result.Series)
	return result
}

func (q *QueryExecutor) executeShowFieldKeysStatement(stmt *influxql.ShowFieldKeysStatement, database string) *influxql.Result {
	var err error

	// Find the database.
	db := q.store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// Make result.
	result := &influxql.Result{
		Series: make(influxql.Rows, 0, len(measurements)),
	}

	// Loop through measurements, adding a result row for each.
	for _, m := range measurements {
		// Create a new row.
		r := &influxql.Row{
			Name:    m.Name,
			Columns: []string{"fieldKey"},
		}

		// Get a list of field names from the measurement then sort them.
		names := m.FieldNames()
		sort.Strings(names)

		// Add the field names to the result row values.
		for _, n := range names {
			v := interface{}(n)
			r.Values = append(r.Values, []interface{}{v})
		}

		// Append the row to the result.
		result.Series = append(result.Series, r)
	}

	return result
}

// measurementsFromSourcesOrDB returns a list of measurements from the
// sources passed in or, if sources is empty, a list of all
// measurement names from the database passed in.
func measurementsFromSourcesOrDB(db *DatabaseIndex, sources ...influxql.Source) (Measurements, error) {
	var measurements Measurements
	if len(sources) > 0 {
		for _, source := range sources {
			if m, ok := source.(*influxql.Measurement); ok {
				measurement := db.measurements[m.Name]
				if measurement == nil {
					return nil, ErrMeasurementNotFound(m.Name)
				}

				measurements = append(measurements, measurement)
			} else {
				return nil, errors.New("identifiers in FROM clause must be measurement names")
			}
		}
	} else {
		// No measurements specified in FROM clause so get all measurements that have series.
		for _, m := range db.Measurements() {
			if m.HasSeries() {
				measurements = append(measurements, m)
			}
		}
	}
	sort.Sort(measurements)

	return measurements, nil
}

// normalizeStatement adds a default database and policy to the measurements in statement.
func (q *QueryExecutor) normalizeStatement(stmt influxql.Statement, defaultDatabase string) (err error) {
	// Track prefixes for replacing field names.
	prefixes := make(map[string]string)

	// Qualify all measurements.
	influxql.WalkFunc(stmt, func(n influxql.Node) {
		if err != nil {
			return
		}
		switch n := n.(type) {
		case *influxql.Measurement:
			e := q.normalizeMeasurement(n, defaultDatabase)
			if e != nil {
				err = e
				return
			}
			prefixes[n.Name] = n.Name
		}
	})
	if err != nil {
		return err
	}

	// Replace all variable references that used measurement prefixes.
	influxql.WalkFunc(stmt, func(n influxql.Node) {
		switch n := n.(type) {
		case *influxql.VarRef:
			for k, v := range prefixes {
				if strings.HasPrefix(n.Val, k+".") {
					n.Val = v + "." + influxql.QuoteIdent(n.Val[len(k)+1:])
				}
			}
		}
	})

	return
}

// normalizeMeasurement inserts the default database or policy into all measurement names,
// if required.
func (q *QueryExecutor) normalizeMeasurement(m *influxql.Measurement, defaultDatabase string) error {
	if m.Name == "" && m.Regex == nil {
		return errors.New("invalid measurement")
	}

	// Measurement does not have an explicit database? Insert default.
	if m.Database == "" {
		m.Database = defaultDatabase
	}

	// The database must now be specified by this point.
	if m.Database == "" {
		return errors.New("database name required")
	}

	// Find database.
	di, err := q.MetaStore.Database(m.Database)
	if err != nil {
		return err
	} else if di == nil {
		return ErrDatabaseNotFound(m.Database)
	}

	// If no retention policy was specified, use the default.
	if m.RetentionPolicy == "" {
		if di.DefaultRetentionPolicy == "" {
			return fmt.Errorf("default retention policy not set for: %s", di.Name)
		}
		m.RetentionPolicy = di.DefaultRetentionPolicy
	}

	return nil
}

func (q *QueryExecutor) executeShowDiagnosticsStatement(stmt *influxql.ShowDiagnosticsStatement) *influxql.Result {
	return &influxql.Result{Err: fmt.Errorf("SHOW DIAGNOSTICS is not implemented yet")}
}

// ErrAuthorize represents an authorization error.
type ErrAuthorize struct {
	q        *QueryExecutor
	query    *influxql.Query
	user     string
	database string
	message  string
}

const authErrLogFmt string = "unauthorized request | user: %q | query: %q | database %q\n"

// newAuthorizationError returns a new instance of AuthorizationError.
func NewErrAuthorize(qe *QueryExecutor, q *influxql.Query, u, db, m string) *ErrAuthorize {
	return &ErrAuthorize{q: qe, query: q, user: u, database: db, message: m}
}

// Error returns the text of the error.
func (e ErrAuthorize) Error() string {
	e.q.Logger.Printf(authErrLogFmt, e.user, e.query.String(), e.database)
	if e.user == "" {
		return fmt.Sprint(e.message)
	}
	return fmt.Sprintf("%s not authorized to execute %s", e.user, e.message)
}

var (
	// ErrInvalidQuery is returned when executing an unknown query type.
	ErrInvalidQuery = errors.New("invalid query")

	// ErrNotExecuted is returned when a statement is not executed in a query.
	// This can occur when a previous statement in the same query has errored.
	ErrNotExecuted = errors.New("not executed")
)

func ErrDatabaseNotFound(name string) error { return fmt.Errorf("database not found: %s", name) }

func ErrMeasurementNotFound(name string) error { return fmt.Errorf("measurement not found: %s", name) }
