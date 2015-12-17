package tsdb

import (
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
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

	// Execute statements relating to statistics and diagnostics.
	MonitorStatementExecutor interface {
		ExecuteStatement(stmt influxql.Statement) *influxql.Result
	}

	// Maps shards for queries.
	ShardMapper interface {
		CreateMapper(shard meta.ShardInfo, stmt influxql.Statement, chunkSize int) (Mapper, error)
	}

	IntoWriter interface {
		WritePointsInto(p *IntoWriteRequest) error
	}

	Logger          *log.Logger
	QueryLogEnabled bool

	// the local data store
	Store *Store
}

// partial copy of cluster.WriteRequest
type IntoWriteRequest struct {
	Database        string
	RetentionPolicy string
	Points          []models.Point
}

// NewQueryExecutor returns an initialized QueryExecutor
func NewQueryExecutor(store *Store) *QueryExecutor {
	return &QueryExecutor{
		Store:  store,
		Logger: log.New(os.Stderr, "[query] ", log.LstdFlags),
	}
}

// SetLogger sets the internal logger to the logger passed in.
func (q *QueryExecutor) SetLogger(l *log.Logger) {
	q.Logger = l
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
func (q *QueryExecutor) ExecuteQuery(query *influxql.Query, database string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
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

			// Log each normalized statement.
			if q.QueryLogEnabled {
				q.Logger.Println(stmt.String())
			}

			var res *influxql.Result
			switch stmt := stmt.(type) {
			case *influxql.SelectStatement:
				if err := q.executeStatement(i, stmt, database, results, chunkSize, closing); err != nil {
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
				if err := q.executeStatement(i, stmt, database, results, chunkSize, closing); err != nil {
					results <- &influxql.Result{Err: err}
					break
				}
			case *influxql.ShowTagKeysStatement:
				if err := q.executeStatement(i, stmt, database, results, chunkSize, closing); err != nil {
					results <- &influxql.Result{Err: err}
					break
				}
			case *influxql.ShowTagValuesStatement:
				res = q.executeShowTagValuesStatement(stmt, database)
			case *influxql.ShowFieldKeysStatement:
				res = q.executeShowFieldKeysStatement(stmt, database)
			case *influxql.DeleteStatement:
				res = &influxql.Result{Err: ErrInvalidQuery}
			case *influxql.DropDatabaseStatement:
				// TODO: handle this in a cluster
				res = q.executeDropDatabaseStatement(stmt)
			case *influxql.ShowStatsStatement, *influxql.ShowDiagnosticsStatement:
				// Send monitor-related queries to the monitor service.
				res = q.MonitorStatementExecutor.ExecuteStatement(stmt)
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
func (q *QueryExecutor) PlanSelect(stmt *influxql.SelectStatement, chunkSize int) (Executor, error) {
	var shardIDs []uint64
	shards := map[uint64]meta.ShardInfo{} // Shards requiring mappers.

	// It is important to "stamp" this time so that everywhere we evaluate `now()` in the statement is EXACTLY the same `now`
	now := time.Now().UTC()

	// Replace instances of "now()" with the current time, and check the resultant times.
	stmt.Condition = influxql.Reduce(stmt.Condition, &influxql.NowValuer{Now: now})
	tmin, tmax := influxql.TimeRange(stmt.Condition)
	if tmax.IsZero() {
		tmax = now
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
				if _, ok := shards[sh.ID]; !ok {
					shards[sh.ID] = sh
					shardIDs = append(shardIDs, sh.ID)
				}
			}
		}
	}

	// Sort shard IDs to make testing deterministic.
	sort.Sort(uint64Slice(shardIDs))

	// Build the Mappers, one per shard.
	mappers := []Mapper{}
	for _, shardID := range shardIDs {
		sh := shards[shardID]

		m, err := q.ShardMapper.CreateMapper(sh, stmt, chunkSize)
		if err != nil {
			return nil, err
		}
		if m == nil {
			// No data for this shard, skip it.
			continue
		}
		mappers = append(mappers, m)
	}

	// Certain operations on the SELECT statement can be performed by the AggregateExecutor without
	// assistance from the Mappers. This allows the AggregateExecutor to prepare aggregation functions
	// and mathematical functions.
	stmt.RewriteDistinct()

	if (stmt.IsRawQuery && !stmt.HasDistinct()) || stmt.IsSimpleDerivative() {
		return NewRawExecutor(stmt, mappers, chunkSize), nil
	} else {
		return NewAggregateExecutor(stmt, mappers), nil
	}
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
			db := q.Store.DatabaseIndex(src.Database)
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
		if stmt.IfExists {
			return &influxql.Result{}
		}
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

	// Remove database from meta-store first so that in-flight writes can complete without error, but new ones will
	// be rejected.
	res := q.MetaStatementExecutor.ExecuteStatement(stmt)

	// Remove the database from the local store
	err = q.Store.DeleteDatabase(stmt.Name, shardIDs)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	return res
}

// executeDropMeasurementStatement removes the measurement and all series data from the local store for the given measurement
func (q *QueryExecutor) executeDropMeasurementStatement(stmt *influxql.DropMeasurementStatement, database string) *influxql.Result {
	// Find the database.
	db := q.Store.DatabaseIndex(database)
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
	if err := q.Store.deleteMeasurement(database, m.Name, m.SeriesKeys()); err != nil {
		return &influxql.Result{Err: err}
	}

	return &influxql.Result{}
}

// executeDropSeriesStatement removes all series from the local store that match the drop query
func (q *QueryExecutor) executeDropSeriesStatement(stmt *influxql.DropSeriesStatement, database string) *influxql.Result {
	// Check for time in WHERE clause (not supported).
	if influxql.HasTimeExpr(stmt.Condition) {
		return &influxql.Result{Err: errors.New("DROP SERIES doesn't support time in WHERE clause")}
	}

	// Find the database.
	db := q.Store.DatabaseIndex(database)
	if db == nil {
		return &influxql.Result{}
	}

	// Expand regex expressions in the FROM clause.
	sources, err := q.expandSources(stmt.Sources)
	if err != nil {
		return &influxql.Result{Err: err}
	} else if stmt.Sources != nil && len(stmt.Sources) != 0 && len(sources) == 0 {
		return &influxql.Result{}
	}

	measurements, err := measurementsFromSourcesOrDB(db, sources...)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	var seriesKeys []string
	for _, m := range measurements {
		var ids SeriesIDs
		var filters FilterExprs
		if stmt.Condition != nil {
			// Get series IDs that match the WHERE clause.
			ids, filters, err = m.walkWhereForSeriesIds(stmt.Condition)
			if err != nil {
				return &influxql.Result{Err: err}
			}

			// Delete boolean literal true filter expressions.
			// These are returned for `WHERE tagKey = 'tagVal'` type expressions and are okay.
			filters.DeleteBoolLiteralTrues()

			// Check for unsupported field filters.
			// Any remaining filters means there were fields (e.g., `WHERE value = 1.2`).
			if filters.Len() > 0 {
				return &influxql.Result{Err: errors.New("DROP SERIES doesn't support fields in WHERE clause")}
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
	if err := q.Store.deleteSeries(database, seriesKeys); err != nil {
		return &influxql.Result{Err: err}
	}
	// remove them from the index
	db.DropSeries(seriesKeys)

	return &influxql.Result{}
}

func (q *QueryExecutor) executeShowSeriesStatement(stmt *influxql.ShowSeriesStatement, database string) *influxql.Result {
	// Check for time in WHERE clause (not supported).
	if influxql.HasTimeExpr(stmt.Condition) {
		return &influxql.Result{Err: errors.New("SHOW SERIES doesn't support time in WHERE clause")}
	}

	// Find the database.
	db := q.Store.DatabaseIndex(database)
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
		Series: make(models.Rows, 0, len(measurements)),
	}

	// Loop through measurements to build result. One result row / measurement.
	for _, m := range measurements {
		var ids SeriesIDs
		var filters FilterExprs

		if stmt.Condition != nil {
			// Get series IDs that match the WHERE clause.
			ids, filters, err = m.walkWhereForSeriesIds(stmt.Condition)
			if err != nil {
				return &influxql.Result{Err: err}
			}

			// Delete boolean literal true filter expressions.
			filters.DeleteBoolLiteralTrues()

			// Check for unsupported field filters.
			if filters.Len() > 0 {
				return &influxql.Result{Err: errors.New("SHOW SERIES doesn't support fields in WHERE clause")}
			}

			// If no series matched, then go to the next measurement.
			if len(ids) == 0 {
				continue
			}
		} else {
			// No WHERE clause so get all series IDs for this measurement.
			ids = m.seriesIDs
		}

		// Make a new row for this measurement.
		r := &models.Row{
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
func (q *QueryExecutor) filterShowSeriesResult(limit, offset int, rows models.Rows) models.Rows {
	var filteredSeries models.Rows
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

func (q *QueryExecutor) planStatement(stmt influxql.Statement, database string, chunkSize int) (Executor, error) {
	switch stmt := stmt.(type) {
	case *influxql.SelectStatement:
		return q.PlanSelect(stmt, chunkSize)
	case *influxql.ShowMeasurementsStatement:
		return q.PlanShowMeasurements(stmt, database, chunkSize)
	case *influxql.ShowTagKeysStatement:
		return q.PlanShowTagKeys(stmt, database, chunkSize)
	default:
		return nil, fmt.Errorf("can't plan statement type: %v", stmt)
	}
}

// PlanShowMeasurements creates an execution plan for a SHOW TAG KEYS statement and returns an Executor.
func (q *QueryExecutor) PlanShowMeasurements(stmt *influxql.ShowMeasurementsStatement, database string, chunkSize int) (Executor, error) {
	// Check for time in WHERE clause (not supported).
	if influxql.HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW MEASUREMENTS doesn't support time in WHERE clause")
	}

	// Get the database info.
	di, err := q.MetaStore.Database(database)
	if err != nil {
		return nil, err
	} else if di == nil {
		return nil, ErrDatabaseNotFound(database)
	}

	// Get info for all shards in the database.
	shards := di.ShardInfos()

	// Build the Mappers, one per shard.
	mappers := []Mapper{}
	for _, sh := range shards {
		m, err := q.ShardMapper.CreateMapper(sh, stmt, chunkSize)
		if err != nil {
			return nil, err
		}
		if m == nil {
			// No data for this shard, skip it.
			continue
		}
		mappers = append(mappers, m)
	}

	executor := NewShowMeasurementsExecutor(stmt, mappers, chunkSize)
	return executor, nil
}

// PlanShowTagKeys creates an execution plan for a SHOW MEASUREMENTS statement and returns an Executor.
func (q *QueryExecutor) PlanShowTagKeys(stmt *influxql.ShowTagKeysStatement, database string, chunkSize int) (Executor, error) {
	// Check for time in WHERE clause (not supported).
	if influxql.HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW TAG KEYS doesn't support time in WHERE clause")
	}

	// Get the database info.
	di, err := q.MetaStore.Database(database)
	if err != nil {
		return nil, err
	} else if di == nil {
		return nil, ErrDatabaseNotFound(database)
	}

	// Get info for all shards in the database.
	shards := di.ShardInfos()

	// Build the Mappers, one per shard.
	mappers := []Mapper{}
	for _, sh := range shards {
		m, err := q.ShardMapper.CreateMapper(sh, stmt, chunkSize)
		if err != nil {
			return nil, err
		}
		if m == nil {
			// No data for this shard, skip it.
			continue
		}
		mappers = append(mappers, m)
	}

	executor := NewShowTagKeysExecutor(stmt, mappers, chunkSize)
	return executor, nil
}

func (q *QueryExecutor) executeStatement(statementID int, stmt influxql.Statement, database string, results chan *influxql.Result, chunkSize int, closing chan struct{}) error {
	// Plan statement execution.
	e, err := q.planStatement(stmt, database, chunkSize)
	if err != nil {
		return err
	}

	// Execute plan.
	ch := e.Execute(closing)
	var writeerr error
	var intoNum int64
	var isinto bool
	// Stream results from the channel. We should send an empty result if nothing comes through.
	resultSent := false
	for row := range ch {
		// We had a write error. Continue draining results from the channel
		// so we don't hang the goroutine in the executor.
		if writeerr != nil {
			continue
		}
		if row.Err != nil {
			return row.Err
		}
		selectstmt, ok := stmt.(*influxql.SelectStatement)
		if ok && selectstmt.Target != nil {
			isinto = true
			// this is a into query. Write results back to database
			writeerr = q.writeInto(row, selectstmt)
			intoNum += int64(len(row.Values))
		} else {
			resultSent = true
			results <- &influxql.Result{StatementID: statementID, Series: []*models.Row{row}}
		}
	}
	if writeerr != nil {
		return writeerr
	} else if isinto {
		results <- &influxql.Result{
			StatementID: statementID,
			Series: []*models.Row{{
				Name: "result",
				// it seems weird to give a time here, but so much stuff breaks if you don't
				Columns: []string{"time", "written"},
				Values: [][]interface{}{{
					time.Unix(0, 0).UTC(),
					intoNum,
				}},
			}},
		}
		return nil
	}

	if !resultSent {
		results <- &influxql.Result{StatementID: statementID, Series: make([]*models.Row, 0)}
	}

	return nil
}

func (q *QueryExecutor) writeInto(row *models.Row, selectstmt *influxql.SelectStatement) error {
	// It might seem a bit weird that this is where we do this, since we will have to
	// convert rows back to points. The Executors (both aggregate and raw) are complex
	// enough that changing them to write back to the DB is going to be clumsy
	//
	// it might seem weird to have the write be in the QueryExecutor, but the interweaving of
	// limitedRowWriter and ExecuteAggregate/Raw makes it ridiculously hard to make sure that the
	// results will be the same as when queried normally.
	measurement := intoMeasurement(selectstmt)
	if measurement == "" {
		measurement = row.Name
	}
	intodb, err := intoDB(selectstmt)
	if err != nil {
		return err
	}
	rp := intoRP(selectstmt)
	points, err := convertRowToPoints(measurement, row)
	if err != nil {
		return err
	}
	req := &IntoWriteRequest{
		Database:        intodb,
		RetentionPolicy: rp,
		Points:          points,
	}
	err = q.IntoWriter.WritePointsInto(req)
	if err != nil {
		return err
	}
	return nil
}

func (q *QueryExecutor) executeShowTagValuesStatement(stmt *influxql.ShowTagValuesStatement, database string) *influxql.Result {
	// Check for time in WHERE clause (not supported).
	if influxql.HasTimeExpr(stmt.Condition) {
		return &influxql.Result{Err: errors.New("SHOW TAG VALUES doesn't support time in WHERE clause")}
	}

	// Find the database.
	db := q.Store.DatabaseIndex(database)
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
		Series: make(models.Rows, 0),
	}

	tagValues := make(map[string]stringSet)
	for _, m := range measurements {
		var ids SeriesIDs

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
		r := &models.Row{
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
	db := q.Store.DatabaseIndex(database)
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
		Series: make(models.Rows, 0, len(measurements)),
	}

	// Loop through measurements, adding a result row for each.
	for _, m := range measurements {
		// Create a new row.
		r := &models.Row{
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
					continue
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
	return
}

// normalizeMeasurement inserts the default database or policy into all measurement names,
// if required.
func (q *QueryExecutor) normalizeMeasurement(m *influxql.Measurement, defaultDatabase string) error {
	// Targets (measurements in an INTO clause) can have blank names, which means it will be
	// the same as the measurement name it came from in the FROM clause.
	if !m.IsTarget && m.Name == "" && m.Regex == nil {
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

type uint64Slice []uint64

func (a uint64Slice) Len() int           { return len(a) }
func (a uint64Slice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a uint64Slice) Less(i, j int) bool { return a[i] < a[j] }
