package tsdb

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/pkg/escape"
	"github.com/influxdb/influxdb/tsdb/internal"

	"github.com/gogo/protobuf/proto"
)

//go:generate protoc --gogo_out=. internal/meta.proto

const (
	maxStringLength = 64 * 1024
)

// DatabaseIndex is the in memory index of a collection of measurements, time series, and their tags.
// Exported functions are goroutine safe while un-exported functions assume the caller will use the appropriate locks
type DatabaseIndex struct {
	// in memory metadata index, built on load and updated when new series come in
	mu           sync.RWMutex
	measurements map[string]*Measurement // measurement name to object and index
	series       map[string]*Series      // map series key to the Series object
	lastID       uint64                  // last used series ID. They're in memory only for this shard
}

func NewDatabaseIndex() *DatabaseIndex {
	return &DatabaseIndex{
		measurements: make(map[string]*Measurement),
		series:       make(map[string]*Series),
	}
}

// Series returns a series by key.
func (d *DatabaseIndex) Series(key string) *Series {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.series[key]
}

// SeriesN returns the number of series.
func (d *DatabaseIndex) SeriesN() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.series)
}

// Measurement returns the measurement object from the index by the name
func (d *DatabaseIndex) Measurement(name string) *Measurement {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.measurements[name]
}

// MeasurementsByName returns a list of measurements.
func (d *DatabaseIndex) MeasurementsByName(names []string) []*Measurement {
	d.mu.RLock()
	defer d.mu.RUnlock()

	a := make([]*Measurement, 0, len(names))
	for _, name := range names {
		if m := d.measurements[name]; m != nil {
			a = append(a, m)
		}
	}
	return a
}

// MeasurementSeriesCounts returns the number of measurements and series currently indexed by the database.
// Useful for reporting and monitoring.
func (d *DatabaseIndex) MeasurementSeriesCounts() (nMeasurements int, nSeries int) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	nMeasurements, nSeries = len(d.measurements), len(d.series)
	return
}

// CreateSeriesIndexIfNotExists adds the series for the given measurement to the index and sets its ID or returns the existing series object
func (s *DatabaseIndex) CreateSeriesIndexIfNotExists(measurementName string, series *Series) *Series {
	// if there is a measurement for this id, it's already been added
	ss := s.series[series.Key]
	if ss != nil {
		return ss
	}

	// get or create the measurement index
	m := s.CreateMeasurementIndexIfNotExists(measurementName)

	// set the in memory ID for query processing on this shard
	series.id = s.lastID + 1
	s.lastID += 1

	series.measurement = m
	s.series[series.Key] = series

	m.AddSeries(series)

	return series
}

// CreateMeasurementIndexIfNotExists creates or retrieves an in memory index object for the measurement
func (s *DatabaseIndex) CreateMeasurementIndexIfNotExists(name string) *Measurement {
	name = escape.UnescapeString(name)
	m := s.measurements[name]
	if m == nil {
		m = NewMeasurement(name, s)
		s.measurements[name] = m
	}
	return m
}

// TagsForSeries returns the tag map for the passed in series
func (s *DatabaseIndex) TagsForSeries(key string) map[string]string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ss := s.series[key]
	if ss == nil {
		return nil
	}
	return ss.Tags
}

// measurementsByExpr takes and expression containing only tags and returns
// a list of matching *Measurement.
func (db *DatabaseIndex) measurementsByExpr(expr influxql.Expr) (Measurements, error) {
	switch e := expr.(type) {
	case *influxql.BinaryExpr:
		switch e.Op {
		case influxql.EQ, influxql.NEQ, influxql.EQREGEX, influxql.NEQREGEX:
			tag, ok := e.LHS.(*influxql.VarRef)
			if !ok {
				return nil, fmt.Errorf("left side of '%s' must be a tag key", e.Op.String())
			}

			tf := &TagFilter{
				Op:  e.Op,
				Key: tag.Val,
			}

			if influxql.IsRegexOp(e.Op) {
				re, ok := e.RHS.(*influxql.RegexLiteral)
				if !ok {
					return nil, fmt.Errorf("right side of '%s' must be a regular expression", e.Op.String())
				}
				tf.Regex = re.Val
			} else {
				s, ok := e.RHS.(*influxql.StringLiteral)
				if !ok {
					return nil, fmt.Errorf("right side of '%s' must be a tag value string", e.Op.String())
				}
				tf.Value = s.Val
			}

			return db.measurementsByTagFilters([]*TagFilter{tf}), nil
		case influxql.OR, influxql.AND:
			lhsIDs, err := db.measurementsByExpr(e.LHS)
			if err != nil {
				return nil, err
			}

			rhsIDs, err := db.measurementsByExpr(e.RHS)
			if err != nil {
				return nil, err
			}

			if e.Op == influxql.OR {
				return lhsIDs.union(rhsIDs), nil
			}

			return lhsIDs.intersect(rhsIDs), nil
		default:
			return nil, fmt.Errorf("invalid operator")
		}
	case *influxql.ParenExpr:
		return db.measurementsByExpr(e.Expr)
	}
	return nil, fmt.Errorf("%#v", expr)
}

// measurementsByTagFilters returns the sorted measurements matching the filters on tag values.
func (db *DatabaseIndex) measurementsByTagFilters(filters []*TagFilter) Measurements {
	// If no filters, then return all measurements.
	if len(filters) == 0 {
		measurements := make(Measurements, 0, len(db.measurements))
		for _, m := range db.measurements {
			measurements = append(measurements, m)
		}
		return measurements
	}

	// Build a list of measurements matching the filters.
	var measurements Measurements
	var tagMatch bool

	// Iterate through all measurements in the database.
	for _, m := range db.measurements {
		// Iterate filters seeing if the measurement has a matching tag.
		for _, f := range filters {
			tagVals, ok := m.seriesByTagKeyValue[f.Key]
			if !ok {
				continue
			}

			tagMatch = false

			// If the operator is non-regex, only check the specified value.
			if f.Op == influxql.EQ || f.Op == influxql.NEQ {
				if _, ok := tagVals[f.Value]; ok {
					tagMatch = true
				}
			} else {
				// Else, the operator is regex and we have to check all tag
				// values against the regular expression.
				for tagVal := range tagVals {
					if f.Regex.MatchString(tagVal) {
						tagMatch = true
						break
					}
				}
			}

			isEQ := (f.Op == influxql.EQ || f.Op == influxql.EQREGEX)

			// tags match | operation is EQ | measurement matches
			// --------------------------------------------------
			//     True   |       True      |      True
			//     True   |       False     |      False
			//     False  |       True      |      False
			//     False  |       False     |      True

			if tagMatch == isEQ {
				measurements = append(measurements, m)
				break
			}
		}
	}

	sort.Sort(measurements)
	return measurements
}

// measurementsByRegex returns the measurements that match the regex.
func (db *DatabaseIndex) measurementsByRegex(re *regexp.Regexp) Measurements {
	var matches Measurements
	for _, m := range db.measurements {
		if re.MatchString(m.Name) {
			matches = append(matches, m)
		}
	}
	return matches
}

// Measurements returns a list of all measurements.
func (db *DatabaseIndex) Measurements() Measurements {
	measurements := make(Measurements, 0, len(db.measurements))
	for _, m := range db.measurements {
		measurements = append(measurements, m)
	}
	return measurements
}

// DropMeasurement removes the measurement and all of its underlying series from the database index
func (db *DatabaseIndex) DropMeasurement(name string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	m := db.measurements[name]
	if m == nil {
		return
	}

	delete(db.measurements, name)
	for _, s := range m.seriesByID {
		delete(db.series, s.Key)
	}
}

// DropSeries removes the series keys and their tags from the index
func (db *DatabaseIndex) DropSeries(keys []string) {
	db.mu.Lock()
	defer db.mu.Unlock()
	for _, k := range keys {
		series := db.series[k]
		if series == nil {
			continue
		}
		series.measurement.DropSeries(series.id)
		delete(db.series, k)
	}
}

// RewriteSelectStatement performs any necessary query re-writing.
func (db *DatabaseIndex) RewriteSelectStatement(stmt *influxql.SelectStatement) (*influxql.SelectStatement, error) {
	// Expand regex expressions in the FROM clause.
	sources, err := db.ExpandSources(stmt.Sources)
	if err != nil {
		return nil, err
	}
	stmt.Sources = sources

	// Expand wildcards in the fields or GROUP BY.
	stmt, err = db.ExpandWildcards(stmt)
	if err != nil {
		return nil, err
	}

	stmt.RewriteDistinct()

	return stmt, nil
}

// expandWildcards returns a new SelectStatement with wildcards expanded
// If only a `SELECT *` is present, without a `GROUP BY *`, both tags and fields expand in the SELECT
// If a `SELECT *` and a `GROUP BY *` are both present, then only fiels are expanded in the `SELECT` and only
// tags are expanded in the `GROUP BY`
func (db *DatabaseIndex) ExpandWildcards(stmt *influxql.SelectStatement) (*influxql.SelectStatement, error) {
	// If there are no wildcards in the statement, return it as-is.
	if !stmt.HasWildcard() {
		return stmt, nil
	}
	// Use sets to avoid duplicate field names.
	fieldSet := map[string]struct{}{}
	dimensionSet := map[string]struct{}{}

	// keep track of where the wildcards are in the select statement
	hasFieldWildcard := stmt.HasFieldWildcard()
	hasDimensionWildcard := stmt.HasDimensionWildcard()

	// Iterate measurements in the FROM clause getting the fields & dimensions for each.
	var fields influxql.Fields
	var dimensions influxql.Dimensions
	for _, src := range stmt.Sources {
		if m, ok := src.(*influxql.Measurement); ok {
			// Lookup the measurement in the database.
			mm := db.Measurement(m.Name)
			if mm == nil {
				// This shard have never received data for the measurement. No Mapper
				// required.
				return stmt, nil
			}

			// Get the fields for this measurement.
			for _, name := range mm.FieldNames() {
				if _, ok := fieldSet[name]; ok {
					continue
				}
				fieldSet[name] = struct{}{}
				fields = append(fields, &influxql.Field{Expr: &influxql.VarRef{Val: name}})
			}

			// Add tags to fields if a field wildcard was provided and a dimension wildcard was not.
			if hasFieldWildcard && !hasDimensionWildcard {
				for _, t := range mm.TagKeys() {
					if _, ok := fieldSet[t]; ok {
						continue
					}
					fieldSet[t] = struct{}{}
					fields = append(fields, &influxql.Field{Expr: &influxql.VarRef{Val: t}})
				}
			}

			// Get the dimensions for this measurement.
			if hasDimensionWildcard {
				for _, t := range mm.TagKeys() {
					if _, ok := dimensionSet[t]; ok {
						continue
					}
					dimensionSet[t] = struct{}{}
					dimensions = append(dimensions, &influxql.Dimension{Expr: &influxql.VarRef{Val: t}})
				}
			}
		}
	}

	// Return a new SelectStatement with the wild cards rewritten.
	return stmt.RewriteWildcards(fields, dimensions), nil
}

// expandSources expands regex sources and removes duplicates.
// NOTE: sources must be normalized (db and rp set) before calling this function.
func (di *DatabaseIndex) ExpandSources(sources influxql.Sources) (influxql.Sources, error) {
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
			// Get measurements from the database that match the regex.
			measurements := di.measurementsByRegex(src.Regex.Val)
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

// Measurement represents a collection of time series in a database. It also contains in memory
// structures for indexing tags. Exported functions are goroutine safe while un-exported functions
// assume the caller will use the appropriate locks
type Measurement struct {
	mu         sync.RWMutex
	Name       string `json:"name,omitempty"`
	fieldNames map[string]struct{}
	index      *DatabaseIndex

	// in-memory index fields
	seriesByID          map[uint64]*Series // lookup table for series by their id
	measurement         *Measurement
	seriesByTagKeyValue map[string]map[string]SeriesIDs // map from tag key to value to sorted set of series ids
	seriesIDs           SeriesIDs                       // sorted list of series IDs in this measurement
}

// NewMeasurement allocates and initializes a new Measurement.
func NewMeasurement(name string, idx *DatabaseIndex) *Measurement {
	return &Measurement{
		Name:       name,
		fieldNames: make(map[string]struct{}),
		index:      idx,

		seriesByID:          make(map[uint64]*Series),
		seriesByTagKeyValue: make(map[string]map[string]SeriesIDs),
		seriesIDs:           make(SeriesIDs, 0),
	}
}

// HasField returns true if the measurement has a field by the given name
func (m *Measurement) HasField(name string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, hasField := m.fieldNames[name]
	return hasField
}

// SeriesByID returns a series by identifier.
func (m *Measurement) SeriesByID(id uint64) *Series {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.seriesByID[id]
}

// SeriesKeys returns the keys of every series in this measurement
func (m *Measurement) SeriesKeys() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var keys []string
	for _, s := range m.seriesByID {
		keys = append(keys, s.Key)
	}
	return keys
}

// ValidateGroupBy ensures that the GROUP BY is not a field.
func (m *Measurement) ValidateGroupBy(stmt *influxql.SelectStatement) error {
	for _, d := range stmt.Dimensions {
		switch e := d.Expr.(type) {
		case *influxql.VarRef:
			if m.HasField(e.Val) {
				return fmt.Errorf("can not use field in GROUP BY clause: %s", e.Val)
			}
		}
	}
	return nil
}

// HasTagKey returns true if at least one series in this measurement has written a value for the passed in tag key
func (m *Measurement) HasTagKey(k string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, hasTag := m.seriesByTagKeyValue[k]
	return hasTag
}

// HasSeries returns true if there is at least 1 series under this measurement
func (m *Measurement) HasSeries() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.seriesByID) > 0
}

// AddSeries will add a series to the measurementIndex. Returns false if already present
func (m *Measurement) AddSeries(s *Series) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.seriesByID[s.id]; ok {
		return false
	}
	m.seriesByID[s.id] = s
	m.seriesIDs = append(m.seriesIDs, s.id)

	// the series ID should always be higher than all others because it's a new
	// series. So don't do the sort if we don't have to.
	if len(m.seriesIDs) > 1 && m.seriesIDs[len(m.seriesIDs)-1] < m.seriesIDs[len(m.seriesIDs)-2] {
		sort.Sort(m.seriesIDs)
	}

	// add this series id to the tag index on the measurement
	for k, v := range s.Tags {
		valueMap := m.seriesByTagKeyValue[k]
		if valueMap == nil {
			valueMap = make(map[string]SeriesIDs)
			m.seriesByTagKeyValue[k] = valueMap
		}
		ids := valueMap[v]
		ids = append(ids, s.id)

		// most of the time the series ID will be higher than all others because it's a new
		// series. So don't do the sort if we don't have to.
		if len(ids) > 1 && ids[len(ids)-1] < ids[len(ids)-2] {
			sort.Sort(ids)
		}
		valueMap[v] = ids
	}

	return true
}

// DropSeries will remove a series from the measurementIndex.
func (m *Measurement) DropSeries(seriesID uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.seriesByID[seriesID]; !ok {
		return
	}
	delete(m.seriesByID, seriesID)

	var ids []uint64
	for _, id := range m.seriesIDs {
		if id != seriesID {
			ids = append(ids, id)
		}
	}
	m.seriesIDs = ids

	// remove this series id to the tag index on the measurement
	// s.seriesByTagKeyValue is defined as map[string]map[string]SeriesIDs
	for k, v := range m.seriesByTagKeyValue {
		values := v
		for kk, vv := range values {
			var ids []uint64
			for _, id := range vv {
				if id != seriesID {
					ids = append(ids, id)
				}
			}
			// Check to see if we have any ids, if not, remove the key
			if len(ids) == 0 {
				delete(values, kk)
			} else {
				values[kk] = ids
			}
		}
		// If we have no values, then we delete the key
		if len(values) == 0 {
			delete(m.seriesByTagKeyValue, k)
		} else {
			m.seriesByTagKeyValue[k] = values
		}
	}

	return
}

// filters walks the where clause of a select statement and returns a map with all series ids
// matching the where clause and any filter expression that should be applied to each
func (m *Measurement) filters(stmt *influxql.SelectStatement) (map[uint64]influxql.Expr, error) {
	if stmt.Condition == nil || influxql.OnlyTimeExpr(stmt.Condition) {
		seriesIdsToExpr := make(map[uint64]influxql.Expr)
		for _, id := range m.seriesIDs {
			seriesIdsToExpr[id] = nil
		}
		return seriesIdsToExpr, nil
	}

	ids, seriesIdsToExpr, err := m.walkWhereForSeriesIds(stmt.Condition)
	if err != nil {
		return nil, err
	}
	// Ensure every id is in the map and replace literal true expressions with
	// nil so the engine doesn't waste time evaluating them.
	for _, id := range ids {
		if expr, ok := seriesIdsToExpr[id]; !ok {
			seriesIdsToExpr[id] = nil
		} else if b, ok := expr.(*influxql.BooleanLiteral); ok && b.Val {
			seriesIdsToExpr[id] = nil
		}
	}
	return seriesIdsToExpr, nil
}

// tagSets returns the unique tag sets that exist for the given tag keys. This is used to determine
// what composite series will be created by a group by. i.e. "group by region" should return:
// {"region":"uswest"}, {"region":"useast"}
// or region, service returns
// {"region": "uswest", "service": "redis"}, {"region": "uswest", "service": "mysql"}, etc...
// This will also populate the TagSet objects with the series IDs that match each tagset and any
// influx filter expression that goes with the series
// TODO: this shouldn't be exported. However, until tx.go and the engine get refactored into tsdb, we need it.
func (m *Measurement) TagSets(stmt *influxql.SelectStatement, dimensions []string) ([]*influxql.TagSet, error) {
	m.index.mu.RLock()
	defer m.index.mu.RUnlock()
	m.mu.RLock()
	defer m.mu.RUnlock()

	// get the unique set of series ids and the filters that should be applied to each
	filters, err := m.filters(stmt)
	if err != nil {
		return nil, err
	}

	// For every series, get the tag values for the requested tag keys i.e. dimensions. This is the
	// TagSet for that series. Series with the same TagSet are then grouped together, because for the
	// purpose of GROUP BY they are part of the same composite series.
	tagSets := make(map[string]*influxql.TagSet)
	for id, filter := range filters {
		s := m.seriesByID[id]
		tags := make(map[string]string)

		// Build the TagSet for this series.
		for _, dim := range dimensions {
			tags[dim] = s.Tags[dim]
		}

		// Convert the TagSet to a string, so it can be added to a map allowing TagSets to be handled
		// as a set.
		tagsAsKey := string(MarshalTags(tags))
		tagSet, ok := tagSets[tagsAsKey]
		if !ok {
			// This TagSet is new, create a new entry for it.
			tagSet = &influxql.TagSet{}
			tagsForSet := make(map[string]string)
			for k, v := range tags {
				tagsForSet[k] = v
			}
			tagSet.Tags = tagsForSet
			tagSet.Key = MarshalTags(tagsForSet)
		}

		// Associate the series and filter with the Tagset.
		tagSet.AddFilter(m.seriesByID[id].Key, filter)

		// Ensure it's back in the map.
		tagSets[tagsAsKey] = tagSet
	}

	// The TagSets have been created, as a map of TagSets. Just send
	// the values back as a slice, sorting for consistency.
	sortedTagSetKeys := make([]string, 0, len(tagSets))
	for k, _ := range tagSets {
		sortedTagSetKeys = append(sortedTagSetKeys, k)
	}
	sort.Strings(sortedTagSetKeys)

	sortedTagsSets := make([]*influxql.TagSet, 0, len(sortedTagSetKeys))
	for _, k := range sortedTagSetKeys {
		sortedTagsSets = append(sortedTagsSets, tagSets[k])
	}

	return sortedTagsSets, nil
}

// mergeSeriesFilters merges two sets of filter expressions and culls series IDs.
func mergeSeriesFilters(op influxql.Token, ids SeriesIDs, lfilters, rfilters FilterExprs) (SeriesIDs, FilterExprs) {
	// Create a map to hold the final set of series filter expressions.
	filters := make(map[uint64]influxql.Expr, 0)
	// Resulting list of series IDs
	var series SeriesIDs

	// Combining logic:
	// +==========+==========+==========+=======================+=======================+
	// | operator |   LHS    |   RHS    |   intermediate expr   |     reduced filter    |
	// +==========+==========+==========+=======================+=======================+
	// |          | <nil>    | <r-expr> | true OR <r-expr>      | true                  |
	// |          |----------+----------+-----------------------+-----------------------+
	// | OR       | <l-expr> | <nil>    | <l-expr> OR true      | true                  |
	// |          |----------+----------+-----------------------+-----------------------+
	// |          | <nil>    | <nil>    | true OR true          | true                  |
	// |          |----------+----------+-----------------------+-----------------------+
	// |          | <l-expr> | <r-expr> | <l-expr> OR <r-expr>  | <l-expr> OR <r-expr>  |
	// +----------+----------+----------+-----------------------+-----------------------+
	// |          | <nil>    | <r-expr> | false AND <r-expr>    | false*                |
	// |          |----------+----------+-----------------------+-----------------------+
	// | AND      | <l-expr> | <nil>    | <l-expr> AND false    | false                 |
	// |          |----------+----------+-----------------------+-----------------------+
	// |          | <nil>    | <nil>    | false AND false       | false                 |
	// |          |----------+----------+-----------------------+-----------------------+
	// |          | <l-expr> | <r-expr> | <l-expr> AND <r-expr> | <l-expr> AND <r-expr> |
	// +----------+----------+----------+-----------------------+-----------------------+
	// *literal false filters and series IDs should be excluded from the results

	def := false
	if op == influxql.OR {
		def = true
	}

	for _, id := range ids {
		// Get LHS and RHS filter expressions for this series ID.
		lfilter, rfilter := lfilters[id], rfilters[id]

		// Set default filters if either LHS or RHS expressions were nil.
		if lfilter == nil {
			lfilter = &influxql.BooleanLiteral{Val: def}
		}
		if rfilter == nil {
			rfilter = &influxql.BooleanLiteral{Val: def}
		}

		// Create the intermediate filter expression for this series ID.
		be := &influxql.BinaryExpr{
			Op:  op,
			LHS: lfilter,
			RHS: rfilter,
		}

		// Reduce the intermediate expression.
		expr := influxql.Reduce(be, nil)

		// If the expression reduced to false, exclude this series ID and filter.
		if b, ok := expr.(*influxql.BooleanLiteral); ok && !b.Val {
			continue
		}

		// Store the series ID and merged filter in the final results.
		filters[id] = expr
		series = append(series, id)
	}
	return series, filters
}

// idsForExpr will return a collection of series ids and a filter expression that should
// be used to filter points from those series.
func (m *Measurement) idsForExpr(n *influxql.BinaryExpr) (SeriesIDs, influxql.Expr, error) {
	name, ok := n.LHS.(*influxql.VarRef)
	value := n.RHS
	if !ok {
		name, ok = n.RHS.(*influxql.VarRef)
		if !ok {
			return nil, nil, fmt.Errorf("invalid expression: %s", n.String())
		}
		value = n.LHS
	}

	// For time literals, return all series IDs and "true" as the filter.
	if _, ok := value.(*influxql.TimeLiteral); ok || name.Val == "time" {
		return m.seriesIDs, &influxql.BooleanLiteral{Val: true}, nil
	}

	// For fields, return all series IDs from this measurement and return
	// the expression passed in, as the filter.
	if m.HasField(name.Val) {
		return m.seriesIDs, n, nil
	}

	tagVals, ok := m.seriesByTagKeyValue[name.Val]
	if !ok {
		return nil, nil, nil
	}

	// if we're looking for series with a specific tag value
	if str, ok := value.(*influxql.StringLiteral); ok {
		var ids SeriesIDs

		if n.Op == influxql.EQ {
			// return series that have a tag of specific value.
			ids = tagVals[str.Val]
		} else if n.Op == influxql.NEQ {
			ids = m.seriesIDs.Reject(tagVals[str.Val])
		}
		return ids, &influxql.BooleanLiteral{Val: true}, nil
	}

	// if we're looking for series with a tag value that matches a regex
	if re, ok := value.(*influxql.RegexLiteral); ok {
		var ids SeriesIDs

		// The operation is a NEQREGEX, code must start by assuming all match, even
		// series without any tags.
		if n.Op == influxql.NEQREGEX {
			ids = m.seriesIDs
		}

		for k := range tagVals {
			match := re.Val.MatchString(k)

			if match && n.Op == influxql.EQREGEX {
				ids = ids.Union(tagVals[k])
			} else if match && n.Op == influxql.NEQREGEX {
				ids = ids.Reject(tagVals[k])
			}
		}
		return ids, &influxql.BooleanLiteral{Val: true}, nil
	}

	return nil, nil, nil
}

// FilterExprs represents a map of series IDs to filter expressions.
type FilterExprs map[uint64]influxql.Expr

// DeleteBoolLiteralTrues deletes all elements whose filter expression is a boolean literal true.
func (fe FilterExprs) DeleteBoolLiteralTrues() {
	for id, expr := range fe {
		if e, ok := expr.(*influxql.BooleanLiteral); ok && e.Val == true {
			delete(fe, id)
		}
	}
}

// Len returns the number of elements.
func (fe FilterExprs) Len() int {
	if fe == nil {
		return 0
	}
	return len(fe)
}

// walkWhereForSeriesIds recursively walks the WHERE clause and returns an ordered set of series IDs and
// a map from those series IDs to filter expressions that should be used to limit points returned in
// the final query result.
func (m *Measurement) walkWhereForSeriesIds(expr influxql.Expr) (SeriesIDs, FilterExprs, error) {
	switch n := expr.(type) {
	case *influxql.BinaryExpr:
		switch n.Op {
		case influxql.EQ, influxql.NEQ, influxql.LT, influxql.LTE, influxql.GT, influxql.GTE, influxql.EQREGEX, influxql.NEQREGEX:
			// Get the series IDs and filter expression for the tag or field comparison.
			ids, expr, err := m.idsForExpr(n)
			if err != nil {
				return nil, nil, err
			}

			filters := FilterExprs{}
			for _, id := range ids {
				filters[id] = expr
			}

			return ids, filters, nil
		case influxql.AND, influxql.OR:
			// Get the series IDs and filter expressions for the LHS.
			lids, lfilters, err := m.walkWhereForSeriesIds(n.LHS)
			if err != nil {
				return nil, nil, err
			}

			// Get the series IDs and filter expressions for the RHS.
			rids, rfilters, err := m.walkWhereForSeriesIds(n.RHS)
			if err != nil {
				return nil, nil, err
			}

			// Combine the series IDs from the LHS and RHS.
			var ids SeriesIDs
			switch n.Op {
			case influxql.AND:
				ids = lids.Intersect(rids)
			case influxql.OR:
				ids = lids.Union(rids)
			}

			// Merge the filter expressions for the LHS and RHS.
			ids, filters := mergeSeriesFilters(n.Op, ids, lfilters, rfilters)

			return ids, filters, nil
		}

		ids, _, err := m.idsForExpr(n)
		return ids, nil, err
	case *influxql.ParenExpr:
		// walk down the tree
		return m.walkWhereForSeriesIds(n.Expr)
	default:
		return nil, nil, nil
	}
}

// expandExpr returns a list of expressions expanded by all possible tag combinations.
func (m *Measurement) expandExpr(expr influxql.Expr) []tagSetExpr {
	// Retrieve list of unique values for each tag.
	valuesByTagKey := m.uniqueTagValues(expr)

	// Convert keys to slices.
	keys := make([]string, 0, len(valuesByTagKey))
	for key := range valuesByTagKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Order uniques by key.
	uniques := make([][]string, len(keys))
	for i, key := range keys {
		uniques[i] = valuesByTagKey[key]
	}

	// Reduce a condition for each combination of tag values.
	return expandExprWithValues(expr, keys, []tagExpr{}, uniques, 0)
}

func expandExprWithValues(expr influxql.Expr, keys []string, tagExprs []tagExpr, uniques [][]string, index int) []tagSetExpr {
	// If we have no more keys left then execute the reduction and return.
	if index == len(keys) {
		// Create a map of tag key/values.
		m := make(map[string]*string, len(keys))
		for i, key := range keys {
			if tagExprs[i].op == influxql.EQ {
				m[key] = &tagExprs[i].values[0]
			} else {
				m[key] = nil
			}
		}

		// TODO: Rewrite full expressions instead of VarRef replacement.

		// Reduce using the current tag key/value set.
		// Ignore it if reduces down to "false".
		e := influxql.Reduce(expr, &tagValuer{tags: m})
		if e, ok := e.(*influxql.BooleanLiteral); ok && e.Val == false {
			return nil
		}

		return []tagSetExpr{{values: copyTagExprs(tagExprs), expr: e}}
	}

	// Otherwise expand for each possible equality value of the key.
	var exprs []tagSetExpr
	for _, v := range uniques[index] {
		exprs = append(exprs, expandExprWithValues(expr, keys, append(tagExprs, tagExpr{keys[index], []string{v}, influxql.EQ}), uniques, index+1)...)
	}
	exprs = append(exprs, expandExprWithValues(expr, keys, append(tagExprs, tagExpr{keys[index], uniques[index], influxql.NEQ}), uniques, index+1)...)

	return exprs
}

// seriesIDsAllOrByExpr walks an expressions for matching series IDs
// or, if no expressions is given, returns all series IDs for the measurement.
func (m *Measurement) seriesIDsAllOrByExpr(expr influxql.Expr) (SeriesIDs, error) {
	// If no expression given or the measurement has no series,
	// we can take just return the ids or nil accordingly.
	if expr == nil {
		return m.seriesIDs, nil
	} else if len(m.seriesIDs) == 0 {
		return nil, nil
	}

	// Get series IDs that match the WHERE clause.
	ids, _, err := m.walkWhereForSeriesIds(expr)
	if err != nil {
		return nil, err
	}

	return ids, nil
}

// tagValuer is used during expression expansion to evaluate all sets of tag values.
type tagValuer struct {
	tags map[string]*string
}

// Value returns the string value of a tag and true if it's listed in the tagset.
func (v *tagValuer) Value(name string) (interface{}, bool) {
	if value, ok := v.tags[name]; ok {
		if value == nil {
			return nil, true
		}
		return *value, true
	}
	return nil, false
}

// tagSetExpr represents a set of tag keys/values and associated expression.
type tagSetExpr struct {
	values []tagExpr
	expr   influxql.Expr
}

// tagExpr represents one or more values assigned to a given tag.
type tagExpr struct {
	key    string
	values []string
	op     influxql.Token // EQ or NEQ
}

func copyTagExprs(a []tagExpr) []tagExpr {
	other := make([]tagExpr, len(a))
	copy(other, a)
	return other
}

// uniqueTagValues returns a list of unique tag values used in an expression.
func (m *Measurement) uniqueTagValues(expr influxql.Expr) map[string][]string {
	// Track unique value per tag.
	tags := make(map[string]map[string]struct{})

	// Find all tag values referenced in the expression.
	influxql.WalkFunc(expr, func(n influxql.Node) {
		switch n := n.(type) {
		case *influxql.BinaryExpr:
			// Ignore operators that are not equality.
			if n.Op != influxql.EQ {
				return
			}

			// Extract ref and string literal.
			var key, value string
			switch lhs := n.LHS.(type) {
			case *influxql.VarRef:
				if rhs, ok := n.RHS.(*influxql.StringLiteral); ok {
					key, value = lhs.Val, rhs.Val
				}
			case *influxql.StringLiteral:
				if rhs, ok := n.RHS.(*influxql.VarRef); ok {
					key, value = rhs.Val, lhs.Val
				}
			}
			if key == "" {
				return
			}

			// Add value to set.
			if tags[key] == nil {
				tags[key] = make(map[string]struct{})
			}
			tags[key][value] = struct{}{}
		}
	})

	// Convert to map of slices.
	out := make(map[string][]string)
	for k, values := range tags {
		out[k] = make([]string, 0, len(values))
		for v := range values {
			out[k] = append(out[k], v)
		}
		sort.Strings(out[k])
	}
	return out
}

// SelectFields returns a list of fields in the SELECT section of stmt.
func (m *Measurement) SelectFields(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInSelect() {
		if m.HasField(name) {
			set.add(name)
			continue
		}
	}
	return set.list()
}

// SelectTags returns a list of non-field tags in the SELECT section of stmt.
func (m *Measurement) SelectTags(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInSelect() {
		if !m.HasField(name) && m.HasTagKey(name) {
			set.add(name)
		}
	}
	return set.list()
}

// WhereFields returns a list of non-"time" fields in the WHERE section of stmt.
func (m *Measurement) WhereFields(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInWhere() {
		if name != "time" && m.HasField(name) {
			set.add(name)
		}
	}
	return set.list()
}

// DimensionTagSets returns list of tag sets from the GROUP BY section of stmt.
func (m *Measurement) DimensionTagSets(stmt *influxql.SelectStatement) ([]*influxql.TagSet, error) {
	_, tagKeys := stmt.Dimensions.Normalize()

	for _, n := range stmt.NamesInDimension() {
		if m.HasTagKey(n) {
			tagKeys = append(tagKeys, n)
		}
	}

	// Get the sorted unique tag sets for this statement.
	tagSets, err := m.TagSets(stmt, tagKeys)
	if err != nil {
		return nil, err
	}
	return tagSets, nil
}

type SelectInfo struct {
	SelectFields []string
	SelectTags   []string
	WhereFields  []string
}

// Measurements represents a list of *Measurement.
type Measurements []*Measurement

func (a Measurements) Len() int           { return len(a) }
func (a Measurements) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a Measurements) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// SelectFields returns a list of fields in the SELECT section of stmt.
func (a Measurements) SelectFields(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInSelect() {
		for _, m := range a {
			if m.HasField(name) {
				set.add(name)
			}
		}
	}
	return set.list()
}

// SelectTags returns a list of non-field tags in the SELECT section of stmt.
func (a Measurements) SelectTags(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInSelect() {
		for _, m := range a {
			if !m.HasField(name) && m.HasTagKey(name) {
				set.add(name)
			}
		}
	}
	return set.list()
}

// WhereFields returns a list of non-"time" fields in the WHERE section of stmt.
func (a Measurements) WhereFields(stmt *influxql.SelectStatement) []string {
	set := newStringSet()
	for _, name := range stmt.NamesInWhere() {
		for _, m := range a {
			if name != "time" && m.HasField(name) {
				set.add(name)
			}
		}
	}
	return set.list()
}

func (a Measurements) intersect(other Measurements) Measurements {
	l := a
	r := other

	// we want to iterate through the shortest one and stop
	if len(other) < len(a) {
		l = other
		r = a
	}

	// they're in sorted order so advance the counter as needed.
	// That is, don't run comparisons against lower values that we've already passed
	var i, j int

	result := make(Measurements, 0, len(l))
	for i < len(l) && j < len(r) {
		if l[i].Name == r[j].Name {
			result = append(result, l[i])
			i++
			j++
		} else if l[i].Name < r[j].Name {
			i++
		} else {
			j++
		}
	}

	return result
}

func (a Measurements) union(other Measurements) Measurements {
	result := make(Measurements, 0, len(a)+len(other))
	var i, j int
	for i < len(a) && j < len(other) {
		if a[i].Name == other[j].Name {
			result = append(result, a[i])
			i++
			j++
		} else if a[i].Name < other[j].Name {
			result = append(result, a[i])
			i++
		} else {
			result = append(result, other[j])
			j++
		}
	}

	// now append the remainder
	if i < len(a) {
		result = append(result, a[i:]...)
	} else if j < len(other) {
		result = append(result, other[j:]...)
	}

	return result
}

// Series belong to a Measurement and represent unique time series in a database
type Series struct {
	Key  string
	Tags map[string]string

	id          uint64
	measurement *Measurement
	shardIDs    map[uint64]bool // shards that have this series defined
}

// NewSeries returns an initialized series struct
func NewSeries(key string, tags map[string]string) *Series {
	return &Series{
		Key:      key,
		Tags:     tags,
		shardIDs: make(map[uint64]bool),
	}
}

// MarshalBinary encodes the object to a binary format.
func (s *Series) MarshalBinary() ([]byte, error) {
	var pb internal.Series
	pb.Key = &s.Key
	for k, v := range s.Tags {
		key := k
		value := v
		pb.Tags = append(pb.Tags, &internal.Tag{Key: &key, Value: &value})
	}
	return proto.Marshal(&pb)
}

// UnmarshalBinary decodes the object from a binary format.
func (s *Series) UnmarshalBinary(buf []byte) error {
	var pb internal.Series
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}
	s.Key = pb.GetKey()
	s.Tags = make(map[string]string)
	for _, t := range pb.Tags {
		s.Tags[t.GetKey()] = t.GetValue()
	}
	return nil
}

func (s *Series) InitializeShards() {
	s.shardIDs = make(map[uint64]bool)
}

// match returns true if all tags match the series' tags.
func (s *Series) match(tags map[string]string) bool {
	for k, v := range tags {
		if s.Tags[k] != v {
			return false
		}
	}
	return true
}

// SeriesIDs is a convenience type for sorting, checking equality, and doing
// union and intersection of collections of series ids.
type SeriesIDs []uint64

func (a SeriesIDs) Len() int           { return len(a) }
func (a SeriesIDs) Less(i, j int) bool { return a[i] < a[j] }
func (a SeriesIDs) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Equals assumes that both are sorted.
func (a SeriesIDs) Equals(other SeriesIDs) bool {
	if len(a) != len(other) {
		return false
	}
	for i, s := range other {
		if a[i] != s {
			return false
		}
	}
	return true
}

// Intersect returns a new collection of series ids in sorted order that is the intersection of the two.
// The two collections must already be sorted.
func (a SeriesIDs) Intersect(other SeriesIDs) SeriesIDs {
	l := a
	r := other

	// we want to iterate through the shortest one and stop
	if len(other) < len(a) {
		l = other
		r = a
	}

	// they're in sorted order so advance the counter as needed.
	// That is, don't run comparisons against lower values that we've already passed
	var i, j int

	ids := make([]uint64, 0, len(l))
	for i < len(l) && j < len(r) {
		if l[i] == r[j] {
			ids = append(ids, l[i])
			i++
			j++
		} else if l[i] < r[j] {
			i++
		} else {
			j++
		}
	}

	return SeriesIDs(ids)
}

// Union returns a new collection of series ids in sorted order that is the union of the two.
// The two collections must already be sorted.
func (a SeriesIDs) Union(other SeriesIDs) SeriesIDs {
	l := a
	r := other
	ids := make([]uint64, 0, len(l)+len(r))
	var i, j int
	for i < len(l) && j < len(r) {
		if l[i] == r[j] {
			ids = append(ids, l[i])
			i++
			j++
		} else if l[i] < r[j] {
			ids = append(ids, l[i])
			i++
		} else {
			ids = append(ids, r[j])
			j++
		}
	}

	// now append the remainder
	if i < len(l) {
		ids = append(ids, l[i:]...)
	} else if j < len(r) {
		ids = append(ids, r[j:]...)
	}

	return ids
}

// Reject returns a new collection of series ids in sorted order with the passed in set removed from the original.
// This is useful for the NOT operator. The two collections must already be sorted.
func (a SeriesIDs) Reject(other SeriesIDs) SeriesIDs {
	l := a
	r := other
	var i, j int

	ids := make([]uint64, 0, len(l))
	for i < len(l) && j < len(r) {
		if l[i] == r[j] {
			i++
			j++
		} else if l[i] < r[j] {
			ids = append(ids, l[i])
			i++
		} else {
			j++
		}
	}

	// Append the remainder
	if i < len(l) {
		ids = append(ids, l[i:]...)
	}

	return SeriesIDs(ids)
}

// TagFilter represents a tag filter when looking up other tags or measurements.
type TagFilter struct {
	Op    influxql.Token
	Key   string
	Value string
	Regex *regexp.Regexp
}

// used to convert the tag set to bytes for use as a lookup key
func MarshalTags(tags map[string]string) []byte {
	// Empty maps marshal to empty bytes.
	if len(tags) == 0 {
		return nil
	}

	// Extract keys and determine final size.
	sz := (len(tags) * 2) - 1 // separators
	keys := make([]string, 0, len(tags))
	for k, v := range tags {
		keys = append(keys, k)
		sz += len(k) + len(v)
	}
	sort.Strings(keys)

	// Generate marshaled bytes.
	b := make([]byte, sz)
	buf := b
	for _, k := range keys {
		copy(buf, k)
		buf[len(k)] = '|'
		buf = buf[len(k)+1:]
	}
	for i, k := range keys {
		v := tags[k]
		copy(buf, v)
		if i < len(keys)-1 {
			buf[len(v)] = '|'
			buf = buf[len(v)+1:]
		}
	}
	return b
}

// timeBetweenInclusive returns true if t is between min and max, inclusive.
func timeBetweenInclusive(t, min, max time.Time) bool {
	return (t.Equal(min) || t.After(min)) && (t.Equal(max) || t.Before(max))
}

// TagKeys returns a list of the measurement's tag names.
func (m *Measurement) TagKeys() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	keys := make([]string, 0, len(m.seriesByTagKeyValue))
	for k := range m.seriesByTagKeyValue {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TagValues returns all the values for the given tag key
func (m *Measurement) TagValues(key string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	values := []string{}
	for v := range m.seriesByTagKeyValue[key] {
		values = append(values, v)
	}
	return values
}

// SetFieldName adds the field name to the measurement.
func (m *Measurement) SetFieldName(name string) {
	m.mu.Lock()
	m.fieldNames[name] = struct{}{}
	m.mu.Unlock()
}

// FieldNames returns a list of the measurement's field names
func (m *Measurement) FieldNames() (a []string) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for n, _ := range m.fieldNames {
		a = append(a, n)
	}
	return
}

func (m *Measurement) tagValuesByKeyAndSeriesID(tagKeys []string, ids SeriesIDs) map[string]stringSet {
	// If no tag keys were passed, get all tag keys for the measurement.
	if len(tagKeys) == 0 {
		for k := range m.seriesByTagKeyValue {
			tagKeys = append(tagKeys, k)
		}
	}

	// Mapping between tag keys to all existing tag values.
	tagValues := make(map[string]stringSet, 0)

	// Iterate all series to collect tag values.
	for _, id := range ids {
		s, ok := m.seriesByID[id]
		if !ok {
			continue
		}

		// Iterate the tag keys we're interested in and collect values
		// from this series, if they exist.
		for _, tagKey := range tagKeys {
			if tagVal, ok := s.Tags[tagKey]; ok {
				if _, ok = tagValues[tagKey]; !ok {
					tagValues[tagKey] = newStringSet()
				}
				tagValues[tagKey].add(tagVal)
			}
		}
	}

	return tagValues
}

// stringSet represents a set of strings.
type stringSet map[string]struct{}

// newStringSet returns an empty stringSet.
func newStringSet() stringSet {
	return make(map[string]struct{})
}

// add adds strings to the set.
func (s stringSet) add(ss ...string) {
	for _, n := range ss {
		s[n] = struct{}{}
	}
}

// contains returns whether the set contains the given string.
func (s stringSet) contains(ss string) bool {
	_, ok := s[ss]
	return ok
}

// list returns the current elements in the set, in sorted order.
func (s stringSet) list() []string {
	l := make([]string, 0, len(s))
	for k := range s {
		l = append(l, k)
	}
	sort.Strings(l)
	return l
}

// union returns the union of this set and another.
func (s stringSet) union(o stringSet) stringSet {
	ns := newStringSet()
	for k := range s {
		ns[k] = struct{}{}
	}
	for k := range o {
		ns[k] = struct{}{}
	}
	return ns
}

// union returns the intersection of this set and another.
func (s stringSet) intersect(o stringSet) stringSet {
	ns := newStringSet()
	for k := range s {
		if _, ok := o[k]; ok {
			ns[k] = struct{}{}
		}
	}
	for k := range o {
		if _, ok := s[k]; ok {
			ns[k] = struct{}{}
		}
	}
	return ns
}

func MeasurementFromSeriesKey(key string) string {
	idx := strings.Index(key, ",")
	if idx == -1 {
		return key
	}
	return key[:strings.Index(key, ",")]
}
