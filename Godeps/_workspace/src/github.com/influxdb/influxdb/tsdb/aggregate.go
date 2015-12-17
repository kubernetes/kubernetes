package tsdb

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/pkg/slices"
)

// AggregateExecutor represents a mapper for execute aggregate SELECT statements.
type AggregateExecutor struct {
	stmt    *influxql.SelectStatement
	mappers []*StatefulMapper
}

// NewAggregateExecutor returns a new AggregateExecutor.
func NewAggregateExecutor(stmt *influxql.SelectStatement, mappers []Mapper) *AggregateExecutor {
	e := &AggregateExecutor{
		stmt:    stmt,
		mappers: make([]*StatefulMapper, 0, len(mappers)),
	}

	for _, m := range mappers {
		e.mappers = append(e.mappers, &StatefulMapper{m, nil, false})
	}

	return e
}

// close closes the executor such that all resources are released.
// Once closed, an executor may not be re-used.
func (e *AggregateExecutor) close() {
	if e != nil {
		for _, m := range e.mappers {
			m.Close()
		}
	}
}

// Execute begins execution of the query and returns a channel to receive rows.
func (e *AggregateExecutor) Execute(closing <-chan struct{}) <-chan *models.Row {
	out := make(chan *models.Row, 0)
	go e.execute(out, closing)
	return out
}

func (e *AggregateExecutor) execute(out chan *models.Row, closing <-chan struct{}) {
	// It's important to close all resources when execution completes.
	defer e.close()

	// Create the functions which will reduce values from mappers for
	// a given interval. The function offsets within this slice match
	// the offsets within the value slices that are returned by the
	// mapper.
	reduceFuncs, err := e.initReduceFuncs()
	if err != nil {
		out <- &models.Row{Err: err}
		return
	}

	// Put together the rows to return, starting with columns.
	columnNames := e.stmt.ColumnNames()

	// Open the mappers.
	if err := e.openMappers(); err != nil {
		out <- &models.Row{Err: err}
		return
	}

	// Filter out empty sets if there are multiple tag sets.
	hasMultipleTagSets := e.hasMultipleTagSets()
	ascending := e.ascending()

	// Prime each mapper's chunk buffer.
	if err := e.initMappers(); err != nil {
		out <- &models.Row{Err: err}
		return
	}

	// Keep looping until all mappers drained.
	for !e.mappersDrained() {
		chunks, err := e.readNextTagset()
		if err != nil {
			out <- &models.Row{Err: err}
			return
		}

		// Prep a row, ready for kicking out.
		row := &models.Row{
			Name:    chunks[0].Name,
			Tags:    chunks[0].Tags,
			Columns: columnNames,
		}

		// Prep for bucketing data by start time of the interval.
		buckets := map[int64][][]interface{}{}

		var chunkValues []*MapperValue
		for _, chunk := range chunks {
			for _, chunkValue := range chunk.Values {
				chunkValues = append(chunkValues, chunkValue)
			}
		}
		sort.Sort(MapperValues(chunkValues))

		for _, chunkValue := range chunkValues {
			startTime := chunkValue.Time
			values := chunkValue.Value.([]interface{})

			if _, ok := buckets[startTime]; !ok {
				buckets[startTime] = make([][]interface{}, len(values))
			}
			for i, v := range values {
				buckets[startTime][i] = append(buckets[startTime][i], v)
			}
		}

		// Now, after the loop above, within each time bucket is a slice. Within the element of each
		// slice is another slice of interface{}, ready for passing to the reducer functions.

		// Work each bucket of time, in time ascending order.
		tMins := make(int64Slice, 0, len(buckets))
		for k, _ := range buckets {
			tMins = append(tMins, k)
		}

		if ascending {
			sort.Sort(tMins)
		} else {
			sort.Sort(sort.Reverse(tMins))
		}

		values := make([][]interface{}, len(tMins))
		for i, t := range tMins {
			values[i] = make([]interface{}, 0, len(columnNames))
			values[i] = append(values[i], time.Unix(0, t).UTC()) // Time value is always first.

			for j, f := range reduceFuncs {
				reducedVal := f(buckets[t][j])
				values[i] = append(values[i], reducedVal)
			}
		}

		// Perform aggregate unwraps
		values, err = e.processFunctions(values, columnNames)
		if err != nil {
			out <- &models.Row{Err: err}
		}

		// Perform any mathematics.
		values = processForMath(e.stmt.Fields, values)

		// Handle any fill options
		values = e.processFill(values)

		// process derivatives
		values = e.processDerivative(values)

		// If we have multiple tag sets we'll want to filter out the empty ones
		if hasMultipleTagSets && resultsEmpty(values) {
			continue
		}

		row.Values = values

		// Check to see if our client disconnected, or it has been to long since
		// we were asked for data...
		select {
		case out <- row:
		case <-closing:
			out <- &models.Row{Err: fmt.Errorf("execute was closed by caller")}
			break
		case <-time.After(30 * time.Second):
			// This should never happen, so if it does, it is a problem
			out <- &models.Row{Err: fmt.Errorf("execute was closed by read timeout")}
			break
		}
	}

	close(out)
}

// initReduceFuncs returns a list of reduce functions for the aggregates in the query.
func (e *AggregateExecutor) initReduceFuncs() ([]reduceFunc, error) {
	calls := e.stmt.FunctionCalls()
	fns := make([]reduceFunc, len(calls))
	for i, c := range calls {
		fn, err := initializeReduceFunc(c)
		if err != nil {
			return nil, err
		}
		fns[i] = fn
	}
	return fns, nil
}

// openMappers opens all the mappers.
func (e *AggregateExecutor) openMappers() error {
	for _, m := range e.mappers {
		if err := m.Open(); err != nil {
			return err
		}
	}
	return nil
}

// initMappers buffers the first chunk of each mapper.
func (e *AggregateExecutor) initMappers() error {
	for _, m := range e.mappers {
		chunk, err := m.NextChunk()
		if err != nil {
			return err
		}
		m.bufferedChunk = chunk

		if m.bufferedChunk == nil {
			m.drained = true
		}
	}
	return nil
}

// hasMultipleTagSets returns true if there is more than one tagset in the mappers.
func (e *AggregateExecutor) hasMultipleTagSets() bool {
	set := make(map[string]struct{})
	for _, m := range e.mappers {
		for _, t := range m.TagSets() {
			set[t] = struct{}{}
			if len(set) > 1 {
				return true
			}
		}
	}
	return false
}

// ascending returns true if statement is sorted in ascending order.
func (e *AggregateExecutor) ascending() bool {
	if len(e.stmt.SortFields) == 0 {
		return true
	}
	return e.stmt.SortFields[0].Ascending
}

// mappersDrained returns whether all the executors Mappers have been drained of data.
func (e *AggregateExecutor) mappersDrained() bool {
	for _, m := range e.mappers {
		if !m.drained {
			return false
		}
	}
	return true
}

// nextMapperTagset returns the alphabetically lowest tagset across all Mappers.
func (e *AggregateExecutor) nextMapperTagSet() string {
	tagset := ""
	for _, m := range e.mappers {
		if m.bufferedChunk != nil {
			if tagset == "" {
				tagset = m.bufferedChunk.key()
			} else if m.bufferedChunk.key() < tagset {
				tagset = m.bufferedChunk.key()
			}
		}
	}
	return tagset
}

// readNextTagset returns all chunks for the next tagset.
func (e *AggregateExecutor) readNextTagset() ([]*MapperOutput, error) {
	// Send out data for the next alphabetically-lowest tagset.
	// All Mappers send out in this order so collect data for this tagset, ignoring all others.
	tagset := e.nextMapperTagSet()
	chunks := []*MapperOutput{}

	// Pull as much as possible from each mapper. Stop when a mapper offers
	// data for a new tagset, or empties completely.
	for _, m := range e.mappers {
		if m.drained {
			continue
		}

		for {
			if m.bufferedChunk == nil {
				chunk, err := m.NextChunk()
				if err != nil {
					return nil, err
				}
				m.bufferedChunk = chunk

				if m.bufferedChunk == nil {
					m.drained = true
					break
				}
			}

			// Got a chunk. Can we use it?
			if m.bufferedChunk.key() != tagset {
				break // No, so just leave it in the buffer.
			}

			// We can, take it.
			chunks = append(chunks, m.bufferedChunk)
			m.bufferedChunk = nil
		}
	}

	return chunks, nil
}

// processFill will take the results and return new results (or the same if no fill modifications are needed)
// with whatever fill options the query has.
func (e *AggregateExecutor) processFill(results [][]interface{}) [][]interface{} {
	// don't do anything if we're supposed to leave the nulls
	if e.stmt.Fill == influxql.NullFill {
		return results
	}

	isCount := e.stmt.HasSimpleCount()
	if e.stmt.Fill == influxql.NoFill {
		// remove any rows that have even one nil value. This one is tricky because they could have multiple
		// aggregates, but this option means that any row that has even one nil gets purged.
		newResults := make([][]interface{}, 0, len(results))
		for _, vals := range results {
			hasNil := false
			// start at 1 because the first value is always time
			for j := 1; j < len(vals); j++ {
				if vals[j] == nil || (isCount && isZero(vals[j])) {
					hasNil = true
					break
				}
			}
			if !hasNil {
				newResults = append(newResults, vals)
			}
		}
		return newResults
	}

	// They're either filling with previous values or a specific number
	for i, vals := range results {
		// start at 1 because the first value is always time
		for j := 1; j < len(vals); j++ {
			if vals[j] == nil || (isCount && isZero(vals[j])) {
				switch e.stmt.Fill {
				case influxql.PreviousFill:
					if i != 0 {
						vals[j] = results[i-1][j]
					}
				case influxql.NumberFill:
					vals[j] = e.stmt.FillValue
				}
			}
		}
	}
	return results
}

// Returns true if the given interface is a zero valued int64 or float64.
func isZero(i interface{}) bool {
	switch v := i.(type) {
	case int64:
		return v == 0
	case float64:
		return v == 0
	default:
		return false
	}
}

// processDerivative returns the derivatives of the results
func (e *AggregateExecutor) processDerivative(results [][]interface{}) [][]interface{} {
	// Return early if we're not supposed to process the derivatives
	if e.stmt.HasDerivative() {
		interval, err := derivativeInterval(e.stmt)
		if err != nil {
			return results // XXX need to handle this better.
		}

		// Determines whether to drop negative differences
		isNonNegative := e.stmt.FunctionCalls()[0].Name == "non_negative_derivative"
		return ProcessAggregateDerivative(results, isNonNegative, interval)
	}
	return results
}

func (e *AggregateExecutor) processFunctions(results [][]interface{}, columnNames []string) ([][]interface{}, error) {
	callInPosition := e.stmt.FunctionCallsByPosition()
	hasTimeField := e.stmt.HasTimeFieldSpecified()

	flatCallInPositions := make([][]*influxql.Call, 0)
	for _, calls := range callInPosition {
		if calls == nil {
			flatCallInPositions = append(flatCallInPositions, calls)
		}
		for _, call := range calls {
			flatCallInPositions = append(flatCallInPositions, []*influxql.Call{call})
		}
	}

	var err error
	for i, calls := range flatCallInPositions {
		// We can only support expanding fields if a single selector call was specified
		// i.e. select tx, max(rx) from foo
		// If you have multiple selectors or aggregates, there is no way of knowing who gets to insert the values, so we don't
		// i.e. select tx, max(rx), min(rx) from foo
		if len(calls) == 1 {
			var c *influxql.Call
			c = calls[0]

			switch c.Name {
			case "top", "bottom":
				results, err = e.processAggregates(results, columnNames, c)
				if err != nil {
					return results, err
				}
			case "first", "last", "min", "max":
				results, err = e.processSelectors(results, i, hasTimeField, columnNames)
				if err != nil {
					return results, err
				}
			}
		}
	}

	return results, nil
}

func (e *AggregateExecutor) processSelectors(results [][]interface{}, callPosition int, hasTimeField bool, columnNames []string) ([][]interface{}, error) {
	// if the columns doesn't have enough columns, expand it
	for i, columns := range results {
		if len(columns) < len(columnNames) {
			columns = append(columns, make([]interface{}, len(columnNames)-len(columns))...)
		} else if len(columns) > len(columnNames) {
			columnNames = append(columnNames, make([]string, len(columns)-len(columnNames))...)
		}
		for j := 1; j < len(columns); j++ {
			switch v := columns[j].(type) {
			case PositionPoint:
				tMin := columns[0].(time.Time)
				results[i] = e.selectorPointToQueryResult(columns, hasTimeField, callPosition, v, tMin, columnNames)
			}
		}
	}
	return results, nil
}

func (e *AggregateExecutor) selectorPointToQueryResult(columns []interface{}, hasTimeField bool, columnIndex int, p PositionPoint, tMin time.Time, columnNames []string) []interface{} {
	callCount := len(e.stmt.FunctionCalls())
	if callCount == 1 {
		tm := time.Unix(0, p.Time).UTC()
		// If we didn't explicity ask for time, and we have a group by, then use TMIN for the time returned
		if len(e.stmt.Dimensions) > 0 && !hasTimeField {
			tm = tMin.UTC()
		}
		columns[0] = tm
	}

	for i, c := range columnNames {
		// skip over time, we already handled that above
		if i == 0 {
			continue
		}
		if (i == columnIndex && hasTimeField) || (i == columnIndex+1 && !hasTimeField) {
			// Check to see if we previously processed this column, if so, continue
			if _, ok := columns[i].(PositionPoint); !ok && columns[i] != nil {
				continue
			}
			columns[i] = p.Value
			continue
		}

		if callCount == 1 {
			// Always favor fields over tags if there is a name collision
			if t, ok := p.Fields[c]; ok {
				columns[i] = t
			} else if t, ok := p.Tags[c]; ok {
				// look in the tags for a value
				columns[i] = t
			}
		}
	}
	return columns
}

func (e *AggregateExecutor) processAggregates(results [][]interface{}, columnNames []string, call *influxql.Call) ([][]interface{}, error) {
	var values [][]interface{}

	// Check if we have a group by, if not, rewrite the entire result by flattening it out
	for _, vals := range results {
		// start at 1 because the first value is always time
		for j := 1; j < len(vals); j++ {
			switch v := vals[j].(type) {
			case PositionPoints:
				tMin := vals[0].(time.Time)
				for _, p := range v {
					result := e.aggregatePointToQueryResult(p, tMin, call, columnNames)
					values = append(values, result)
				}
			case nil:
				continue
			default:
				return nil, fmt.Errorf("unrechable code - processAggregates for type %T %v", v, v)
			}
		}
	}
	return values, nil
}

func (e *AggregateExecutor) aggregatePointToQueryResult(p PositionPoint, tMin time.Time, call *influxql.Call, columnNames []string) []interface{} {
	tm := time.Unix(0, p.Time).UTC()
	// If we didn't explicity ask for time, and we have a group by, then use TMIN for the time returned
	if len(e.stmt.Dimensions) > 0 && !e.stmt.HasTimeFieldSpecified() {
		tm = tMin.UTC()
	}
	vals := []interface{}{tm}
	for _, c := range columnNames {
		if c == call.Name {
			vals = append(vals, p.Value)
			continue
		}
		// TODO in the future fields will also be available to us.
		// we should always favor fields over tags if there is a name collision

		// look in the tags for a value
		if t, ok := p.Tags[c]; ok {
			vals = append(vals, t)
		}
	}
	return vals
}

// AggregateMapper runs the map phase for aggregate SELECT queries.
type AggregateMapper struct {
	shard      *Shard
	stmt       *influxql.SelectStatement
	qmin, qmax int64 // query time range

	tx          Tx
	cursors     []CursorSet
	cursorIndex int

	interval     int   // Current interval for which data is being fetched.
	intervalN    int   // Maximum number of intervals to return.
	intervalSize int64 // Size of each interval.
	qminWindow   int64 // Minimum time of the query floored to start of interval.

	mapFuncs   []mapFunc // The mapping functions.
	fieldNames []string  // the field name being read for mapping.

	selectFields []string
	selectTags   []string
	whereFields  []string
}

// NewAggregateMapper returns a new instance of AggregateMapper.
func NewAggregateMapper(sh *Shard, stmt *influxql.SelectStatement) *AggregateMapper {
	return &AggregateMapper{
		shard: sh,
		stmt:  stmt,
	}
}

// Open opens and initializes the mapper.
func (m *AggregateMapper) Open() error {
	// Ignore if node has the shard but hasn't written to it yet.
	if m.shard == nil {
		return nil
	}

	// Rewrite statement.
	stmt, err := m.shard.index.RewriteSelectStatement(m.stmt)
	if err != nil {
		return err
	}
	m.stmt = stmt

	// Set all time-related parameters on the mapper.
	m.qmin, m.qmax = influxql.TimeRangeAsEpochNano(m.stmt.Condition)

	if err := m.initializeMapFunctions(); err != nil {
		return err
	}

	// For GROUP BY time queries, limit the number of data points returned by the limit and offset
	d, err := m.stmt.GroupByInterval()
	if err != nil {
		return err
	}

	m.intervalSize = d.Nanoseconds()
	if m.qmin == 0 || m.intervalSize == 0 {
		m.intervalN = 1
		m.intervalSize = m.qmax - m.qmin
	} else {
		intervalTop := m.qmax/m.intervalSize*m.intervalSize + m.intervalSize
		intervalBottom := m.qmin / m.intervalSize * m.intervalSize
		m.intervalN = int((intervalTop - intervalBottom) / m.intervalSize)
	}

	if m.stmt.Limit > 0 || m.stmt.Offset > 0 {
		// ensure that the offset isn't higher than the number of points we'd get
		if m.stmt.Offset > m.intervalN {
			return nil
		}

		// Take the lesser of either the pre computed number of GROUP BY buckets that
		// will be in the result or the limit passed in by the user
		if m.stmt.Limit < m.intervalN {
			m.intervalN = m.stmt.Limit
		}
	}

	// If we are exceeding our MaxGroupByPoints error out
	if m.intervalN > MaxGroupByPoints {
		return errors.New("too many points in the group by interval. maybe you forgot to specify a where time clause?")
	}

	// Ensure that the start time for the results is on the start of the window.
	m.qminWindow = m.qmin
	if m.intervalSize > 0 && m.intervalN > 1 {
		m.qminWindow = m.qminWindow / m.intervalSize * m.intervalSize
	}

	// Get a read-only transaction.
	tx, err := m.shard.engine.Begin(false)
	if err != nil {
		return err
	}
	m.tx = tx

	// Collect measurements.
	mms := Measurements(m.shard.index.MeasurementsByName(m.stmt.SourceNames()))
	m.selectFields = mms.SelectFields(m.stmt)
	m.selectTags = mms.SelectTags(m.stmt)
	m.whereFields = mms.WhereFields(m.stmt)

	// Open cursors for each measurement.
	for _, mm := range mms {
		if err := m.openMeasurement(mm); err != nil {
			return err
		}
	}

	return nil
}

func (m *AggregateMapper) openMeasurement(mm *Measurement) error {
	// Validate that ANY GROUP BY is not a field for the measurement.
	if err := mm.ValidateGroupBy(m.stmt); err != nil {
		return err
	}

	// Validate the fields and tags asked for exist and keep track of which are in the select vs the where
	selectFields := mm.SelectFields(m.stmt)
	selectTags := mm.SelectTags(m.stmt)

	// If we only have tags in our select clause we just return
	if len(selectFields) == 0 && len(selectTags) > 0 {
		return fmt.Errorf("statement must have at least one field in select clause")
	}

	// Calculate tag sets and apply SLIMIT/SOFFSET.
	tagSets, err := mm.DimensionTagSets(m.stmt)
	if err != nil {
		return err
	}
	tagSets = m.stmt.LimitTagSets(tagSets)

	// Create all cursors for reading the data from this shard.
	for _, t := range tagSets {
		cursorSet := CursorSet{
			Measurement: mm.Name,
			Tags:        t.Tags,
		}
		if len(t.Tags) == 0 {
			cursorSet.Key = mm.Name
		} else {
			cursorSet.Key = strings.Join([]string{mm.Name, string(MarshalTags(t.Tags))}, "|")
		}

		for i, key := range t.SeriesKeys {
			fields := slices.Union(slices.Union(selectFields, m.fieldNames, false), m.whereFields, false)
			c := m.tx.Cursor(key, fields, m.shard.FieldCodec(mm.Name), true)
			if c == nil {
				continue
			}

			seriesTags := m.shard.index.TagsForSeries(key)
			cursorSet.Cursors = append(cursorSet.Cursors, NewTagsCursor(c, t.Filters[i], seriesTags))
		}

		// tsc.Init(m.qmin)
		m.cursors = append(m.cursors, cursorSet)
	}

	sort.Sort(CursorSets(m.cursors))

	return nil
}

// initializeMapFunctions initialize the mapping functions for the mapper.
func (m *AggregateMapper) initializeMapFunctions() error {
	// Set up each mapping function for this statement.
	aggregates := m.stmt.FunctionCalls()
	m.mapFuncs = make([]mapFunc, len(aggregates))
	m.fieldNames = make([]string, len(m.mapFuncs))

	for i, c := range aggregates {
		mfn, err := initializeMapFunc(c)
		if err != nil {
			return err
		}
		m.mapFuncs[i] = mfn

		// Check for calls like `derivative(mean(value), 1d)`
		var nested *influxql.Call = c
		if fn, ok := c.Args[0].(*influxql.Call); ok {
			nested = fn
		}
		switch lit := nested.Args[0].(type) {
		case *influxql.VarRef:
			m.fieldNames[i] = lit.Val
		case *influxql.Distinct:
			if c.Name != "count" {
				return fmt.Errorf("aggregate call didn't contain a field %s", c.String())
			}
			m.fieldNames[i] = lit.Val
		default:
			return fmt.Errorf("aggregate call didn't contain a field %s", c.String())
		}
	}

	return nil
}

// Close closes the mapper.
func (m *AggregateMapper) Close() {
	if m != nil && m.tx != nil {
		m.tx.Rollback()
	}
	return
}

// TagSets returns the list of tag sets for which this mapper has data.
func (m *AggregateMapper) TagSets() []string { return CursorSets(m.cursors).Keys() }

// Fields returns all SELECT fields.
func (m *AggregateMapper) Fields() []string { return append(m.selectFields, m.selectTags...) }

// NextChunk returns the next interval of data.
// Tagsets are always processed in the same order as AvailTagsSets().
// When there is no more data for any tagset nil is returned.
func (m *AggregateMapper) NextChunk() (interface{}, error) {
	var tmin, tmax int64
	for {
		// All tagset cursors processed. NextChunk'ing complete.
		if m.cursorIndex == len(m.cursors) {
			return nil, nil
		}

		// All intervals complete for this tagset. Move to the next tagset.
		tmin, tmax = m.nextInterval()
		if tmin < 0 {
			m.interval = 0
			m.cursorIndex++
			continue
		}
		break
	}

	// Prep the return data for this tagset.
	// This will hold data for a single interval for a single tagset.
	cursorSet := m.cursors[m.cursorIndex]
	output := &MapperOutput{
		Name:      cursorSet.Measurement,
		Tags:      cursorSet.Tags,
		Fields:    m.selectFields,
		CursorKey: cursorSet.Key,
	}

	// Always clamp tmin and tmax. This can happen as bucket-times are bucketed to the nearest
	// interval. This is necessary to grab the "partial" buckets at the beginning and end of the time range
	qmin, qmax := tmin, tmax
	if qmin < m.qmin {
		qmin = m.qmin
	}
	if qmax > m.qmax {
		qmax = m.qmax + 1
	}

	mapperValue := &MapperValue{
		Time:  tmin,
		Value: make([]interface{}, len(m.mapFuncs)),
	}

	for i := range m.mapFuncs {
		// Build a map input from the cursor.
		input := &MapInput{
			TMin:  -1,
			Items: readMapItems(cursorSet.Cursors, m.fieldNames[i], qmin, qmin, qmax),
		}

		if len(m.stmt.Dimensions) > 0 && !m.stmt.HasTimeFieldSpecified() {
			input.TMin = tmin
		}

		// Execute the map function which walks the entire interval, and aggregates the result.
		value := m.mapFuncs[i](input)
		if value == nil {
			continue
		}
		mapperValue.Value.([]interface{})[i] = value
	}
	output.Values = append(output.Values, mapperValue)

	return output, nil
}

func readMapItems(cursors []*TagsCursor, field string, seek, tmin, tmax int64) []MapItem {
	var items []MapItem

	for _, c := range cursors {
		seeked := false

		for {
			var timestamp int64
			var value interface{}

			if !seeked {
				timestamp, value = c.SeekTo(seek)
				seeked = true
			} else {
				timestamp, value = c.Next()
			}

			// We're done if the point is outside the query's time range [tmin:tmax).
			if timestamp != tmin && (timestamp < tmin || timestamp >= tmax) {
				break
			}

			// Convert values to fields map.
			fields, ok := value.(map[string]interface{})
			if !ok {
				fields = map[string]interface{}{"": value}
			}

			// Value didn't match, look for the next one.
			if value == nil {
				continue
			}

			// Filter value.
			if c.filter != nil {
				// Convert value to a map for filter evaluation.
				m, ok := value.(map[string]interface{})
				if !ok {
					m = map[string]interface{}{field: value}
				}

				// If filter fails then skip to the next value.
				if !influxql.EvalBool(c.filter, m) {
					continue
				}
			}

			// Filter out single field, if specified.
			if m, ok := value.(map[string]interface{}); ok {
				value = m[field]
			}
			if value == nil {
				continue
			}

			items = append(items, MapItem{
				Timestamp: timestamp,
				Value:     value,
				Fields:    fields,
				Tags:      c.tags,
			})
		}
	}
	sort.Sort(MapItems(items))

	return items
}

// nextInterval returns the next interval for which to return data.
// If start is less than 0 there are no more intervals.
func (m *AggregateMapper) nextInterval() (start, end int64) {
	t := m.qminWindow + int64(m.interval+m.stmt.Offset)*m.intervalSize

	// On to next interval.
	m.interval++
	if t > m.qmax || m.interval > m.intervalN {
		start, end = -1, 1
	} else {
		start, end = t, t+m.intervalSize
	}
	return
}

type CursorSet struct {
	Measurement string
	Tags        map[string]string
	Key         string
	Cursors     []*TagsCursor
}

// CursorSets represents a sortable slice of CursorSet.
type CursorSets []CursorSet

func (a CursorSets) Len() int           { return len(a) }
func (a CursorSets) Less(i, j int) bool { return a[i].Key < a[j].Key }
func (a CursorSets) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func (a CursorSets) Keys() []string {
	keys := make([]string, len(a))
	for i := range a {
		keys[i] = a[i].Key
	}
	sort.Strings(keys)
	return keys
}

type int64Slice []int64

func (a int64Slice) Len() int           { return len(a) }
func (a int64Slice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a int64Slice) Less(i, j int) bool { return a[i] < a[j] }
