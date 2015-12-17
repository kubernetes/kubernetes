package tsdb

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
)

const (
	// Return an error if the user is trying to select more than this number of points in a group by statement.
	// Most likely they specified a group by interval without time boundaries.
	MaxGroupByPoints = 100000

	// Since time is always selected, the column count when selecting only a single other value will be 2
	SelectColumnCountWithOneValue = 2

	// IgnoredChunkSize is what gets passed into Mapper.Begin for aggregate queries as they don't chunk points out
	IgnoredChunkSize = 0
)

type RawExecutor struct {
	stmt           *influxql.SelectStatement
	mappers        []*StatefulMapper
	chunkSize      int
	limitedTagSets map[string]struct{} // Set tagsets for which data has reached the LIMIT.
}

// NewRawExecutor returns a new RawExecutor.
func NewRawExecutor(stmt *influxql.SelectStatement, mappers []Mapper, chunkSize int) *RawExecutor {
	a := []*StatefulMapper{}
	for _, m := range mappers {
		a = append(a, &StatefulMapper{m, nil, false})
	}
	return &RawExecutor{
		stmt:           stmt,
		mappers:        a,
		chunkSize:      chunkSize,
		limitedTagSets: make(map[string]struct{}),
	}
}

// Close closes the executor such that all resources are released.
// Once closed, an executor may not be re-used.
func (e *RawExecutor) close() {
	if e != nil {
		for _, m := range e.mappers {
			m.Close()
		}
	}
}

// Execute begins execution of the query and returns a channel to receive rows.
func (e *RawExecutor) Execute(closing <-chan struct{}) <-chan *models.Row {
	out := make(chan *models.Row, 0)
	go e.execute(out, closing)
	return out
}

func (e *RawExecutor) execute(out chan *models.Row, closing <-chan struct{}) {
	// It's important that all resources are released when execution completes.
	defer e.close()

	// Open the mappers.
	for _, m := range e.mappers {
		if err := m.Open(); err != nil {
			out <- &models.Row{Err: err}
			return
		}
	}

	// Get the distinct fields across all mappers.
	var selectFields, aliasFields []string
	if e.stmt.HasWildcard() {
		sf := newStringSet()
		for _, m := range e.mappers {
			sf.add(m.Fields()...)
		}
		selectFields = sf.list()
		aliasFields = selectFields
	} else {
		selectFields = e.stmt.Fields.Names()
		aliasFields = e.stmt.Fields.AliasNames()
	}

	// Used to read ahead chunks from mappers.
	var rowWriter *limitedRowWriter
	var currTagset string

	// Keep looping until all mappers drained.
	var err error
	for {
		// Get the next chunk from each Mapper.
		for _, m := range e.mappers {
			if m.drained {
				continue
			}

			// Set the next buffered chunk on the mapper, or mark it drained.
			for {
				if m.bufferedChunk == nil {
					m.bufferedChunk, err = m.NextChunk()
					if err != nil {
						out <- &models.Row{Err: err}
						return
					}
					if m.bufferedChunk == nil {
						// Mapper can do no more for us.
						m.drained = true
						break
					}

					// If the SELECT query is on more than 1 field, but the chunks values from the Mappers
					// only contain a single value, create k-v pairs using the field name of the chunk
					// and the value of the chunk. If there is only 1 SELECT field across all mappers then
					// there is no need to create k-v pairs, and there is no need to distinguish field data,
					// as it is all for the *same* field.
					if len(selectFields) > 1 && len(m.bufferedChunk.Fields) == 1 {
						fieldKey := m.bufferedChunk.Fields[0]

						for i := range m.bufferedChunk.Values {
							field := map[string]interface{}{fieldKey: m.bufferedChunk.Values[i].Value}
							m.bufferedChunk.Values[i].Value = field
						}
					}
				}

				if e.tagSetIsLimited(m.bufferedChunk.Name) {
					// chunk's tagset is limited, so no good. Try again.
					m.bufferedChunk = nil
					continue
				}
				// This mapper has a chunk available, and it is not limited.
				break
			}
		}

		// All Mappers done?
		if e.mappersDrained() {
			rowWriter.Flush()
			break
		}

		// Send out data for the next alphabetically-lowest tagset. All Mappers emit data in this order,
		// so by always continuing with the lowest tagset until it is finished, we process all data in
		// the required order, and don't "miss" any.
		tagset := e.nextMapperTagSet()
		if tagset != currTagset {
			currTagset = tagset
			// Tagset has changed, time for a new rowWriter. Be sure to kick out any residual values.
			rowWriter.Flush()
			rowWriter = nil
		}

		ascending := true
		if len(e.stmt.SortFields) > 0 {
			ascending = e.stmt.SortFields[0].Ascending
		}

		var timeBoundary int64

		if ascending {
			// Process the mapper outputs. We can send out everything up to the min of the last time
			// of the chunks for the next tagset.
			timeBoundary = e.nextMapperLowestTime(tagset)
		} else {
			timeBoundary = e.nextMapperHighestTime(tagset)
		}

		// Now empty out all the chunks up to the min time. Create new output struct for this data.
		var chunkedOutput *MapperOutput
		for _, m := range e.mappers {
			if m.drained {
				continue
			}

			chunkBoundary := false
			if ascending {
				chunkBoundary = m.bufferedChunk.Values[0].Time > timeBoundary
			} else {
				chunkBoundary = m.bufferedChunk.Values[0].Time < timeBoundary
			}

			// This mapper's next chunk is not for the next tagset, or the very first value of
			// the chunk is at a higher acceptable timestamp. Skip it.
			if m.bufferedChunk.key() != tagset || chunkBoundary {
				continue
			}

			// Find the index of the point up to the min.
			ind := len(m.bufferedChunk.Values)
			for i, mo := range m.bufferedChunk.Values {
				if ascending && mo.Time > timeBoundary {
					ind = i
					break
				} else if !ascending && mo.Time < timeBoundary {
					ind = i
					break
				}

			}

			// Add up to the index to the values
			if chunkedOutput == nil {
				chunkedOutput = &MapperOutput{
					Name:      m.bufferedChunk.Name,
					Tags:      m.bufferedChunk.Tags,
					CursorKey: m.bufferedChunk.key(),
				}
				chunkedOutput.Values = m.bufferedChunk.Values[:ind]
			} else {
				chunkedOutput.Values = append(chunkedOutput.Values, m.bufferedChunk.Values[:ind]...)
			}

			// Clear out the values being sent out, keep the remainder.
			m.bufferedChunk.Values = m.bufferedChunk.Values[ind:]

			// If we emptied out all the values, clear the mapper's buffered chunk.
			if len(m.bufferedChunk.Values) == 0 {
				m.bufferedChunk = nil
			}
		}

		if ascending {
			// Sort the values by time first so we can then handle offset and limit
			sort.Sort(MapperValues(chunkedOutput.Values))
		} else {
			sort.Sort(sort.Reverse(MapperValues(chunkedOutput.Values)))
		}

		// Now that we have full name and tag details, initialize the rowWriter.
		// The Name and Tags will be the same for all mappers.
		if rowWriter == nil {
			rowWriter = &limitedRowWriter{
				limit:       e.stmt.Limit,
				offset:      e.stmt.Offset,
				chunkSize:   e.chunkSize,
				name:        chunkedOutput.Name,
				tags:        chunkedOutput.Tags,
				selectNames: selectFields,
				aliasNames:  aliasFields,
				fields:      e.stmt.Fields,
				c:           out,
			}
		}
		if e.stmt.HasDerivative() {
			interval, err := derivativeInterval(e.stmt)
			if err != nil {
				out <- &models.Row{Err: err}
				return
			}
			rowWriter.transformer = &RawQueryDerivativeProcessor{
				IsNonNegative:      e.stmt.FunctionCalls()[0].Name == "non_negative_derivative",
				DerivativeInterval: interval,
			}
		}

		// Emit the data via the limiter.
		if limited := rowWriter.Add(chunkedOutput.Values); limited {
			// Limit for this tagset was reached, mark it and start draining a new tagset.
			e.limitTagSet(chunkedOutput.key())
			continue
		}
		// Check to see if our client disconnected, or it has been to long since
		// we were asked for data...
		select {
		case <-closing:
			out <- &models.Row{Err: fmt.Errorf("execute was closed by caller")}
			break
		default:
			// do nothing
		}
	}

	close(out)
}

// mappersDrained returns whether all the executors Mappers have been drained of data.
func (e *RawExecutor) mappersDrained() bool {
	for _, m := range e.mappers {
		if !m.drained {
			return false
		}
	}
	return true
}

// nextMapperTagset returns the alphabetically lowest tagset across all Mappers.
func (e *RawExecutor) nextMapperTagSet() string {
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

// nextMapperLowestTime returns the lowest minimum time across all Mappers, for the given tagset.
func (e *RawExecutor) nextMapperLowestTime(tagset string) int64 {
	minTime := int64(math.MaxInt64)
	for _, m := range e.mappers {
		if !m.drained && m.bufferedChunk != nil {
			if m.bufferedChunk.key() != tagset {
				continue
			}
			t := m.bufferedChunk.Values[len(m.bufferedChunk.Values)-1].Time
			if t < minTime {
				minTime = t
			}
		}
	}
	return minTime
}

// nextMapperHighestTime returns the highest time across all Mappers, for the given tagset.
func (e *RawExecutor) nextMapperHighestTime(tagset string) int64 {
	maxTime := int64(math.MinInt64)
	for _, m := range e.mappers {
		if !m.drained && m.bufferedChunk != nil {
			if m.bufferedChunk.key() != tagset {
				continue
			}
			t := m.bufferedChunk.Values[0].Time
			if t > maxTime {
				maxTime = t
			}
		}
	}
	return maxTime
}

// tagSetIsLimited returns whether data for the given tagset has been LIMITed.
func (e *RawExecutor) tagSetIsLimited(tagset string) bool {
	_, ok := e.limitedTagSets[tagset]
	return ok
}

// limitTagSet marks the given taset as LIMITed.
func (e *RawExecutor) limitTagSet(tagset string) {
	e.limitedTagSets[tagset] = struct{}{}
}

// limitedRowWriter accepts raw mapper values, and will emit those values as rows in chunks
// of the given size. If the chunk size is 0, no chunking will be performed. In addition if
// limit is reached, outstanding values will be emitted. If limit is zero, no limit is enforced.
type limitedRowWriter struct {
	chunkSize   int
	limit       int
	offset      int
	name        string
	tags        map[string]string
	fields      influxql.Fields
	selectNames []string
	aliasNames  []string
	c           chan *models.Row

	currValues  []*MapperValue
	totalOffSet int
	totalSent   int

	transformer interface {
		Process(input []*MapperValue) []*MapperValue
	}
}

// Add accepts a slice of values, and will emit those values as per chunking requirements.
// If limited is returned as true, the limit was also reached and no more values should be
// added. In that case only up the limit of values are emitted.
func (r *limitedRowWriter) Add(values []*MapperValue) (limited bool) {
	if r.currValues == nil {
		r.currValues = make([]*MapperValue, 0, r.chunkSize)
	}

	// Enforce offset.
	if r.totalOffSet < r.offset {
		// Still some offsetting to do.
		offsetRequired := r.offset - r.totalOffSet
		if offsetRequired >= len(values) {
			r.totalOffSet += len(values)
			return false
		} else {
			// Drop leading values and keep going.
			values = values[offsetRequired:]
			r.totalOffSet += offsetRequired
		}
	}
	r.currValues = append(r.currValues, values...)

	// Check limit.
	limitReached := r.limit > 0 && r.totalSent+len(r.currValues) >= r.limit
	if limitReached {
		// Limit will be satified with current values. Truncate 'em.
		r.currValues = r.currValues[:r.limit-r.totalSent]
	}

	// Is chunking in effect?
	if r.chunkSize != IgnoredChunkSize {
		// Chunking level reached?
		for len(r.currValues) >= r.chunkSize {
			index := len(r.currValues) - (len(r.currValues) - r.chunkSize)
			r.c <- r.processValues(r.currValues[:index])
			r.currValues = r.currValues[index:]
		}

		// After values have been sent out by chunking, there may still be some
		// values left, if the remainder is less than the chunk size. But if the
		// limit has been reached, kick them out.
		if len(r.currValues) > 0 && limitReached {
			r.c <- r.processValues(r.currValues)
			r.currValues = nil
		}
	} else if limitReached {
		// No chunking in effect, but the limit has been reached.
		r.c <- r.processValues(r.currValues)
		r.currValues = nil
	}

	return limitReached
}

// Flush instructs the limitedRowWriter to emit any pending values as a single row,
// adhering to any limits. Chunking is not enforced.
func (r *limitedRowWriter) Flush() {
	if r == nil {
		return
	}

	// If at least some rows were sent, and no values are pending, then don't
	// emit anything, since at least 1 row was previously emitted. This ensures
	// that if no rows were ever sent, at least 1 will be emitted, even an empty row.
	if r.totalSent != 0 && len(r.currValues) == 0 {
		return
	}

	if r.limit > 0 && len(r.currValues) > r.limit {
		r.currValues = r.currValues[:r.limit]
	}
	r.c <- r.processValues(r.currValues)
	r.currValues = nil
}

// processValues emits the given values in a single row.
func (r *limitedRowWriter) processValues(values []*MapperValue) *models.Row {
	defer func() {
		r.totalSent += len(values)
	}()

	selectNames := r.selectNames
	aliasNames := r.aliasNames

	if r.transformer != nil {
		values = r.transformer.Process(values)
	}

	// ensure that time is in the select names and in the first position
	hasTime := false
	for i, n := range selectNames {
		if n == "time" {
			// Swap time to the first argument for names
			if i != 0 {
				selectNames[0], selectNames[i] = selectNames[i], selectNames[0]
			}
			hasTime = true
			break
		}
	}

	// time should always be in the list of names they get back
	if !hasTime {
		selectNames = append([]string{"time"}, selectNames...)
		aliasNames = append([]string{"time"}, aliasNames...)
	}

	// since selectNames can contain tags, we need to strip them out
	selectFields := make([]string, 0, len(selectNames))
	aliasFields := make([]string, 0, len(selectNames))

	for _, n := range selectNames {
		if _, found := r.tags[n]; !found {
			selectFields = append(selectFields, n)
		}
	}
	for i, n := range aliasNames {
		if _, found := r.tags[n]; !found {
			aliasFields = append(aliasFields, aliasNames[i])
		}
	}

	row := &models.Row{
		Name:    r.name,
		Tags:    r.tags,
		Columns: aliasFields,
	}

	// Kick out an empty row it no results available.
	if len(values) == 0 {
		return row
	}

	// if they've selected only a single value we have to handle things a little differently
	singleValue := len(selectFields) == SelectColumnCountWithOneValue

	// the results will have all of the raw mapper results, convert into the row
	for _, v := range values {
		vals := make([]interface{}, len(selectFields))

		if singleValue {
			vals[0] = time.Unix(0, v.Time).UTC()
			switch val := v.Value.(type) {
			case map[string]interface{}:
				vals[1] = val[selectFields[1]]
			default:
				vals[1] = val
			}
		} else {
			fields := v.Value.(map[string]interface{})

			// time is always the first value
			vals[0] = time.Unix(0, v.Time).UTC()

			// populate the other values
			for i := 1; i < len(selectFields); i++ {
				f, ok := fields[selectFields[i]]
				if ok {
					vals[i] = f
					continue
				}
				if v.Tags != nil {
					f, ok = v.Tags[selectFields[i]]
					if ok {
						vals[i] = f
					}
				}
			}
		}

		row.Values = append(row.Values, vals)
	}

	// Perform any mathematical post-processing.
	row.Values = processForMath(r.fields, row.Values)

	return row
}

type RawQueryDerivativeProcessor struct {
	LastValueFromPreviousChunk *MapperValue
	IsNonNegative              bool // Whether to drop negative differences
	DerivativeInterval         time.Duration
}

func (rqdp *RawQueryDerivativeProcessor) canProcess(input *MapperValue) bool {
	// Cannot process a nil value
	if input == nil {
		return false
	}

	// See if the field value is numeric, if it's not, we can't process the derivative
	validType := false
	switch input.Value.(type) {
	case int64:
		validType = true
	case float64:
		validType = true
	}

	return validType
}

func (rqdp *RawQueryDerivativeProcessor) Process(input []*MapperValue) []*MapperValue {
	if len(input) == 0 {
		return input
	}

	if len(input) == 1 {
		return []*MapperValue{
			&MapperValue{
				Time:  input[0].Time,
				Value: 0.0,
			},
		}
	}

	if rqdp.LastValueFromPreviousChunk == nil {
		rqdp.LastValueFromPreviousChunk = input[0]
	}

	derivativeValues := []*MapperValue{}
	for i := 1; i < len(input); i++ {
		v := input[i]

		// If we can't use the current or prev value (wrong time, nil), just append
		// nil
		if !rqdp.canProcess(v) || !rqdp.canProcess(rqdp.LastValueFromPreviousChunk) {
			derivativeValues = append(derivativeValues, &MapperValue{
				Time:  v.Time,
				Value: nil,
			})
			continue
		}

		// Calculate the derivative of successive points by dividing the difference
		// of each value by the elapsed time normalized to the interval
		diff := int64toFloat64(v.Value) - int64toFloat64(rqdp.LastValueFromPreviousChunk.Value)

		elapsed := v.Time - rqdp.LastValueFromPreviousChunk.Time

		value := 0.0
		if elapsed > 0 {
			value = diff / (float64(elapsed) / float64(rqdp.DerivativeInterval))
		}

		rqdp.LastValueFromPreviousChunk = v

		// Drop negative values for non-negative derivatives
		if rqdp.IsNonNegative && diff < 0 {
			continue
		}

		derivativeValues = append(derivativeValues, &MapperValue{
			Time:  v.Time,
			Value: value,
		})
	}

	return derivativeValues
}

// processForMath will apply any math that was specified in the select statement
// against the passed in results
func processForMath(fields influxql.Fields, results [][]interface{}) [][]interface{} {
	hasMath := false
	for _, f := range fields {
		if _, ok := f.Expr.(*influxql.BinaryExpr); ok {
			hasMath = true
		} else if _, ok := f.Expr.(*influxql.ParenExpr); ok {
			hasMath = true
		}
	}

	if !hasMath {
		return results
	}

	processors := make([]influxql.Processor, len(fields))
	startIndex := 1
	for i, f := range fields {
		processors[i], startIndex = influxql.GetProcessor(f.Expr, startIndex)
	}

	mathResults := make([][]interface{}, len(results))
	for i, _ := range mathResults {
		mathResults[i] = make([]interface{}, len(fields)+1)
		// put the time in
		mathResults[i][0] = results[i][0]
		for j, p := range processors {
			mathResults[i][j+1] = p(results[i])
		}
	}

	return mathResults
}

// ProcessAggregateDerivative returns the derivatives of an aggregate result set
func ProcessAggregateDerivative(results [][]interface{}, isNonNegative bool, interval time.Duration) [][]interface{} {
	// Return early if we can't calculate derivatives
	if len(results) == 0 {
		return results
	}

	// If we only have 1 value, then the value did not change, so return
	// a single row w/ 0.0
	if len(results) == 1 {
		return [][]interface{}{
			[]interface{}{results[0][0], 0.0},
		}
	}

	// Otherwise calculate the derivatives as the difference between consecutive
	// points divided by the elapsed time.  Then normalize to the requested
	// interval.
	derivatives := [][]interface{}{}
	for i := 1; i < len(results); i++ {
		prev := results[i-1]
		cur := results[i]

		// If current value is nil, append nil for the value
		if prev[1] == nil || cur[1] == nil {
			derivatives = append(derivatives, []interface{}{
				cur[0], nil,
			})
			continue
		}

		// Check the value's type to ensure it's an numeric, if not, return a nil result. We only check the first value
		// because derivatives cannot be combined with other aggregates currently.
		prevValue, prevOK := toFloat64(prev[1])
		curValue, curOK := toFloat64(cur[1])
		if !prevOK || !curOK {
			derivatives = append(derivatives, []interface{}{
				cur[0], nil,
			})
			continue
		}

		elapsed := cur[0].(time.Time).Sub(prev[0].(time.Time))
		diff := curValue - prevValue
		value := 0.0
		if elapsed > 0 {
			value = float64(diff) / (float64(elapsed) / float64(interval))
		}

		// Drop negative values for non-negative derivatives
		if isNonNegative && diff < 0 {
			continue
		}

		val := []interface{}{
			cur[0],
			value,
		}
		derivatives = append(derivatives, val)
	}

	return derivatives
}

// derivativeInterval returns the time interval for the one (and only) derivative func
func derivativeInterval(stmt *influxql.SelectStatement) (time.Duration, error) {
	if len(stmt.FunctionCalls()[0].Args) == 2 {
		return stmt.FunctionCalls()[0].Args[1].(*influxql.DurationLiteral).Val, nil
	}
	interval, err := stmt.GroupByInterval()
	if err != nil {
		return 0, err
	}
	if interval > 0 {
		return interval, nil
	}
	return time.Second, nil
}

// resultsEmpty will return true if the all the result values are empty or contain only nulls
func resultsEmpty(resultValues [][]interface{}) bool {
	for _, vals := range resultValues {
		// start the loop at 1 because we want to skip over the time value
		for i := 1; i < len(vals); i++ {
			if vals[i] != nil {
				return false
			}
		}
	}
	return true
}

// Convert commonly understood types to a float64
// Valid types are int64, float64 or PositionPoint with a Value of int64 or float64
// The second retuned boolean indicates if the conversion was successful.
func toFloat64(v interface{}) (float64, bool) {
	switch value := v.(type) {
	case int64:
		return float64(value), true
	case float64:
		return value, true
	case PositionPoint:
		return toFloat64(value.Value)
	}
	return 0, false
}

func int64toFloat64(v interface{}) float64 {
	switch value := v.(type) {
	case int64:
		return float64(value)
	case float64:
		return value
	}
	panic(fmt.Sprintf("expected either int64 or float64, got %T", v))
}

// RawMapper runs the map phase for non-aggregate, raw SELECT queries.
type RawMapper struct {
	shard      *Shard
	stmt       *influxql.SelectStatement
	qmin, qmax int64 // query time range

	tx          Tx
	cursors     []*TagSetCursor
	cursorIndex int

	selectFields []string
	selectTags   []string
	whereFields  []string

	ChunkSize int
}

// NewRawMapper returns a new instance of RawMapper.
func NewRawMapper(sh *Shard, stmt *influxql.SelectStatement) *RawMapper {
	return &RawMapper{
		shard: sh,
		stmt:  stmt,
	}
}

// Open opens and initializes the mapper.
func (m *RawMapper) Open() error {
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

	// Remove cursors if there are not SELECT fields.
	if len(m.selectFields) == 0 {
		m.cursors = nil
	}

	return nil
}

func (m *RawMapper) openMeasurement(mm *Measurement) error {
	// Validate that ANY GROUP BY is not a field for the measurement.
	if err := mm.ValidateGroupBy(m.stmt); err != nil {
		return err
	}

	// Validate the fields and tags asked for exist and keep track of which are in the select vs the where
	selectFields := mm.SelectFields(m.stmt)
	selectTags := mm.SelectTags(m.stmt)
	fields := uniqueStrings(m.selectFields, m.whereFields)

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
	ascending := m.stmt.TimeAscending()
	for _, t := range tagSets {
		cursors := []*TagsCursor{}

		for i, key := range t.SeriesKeys {
			c := m.tx.Cursor(key, fields, m.shard.FieldCodec(mm.Name), ascending)
			if c == nil {
				continue
			}

			seriesTags := m.shard.index.TagsForSeries(key)
			cm := NewTagsCursor(c, t.Filters[i], seriesTags)
			cursors = append(cursors, cm)
		}

		tsc := NewTagSetCursor(mm.Name, t.Tags, cursors, ascending)
		tsc.SelectFields = m.selectFields
		if ascending {
			tsc.Init(m.qmin)
		} else {
			tsc.Init(m.qmax)
		}

		m.cursors = append(m.cursors, tsc)
	}

	sort.Sort(TagSetCursors(m.cursors))

	return nil
}

// Close closes the mapper.
func (m *RawMapper) Close() {
	if m != nil && m.tx != nil {
		m.tx.Rollback()
	}
}

// TagSets returns the list of tag sets for which this mapper has data.
func (m *RawMapper) TagSets() []string { return TagSetCursors(m.cursors).Keys() }

// Fields returns all SELECT fields.
func (m *RawMapper) Fields() []string { return append(m.selectFields, m.selectTags...) }

// NextChunk returns the next chunk of data.
// Data is ordered the same as TagSets. Each chunk contains one tag set.
// If there is no more data for any tagset, nil will be returned.
func (m *RawMapper) NextChunk() (interface{}, error) {
	var output *MapperOutput
	for {
		// All tagset cursors processed. NextChunk'ing complete.
		if m.cursorIndex == len(m.cursors) {
			return nil, nil
		}

		cursor := m.cursors[m.cursorIndex]

		k, v := cursor.Next(m.qmin, m.qmax)
		if v == nil {
			// Tagset cursor is empty, move to next one.
			m.cursorIndex++
			if output != nil {
				// There is data, so return it and continue when next called.
				return output, nil
			} else {
				// Just go straight to the next cursor.
				continue
			}
		}

		if output == nil {
			output = &MapperOutput{
				Name:      cursor.measurement,
				Tags:      cursor.tags,
				Fields:    m.selectFields,
				CursorKey: cursor.key(),
			}
		}

		output.Values = append(output.Values, &MapperValue{
			Time:  k,
			Value: v,
			Tags:  cursor.Tags(),
		})

		if len(output.Values) == m.ChunkSize {
			return output, nil
		}
	}
}
