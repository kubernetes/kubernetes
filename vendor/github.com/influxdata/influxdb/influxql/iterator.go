package influxql

import (
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/influxdata/influxdb/models"

	"github.com/gogo/protobuf/proto"
	internal "github.com/influxdata/influxdb/influxql/internal"
)

// ErrUnknownCall is returned when operating on an unknown function call.
var ErrUnknownCall = errors.New("unknown call")

const (
	// MinTime is used as the minimum time value when computing an unbounded range.
	// This time is one less than the MinNanoTime so that the first minimum
	// time can be used as a sentinel value to signify that it is the default
	// value rather than explicitly set by the user.
	MinTime = models.MinNanoTime - 1

	// MaxTime is used as the maximum time value when computing an unbounded range.
	// This time is 2262-04-11 23:47:16.854775806 +0000 UTC
	MaxTime = models.MaxNanoTime
)

// Iterator represents a generic interface for all Iterators.
// Most iterator operations are done on the typed sub-interfaces.
type Iterator interface {
	Stats() IteratorStats
	Close() error
}

// Iterators represents a list of iterators.
type Iterators []Iterator

// Stats returns the aggregation of all iterator stats.
func (a Iterators) Stats() IteratorStats {
	var stats IteratorStats
	for _, itr := range a {
		stats.Add(itr.Stats())
	}
	return stats
}

// Close closes all iterators.
func (a Iterators) Close() error {
	for _, itr := range a {
		itr.Close()
	}
	return nil
}

// filterNonNil returns a slice of iterators that removes all nil iterators.
func (a Iterators) filterNonNil() []Iterator {
	other := make([]Iterator, 0, len(a))
	for _, itr := range a {
		if itr == nil {
			continue
		}
		other = append(other, itr)
	}
	return other
}

// castType determines what type to cast the set of iterators to.
// An iterator type is chosen using this hierarchy:
//   float > integer > string > boolean
func (a Iterators) castType() DataType {
	if len(a) == 0 {
		return Unknown
	}

	typ := DataType(Boolean)
	for _, input := range a {
		switch input.(type) {
		case FloatIterator:
			// Once a float iterator is found, short circuit the end.
			return Float
		case IntegerIterator:
			if typ > Integer {
				typ = Integer
			}
		case StringIterator:
			if typ > String {
				typ = String
			}
		case BooleanIterator:
			// Boolean is the lowest type.
		}
	}
	return typ
}

// cast casts an array of iterators to a single type.
// Iterators that are not compatible or cannot be cast to the
// chosen iterator type are closed and dropped.
func (a Iterators) cast() interface{} {
	typ := a.castType()
	switch typ {
	case Float:
		return newFloatIterators(a)
	case Integer:
		return newIntegerIterators(a)
	case String:
		return newStringIterators(a)
	case Boolean:
		return newBooleanIterators(a)
	}
	return a
}

// Merge combines all iterators into a single iterator.
// A sorted merge iterator or a merge iterator can be used based on opt.
func (a Iterators) Merge(opt IteratorOptions) (Iterator, error) {
	// Merge into a single iterator.
	if opt.MergeSorted() {
		itr := NewSortedMergeIterator(a, opt)
		if itr != nil && opt.InterruptCh != nil {
			itr = NewInterruptIterator(itr, opt.InterruptCh)
		}
		return itr, nil
	}

	itr := NewMergeIterator(a, opt)
	if itr == nil {
		return nil, nil
	}

	if opt.Expr != nil {
		if expr, ok := opt.Expr.(*Call); ok && expr.Name == "count" {
			opt.Expr = &Call{
				Name: "sum",
				Args: expr.Args,
			}
		}
	}

	if opt.InterruptCh != nil {
		itr = NewInterruptIterator(itr, opt.InterruptCh)
	}
	return NewCallIterator(itr, opt)
}

// NewMergeIterator returns an iterator to merge itrs into one.
// Inputs must either be merge iterators or only contain a single name/tag in
// sorted order. The iterator will output all points by window, name/tag, then
// time. This iterator is useful when you need all of the points for an
// interval.
func NewMergeIterator(inputs []Iterator, opt IteratorOptions) Iterator {
	inputs = Iterators(inputs).filterNonNil()
	if n := len(inputs); n == 0 {
		return nil
	} else if n == 1 {
		return inputs[0]
	}

	// Aggregate functions can use a more relaxed sorting so that points
	// within a window are grouped. This is much more efficient.
	switch inputs := Iterators(inputs).cast().(type) {
	case []FloatIterator:
		return newFloatMergeIterator(inputs, opt)
	case []IntegerIterator:
		return newIntegerMergeIterator(inputs, opt)
	case []StringIterator:
		return newStringMergeIterator(inputs, opt)
	case []BooleanIterator:
		return newBooleanMergeIterator(inputs, opt)
	default:
		panic(fmt.Sprintf("unsupported merge iterator type: %T", inputs))
	}
}

// NewParallelMergeIterator returns an iterator that breaks input iterators
// into groups and processes them in parallel.
func NewParallelMergeIterator(inputs []Iterator, opt IteratorOptions, parallelism int) Iterator {
	inputs = Iterators(inputs).filterNonNil()
	if len(inputs) == 0 {
		return nil
	} else if len(inputs) == 1 {
		return inputs[0]
	}

	// Limit parallelism to the number of inputs.
	if len(inputs) < parallelism {
		parallelism = len(inputs)
	}

	// Determine the number of inputs per output iterator.
	n := len(inputs) / parallelism

	// Group iterators together.
	outputs := make([]Iterator, parallelism)
	for i := range outputs {
		var slice []Iterator
		if i < len(outputs)-1 {
			slice = inputs[i*n : (i+1)*n]
		} else {
			slice = inputs[i*n:]
		}

		outputs[i] = newParallelIterator(NewMergeIterator(slice, opt))
	}

	// Merge all groups together.
	return NewMergeIterator(outputs, opt)
}

// NewSortedMergeIterator returns an iterator to merge itrs into one.
// Inputs must either be sorted merge iterators or only contain a single
// name/tag in sorted order. The iterator will output all points by name/tag,
// then time. This iterator is useful when you need all points for a name/tag
// to be in order.
func NewSortedMergeIterator(inputs []Iterator, opt IteratorOptions) Iterator {
	inputs = Iterators(inputs).filterNonNil()
	if len(inputs) == 0 {
		return nil
	}

	switch inputs := Iterators(inputs).cast().(type) {
	case []FloatIterator:
		return newFloatSortedMergeIterator(inputs, opt)
	case []IntegerIterator:
		return newIntegerSortedMergeIterator(inputs, opt)
	case []StringIterator:
		return newStringSortedMergeIterator(inputs, opt)
	case []BooleanIterator:
		return newBooleanSortedMergeIterator(inputs, opt)
	default:
		panic(fmt.Sprintf("unsupported sorted merge iterator type: %T", inputs))
	}
}

// newParallelIterator returns an iterator that runs in a separate goroutine.
func newParallelIterator(input Iterator) Iterator {
	if input == nil {
		return nil
	}

	switch itr := input.(type) {
	case FloatIterator:
		return newFloatParallelIterator(itr)
	case IntegerIterator:
		return newIntegerParallelIterator(itr)
	case StringIterator:
		return newStringParallelIterator(itr)
	case BooleanIterator:
		return newBooleanParallelIterator(itr)
	default:
		panic(fmt.Sprintf("unsupported parallel iterator type: %T", itr))
	}
}

// NewLimitIterator returns an iterator that limits the number of points per grouping.
func NewLimitIterator(input Iterator, opt IteratorOptions) Iterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatLimitIterator(input, opt)
	case IntegerIterator:
		return newIntegerLimitIterator(input, opt)
	case StringIterator:
		return newStringLimitIterator(input, opt)
	case BooleanIterator:
		return newBooleanLimitIterator(input, opt)
	default:
		panic(fmt.Sprintf("unsupported limit iterator type: %T", input))
	}
}

// NewDedupeIterator returns an iterator that only outputs unique points.
// This iterator maintains a serialized copy of each row so it is inefficient
// to use on large datasets. It is intended for small datasets such as meta queries.
func NewDedupeIterator(input Iterator) Iterator {
	if input == nil {
		return nil
	}

	switch input := input.(type) {
	case FloatIterator:
		return newFloatDedupeIterator(input)
	case IntegerIterator:
		return newIntegerDedupeIterator(input)
	case StringIterator:
		return newStringDedupeIterator(input)
	case BooleanIterator:
		return newBooleanDedupeIterator(input)
	default:
		panic(fmt.Sprintf("unsupported dedupe iterator type: %T", input))
	}
}

// NewFillIterator returns an iterator that fills in missing points in an aggregate.
func NewFillIterator(input Iterator, expr Expr, opt IteratorOptions) Iterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatFillIterator(input, expr, opt)
	case IntegerIterator:
		return newIntegerFillIterator(input, expr, opt)
	case StringIterator:
		return newStringFillIterator(input, expr, opt)
	case BooleanIterator:
		return newBooleanFillIterator(input, expr, opt)
	default:
		panic(fmt.Sprintf("unsupported fill iterator type: %T", input))
	}
}

// NewIntervalIterator returns an iterator that sets the time on each point to the interval.
func NewIntervalIterator(input Iterator, opt IteratorOptions) Iterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatIntervalIterator(input, opt)
	case IntegerIterator:
		return newIntegerIntervalIterator(input, opt)
	case StringIterator:
		return newStringIntervalIterator(input, opt)
	case BooleanIterator:
		return newBooleanIntervalIterator(input, opt)
	default:
		panic(fmt.Sprintf("unsupported fill iterator type: %T", input))
	}
}

// NewInterruptIterator returns an iterator that will stop producing output when a channel
// has been closed on the passed in channel.
func NewInterruptIterator(input Iterator, closing <-chan struct{}) Iterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatInterruptIterator(input, closing)
	case IntegerIterator:
		return newIntegerInterruptIterator(input, closing)
	case StringIterator:
		return newStringInterruptIterator(input, closing)
	case BooleanIterator:
		return newBooleanInterruptIterator(input, closing)
	default:
		panic(fmt.Sprintf("unsupported interrupt iterator type: %T", input))
	}
}

// NewCloseInterruptIterator returns an iterator that will invoke the Close() method on an
// iterator when a channel has been closed.
func NewCloseInterruptIterator(input Iterator, closing <-chan struct{}) Iterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatCloseInterruptIterator(input, closing)
	case IntegerIterator:
		return newIntegerCloseInterruptIterator(input, closing)
	case StringIterator:
		return newStringCloseInterruptIterator(input, closing)
	case BooleanIterator:
		return newBooleanCloseInterruptIterator(input, closing)
	default:
		panic(fmt.Sprintf("unsupported close iterator iterator type: %T", input))
	}
}

// AuxIterator represents an iterator that can split off separate auxilary iterators.
type AuxIterator interface {
	Iterator
	IteratorCreator

	// Auxilary iterator
	Iterator(name string, typ DataType) Iterator

	// Start starts writing to the created iterators.
	Start()

	// Backgrounds the iterator so that, when start is called, it will
	// continuously read from the iterator.
	Background()
}

// NewAuxIterator returns a new instance of AuxIterator.
func NewAuxIterator(input Iterator, opt IteratorOptions) AuxIterator {
	switch input := input.(type) {
	case FloatIterator:
		return newFloatAuxIterator(input, opt)
	case IntegerIterator:
		return newIntegerAuxIterator(input, opt)
	case StringIterator:
		return newStringAuxIterator(input, opt)
	case BooleanIterator:
		return newBooleanAuxIterator(input, opt)
	default:
		panic(fmt.Sprintf("unsupported aux iterator type: %T", input))
	}
}

// auxIteratorField represents an auxilary field within an AuxIterator.
type auxIteratorField struct {
	name string     // field name
	typ  DataType   // detected data type
	itrs []Iterator // auxillary iterators
	mu   sync.Mutex
	opt  IteratorOptions
}

func (f *auxIteratorField) append(itr Iterator) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.itrs = append(f.itrs, itr)
}

func (f *auxIteratorField) close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, itr := range f.itrs {
		itr.Close()
	}
}

type auxIteratorFields []*auxIteratorField

// newAuxIteratorFields returns a new instance of auxIteratorFields from a list of field names.
func newAuxIteratorFields(opt IteratorOptions) auxIteratorFields {
	fields := make(auxIteratorFields, len(opt.Aux))
	for i, ref := range opt.Aux {
		fields[i] = &auxIteratorField{name: ref.Val, typ: ref.Type, opt: opt}
	}
	return fields
}

func (a auxIteratorFields) close() {
	for _, f := range a {
		f.close()
	}
}

// iterator creates a new iterator for a named auxilary field.
func (a auxIteratorFields) iterator(name string, typ DataType) Iterator {
	for _, f := range a {
		// Skip field if it's name doesn't match.
		// Exit if no points were received by the iterator.
		if f.name != name || (typ != Unknown && f.typ != typ) {
			continue
		}

		// Create channel iterator by data type.
		switch f.typ {
		case Float:
			itr := &floatChanIterator{cond: sync.NewCond(&sync.Mutex{})}
			f.append(itr)
			return itr
		case Integer:
			itr := &integerChanIterator{cond: sync.NewCond(&sync.Mutex{})}
			f.append(itr)
			return itr
		case String, Tag:
			itr := &stringChanIterator{cond: sync.NewCond(&sync.Mutex{})}
			f.append(itr)
			return itr
		case Boolean:
			itr := &booleanChanIterator{cond: sync.NewCond(&sync.Mutex{})}
			f.append(itr)
			return itr
		default:
			break
		}
	}

	return &nilFloatIterator{}
}

// send sends a point to all field iterators.
func (a auxIteratorFields) send(p Point) (ok bool) {
	values := p.aux()
	for i, f := range a {
		v := values[i]

		tags := p.tags()
		tags = tags.Subset(f.opt.Dimensions)

		// Send new point for each aux iterator.
		// Primitive pointers represent nil values.
		for _, itr := range f.itrs {
			switch itr := itr.(type) {
			case *floatChanIterator:
				ok = itr.setBuf(p.name(), tags, p.time(), v) || ok
			case *integerChanIterator:
				ok = itr.setBuf(p.name(), tags, p.time(), v) || ok
			case *stringChanIterator:
				ok = itr.setBuf(p.name(), tags, p.time(), v) || ok
			case *booleanChanIterator:
				ok = itr.setBuf(p.name(), tags, p.time(), v) || ok
			default:
				panic(fmt.Sprintf("invalid aux itr type: %T", itr))
			}
		}
	}
	return ok
}

func (a auxIteratorFields) sendError(err error) {
	for _, f := range a {
		for _, itr := range f.itrs {
			switch itr := itr.(type) {
			case *floatChanIterator:
				itr.setErr(err)
			case *integerChanIterator:
				itr.setErr(err)
			case *stringChanIterator:
				itr.setErr(err)
			case *booleanChanIterator:
				itr.setErr(err)
			default:
				panic(fmt.Sprintf("invalid aux itr type: %T", itr))
			}
		}
	}
}

// DrainIterator reads all points from an iterator.
func DrainIterator(itr Iterator) {
	defer itr.Close()
	switch itr := itr.(type) {
	case FloatIterator:
		for p, _ := itr.Next(); p != nil; p, _ = itr.Next() {
		}
	case IntegerIterator:
		for p, _ := itr.Next(); p != nil; p, _ = itr.Next() {
		}
	case StringIterator:
		for p, _ := itr.Next(); p != nil; p, _ = itr.Next() {
		}
	case BooleanIterator:
		for p, _ := itr.Next(); p != nil; p, _ = itr.Next() {
		}
	default:
		panic(fmt.Sprintf("unsupported iterator type for draining: %T", itr))
	}
}

// DrainIterators reads all points from all iterators.
func DrainIterators(itrs []Iterator) {
	defer Iterators(itrs).Close()
	for {
		var hasData bool

		for _, itr := range itrs {
			switch itr := itr.(type) {
			case FloatIterator:
				if p, _ := itr.Next(); p != nil {
					hasData = true
				}
			case IntegerIterator:
				if p, _ := itr.Next(); p != nil {
					hasData = true
				}
			case StringIterator:
				if p, _ := itr.Next(); p != nil {
					hasData = true
				}
			case BooleanIterator:
				if p, _ := itr.Next(); p != nil {
					hasData = true
				}
			default:
				panic(fmt.Sprintf("unsupported iterator type for draining: %T", itr))
			}
		}

		// Exit once all iterators return a nil point.
		if !hasData {
			break
		}
	}
}

// NewReaderIterator returns an iterator that streams from a reader.
func NewReaderIterator(r io.Reader, typ DataType, stats IteratorStats) Iterator {
	switch typ {
	case Float:
		return newFloatReaderIterator(r, stats)
	case Integer:
		return newIntegerReaderIterator(r, stats)
	case String:
		return newStringReaderIterator(r, stats)
	case Boolean:
		return newBooleanReaderIterator(r, stats)
	default:
		return &nilFloatIterator{}
	}
}

// IteratorCreator represents an interface for objects that can create Iterators.
type IteratorCreator interface {
	// Creates a simple iterator for use in an InfluxQL query.
	CreateIterator(opt IteratorOptions) (Iterator, error)

	// Returns the unique fields and dimensions across a list of sources.
	FieldDimensions(sources Sources) (fields map[string]DataType, dimensions map[string]struct{}, err error)

	// Expands regex sources to all matching sources.
	ExpandSources(sources Sources) (Sources, error)
}

// IteratorCreators represents a list of iterator creators.
type IteratorCreators []IteratorCreator

// Close closes all iterator creators that implement io.Closer.
func (a IteratorCreators) Close() error {
	for _, ic := range a {
		if ic, ok := ic.(io.Closer); ok {
			ic.Close()
		}
	}
	return nil
}

// CreateIterator returns a single combined iterator from multiple iterator creators.
func (a IteratorCreators) CreateIterator(opt IteratorOptions) (Iterator, error) {
	// Create iterators for each shard.
	// Ensure that they are closed if an error occurs.
	itrs := make([]Iterator, 0, len(a))
	if err := func() error {
		for _, ic := range a {
			itr, err := ic.CreateIterator(opt)
			if err != nil {
				return err
			} else if itr == nil {
				continue
			}
			itrs = append(itrs, itr)
		}
		return nil
	}(); err != nil {
		Iterators(itrs).Close()
		return nil, err
	}

	if len(itrs) == 0 {
		return nil, nil
	}

	return Iterators(itrs).Merge(opt)
}

// FieldDimensions returns unique fields and dimensions from multiple iterator creators.
func (a IteratorCreators) FieldDimensions(sources Sources) (fields map[string]DataType, dimensions map[string]struct{}, err error) {
	fields = make(map[string]DataType)
	dimensions = make(map[string]struct{})

	for _, ic := range a {
		f, d, err := ic.FieldDimensions(sources)
		if err != nil {
			return nil, nil, err
		}
		for k, typ := range f {
			if _, ok := fields[k]; typ != Unknown && (!ok || typ < fields[k]) {
				fields[k] = typ
			}
		}
		for k := range d {
			dimensions[k] = struct{}{}
		}
	}
	return
}

// ExpandSources expands sources across all iterator creators and returns a unique result.
func (a IteratorCreators) ExpandSources(sources Sources) (Sources, error) {
	m := make(map[string]Source)

	for _, ic := range a {
		expanded, err := ic.ExpandSources(sources)
		if err != nil {
			return nil, err
		}

		for _, src := range expanded {
			switch src := src.(type) {
			case *Measurement:
				m[src.String()] = src
			default:
				return nil, fmt.Errorf("IteratorCreators.ExpandSources: unsupported source type: %T", src)
			}
		}
	}

	// Convert set to sorted slice.
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)

	// Convert set to a list of Sources.
	sorted := make(Sources, 0, len(m))
	for _, name := range names {
		sorted = append(sorted, m[name])
	}

	return sorted, nil
}

// IteratorOptions is an object passed to CreateIterator to specify creation options.
type IteratorOptions struct {
	// Expression to iterate for.
	// This can be VarRef or a Call.
	Expr Expr

	// Auxilary tags or values to also retrieve for the point.
	Aux []VarRef

	// Data sources from which to retrieve data.
	Sources []Source

	// Group by interval and tags.
	Interval   Interval
	Dimensions []string

	// Fill options.
	Fill      FillOption
	FillValue interface{}

	// Condition to filter by.
	Condition Expr

	// Time range for the iterator.
	StartTime int64
	EndTime   int64

	// Sorted in time ascending order if true.
	Ascending bool

	// Limits the number of points per series.
	Limit, Offset int

	// Limits the number of series.
	SLimit, SOffset int

	// Removes duplicate rows from raw queries.
	Dedupe bool

	// If this channel is set and is closed, the iterator should try to exit
	// and close as soon as possible.
	InterruptCh <-chan struct{}
}

// newIteratorOptionsStmt creates the iterator options from stmt.
func newIteratorOptionsStmt(stmt *SelectStatement, sopt *SelectOptions) (opt IteratorOptions, err error) {
	// Determine time range from the condition.
	startTime, endTime, err := TimeRange(stmt.Condition)
	if err != nil {
		return IteratorOptions{}, err
	}

	if !startTime.IsZero() {
		opt.StartTime = startTime.UnixNano()
	} else {
		if sopt != nil {
			opt.StartTime = sopt.MinTime.UnixNano()
		} else {
			opt.StartTime = MinTime
		}
	}
	if !endTime.IsZero() {
		opt.EndTime = endTime.UnixNano()
	} else {
		if sopt != nil {
			opt.EndTime = sopt.MaxTime.UnixNano()
		} else {
			opt.EndTime = MaxTime
		}
	}

	// Determine group by interval.
	interval, err := stmt.GroupByInterval()
	if err != nil {
		return opt, err
	}
	// Set duration to zero if a negative interval has been used.
	if interval < 0 {
		interval = 0
	} else if interval > 0 {
		opt.Interval.Offset, err = stmt.GroupByOffset()
		if err != nil {
			return opt, err
		}
	}
	opt.Interval.Duration = interval

	// Determine dimensions.
	for _, d := range stmt.Dimensions {
		if d, ok := d.Expr.(*VarRef); ok {
			opt.Dimensions = append(opt.Dimensions, d.Val)
		}
	}

	opt.Sources = stmt.Sources
	opt.Condition = stmt.Condition
	opt.Ascending = stmt.TimeAscending()
	opt.Dedupe = stmt.Dedupe

	opt.Fill, opt.FillValue = stmt.Fill, stmt.FillValue
	if opt.Fill == NullFill && stmt.Target != nil {
		// Set the fill option to none if a target has been given.
		// Null values will get ignored when being written to the target
		// so fill(null) wouldn't write any null values to begin with.
		opt.Fill = NoFill
	}
	opt.Limit, opt.Offset = stmt.Limit, stmt.Offset
	opt.SLimit, opt.SOffset = stmt.SLimit, stmt.SOffset
	if sopt != nil {
		opt.InterruptCh = sopt.InterruptCh
	}

	return opt, nil
}

// MergeSorted returns true if the options require a sorted merge.
// This is only needed when the expression is a variable reference or there is no expr.
func (opt IteratorOptions) MergeSorted() bool {
	if opt.Expr == nil {
		return true
	}
	_, ok := opt.Expr.(*VarRef)
	return ok
}

// SeekTime returns the time the iterator should start from.
// For ascending iterators this is the start time, for descending iterators it's the end time.
func (opt IteratorOptions) SeekTime() int64 {
	if opt.Ascending {
		return opt.StartTime
	}
	return opt.EndTime
}

// Window returns the time window [start,end) that t falls within.
func (opt IteratorOptions) Window(t int64) (start, end int64) {
	if opt.Interval.IsZero() {
		return opt.StartTime, opt.EndTime + 1
	}

	// Subtract the offset to the time so we calculate the correct base interval.
	t -= int64(opt.Interval.Offset)

	// Truncate time by duration.
	dt := t % int64(opt.Interval.Duration)
	if dt < 0 {
		// Negative modulo rounds up instead of down, so offset
		// with the duration.
		dt += int64(opt.Interval.Duration)
	}
	t -= dt

	// Apply the offset.
	start = t + int64(opt.Interval.Offset)
	end = start + int64(opt.Interval.Duration)
	return
}

// DerivativeInterval returns the time interval for the derivative function.
func (opt IteratorOptions) DerivativeInterval() Interval {
	// Use the interval on the derivative() call, if specified.
	if expr, ok := opt.Expr.(*Call); ok && len(expr.Args) == 2 {
		return Interval{Duration: expr.Args[1].(*DurationLiteral).Val}
	}

	// Otherwise use the group by interval, if specified.
	if opt.Interval.Duration > 0 {
		return Interval{Duration: opt.Interval.Duration}
	}

	return Interval{Duration: time.Second}
}

// ElapsedInterval returns the time interval for the elapsed function.
func (opt IteratorOptions) ElapsedInterval() Interval {
	// Use the interval on the elapsed() call, if specified.
	if expr, ok := opt.Expr.(*Call); ok && len(expr.Args) == 2 {
		return Interval{Duration: expr.Args[1].(*DurationLiteral).Val}
	}

	return Interval{Duration: time.Nanosecond}
}

// MarshalBinary encodes opt into a binary format.
func (opt *IteratorOptions) MarshalBinary() ([]byte, error) {
	return proto.Marshal(encodeIteratorOptions(opt))
}

// UnmarshalBinary decodes from a binary format in to opt.
func (opt *IteratorOptions) UnmarshalBinary(buf []byte) error {
	var pb internal.IteratorOptions
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}

	other, err := decodeIteratorOptions(&pb)
	if err != nil {
		return err
	}
	*opt = *other

	return nil
}

func encodeIteratorOptions(opt *IteratorOptions) *internal.IteratorOptions {
	pb := &internal.IteratorOptions{
		Interval:   encodeInterval(opt.Interval),
		Dimensions: opt.Dimensions,
		Fill:       proto.Int32(int32(opt.Fill)),
		StartTime:  proto.Int64(opt.StartTime),
		EndTime:    proto.Int64(opt.EndTime),
		Ascending:  proto.Bool(opt.Ascending),
		Limit:      proto.Int64(int64(opt.Limit)),
		Offset:     proto.Int64(int64(opt.Offset)),
		SLimit:     proto.Int64(int64(opt.SLimit)),
		SOffset:    proto.Int64(int64(opt.SOffset)),
		Dedupe:     proto.Bool(opt.Dedupe),
	}

	// Set expression, if set.
	if opt.Expr != nil {
		pb.Expr = proto.String(opt.Expr.String())
	}

	// Convert and encode aux fields as variable references.
	pb.Fields = make([]*internal.VarRef, len(opt.Aux))
	pb.Aux = make([]string, len(opt.Aux))
	for i, ref := range opt.Aux {
		pb.Fields[i] = encodeVarRef(ref)
		pb.Aux[i] = ref.Val
	}

	// Convert and encode sources to measurements.
	sources := make([]*internal.Measurement, len(opt.Sources))
	for i, source := range opt.Sources {
		mm := source.(*Measurement)
		sources[i] = encodeMeasurement(mm)
	}
	pb.Sources = sources

	// Fill value can only be a number. Set it if available.
	if v, ok := opt.FillValue.(float64); ok {
		pb.FillValue = proto.Float64(v)
	}

	// Set condition, if set.
	if opt.Condition != nil {
		pb.Condition = proto.String(opt.Condition.String())
	}

	return pb
}

func decodeIteratorOptions(pb *internal.IteratorOptions) (*IteratorOptions, error) {
	opt := &IteratorOptions{
		Interval:   decodeInterval(pb.GetInterval()),
		Dimensions: pb.GetDimensions(),
		Fill:       FillOption(pb.GetFill()),
		FillValue:  pb.GetFillValue(),
		StartTime:  pb.GetStartTime(),
		EndTime:    pb.GetEndTime(),
		Ascending:  pb.GetAscending(),
		Limit:      int(pb.GetLimit()),
		Offset:     int(pb.GetOffset()),
		SLimit:     int(pb.GetSLimit()),
		SOffset:    int(pb.GetSOffset()),
		Dedupe:     pb.GetDedupe(),
	}

	// Set expression, if set.
	if pb.Expr != nil {
		expr, err := ParseExpr(pb.GetExpr())
		if err != nil {
			return nil, err
		}
		opt.Expr = expr
	}

	// Convert and decode variable references.
	if fields := pb.GetFields(); fields != nil {
		opt.Aux = make([]VarRef, len(fields))
		for i, ref := range fields {
			opt.Aux[i] = decodeVarRef(ref)
		}
	} else {
		opt.Aux = make([]VarRef, len(pb.GetAux()))
		for i, name := range pb.GetAux() {
			opt.Aux[i] = VarRef{Val: name}
		}
	}

	// Convert and dencode sources to measurements.
	sources := make([]Source, len(pb.GetSources()))
	for i, source := range pb.GetSources() {
		mm, err := decodeMeasurement(source)
		if err != nil {
			return nil, err
		}
		sources[i] = mm
	}
	opt.Sources = sources

	// Set condition, if set.
	if pb.Condition != nil {
		expr, err := ParseExpr(pb.GetCondition())
		if err != nil {
			return nil, err
		}
		opt.Condition = expr
	}

	return opt, nil
}

// selectInfo represents an object that stores info about select fields.
type selectInfo struct {
	calls map[*Call]struct{}
	refs  map[*VarRef]struct{}
}

// newSelectInfo creates a object with call and var ref info from stmt.
func newSelectInfo(stmt *SelectStatement) *selectInfo {
	info := &selectInfo{
		calls: make(map[*Call]struct{}),
		refs:  make(map[*VarRef]struct{}),
	}
	Walk(info, stmt.Fields)
	return info
}

func (v *selectInfo) Visit(n Node) Visitor {
	switch n := n.(type) {
	case *Call:
		v.calls[n] = struct{}{}
		return nil
	case *VarRef:
		v.refs[n] = struct{}{}
		return nil
	}
	return v
}

// Interval represents a repeating interval for a query.
type Interval struct {
	Duration time.Duration
	Offset   time.Duration
}

// IsZero returns true if the interval has no duration.
func (i Interval) IsZero() bool { return i.Duration == 0 }

func encodeInterval(i Interval) *internal.Interval {
	return &internal.Interval{
		Duration: proto.Int64(i.Duration.Nanoseconds()),
		Offset:   proto.Int64(i.Offset.Nanoseconds()),
	}
}

func decodeInterval(pb *internal.Interval) Interval {
	return Interval{
		Duration: time.Duration(pb.GetDuration()),
		Offset:   time.Duration(pb.GetOffset()),
	}
}

func encodeVarRef(ref VarRef) *internal.VarRef {
	return &internal.VarRef{
		Val:  proto.String(ref.Val),
		Type: proto.Int32(int32(ref.Type)),
	}
}

func decodeVarRef(pb *internal.VarRef) VarRef {
	return VarRef{
		Val:  pb.GetVal(),
		Type: DataType(pb.GetType()),
	}
}

type nilFloatIterator struct{}

func (*nilFloatIterator) Stats() IteratorStats       { return IteratorStats{} }
func (*nilFloatIterator) Close() error               { return nil }
func (*nilFloatIterator) Next() (*FloatPoint, error) { return nil, nil }

// integerFloatTransformIterator executes a function to modify an existing point for every
// output of the input iterator.
type integerFloatTransformIterator struct {
	input IntegerIterator
	fn    integerFloatTransformFunc
}

// Stats returns stats from the input iterator.
func (itr *integerFloatTransformIterator) Stats() IteratorStats { return itr.input.Stats() }

// Close closes the iterator and all child iterators.
func (itr *integerFloatTransformIterator) Close() error { return itr.input.Close() }

// Next returns the minimum value for the next available interval.
func (itr *integerFloatTransformIterator) Next() (*FloatPoint, error) {
	p, err := itr.input.Next()
	if err != nil {
		return nil, err
	} else if p != nil {
		return itr.fn(p), nil
	}
	return nil, nil
}

// integerFloatTransformFunc creates or modifies a point.
// The point passed in may be modified and returned rather than allocating a
// new point if possible.
type integerFloatTransformFunc func(p *IntegerPoint) *FloatPoint

type integerFloatCastIterator struct {
	input IntegerIterator
}

func (itr *integerFloatCastIterator) Stats() IteratorStats { return itr.input.Stats() }
func (itr *integerFloatCastIterator) Close() error         { return itr.input.Close() }
func (itr *integerFloatCastIterator) Next() (*FloatPoint, error) {
	p, err := itr.input.Next()
	if p == nil || err != nil {
		return nil, err
	}

	return &FloatPoint{
		Name:  p.Name,
		Tags:  p.Tags,
		Time:  p.Time,
		Nil:   p.Nil,
		Value: float64(p.Value),
		Aux:   p.Aux,
	}, nil
}

// IteratorStats represents statistics about an iterator.
// Some statistics are available immediately upon iterator creation while
// some are derived as the iterator processes data.
type IteratorStats struct {
	SeriesN int // series represented
	PointN  int // points returned
}

// Add aggregates fields from s and other together. Overwrites s.
func (s *IteratorStats) Add(other IteratorStats) {
	s.SeriesN += other.SeriesN
	s.PointN += other.PointN
}

func encodeIteratorStats(stats *IteratorStats) *internal.IteratorStats {
	return &internal.IteratorStats{
		SeriesN: proto.Int64(int64(stats.SeriesN)),
		PointN:  proto.Int64(int64(stats.PointN)),
	}
}

func decodeIteratorStats(pb *internal.IteratorStats) IteratorStats {
	return IteratorStats{
		SeriesN: int(pb.GetSeriesN()),
		PointN:  int(pb.GetPointN()),
	}
}

// floatFastDedupeIterator outputs unique points where the point has a single aux field.
type floatFastDedupeIterator struct {
	input FloatIterator
	m     map[fastDedupeKey]struct{} // lookup of points already sent
}

// newFloatFastDedupeIterator returns a new instance of floatFastDedupeIterator.
func newFloatFastDedupeIterator(input FloatIterator) *floatFastDedupeIterator {
	return &floatFastDedupeIterator{
		input: input,
		m:     make(map[fastDedupeKey]struct{}),
	}
}

// Stats returns stats from the input iterator.
func (itr *floatFastDedupeIterator) Stats() IteratorStats { return itr.input.Stats() }

// Close closes the iterator and all child iterators.
func (itr *floatFastDedupeIterator) Close() error { return itr.input.Close() }

// Next returns the next unique point from the input iterator.
func (itr *floatFastDedupeIterator) Next() (*FloatPoint, error) {
	for {
		// Read next point.
		// Skip if there are not any aux fields.
		p, err := itr.input.Next()
		if p == nil || err != nil {
			return nil, err
		} else if len(p.Aux) == 0 {
			continue
		}

		// If the point has already been output then move to the next point.
		key := fastDedupeKey{name: p.Name}
		key.values[0] = p.Aux[0]
		if len(p.Aux) > 1 {
			key.values[1] = p.Aux[1]
		}
		if _, ok := itr.m[key]; ok {
			continue
		}

		// Otherwise mark it as emitted and return point.
		itr.m[key] = struct{}{}
		return p, nil
	}
}

type fastDedupeKey struct {
	name   string
	values [2]interface{}
}

type reverseStringSlice []string

func (p reverseStringSlice) Len() int           { return len(p) }
func (p reverseStringSlice) Less(i, j int) bool { return p[i] > p[j] }
func (p reverseStringSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
