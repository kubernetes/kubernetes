package influxql

import (
	"bytes"
	"container/heap"
	"fmt"
	"math"
	"sort"
	"time"
)

/*
This file contains iterator implementations for each function call available
in InfluxQL. Call iterators are separated into two groups:

1. Map/reduce-style iterators - these are passed to IteratorCreator so that
   processing can be at the low-level storage and aggregates are returned.

2. Raw aggregate iterators - these require the full set of data for a window.
   These are handled by the select() function and raw points are streamed in
   from the low-level storage.

There are helpers to aid in building aggregate iterators. For simple map/reduce
iterators, you can use the reduceIterator types and pass a reduce function. This
reduce function is passed a previous and current value and the new timestamp,
value, and auxilary fields are returned from it.

For raw aggregate iterators, you can use the reduceSliceIterators which pass
in a slice of all points to the function and return a point. For more complex
iterator types, you may need to create your own iterators by hand.

Once your iterator is complete, you'll need to add it to the NewCallIterator()
function if it is to be available to IteratorCreators and add it to the select()
function to allow it to be included during planning.
*/

// NewCallIterator returns a new iterator for a Call.
func NewCallIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	name := opt.Expr.(*Call).Name
	switch name {
	case "count":
		return newCountIterator(input, opt)
	case "min":
		return newMinIterator(input, opt)
	case "max":
		return newMaxIterator(input, opt)
	case "sum":
		return newSumIterator(input, opt)
	case "first":
		return newFirstIterator(input, opt)
	case "last":
		return newLastIterator(input, opt)
	case "mean":
		return newMeanIterator(input, opt)
	default:
		return nil, fmt.Errorf("unsupported function call: %s", name)
	}
}

// newCountIterator returns an iterator for operating on a count() call.
func newCountIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	// FIXME: Wrap iterator in int-type iterator and always output int value.

	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, IntegerPointEmitter) {
			fn := NewFloatFuncIntegerReducer(FloatCountReduce, &IntegerPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &floatReduceIntegerIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerCountReduce, &IntegerPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, IntegerPointEmitter) {
			fn := NewStringFuncIntegerReducer(StringCountReduce, &IntegerPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &stringReduceIntegerIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, IntegerPointEmitter) {
			fn := NewBooleanFuncIntegerReducer(BooleanCountReduce, &IntegerPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &booleanReduceIntegerIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported count iterator type: %T", input)
	}
}

// FloatCountReduce returns the count of points.
func FloatCountReduce(prev *IntegerPoint, curr *FloatPoint) (int64, int64, []interface{}) {
	if prev == nil {
		return ZeroTime, 1, nil
	}
	return ZeroTime, prev.Value + 1, nil
}

// IntegerCountReduce returns the count of points.
func IntegerCountReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil {
		return ZeroTime, 1, nil
	}
	return ZeroTime, prev.Value + 1, nil
}

// StringCountReduce returns the count of points.
func StringCountReduce(prev *IntegerPoint, curr *StringPoint) (int64, int64, []interface{}) {
	if prev == nil {
		return ZeroTime, 1, nil
	}
	return ZeroTime, prev.Value + 1, nil
}

// BooleanCountReduce returns the count of points.
func BooleanCountReduce(prev *IntegerPoint, curr *BooleanPoint) (int64, int64, []interface{}) {
	if prev == nil {
		return ZeroTime, 1, nil
	}
	return ZeroTime, prev.Value + 1, nil
}

// newMinIterator returns an iterator for operating on a min() call.
func newMinIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatFuncReducer(FloatMinReduce, nil)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerMinReduce, nil)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanFuncReducer(BooleanMinReduce, nil)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported min iterator type: %T", input)
	}
}

// FloatMinReduce returns the minimum value between prev & curr.
func FloatMinReduce(prev, curr *FloatPoint) (int64, float64, []interface{}) {
	if prev == nil || curr.Value < prev.Value || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// IntegerMinReduce returns the minimum value between prev & curr.
func IntegerMinReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil || curr.Value < prev.Value || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// BooleanMinReduce returns the minimum value between prev & curr.
func BooleanMinReduce(prev, curr *BooleanPoint) (int64, bool, []interface{}) {
	if prev == nil || (curr.Value != prev.Value && !curr.Value) || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// newMaxIterator returns an iterator for operating on a max() call.
func newMaxIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatFuncReducer(FloatMaxReduce, nil)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerMaxReduce, nil)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanFuncReducer(BooleanMaxReduce, nil)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported max iterator type: %T", input)
	}
}

// FloatMaxReduce returns the maximum value between prev & curr.
func FloatMaxReduce(prev, curr *FloatPoint) (int64, float64, []interface{}) {
	if prev == nil || curr.Value > prev.Value || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// IntegerMaxReduce returns the maximum value between prev & curr.
func IntegerMaxReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil || curr.Value > prev.Value || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// BooleanMaxReduce returns the minimum value between prev & curr.
func BooleanMaxReduce(prev, curr *BooleanPoint) (int64, bool, []interface{}) {
	if prev == nil || (curr.Value != prev.Value && curr.Value) || (curr.Value == prev.Value && curr.Time < prev.Time) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// newSumIterator returns an iterator for operating on a sum() call.
func newSumIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatFuncReducer(FloatSumReduce, &FloatPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerSumReduce, &IntegerPoint{Value: 0, Time: ZeroTime})
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported sum iterator type: %T", input)
	}
}

// FloatSumReduce returns the sum prev value & curr value.
func FloatSumReduce(prev, curr *FloatPoint) (int64, float64, []interface{}) {
	if prev == nil {
		return ZeroTime, curr.Value, nil
	}
	return prev.Time, prev.Value + curr.Value, nil
}

// IntegerSumReduce returns the sum prev value & curr value.
func IntegerSumReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil {
		return ZeroTime, curr.Value, nil
	}
	return prev.Time, prev.Value + curr.Value, nil
}

// newFirstIterator returns an iterator for operating on a first() call.
func newFirstIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatFuncReducer(FloatFirstReduce, nil)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerFirstReduce, nil)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, StringPointEmitter) {
			fn := NewStringFuncReducer(StringFirstReduce, nil)
			return fn, fn
		}
		return &stringReduceStringIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanFuncReducer(BooleanFirstReduce, nil)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported first iterator type: %T", input)
	}
}

// FloatFirstReduce returns the first point sorted by time.
func FloatFirstReduce(prev, curr *FloatPoint) (int64, float64, []interface{}) {
	if prev == nil || curr.Time < prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// IntegerFirstReduce returns the first point sorted by time.
func IntegerFirstReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil || curr.Time < prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// StringFirstReduce returns the first point sorted by time.
func StringFirstReduce(prev, curr *StringPoint) (int64, string, []interface{}) {
	if prev == nil || curr.Time < prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// BooleanFirstReduce returns the first point sorted by time.
func BooleanFirstReduce(prev, curr *BooleanPoint) (int64, bool, []interface{}) {
	if prev == nil || curr.Time < prev.Time || (curr.Time == prev.Time && !curr.Value && prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// newLastIterator returns an iterator for operating on a last() call.
func newLastIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatFuncReducer(FloatLastReduce, nil)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerFuncReducer(IntegerLastReduce, nil)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, StringPointEmitter) {
			fn := NewStringFuncReducer(StringLastReduce, nil)
			return fn, fn
		}
		return &stringReduceStringIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanFuncReducer(BooleanLastReduce, nil)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported last iterator type: %T", input)
	}
}

// FloatLastReduce returns the last point sorted by time.
func FloatLastReduce(prev, curr *FloatPoint) (int64, float64, []interface{}) {
	if prev == nil || curr.Time > prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// IntegerLastReduce returns the last point sorted by time.
func IntegerLastReduce(prev, curr *IntegerPoint) (int64, int64, []interface{}) {
	if prev == nil || curr.Time > prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// StringLastReduce returns the first point sorted by time.
func StringLastReduce(prev, curr *StringPoint) (int64, string, []interface{}) {
	if prev == nil || curr.Time > prev.Time || (curr.Time == prev.Time && curr.Value > prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// BooleanLastReduce returns the first point sorted by time.
func BooleanLastReduce(prev, curr *BooleanPoint) (int64, bool, []interface{}) {
	if prev == nil || curr.Time > prev.Time || (curr.Time == prev.Time && curr.Value && !prev.Value) {
		return curr.Time, curr.Value, curr.Aux
	}
	return prev.Time, prev.Value, prev.Aux
}

// NewDistinctIterator returns an iterator for operating on a distinct() call.
func NewDistinctIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatDistinctReducer()
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerDistinctReducer()
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, StringPointEmitter) {
			fn := NewStringDistinctReducer()
			return fn, fn
		}
		return &stringReduceStringIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanDistinctReducer()
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported distinct iterator type: %T", input)
	}
}

// newMeanIterator returns an iterator for operating on a mean() call.
func newMeanIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatMeanReducer()
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewIntegerMeanReducer()
			return fn, fn
		}
		return &integerReduceFloatIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported mean iterator type: %T", input)
	}
}

// NewMedianIterator returns an iterator for operating on a median() call.
func NewMedianIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	return newMedianIterator(input, opt)
}

// newMedianIterator returns an iterator for operating on a median() call.
func newMedianIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(FloatMedianReduceSlice)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewIntegerSliceFuncFloatReducer(IntegerMedianReduceSlice)
			return fn, fn
		}
		return &integerReduceFloatIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported median iterator type: %T", input)
	}
}

// FloatMedianReduceSlice returns the median value within a window.
func FloatMedianReduceSlice(a []FloatPoint) []FloatPoint {
	if len(a) == 1 {
		return a
	}

	// OPTIMIZE(benbjohnson): Use getSortedRange() from v0.9.5.1.

	// Return the middle value from the points.
	// If there are an even number of points then return the mean of the two middle points.
	sort.Sort(floatPointsByValue(a))
	if len(a)%2 == 0 {
		lo, hi := a[len(a)/2-1], a[(len(a)/2)]
		return []FloatPoint{{Time: ZeroTime, Value: lo.Value + (hi.Value-lo.Value)/2}}
	}
	return []FloatPoint{{Time: ZeroTime, Value: a[len(a)/2].Value}}
}

// IntegerMedianReduceSlice returns the median value within a window.
func IntegerMedianReduceSlice(a []IntegerPoint) []FloatPoint {
	if len(a) == 1 {
		return []FloatPoint{{Time: ZeroTime, Value: float64(a[0].Value)}}
	}

	// OPTIMIZE(benbjohnson): Use getSortedRange() from v0.9.5.1.

	// Return the middle value from the points.
	// If there are an even number of points then return the mean of the two middle points.
	sort.Sort(integerPointsByValue(a))
	if len(a)%2 == 0 {
		lo, hi := a[len(a)/2-1], a[(len(a)/2)]
		return []FloatPoint{{Time: ZeroTime, Value: float64(lo.Value) + float64(hi.Value-lo.Value)/2}}
	}
	return []FloatPoint{{Time: ZeroTime, Value: float64(a[len(a)/2].Value)}}
}

// newModeIterator returns an iterator for operating on a mode() call.
func NewModeIterator(input Iterator, opt IteratorOptions) (Iterator, error) {

	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(FloatModeReduceSlice)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSliceFuncReducer(IntegerModeReduceSlice)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, StringPointEmitter) {
			fn := NewStringSliceFuncReducer(StringModeReduceSlice)
			return fn, fn
		}
		return &stringReduceStringIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanSliceFuncReducer(BooleanModeReduceSlice)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil

	default:
		return nil, fmt.Errorf("unsupported median iterator type: %T", input)
	}
}

// FloatModeReduceSlice returns the mode value within a window.
func FloatModeReduceSlice(a []FloatPoint) []FloatPoint {
	if len(a) == 1 {
		return a
	}

	// fmt.Println(a[0])
	sort.Sort(floatPointsByValue(a))

	mostFreq := 0
	currFreq := 0
	currMode := a[0].Value
	mostMode := a[0].Value
	mostTime := a[0].Time
	currTime := a[0].Time

	for _, p := range a {
		if p.Value != currMode {
			currFreq = 1
			currMode = p.Value
			currTime = p.Time
			continue
		}
		currFreq++
		if mostFreq > currFreq || (mostFreq == currFreq && currTime > mostTime) {
			continue
		}
		mostFreq = currFreq
		mostMode = p.Value
		mostTime = p.Time
	}

	return []FloatPoint{{Time: ZeroTime, Value: mostMode}}
}

// IntegerModeReduceSlice returns the mode value within a window.
func IntegerModeReduceSlice(a []IntegerPoint) []IntegerPoint {
	if len(a) == 1 {
		return a
	}
	sort.Sort(integerPointsByValue(a))

	mostFreq := 0
	currFreq := 0
	currMode := a[0].Value
	mostMode := a[0].Value
	mostTime := a[0].Time
	currTime := a[0].Time

	for _, p := range a {
		if p.Value != currMode {
			currFreq = 1
			currMode = p.Value
			currTime = p.Time
			continue
		}
		currFreq++
		if mostFreq > currFreq || (mostFreq == currFreq && currTime > mostTime) {
			continue
		}
		mostFreq = currFreq
		mostMode = p.Value
		mostTime = p.Time
	}

	return []IntegerPoint{{Time: ZeroTime, Value: mostMode}}
}

// StringModeReduceSlice returns the mode value within a window.
func StringModeReduceSlice(a []StringPoint) []StringPoint {
	if len(a) == 1 {
		return a
	}

	sort.Sort(stringPointsByValue(a))

	mostFreq := 0
	currFreq := 0
	currMode := a[0].Value
	mostMode := a[0].Value
	mostTime := a[0].Time
	currTime := a[0].Time

	for _, p := range a {
		if p.Value != currMode {
			currFreq = 1
			currMode = p.Value
			currTime = p.Time
			continue
		}
		currFreq++
		if mostFreq > currFreq || (mostFreq == currFreq && currTime > mostTime) {
			continue
		}
		mostFreq = currFreq
		mostMode = p.Value
		mostTime = p.Time
	}

	return []StringPoint{{Time: ZeroTime, Value: mostMode}}
}

// BooleanModeReduceSlice returns the mode value within a window.
func BooleanModeReduceSlice(a []BooleanPoint) []BooleanPoint {
	if len(a) == 1 {
		return a
	}

	trueFreq := 0
	falsFreq := 0
	mostMode := false

	for _, p := range a {
		if p.Value {
			trueFreq++
		} else {
			falsFreq++
		}
	}
	// In case either of true or false are mode then retuned mode value wont be
	// of metric with oldest timestamp
	if trueFreq >= falsFreq {
		mostMode = true
	}

	return []BooleanPoint{{Time: ZeroTime, Value: mostMode}}
}

// newStddevIterator returns an iterator for operating on a stddev() call.
func newStddevIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(FloatStddevReduceSlice)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewIntegerSliceFuncFloatReducer(IntegerStddevReduceSlice)
			return fn, fn
		}
		return &integerReduceFloatIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported stddev iterator type: %T", input)
	}
}

// FloatStddevReduceSlice returns the stddev value within a window.
func FloatStddevReduceSlice(a []FloatPoint) []FloatPoint {
	// If there is only one point then return 0.
	if len(a) < 2 {
		return []FloatPoint{{Time: ZeroTime, Nil: true}}
	}

	// Calculate the mean.
	var mean float64
	var count int
	for _, p := range a {
		if math.IsNaN(p.Value) {
			continue
		}
		count++
		mean += (p.Value - mean) / float64(count)
	}

	// Calculate the variance.
	var variance float64
	for _, p := range a {
		if math.IsNaN(p.Value) {
			continue
		}
		variance += math.Pow(p.Value-mean, 2)
	}
	return []FloatPoint{{
		Time:  ZeroTime,
		Value: math.Sqrt(variance / float64(count-1)),
	}}
}

// IntegerStddevReduceSlice returns the stddev value within a window.
func IntegerStddevReduceSlice(a []IntegerPoint) []FloatPoint {
	// If there is only one point then return 0.
	if len(a) < 2 {
		return []FloatPoint{{Time: ZeroTime, Nil: true}}
	}

	// Calculate the mean.
	var mean float64
	var count int
	for _, p := range a {
		count++
		mean += (float64(p.Value) - mean) / float64(count)
	}

	// Calculate the variance.
	var variance float64
	for _, p := range a {
		variance += math.Pow(float64(p.Value)-mean, 2)
	}
	return []FloatPoint{{
		Time:  ZeroTime,
		Value: math.Sqrt(variance / float64(count-1)),
	}}
}

// newSpreadIterator returns an iterator for operating on a spread() call.
func newSpreadIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(FloatSpreadReduceSlice)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSliceFuncReducer(IntegerSpreadReduceSlice)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported spread iterator type: %T", input)
	}
}

// FloatSpreadReduceSlice returns the spread value within a window.
func FloatSpreadReduceSlice(a []FloatPoint) []FloatPoint {
	// Find min & max values.
	min, max := a[0].Value, a[0].Value
	for _, p := range a[1:] {
		min = math.Min(min, p.Value)
		max = math.Max(max, p.Value)
	}
	return []FloatPoint{{Time: ZeroTime, Value: max - min}}
}

// IntegerSpreadReduceSlice returns the spread value within a window.
func IntegerSpreadReduceSlice(a []IntegerPoint) []IntegerPoint {
	// Find min & max values.
	min, max := a[0].Value, a[0].Value
	for _, p := range a[1:] {
		if p.Value < min {
			min = p.Value
		}
		if p.Value > max {
			max = p.Value
		}
	}
	return []IntegerPoint{{Time: ZeroTime, Value: max - min}}
}

func newTopIterator(input Iterator, opt IteratorOptions, n *IntegerLiteral, tags []int) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		aggregateFn := NewFloatTopReduceSliceFunc(int(n.Val), tags, opt.Interval)
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(aggregateFn)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		aggregateFn := NewIntegerTopReduceSliceFunc(int(n.Val), tags, opt.Interval)
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSliceFuncReducer(aggregateFn)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported top iterator type: %T", input)
	}
}

// NewFloatTopReduceSliceFunc returns the top values within a window.
func NewFloatTopReduceSliceFunc(n int, tags []int, interval Interval) FloatReduceSliceFunc {
	return func(a []FloatPoint) []FloatPoint {
		// Filter by tags if they exist.
		if len(tags) > 0 {
			a = filterFloatByUniqueTags(a, tags, func(cur, p *FloatPoint) bool {
				return p.Value > cur.Value || (p.Value == cur.Value && p.Time < cur.Time)
			})
		}

		// If we ask for more elements than exist, restrict n to be the length of the array.
		size := n
		if size > len(a) {
			size = len(a)
		}

		// Construct a heap preferring higher values and breaking ties
		// based on the earliest time for a point.
		h := floatPointsSortBy(a, func(a, b *FloatPoint) bool {
			if a.Value != b.Value {
				return a.Value > b.Value
			}
			return a.Time < b.Time
		})
		heap.Init(h)

		// Pop the first n elements and then sort by time.
		points := make([]FloatPoint, 0, size)
		for i := 0; i < size; i++ {
			p := heap.Pop(h).(FloatPoint)
			points = append(points, p)
		}

		// Either zero out all values or sort the points by time
		// depending on if a time interval was given or not.
		if !interval.IsZero() {
			for i := range points {
				points[i].Time = ZeroTime
			}
		} else {
			sort.Stable(floatPointsByTime(points))
		}
		return points
	}
}

// NewIntegerTopReduceSliceFunc returns the top values within a window.
func NewIntegerTopReduceSliceFunc(n int, tags []int, interval Interval) IntegerReduceSliceFunc {
	return func(a []IntegerPoint) []IntegerPoint {
		// Filter by tags if they exist.
		if len(tags) > 0 {
			a = filterIntegerByUniqueTags(a, tags, func(cur, p *IntegerPoint) bool {
				return p.Value > cur.Value || (p.Value == cur.Value && p.Time < cur.Time)
			})
		}

		// If we ask for more elements than exist, restrict n to be the length of the array.
		size := n
		if size > len(a) {
			size = len(a)
		}

		// Construct a heap preferring higher values and breaking ties
		// based on the earliest time for a point.
		h := integerPointsSortBy(a, func(a, b *IntegerPoint) bool {
			if a.Value != b.Value {
				return a.Value > b.Value
			}
			return a.Time < b.Time
		})
		heap.Init(h)

		// Pop the first n elements and then sort by time.
		points := make([]IntegerPoint, 0, size)
		for i := 0; i < size; i++ {
			p := heap.Pop(h).(IntegerPoint)
			points = append(points, p)
		}

		// Either zero out all values or sort the points by time
		// depending on if a time interval was given or not.
		if !interval.IsZero() {
			for i := range points {
				points[i].Time = ZeroTime
			}
		} else {
			sort.Stable(integerPointsByTime(points))
		}
		return points
	}
}

func newBottomIterator(input Iterator, opt IteratorOptions, n *IntegerLiteral, tags []int) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		aggregateFn := NewFloatBottomReduceSliceFunc(int(n.Val), tags, opt.Interval)
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(aggregateFn)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		aggregateFn := NewIntegerBottomReduceSliceFunc(int(n.Val), tags, opt.Interval)
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSliceFuncReducer(aggregateFn)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported bottom iterator type: %T", input)
	}
}

// NewFloatBottomReduceSliceFunc returns the bottom values within a window.
func NewFloatBottomReduceSliceFunc(n int, tags []int, interval Interval) FloatReduceSliceFunc {
	return func(a []FloatPoint) []FloatPoint {
		// Filter by tags if they exist.
		if len(tags) > 0 {
			a = filterFloatByUniqueTags(a, tags, func(cur, p *FloatPoint) bool {
				return p.Value < cur.Value || (p.Value == cur.Value && p.Time < cur.Time)
			})
		}

		// If we ask for more elements than exist, restrict n to be the length of the array.
		size := n
		if size > len(a) {
			size = len(a)
		}

		// Construct a heap preferring lower values and breaking ties
		// based on the earliest time for a point.
		h := floatPointsSortBy(a, func(a, b *FloatPoint) bool {
			if a.Value != b.Value {
				return a.Value < b.Value
			}
			return a.Time < b.Time
		})
		heap.Init(h)

		// Pop the first n elements and then sort by time.
		points := make([]FloatPoint, 0, size)
		for i := 0; i < size; i++ {
			p := heap.Pop(h).(FloatPoint)
			points = append(points, p)
		}

		// Either zero out all values or sort the points by time
		// depending on if a time interval was given or not.
		if !interval.IsZero() {
			for i := range points {
				points[i].Time = ZeroTime
			}
		} else {
			sort.Stable(floatPointsByTime(points))
		}
		return points
	}
}

// NewIntegerBottomReduceSliceFunc returns the bottom values within a window.
func NewIntegerBottomReduceSliceFunc(n int, tags []int, interval Interval) IntegerReduceSliceFunc {
	return func(a []IntegerPoint) []IntegerPoint {
		// Filter by tags if they exist.
		if len(tags) > 0 {
			a = filterIntegerByUniqueTags(a, tags, func(cur, p *IntegerPoint) bool {
				return p.Value < cur.Value || (p.Value == cur.Value && p.Time < cur.Time)
			})
		}

		// If we ask for more elements than exist, restrict n to be the length of the array.
		size := n
		if size > len(a) {
			size = len(a)
		}

		// Construct a heap preferring lower values and breaking ties
		// based on the earliest time for a point.
		h := integerPointsSortBy(a, func(a, b *IntegerPoint) bool {
			if a.Value != b.Value {
				return a.Value < b.Value
			}
			return a.Time < b.Time
		})
		heap.Init(h)

		// Pop the first n elements and then sort by time.
		points := make([]IntegerPoint, 0, size)
		for i := 0; i < size; i++ {
			p := heap.Pop(h).(IntegerPoint)
			points = append(points, p)
		}

		// Either zero out all values or sort the points by time
		// depending on if a time interval was given or not.
		if !interval.IsZero() {
			for i := range points {
				points[i].Time = ZeroTime
			}
		} else {
			sort.Stable(integerPointsByTime(points))
		}
		return points
	}
}

func filterFloatByUniqueTags(a []FloatPoint, tags []int, cmpFunc func(cur, p *FloatPoint) bool) []FloatPoint {
	pointMap := make(map[string]FloatPoint)
	for _, p := range a {
		keyBuf := bytes.NewBuffer(nil)
		for i, index := range tags {
			if i > 0 {
				keyBuf.WriteString(",")
			}
			fmt.Fprintf(keyBuf, "%s", p.Aux[index])
		}
		key := keyBuf.String()

		cur, ok := pointMap[key]
		if ok {
			if cmpFunc(&cur, &p) {
				pointMap[key] = p
			}
		} else {
			pointMap[key] = p
		}
	}

	// Recreate the original array with our new filtered list.
	points := make([]FloatPoint, 0, len(pointMap))
	for _, p := range pointMap {
		points = append(points, p)
	}
	return points
}

func filterIntegerByUniqueTags(a []IntegerPoint, tags []int, cmpFunc func(cur, p *IntegerPoint) bool) []IntegerPoint {
	pointMap := make(map[string]IntegerPoint)
	for _, p := range a {
		keyBuf := bytes.NewBuffer(nil)
		for i, index := range tags {
			if i > 0 {
				keyBuf.WriteString(",")
			}
			fmt.Fprintf(keyBuf, "%s", p.Aux[index])
		}
		key := keyBuf.String()

		cur, ok := pointMap[key]
		if ok {
			if cmpFunc(&cur, &p) {
				pointMap[key] = p
			}
		} else {
			pointMap[key] = p
		}
	}

	// Recreate the original array with our new filtered list.
	points := make([]IntegerPoint, 0, len(pointMap))
	for _, p := range pointMap {
		points = append(points, p)
	}
	return points
}

// newPercentileIterator returns an iterator for operating on a percentile() call.
func newPercentileIterator(input Iterator, opt IteratorOptions, percentile float64) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		floatPercentileReduceSlice := NewFloatPercentileReduceSliceFunc(percentile)
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSliceFuncReducer(floatPercentileReduceSlice)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		integerPercentileReduceSlice := NewIntegerPercentileReduceSliceFunc(percentile)
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSliceFuncReducer(integerPercentileReduceSlice)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported percentile iterator type: %T", input)
	}
}

// NewFloatPercentileReduceSliceFunc returns the percentile value within a window.
func NewFloatPercentileReduceSliceFunc(percentile float64) FloatReduceSliceFunc {
	return func(a []FloatPoint) []FloatPoint {
		length := len(a)
		i := int(math.Floor(float64(length)*percentile/100.0+0.5)) - 1

		if i < 0 || i >= length {
			return nil
		}

		sort.Sort(floatPointsByValue(a))
		return []FloatPoint{{Time: a[i].Time, Value: a[i].Value, Aux: a[i].Aux}}
	}
}

// NewIntegerPercentileReduceSliceFunc returns the percentile value within a window.
func NewIntegerPercentileReduceSliceFunc(percentile float64) IntegerReduceSliceFunc {
	return func(a []IntegerPoint) []IntegerPoint {
		length := len(a)
		i := int(math.Floor(float64(length)*percentile/100.0+0.5)) - 1

		if i < 0 || i >= length {
			return nil
		}

		sort.Sort(integerPointsByValue(a))
		return []IntegerPoint{{Time: ZeroTime, Value: a[i].Value, Aux: a[i].Aux}}
	}
}

// newDerivativeIterator returns an iterator for operating on a derivative() call.
func newDerivativeIterator(input Iterator, opt IteratorOptions, interval Interval, isNonNegative bool) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatDerivativeReducer(interval, isNonNegative, opt.Ascending)
			return fn, fn
		}
		return newFloatStreamFloatIterator(input, createFn, opt), nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewIntegerDerivativeReducer(interval, isNonNegative, opt.Ascending)
			return fn, fn
		}
		return newIntegerStreamFloatIterator(input, createFn, opt), nil
	default:
		return nil, fmt.Errorf("unsupported derivative iterator type: %T", input)
	}
}

// newDifferenceIterator returns an iterator for operating on a difference() call.
func newDifferenceIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatDifferenceReducer()
			return fn, fn
		}
		return newFloatStreamFloatIterator(input, createFn, opt), nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerDifferenceReducer()
			return fn, fn
		}
		return newIntegerStreamIntegerIterator(input, createFn, opt), nil
	default:
		return nil, fmt.Errorf("unsupported difference iterator type: %T", input)
	}
}

// newElapsedIterator returns an iterator for operating on a elapsed() call.
func newElapsedIterator(input Iterator, opt IteratorOptions, interval Interval) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, IntegerPointEmitter) {
			fn := NewFloatElapsedReducer(interval)
			return fn, fn
		}
		return newFloatStreamIntegerIterator(input, createFn, opt), nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerElapsedReducer(interval)
			return fn, fn
		}
		return newIntegerStreamIntegerIterator(input, createFn, opt), nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, IntegerPointEmitter) {
			fn := NewBooleanElapsedReducer(interval)
			return fn, fn
		}
		return newBooleanStreamIntegerIterator(input, createFn, opt), nil
	case StringIterator:
		createFn := func() (StringPointAggregator, IntegerPointEmitter) {
			fn := NewStringElapsedReducer(interval)
			return fn, fn
		}
		return newStringStreamIntegerIterator(input, createFn, opt), nil
	default:
		return nil, fmt.Errorf("unsupported elapsed iterator type: %T", input)
	}
}

// newMovingAverageIterator returns an iterator for operating on a moving_average() call.
func newMovingAverageIterator(input Iterator, n int, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatMovingAverageReducer(n)
			return fn, fn
		}
		return newFloatStreamFloatIterator(input, createFn, opt), nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewIntegerMovingAverageReducer(n)
			return fn, fn
		}
		return newIntegerStreamFloatIterator(input, createFn, opt), nil
	default:
		return nil, fmt.Errorf("unsupported moving average iterator type: %T", input)
	}
}

// newCumulativeSumIterator returns an iterator for operating on a cumulative_sum() call.
func newCumulativeSumIterator(input Iterator, opt IteratorOptions) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatCumulativeSumReducer()
			return fn, fn
		}
		return newFloatStreamFloatIterator(input, createFn, opt), nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerCumulativeSumReducer()
			return fn, fn
		}
		return newIntegerStreamIntegerIterator(input, createFn, opt), nil
	default:
		return nil, fmt.Errorf("unsupported cumulative sum iterator type: %T", input)
	}
}

// newHoltWintersIterator returns an iterator for operating on a elapsed() call.
func newHoltWintersIterator(input Iterator, opt IteratorOptions, h, m int, includeFitData bool, interval time.Duration) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatHoltWintersReducer(h, m, includeFitData, interval)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, FloatPointEmitter) {
			fn := NewFloatHoltWintersReducer(h, m, includeFitData, interval)
			return fn, fn
		}
		return &integerReduceFloatIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported elapsed iterator type: %T", input)
	}
}

// NewSampleIterator returns an iterator
func NewSampleIterator(input Iterator, opt IteratorOptions, size int) (Iterator, error) {
	return newSampleIterator(input, opt, size)
}

// newSampleIterator returns an iterator
func newSampleIterator(input Iterator, opt IteratorOptions, size int) (Iterator, error) {
	switch input := input.(type) {
	case FloatIterator:
		createFn := func() (FloatPointAggregator, FloatPointEmitter) {
			fn := NewFloatSampleReducer(size)
			return fn, fn
		}
		return &floatReduceFloatIterator{input: newBufFloatIterator(input), opt: opt, create: createFn}, nil
	case IntegerIterator:
		createFn := func() (IntegerPointAggregator, IntegerPointEmitter) {
			fn := NewIntegerSampleReducer(size)
			return fn, fn
		}
		return &integerReduceIntegerIterator{input: newBufIntegerIterator(input), opt: opt, create: createFn}, nil
	case BooleanIterator:
		createFn := func() (BooleanPointAggregator, BooleanPointEmitter) {
			fn := NewBooleanSampleReducer(size)
			return fn, fn
		}
		return &booleanReduceBooleanIterator{input: newBufBooleanIterator(input), opt: opt, create: createFn}, nil
	case StringIterator:
		createFn := func() (StringPointAggregator, StringPointEmitter) {
			fn := NewStringSampleReducer(size)
			return fn, fn
		}
		return &stringReduceStringIterator{input: newBufStringIterator(input), opt: opt, create: createFn}, nil
	default:
		return nil, fmt.Errorf("unsupported elapsed iterator type: %T", input)
	}
}
