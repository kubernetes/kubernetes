package stats

import (
	"math"
	"time"
)

// timeseries holds the history of a changing value over a predefined period of
// time.
type timeseries struct {
	size       int           // The number of time slots. Equivalent to len(slots).
	resolution time.Duration // The time resolution of each slot.
	stepCount  int64         // The number of intervals seen since creation.
	head       int           // The position of the current time in slots.
	time       time.Time     // The time at the beginning of the current time slot.
	slots      []int64       // A circular buffer of time slots.
}

// newTimeSeries returns a newly allocated timeseries that covers the requested
// period with the given resolution.
func newTimeSeries(initialTime time.Time, period, resolution time.Duration) *timeseries {
	size := int(period.Nanoseconds()/resolution.Nanoseconds()) + 1
	return &timeseries{
		size:       size,
		resolution: resolution,
		stepCount:  1,
		time:       initialTime,
		slots:      make([]int64, size),
	}
}

// advanceTimeWithFill moves the timeseries forward to time t and fills in any
// slots that get skipped in the process with the given value. Values older than
// the timeseries period are lost.
func (ts *timeseries) advanceTimeWithFill(t time.Time, value int64) {
	advanceTo := t.Truncate(ts.resolution)
	if !advanceTo.After(ts.time) {
		// This is shortcut for the most common case of a busy counter
		// where updates come in many times per ts.resolution.
		ts.time = advanceTo
		return
	}
	steps := int(advanceTo.Sub(ts.time).Nanoseconds() / ts.resolution.Nanoseconds())
	ts.stepCount += int64(steps)
	if steps > ts.size {
		steps = ts.size
	}
	for steps > 0 {
		ts.head = (ts.head + 1) % ts.size
		ts.slots[ts.head] = value
		steps--
	}
	ts.time = advanceTo
}

// advanceTime moves the timeseries forward to time t and fills in any slots
// that get skipped in the process with the head value. Values older than the
// timeseries period are lost.
func (ts *timeseries) advanceTime(t time.Time) {
	ts.advanceTimeWithFill(t, ts.slots[ts.head])
}

// set sets the current value of the timeseries.
func (ts *timeseries) set(value int64) {
	ts.slots[ts.head] = value
}

// incr sets the current value of the timeseries.
func (ts *timeseries) incr(delta int64) {
	ts.slots[ts.head] += delta
}

// headValue returns the latest value from the timeseries.
func (ts *timeseries) headValue() int64 {
	return ts.slots[ts.head]
}

// headTime returns the time of the latest value from the timeseries.
func (ts *timeseries) headTime() time.Time {
	return ts.time
}

// tailValue returns the oldest value from the timeseries.
func (ts *timeseries) tailValue() int64 {
	if ts.stepCount < int64(ts.size) {
		return 0
	}
	return ts.slots[(ts.head+1)%ts.size]
}

// tailTime returns the time of the oldest value from the timeseries.
func (ts *timeseries) tailTime() time.Time {
	size := int64(ts.size)
	if ts.stepCount < size {
		size = ts.stepCount
	}
	return ts.time.Add(-time.Duration(size-1) * ts.resolution)
}

// delta returns the difference between the newest and oldest values from the
// timeseries.
func (ts *timeseries) delta() int64 {
	return ts.headValue() - ts.tailValue()
}

// rate returns the rate of change between the oldest and newest values from
// the timeseries in units per second.
func (ts *timeseries) rate() float64 {
	deltaTime := ts.headTime().Sub(ts.tailTime()).Seconds()
	if deltaTime == 0 {
		return 0
	}
	return float64(ts.delta()) / deltaTime
}

// min returns the smallest value from the timeseries.
func (ts *timeseries) min() int64 {
	to := ts.size
	if ts.stepCount < int64(ts.size) {
		to = ts.head + 1
	}
	tail := (ts.head + 1) % ts.size
	min := int64(math.MaxInt64)
	for b := 0; b < to; b++ {
		if b != tail && ts.slots[b] < min {
			min = ts.slots[b]
		}
	}
	return min
}

// max returns the largest value from the timeseries.
func (ts *timeseries) max() int64 {
	to := ts.size
	if ts.stepCount < int64(ts.size) {
		to = ts.head + 1
	}
	tail := (ts.head + 1) % ts.size
	max := int64(math.MinInt64)
	for b := 0; b < to; b++ {
		if b != tail && ts.slots[b] > max {
			max = ts.slots[b]
		}
	}
	return max
}

// reset resets the timeseries to an empty state.
func (ts *timeseries) reset(t time.Time) {
	ts.head = 0
	ts.time = t
	ts.stepCount = 1
	ts.slots = make([]int64, ts.size)
}
