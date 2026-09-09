// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package timeseries implements a time series structure for stats collection.
package timeseries // import "golang.org/x/net/internal/timeseries"

import (
	"fmt"
	"log"
	"time"
)

const (
	timeSeriesNumBuckets       = 64
	minuteHourSeriesNumBuckets = 60
)

var timeSeriesResolutions = []time.Duration{
	1 * time.Second,
	10 * time.Second,
	1 * time.Minute,
	10 * time.Minute,
	1 * time.Hour,
	6 * time.Hour,
	24 * time.Hour,          // 1 day
	7 * 24 * time.Hour,      // 1 week
	4 * 7 * 24 * time.Hour,  // 4 weeks
	16 * 7 * 24 * time.Hour, // 16 weeks
}

var minuteHourSeriesResolutions = []time.Duration{
	1 * time.Second,
	1 * time.Minute,
}

// An Observable is a kind of data that can be aggregated in a time series.
type Observable interface {
	Multiply(ratio float64)    // Multiplies the data in self by a given ratio
	Add(other Observable)      // Adds the data from a different observation to self
	Clear()                    // Clears the observation so it can be reused.
	CopyFrom(other Observable) // Copies the contents of a given observation to self
}

// Float attaches the methods of Observable to a float64.
type Float float64

// NewFloat returns a Float.
func NewFloat() Observable {
	f := Float(0)
	return &f
}

// String returns the float as a string.
func (f *Float) String() string { return fmt.Sprintf("%g", f.Value()) }

// Value returns the float's value.
func (f *Float) Value() float64 { return float64(*f) }

func (f *Float) Multiply(ratio float64) { *f *= Float(ratio) }

func (f *Float) Add(other Observable) {
	o := other.(*Float)
	*f += *o
}

func (f *Float) Clear() { *f = 0 }

func (f *Float) CopyFrom(other Observable) {
	o := other.(*Float)
	*f = *o
}

// A Clock tells the current time.
type Clock interface {
	Time() time.Time
}

type defaultClock int

var defaultClockInstance defaultClock

func (defaultClock) Time() time.Time { return time.Now() }

// Information kept per level. Each level consists of a circular list of
// observations. The start of the level may be derived from end and the
// len(buckets) * sizeInMillis.
type tsLevel struct {
	oldest   int               // index to oldest bucketed Observable
	newest   int               // index to newest bucketed Observable
	end      time.Time         // end timestamp for this level
	size     time.Duration     // duration of the bucketed Observable
	buckets  []Observable      // collections of observations
	provider func() Observable // used for creating new Observable
}

func (l *tsLevel) Clear() {
	l.oldest = 0
	l.newest = len(l.buckets) - 1
	l.end = time.Time{}
	for i := range l.buckets {
		if l.buckets[i] != nil {
			l.buckets[i].Clear()
			l.buckets[i] = nil
		}
	}
}

func (l *tsLevel) InitLevel(size time.Duration, numBuckets int, f func() Observable) {
	l.size = size
	l.provider = f
	l.buckets = make([]Observable, numBuckets)
}

// Keeps a sequence of levels. Each level is responsible for storing data at
// a given resolution. For example, the first level stores data at a one
// minute resolution while the second level stores data at a one hour
// resolution.

// Each level is represented by a sequence of buckets. Each bucket spans an
// interval equal to the resolution of the level. New observations are added
// to the last bucket.
type timeSeries struct {
	provider    func() Observable // make more Observable
	numBuckets  int               // number of buckets in each level
	levels      []*tsLevel        // levels of bucketed Observable
	lastAdd     time.Time         // time of last Observable tracked
	total       Observable        // convenient aggregation of all Observable
	clock       Clock             // Clock for getting current time
	pending     Observable        // observations not yet bucketed
	pendingTime time.Time         // what time are we keeping in pending
	dirty       bool              // if there are pending observations
}

// init initializes a level according to the supplied criteria.
func (ts *timeSeries) init(resolutions []time.Duration, f func() Observable, numBuckets int, clock Clock) {
	ts.provider = f
	ts.numBuckets = numBuckets
	ts.clock = clock
	ts.levels = make([]*tsLevel, len(resolutions))

	for i := range resolutions {
		if i > 0 && resolutions[i-1] >= resolutions[i] {
			log.Print("timeseries: resolutions must be monotonically increasing")
			break
		}
		newLevel := new(tsLevel)
		newLevel.InitLevel(resolutions[i], ts.numBuckets, ts.provider)
		ts.levels[i] = newLevel
	}

	ts.Clear()
}

// Clear removes all observations from the time series.
func (ts *timeSeries) Clear() {
	ts.lastAdd = time.Time{}
	ts.total = ts.resetObservation(ts.total)
	ts.pending = ts.resetObservation(ts.pending)
	ts.pendingTime = time.Time{}
	ts.dirty = false

	for i := range ts.levels {
		ts.levels[i].Clear()
	}
}

// Add records an observation at the current time.
func (ts *timeSeries) Add(observation Observable) {
	ts.AddWithTime(observation, ts.clock.Time())
}

// AddWithTime records an observation at the specified time.
func (ts *timeSeries) AddWithTime(observation Observable, t time.Time) {

	smallBucketDuration := ts.levels[0].size

	if t.After(ts.lastAdd) {
		ts.lastAdd = t
	}

	if t.After(ts.pendingTime) {
		ts.advance(t)
		ts.mergePendingUpdates()
		ts.pendingTime = ts.levels[0].end
		ts.pending.CopyFrom(observation)
		ts.dirty = true
	} else if t.After(ts.pendingTime.Add(-1 * smallBucketDuration)) {
		// The observation is close enough to go into the pending bucket.
		// This compensates for clock skewing and small scheduling delays
		// by letting the update stay in the fast path.
		ts.pending.Add(observation)
		ts.dirty = true
	} else {
		ts.mergeValue(observation, t)
	}
}

// mergeValue inserts the observation at the specified time in the past into all levels.
func (ts *timeSeries) mergeValue(observation Observable, t time.Time) {
	for _, level := range ts.levels {
		index := (ts.numBuckets - 1) - int(level.end.Sub(t)/level.size)
		if 0 <= index && index < ts.numBuckets {
			bucketNumber := (level.oldest + index) % ts.numBuckets
			if level.buckets[bucketNumber] == nil {
				level.buckets[bucketNumber] = level.provider()
			}
			level.buckets[bucketNumber].Add(observation)
		}
	}
	ts.total.Add(observation)
}

// mergePendingUpdates applies the pending updates into all levels.
func (ts *timeSeries) mergePendingUpdates() {
	if ts.dirty {
		ts.mergeValue(ts.pending, ts.pendingTime)
		ts.pending = ts.resetObservation(ts.pending)
		ts.dirty = false
	}
}

// advance cycles the buckets at each level until the latest bucket in
// each level can hold the time specified.
func (ts *timeSeries) advance(t time.Time) {
	if !t.After(ts.levels[0].end) {
		return
	}
	for i := 0; i < len(ts.levels); i++ {
		level := ts.levels[i]
		if !level.end.Before(t) {
			break
		}

		// If the time is sufficiently far, just clear the level and advance
		// directly.
		if !t.Before(level.end.Add(level.size * time.Duration(ts.numBuckets))) {
			for _, b := range level.buckets {
				ts.resetObservation(b)
			}
			level.end = time.Unix(0, (t.UnixNano()/level.size.Nanoseconds())*level.size.Nanoseconds())
		}

		for t.After(level.end) {
			level.end = level.end.Add(level.size)
			level.newest = level.oldest
			level.oldest = (level.oldest + 1) % ts.numBuckets
			ts.resetObservation(level.buckets[level.newest])
		}

		t = level.end
	}
}

// Latest returns the sum of the num latest buckets from the level.
func (ts *timeSeries) Latest(level, num int) Observable {
	now := ts.clock.Time()
	if ts.levels[0].end.Before(now) {
		ts.advance(now)
	}

	ts.mergePendingUpdates()

	result := ts.provider()
	l := ts.levels[level]
	index := l.newest

	for i := 0; i < num; i++ {
		if l.buckets[index] != nil {
			result.Add(l.buckets[index])
		}
		if index == 0 {
			index = ts.numBuckets
		}
		index--
	}

	return result
}

// LatestBuckets returns a copy of the num latest buckets from level.
func (ts *timeSeries) LatestBuckets(level, num int) []Observable {
	if level < 0 || level > len(ts.levels) {
		log.Print("timeseries: bad level argument: ", level)
		return nil
	}
	if num < 0 || num >= ts.numBuckets {
		log.Print("timeseries: bad num argument: ", num)
		return nil
	}

	results := make([]Observable, num)
	now := ts.clock.Time()
	if ts.levels[0].end.Before(now) {
		ts.advance(now)
	}

	ts.mergePendingUpdates()

	l := ts.levels[level]
	index := l.newest

	for i := 0; i < num; i++ {
		result := ts.provider()
		results[i] = result
		if l.buckets[index] != nil {
			result.CopyFrom(l.buckets[index])
		}

		if index == 0 {
			index = ts.numBuckets
		}
		index -= 1
	}
	return results
}

// ScaleBy updates observations by scaling by factor.
func (ts *timeSeries) ScaleBy(factor float64) {
	for _, l := range ts.levels {
		for i := 0; i < ts.numBuckets; i++ {
			l.buckets[i].Multiply(factor)
		}
	}

	ts.total.Multiply(factor)
	ts.pending.Multiply(factor)
}

// Range returns the sum of observations added over the specified time range.
// If start or finish times don't fall on bucket boundaries of the same
// level, then return values are approximate answers.
func (ts *timeSeries) Range(start, finish time.Time) Observable {
	return ts.ComputeRange(start, finish, 1)[0]
}

// Recent returns the sum of observations from the last delta.
func (ts *timeSeries) Recent(delta time.Duration) Observable {
	now := ts.clock.Time()
	return ts.Range(now.Add(-delta), now)
}

// Total returns the total of all observations.
func (ts *timeSeries) Total() Observable {
	ts.mergePendingUpdates()
	return ts.total
}

// ComputeRange computes a specified number of values into a slice using
// the observations recorded over the specified time period. The return
// values are approximate if the start or finish times don't fall on the
// bucket boundaries at the same level or if the number of buckets spanning
// the range is not an integral multiple of num.
func (ts *timeSeries) ComputeRange(start, finish time.Time, num int) []Observable {
	if start.After(finish) {
		log.Printf("timeseries: start > finish, %v>%v", start, finish)
		return nil
	}

	if num < 0 {
		log.Printf("timeseries: num < 0, %v", num)
		return nil
	}

	results := make([]Observable, num)

	for _, l := range ts.levels {
		if !start.Before(l.end.Add(-l.size * time.Duration(ts.numBuckets))) {
			ts.extract(l, start, finish, num, results)
			return results
		}
	}

	// Failed to find a level that covers the desired range. So just
	// extract from the last level, even if it doesn't cover the entire
	// desired range.
	ts.extract(ts.levels[len(ts.levels)-1], start, finish, num, results)

	return results
}

// RecentList returns the specified number of values in slice over the most
// recent time period of the specified range.
func (ts *timeSeries) RecentList(delta time.Duration, num int) []Observable {
	if delta < 0 {
		return nil
	}
	now := ts.clock.Time()
	return ts.ComputeRange(now.Add(-delta), now, num)
}

// extract returns a slice of specified number of observations from a given
// level over a given range.
func (ts *timeSeries) extract(l *tsLevel, start, finish time.Time, num int, results []Observable) {
	ts.mergePendingUpdates()

	srcInterval := l.size
	dstInterval := finish.Sub(start) / time.Duration(num)
	dstStart := start
	srcStart := l.end.Add(-srcInterval * time.Duration(ts.numBuckets))

	srcIndex := 0

	// Where should scanning start?
	if dstStart.After(srcStart) {
		advance := int(dstStart.Sub(srcStart) / srcInterval)
		srcIndex += advance
		srcStart = srcStart.Add(time.Duration(advance) * srcInterval)
	}

	// The i'th value is computed as show below.
	// interval = (finish/start)/num
	// i'th value = sum of observation in range
	//   [ start + i       * interval,
	//     start + (i + 1) * interval )
	for i := 0; i < num; i++ {
		results[i] = ts.resetObservation(results[i])
		dstEnd := dstStart.Add(dstInterval)
		for srcIndex < ts.numBuckets && srcStart.Before(dstEnd) {
			srcEnd := srcStart.Add(srcInterval)
			if srcEnd.After(ts.lastAdd) {
				srcEnd = ts.lastAdd
			}

			if !srcEnd.Before(dstStart) {
				srcValue := l.buckets[(srcIndex+l.oldest)%ts.numBuckets]
				if !srcStart.Before(dstStart) && !srcEnd.After(dstEnd) {
					// dst completely contains src.
					if srcValue != nil {
						results[i].Add(srcValue)
					}
				} else {
					// dst partially overlaps src.
					overlapStart := maxTime(srcStart, dstStart)
					overlapEnd := minTime(srcEnd, dstEnd)
					base := srcEnd.Sub(srcStart)
					fraction := overlapEnd.Sub(overlapStart).Seconds() / base.Seconds()

					used := ts.provider()
					if srcValue != nil {
						used.CopyFrom(srcValue)
					}
					used.Multiply(fraction)
					results[i].Add(used)
				}

				if srcEnd.After(dstEnd) {
					break
				}
			}
			srcIndex++
			srcStart = srcStart.Add(srcInterval)
		}
		dstStart = dstStart.Add(dstInterval)
	}
}

// resetObservation clears the content so the struct may be reused.
func (ts *timeSeries) resetObservation(observation Observable) Observable {
	if observation == nil {
		observation = ts.provider()
	} else {
		observation.Clear()
	}
	return observation
}

// TimeSeries tracks data at granularities from 1 second to 16 weeks.
type TimeSeries struct {
	timeSeries
}

// NewTimeSeries creates a new TimeSeries using the function provided for creating new Observable.
func NewTimeSeries(f func() Observable) *TimeSeries {
	return NewTimeSeriesWithClock(f, defaultClockInstance)
}

// NewTimeSeriesWithClock creates a new TimeSeries using the function provided for creating new Observable and the clock for
// assigning timestamps.
func NewTimeSeriesWithClock(f func() Observable, clock Clock) *TimeSeries {
	ts := new(TimeSeries)
	ts.timeSeries.init(timeSeriesResolutions, f, timeSeriesNumBuckets, clock)
	return ts
}

// MinuteHourSeries tracks data at granularities of 1 minute and 1 hour.
type MinuteHourSeries struct {
	timeSeries
}

// NewMinuteHourSeries creates a new MinuteHourSeries using the function provided for creating new Observable.
func NewMinuteHourSeries(f func() Observable) *MinuteHourSeries {
	return NewMinuteHourSeriesWithClock(f, defaultClockInstance)
}

// NewMinuteHourSeriesWithClock creates a new MinuteHourSeries using the function provided for creating new Observable and the clock for
// assigning timestamps.
func NewMinuteHourSeriesWithClock(f func() Observable, clock Clock) *MinuteHourSeries {
	ts := new(MinuteHourSeries)
	ts.timeSeries.init(minuteHourSeriesResolutions, f,
		minuteHourSeriesNumBuckets, clock)
	return ts
}

func (ts *MinuteHourSeries) Minute() Observable {
	return ts.timeSeries.Latest(0, 60)
}

func (ts *MinuteHourSeries) Hour() Observable {
	return ts.timeSeries.Latest(1, 60)
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}

func maxTime(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}
