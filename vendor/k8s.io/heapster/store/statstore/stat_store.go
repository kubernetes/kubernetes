// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package statstore

import (
	"container/list"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// StatStore is an in-memory rolling timeseries window that allows the extraction of -
// segments of the stored timeseries and derived stats.
// StatStore only retains information about the latest TimePoints that are within its
// specified duration.

// The tradeoff between space and precision can be configured through the epsilon and resolution -
// parameters.
// @epsilon: the acceptable error margin, in absolute value, such as 1024 bytes.
// @resolution: the desired resolution of the StatStore, such as 2 * time.Minute

// Compression in the StatStore is performed by storing values of consecutive time resolutions -
// that differ less than epsilon in the same "bucket", represented by the tpBucket struct.

// For example, a timeseries may be represented in the following manner.
// Each line parallel to the x-axis represents a single tpBucket.
// This example leads to 4 tpBuckets being stored in the StatStore, even though the length of the
// window is 7 resolutions.
//
// Legend:
//	ε : epsilon
//	δ : resolution
//      W : window length
//
//        value ^
//		|
//		|
//	     4ε |      	 ----
//	     	|	 |  |
//	     2ε |--------|  |		 |----|
//	      ε	|	    |------------|
//		|_____________________________|______>
//					       W   time
//		|--------|--|------------|----|
//		   2δ     δ       3δ        δ

// StatStore assumes that values are inserted in a chronologically ascending order through the -
// Put method. If a TimePoint with a past Timestamp is inserted, it is ignored.
// The last one resolution's worth of Timepoints are held in a putState structure, as -
// we are not confident of that resolution's average until values from the next resolution have -
// arrived. Due to this assumption, the data extraction methods of the StatStore ignore values -
// currently in the putState struct.

func NewStatStore(epsilon uint64, resolution time.Duration, windowDuration uint, supportedPercentiles []float64) *StatStore {
	return &StatStore{
		buffer:               list.New(),
		epsilon:              epsilon,
		resolution:           resolution,
		windowDuration:       windowDuration,
		supportedPercentiles: supportedPercentiles,
	}
}

// TimePoint is a single point of a timeseries, representing a time-value pair.
type TimePoint struct {
	Timestamp time.Time
	Value     uint64
}

// StatStore is a TimeStore-like object that does not implement the TimeStore interface.
type StatStore struct {
	// start is the start of the represented time window.
	start time.Time

	// buffer is a list of tpBucket that is sequenced in a time-descending order, meaning that
	// Front points to the latest tpBucket and Back to the oldest one.
	buffer *list.List

	// A RWMutex guards all operations on the StatStore.
	sync.RWMutex

	// epsilon is the acceptable error difference for the storage of TimePoints.
	// Increasing epsilon decreases memory usage of the StatStore, at the cost of precision.
	// The precision of max is not affected by epsilon.
	epsilon uint64

	// resolution is the standardized duration between points in the StatStore.
	// the Get operation returns TimePoints at every multiple of resolution,
	// even if TimePoints have not been Put for all such times.
	resolution time.Duration

	// windowDuration is the maximum number of time resolutions that is stored in the StatStore
	// e.g. windowDuration 60, with a resolution of time.Minute represents an 1-hour window
	windowDuration uint

	// tpCount is the number of TimePoints that are represented in the StatStore.
	// If tpCount is equal to windowDuration, then the StatStore window is considered full.
	tpCount uint

	// lastPut maintains the state of values inserted within the last resolution.
	// When values of a later resolution are added to the StatStore, lastPut is flushed to
	// the last tpBucket, if its average is within epsilon. Otherwise, a new bucket is -
	// created.
	lastPut putState

	// suportedPercentiles is a slice of values from (0,1) that represents the percentiles
	// that are calculated by the StatStore.
	supportedPercentiles []float64

	// validCache is true if lastPut has not been flushed since the calculation of the -
	// cached derived stats.
	validCache bool

	// cachedAverage, cachedMax and cachedPercentiles are the cached derived stats that -
	// are exposed by the StatStore. They are calculated upon the first request of any -
	// derived stat, and invalidated when lastPut is flushed into the StatStore
	cachedAverage     uint64
	cachedMax         uint64
	cachedPercentiles []uint64
}

// tpBucket is a bucket that represents a set of consecutive TimePoints with different
// timestamps whose values differ less than epsilon.
// tpBucket essentially represents a time window with a constant value.
type tpBucket struct {
	// count is the number of TimePoints represented in the tpBucket.
	count uint

	// value is the approximate value of all in the tpBucket, +- epsilon.
	value uint64

	// max is the maximum value of all TimePoints that have been used to generate the tpBucket.
	max uint64

	// maxIdx is the number of resolutions after the start time where the max value is located.
	maxIdx uint
}

// putState is a structure that maintains context of the values in the resolution that is currently
// being inserted.
// Assumes that Puts are performed in a time-ascending order.
type putState struct {
	actualCount uint
	average     float64
	max         uint64
	stamp       time.Time
}

// IsEmpty returns true if the StatStore is empty
func (ss *StatStore) IsEmpty() bool {
	if ss.buffer.Front() == nil {
		return true
	}
	return false
}

// MaxSize returns the total duration of data that can be stored in the StatStore.
func (ss *StatStore) MaxSize() time.Duration {
	return time.Duration(ss.windowDuration) * ss.resolution
}

func (ss *StatStore) Put(tp TimePoint) error {
	ss.Lock()
	defer ss.Unlock()

	// Flatten timestamp to the last multiple of resolution
	ts := tp.Timestamp.Truncate(ss.resolution)

	lastPutTime := ss.lastPut.stamp

	// Handle the case where the buffer and lastPut are both empty
	if lastPutTime.Equal(time.Time{}) {
		ss.resetLastPut(ts, tp.Value)
		return nil
	}

	if ts.Before(lastPutTime) {
		// Ignore TimePoints with Timestamps in the past
		return fmt.Errorf("the provided timepoint has a timestamp in the past")
	}

	if ts.Equal(lastPutTime) {
		// update lastPut with the new TimePoint
		newVal := tp.Value
		if newVal > ss.lastPut.max {
			ss.lastPut.max = newVal
		}
		oldAvg := ss.lastPut.average
		n := float64(ss.lastPut.actualCount)
		ss.lastPut.average = (float64(newVal) + (n * oldAvg)) / (n + 1)
		ss.lastPut.actualCount++
		return nil
	}

	ss.flush(ts, tp.Value)
	return nil
}

// resetLastPut initializes the lastPut field of the StatStore, given a time and a value.
func (ss *StatStore) resetLastPut(timestamp time.Time, value uint64) {
	ss.lastPut.stamp = timestamp
	ss.lastPut.actualCount = 1
	ss.lastPut.average = float64(value)
	ss.lastPut.max = value
}

// newBucket appends a new bucket to the StatStore, using the values of lastPut.
// newBucket should be always called BEFORE resetting lastPut.
// newBuckets are created by rounding up the lastPut average to the closest epsilon.
// numRes represents the number of resolutions from the newest TimePoint to the lastPut.
// numRes resolutions will be represented in the newly created bucket.
func (ss *StatStore) newBucket(numRes uint) {
	// Calculate the value of the new bucket based on the average of lastPut.
	newVal := (uint64(ss.lastPut.average) / ss.epsilon) * ss.epsilon
	if (uint64(ss.lastPut.average) % ss.epsilon) != 0 {
		newVal += ss.epsilon
	}
	newEntry := tpBucket{
		count:  numRes,
		value:  newVal,
		max:    ss.lastPut.max,
		maxIdx: 0,
	}
	ss.buffer.PushFront(newEntry)
	ss.tpCount += numRes

	// If this was the first bucket, update ss.start
	if ss.start.Equal(time.Time{}) {
		ss.start = ss.lastPut.stamp
	}
}

// flush causes the lastPut struct to be flushed to the StatStore list.
func (ss *StatStore) flush(ts time.Time, val uint64) {
	// The new point is in the future, lastPut needs to be flushed to the StatStore.
	ss.validCache = false

	// Determine how many resolutions in the future the new point is at.
	// The StatStore always represents values up until 1 resolution from lastPut.
	// If the TimePoint is more than one resolutions in the future, the last bucket is -
	// extended to be exactly one resolution behind the new lastPut timestamp.
	numRes := uint(0)
	curr := ts
	for curr.After(ss.lastPut.stamp) {
		curr = curr.Add(-ss.resolution)
		numRes++
	}

	// Create a new bucket if the buffer is empty
	if ss.IsEmpty() {
		ss.newBucket(numRes)
		ss.resetLastPut(ts, val)
		for ss.tpCount > ss.windowDuration {
			ss.rewind()
		}
		return
	}

	lastElem := ss.buffer.Front()
	lastEntry := lastElem.Value.(tpBucket)
	lastAvg := ss.lastPut.average

	// Place lastPut in the latest bucket if the difference from its average
	// is less than epsilon
	if uint64(math.Abs(float64(lastEntry.value)-lastAvg)) < ss.epsilon {
		lastEntry.count += numRes
		ss.tpCount += numRes
		if ss.lastPut.max > lastEntry.max {
			lastEntry.max = ss.lastPut.max
			lastEntry.maxIdx = lastEntry.count - 1
		}

		// update in list
		lastElem.Value = lastEntry
	} else {
		// Create a new bucket
		ss.newBucket(numRes)
	}

	// Delete the earliest represented TimePoints if the window is full
	for ss.tpCount > ss.windowDuration {
		ss.rewind()
	}

	ss.resetLastPut(ts, val)
}

// rewind deletes the oldest one resolution of data in the StatStore.
func (ss *StatStore) rewind() {
	firstElem := ss.buffer.Back()
	firstEntry := firstElem.Value.(tpBucket)
	// Decrement number of TimePoints in the earliest tpBucket
	firstEntry.count--
	// Decrement total number of TimePoints in the StatStore
	ss.tpCount--

	// Update the max
	if firstEntry.maxIdx == 0 {
		// The Max value was just removed, lose precision for other maxes in this bucket
		firstEntry.max = firstEntry.value
		firstEntry.maxIdx = firstEntry.count - 1
	} else {
		firstEntry.maxIdx--
	}

	if firstEntry.count == 0 {
		// Delete the entry if no TimePoints are represented any more
		ss.buffer.Remove(firstElem)
	} else {
		firstElem.Value = firstEntry
	}

	// Update the start time of the StatStore
	ss.start = ss.start.Add(ss.resolution)
}

// Get generates a []TimePoint from the appropriate tpEntries.
// Get receives a start and end time as parameters.
// If start or end are equal to time.Time{}, then we consider no such bound.
func (ss *StatStore) Get(start, end time.Time) []TimePoint {
	ss.RLock()
	defer ss.RUnlock()

	var result []TimePoint

	if start.After(end) && end.After(time.Time{}) {
		return result
	}

	// Generate a TimePoint for the lastPut, if within range
	low := start.Equal(time.Time{}) || start.Before(ss.lastPut.stamp)
	hi := end.Equal(time.Time{}) || !end.Before(ss.lastPut.stamp)
	if ss.lastPut.actualCount > 0 && low && hi {
		newTP := TimePoint{
			Timestamp: ss.lastPut.stamp,
			Value:     uint64(ss.lastPut.max), // expose the max to avoid conflicts when viewing derived stats
		}
		result = append(result, newTP)
	}

	if ss.IsEmpty() {
		return result
	}

	// Generate TimePoints from the buckets in the buffer
	skipped := 0
	for elem := ss.buffer.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(tpBucket)

		// calculate the start time of the entry
		offset := int(ss.tpCount) - skipped - int(entry.count)
		entryStart := ss.start.Add(time.Duration(offset) * ss.resolution)

		// ignore tpEntries later than the requested end time
		if end.After(time.Time{}) && entryStart.After(end) {
			skipped += int(entry.count)
			continue
		}

		// break if we have reached a tpBucket with no values before or equal to
		// the start time.
		if !entryStart.Add(time.Duration(entry.count-1) * ss.resolution).After(start) {
			break
		}

		// generate as many TimePoints as required from this bucket
		newSkip := 0
		for curr := 1; curr <= int(entry.count); curr++ {
			offset = int(ss.tpCount) - skipped - curr
			newStamp := ss.start.Add(time.Duration(offset) * ss.resolution)
			if end.After(time.Time{}) && newStamp.After(end) {
				continue
			}

			if newStamp.Before(start) {
				break
			}

			// this TimePoint is within (start, end), generate it
			newSkip++
			newTP := TimePoint{
				Timestamp: newStamp,
				Value:     entry.value,
			}
			result = append(result, newTP)
		}
		skipped += newSkip
	}

	return result
}

// Last returns the latest TimePoint, representing the average value of lastPut.
// Last also returns the max value of all Puts represented in lastPut.
// Last returns an error if no Put operations have been performed on the StatStore.
func (ss *StatStore) Last() (TimePoint, uint64, error) {
	ss.RLock()
	defer ss.RUnlock()

	if ss.lastPut.stamp.Equal(time.Time{}) {
		return TimePoint{}, uint64(0), fmt.Errorf("the StatStore is empty")
	}

	tp := TimePoint{
		Timestamp: ss.lastPut.stamp,
		Value:     uint64(ss.lastPut.average),
	}

	return tp, ss.lastPut.max, nil
}

// fillCache caches the average, max and percentiles of the StatStore.
// Assumes a write lock is taken by the caller.
func (ss *StatStore) fillCache() {
	// Calculate the average and max, flatten values into a slice
	sum := uint64(0)
	curMax := ss.lastPut.max
	vals := []float64{}
	for elem := ss.buffer.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(tpBucket)

		// Calculate the weighted sum of all tpBuckets
		sum += uint64(entry.count) * entry.value

		// Compare the bucket value with the current max
		if entry.value > curMax {
			curMax = entry.value
		}

		// Create a slice of values to generate percentiles
		for i := uint(0); i < entry.count; i++ {
			vals = append(vals, float64(entry.value))
		}
	}
	ss.cachedAverage = sum / uint64(ss.tpCount)
	ss.cachedMax = curMax

	// Calculate all supported percentiles
	sort.Float64s(vals)
	ss.cachedPercentiles = []uint64{}
	for _, spc := range ss.supportedPercentiles {
		pcIdx := int(math.Trunc(spc * float64(ss.tpCount)))
		ss.cachedPercentiles = append(ss.cachedPercentiles, uint64(vals[pcIdx]))
	}

	ss.validCache = true
}

// Average returns a weighted average across all buckets, using the count of -
// resolutions at each bucket as the weight.
func (ss *StatStore) Average() (uint64, error) {
	ss.Lock()
	defer ss.Unlock()

	if ss.IsEmpty() {
		return uint64(0), fmt.Errorf("the StatStore is empty")
	}

	if !ss.validCache {
		ss.fillCache()
	}
	return ss.cachedAverage, nil
}

// Max returns the maximum element currently in the StatStore.
// Max does NOT consider the case where the maximum is in the last one minute.
func (ss *StatStore) Max() (uint64, error) {
	ss.Lock()
	defer ss.Unlock()

	if ss.IsEmpty() {
		return uint64(0), fmt.Errorf("the StatStore is empty")
	}

	if !ss.validCache {
		ss.fillCache()
	}
	return ss.cachedMax, nil
}

// Percentile returns the requested percentile from the StatStore.
func (ss *StatStore) Percentile(p float64) (uint64, error) {
	ss.Lock()
	defer ss.Unlock()

	if ss.IsEmpty() {
		return uint64(0), fmt.Errorf("the StatStore is empty")
	}

	// Check if the specific percentile is supported
	found := false
	idx := 0
	for i, spc := range ss.supportedPercentiles {
		if p == spc {
			found = true
			idx = i
			break
		}
	}

	if !found {
		return uint64(0), fmt.Errorf("the requested percentile is not supported")
	}

	if !ss.validCache {
		ss.fillCache()
	}

	return ss.cachedPercentiles[idx], nil
}
