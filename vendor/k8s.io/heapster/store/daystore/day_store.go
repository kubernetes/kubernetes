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

package daystore

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	statstore "k8s.io/heapster/store/statstore"

	"k8s.io/heapster/third_party/window"
)

// DayStore is an in-memory window of derived stats for each hour of a day.
// DayStore holds 24 hours of derived stats, plus an hour-long StatStore that holds -
// historical data on the current hour.
// DayStore can calculate the Average, Max and 95th Percentile over the past 24 hours.
// The DayStore needs to be populated in a chronological order.

// Note on how derived stats are extracted:
// If the DayStore holds less than 1 hour of data, then the average, max and 95th are -
// calculated only from the the Hour store.
//
// If the DayStore holds more than 1 hour of data, then the average and 95th are calculated -
// only from the 24 past hourly stats that have been collected.
// Assuming a resolution of 1 minute, this behavior may ignore up to 59 minutes of the latest data.
// This behavior is required to avoid calculating less than a full day's data.
// In order to correctly capture spikes, the Max takes into account the latest data, causing the -
// max to reflect values in the past [24 hours, 25 hours)
type DayStore struct {
	// a RWMutex guards all operations on the underlying window and cached values.
	sync.RWMutex

	// Hour is a StatStore with data from the last one hour
	Hour *statstore.StatStore

	window *window.MovingWindow

	// size is the number of items currently stored in the window
	size int

	// lastFlush is the time when the previous hourly stats were flushed into the DayStore
	lastFlush time.Time

	// validAvgPct marks whether the cached average and percentiles are correct.
	validAvgPct bool
	// validMax marks whether the cached max value is correct.
	validMax bool

	cachedAverage     uint64
	cachedMax         uint64
	cachedNinetyFifth uint64
}

// hourEntry is the set of derived stats that are maintained per hour.
type hourEntry struct {
	average     uint64
	max         uint64
	ninetyFifth uint64
}

// NewDayStore is a DayStore constructor.
// The recommended minimum resolution is at least one minute.
func NewDayStore(epsilon uint64, resolution time.Duration) *DayStore {
	// Calculate how many resolutions correspond to an hour
	hourNS := time.Hour.Nanoseconds()
	resNS := resolution.Nanoseconds()
	intervals := uint(hourNS / resNS)

	if hourNS%resNS != 0 {
		intervals++
	}

	return &DayStore{
		window: window.New(24, 1),
		Hour:   statstore.NewStatStore(epsilon, resolution, intervals, []float64{0.95}),
	}
}

// Put stores a TimePoint into the Hour StatStore, while checking whether it -
// is time to flush the last hour's stats in the window.
// Put operations need to be performed in a chronological (time-ascending) order
func (ds *DayStore) Put(tp statstore.TimePoint) error {
	ds.Lock()
	defer ds.Unlock()

	err := ds.Hour.Put(tp)
	if err != nil {
		return err
	}

	if tp.Value > ds.cachedMax {
		ds.validMax = false
	}

	// Check if this is the first TimePoint ever, in which case flush in one hour from now.
	if ds.lastFlush.Equal(time.Time{}) {
		ds.lastFlush = tp.Timestamp
		return nil
	}

	// The new TimePoint is not newer by at least one hour since the last flush
	if tp.Timestamp.Add(-time.Hour).Before(ds.lastFlush) {
		return nil
	}

	// create an hourEntry for the existing hour
	ds.validAvgPct = false
	avg, _ := ds.Hour.Average()
	max, _ := ds.Hour.Max()
	nf, _ := ds.Hour.Percentile(0.95)
	newEntry := hourEntry{
		average:     avg,
		max:         max,
		ninetyFifth: nf,
	}

	// check if the TimePoint is multiple hours in the future
	// insert the hourEntry the appropriate amount of hours
	distance := tp.Timestamp.Sub(ds.lastFlush)
	nextflush := tp.Timestamp
	for distance.Nanoseconds() >= time.Hour.Nanoseconds() {
		ds.lastFlush = nextflush
		nextflush = ds.lastFlush.Add(time.Hour)
		if ds.size < 24 {
			ds.size++
		}
		ds.window.PushBack(newEntry)
		distance = time.Time{}.Add(distance).Add(-time.Hour).Sub(time.Time{})
	}
	return nil
}

// fillMax caches the max of the DayStore.
func (ds *DayStore) fillMax() {
	// generate a slice of the window
	day := ds.window.Slice()

	// calculate th max of the hourly maxes
	curMax, _ := ds.Hour.Max()
	for _, elem := range day {
		he := elem.(hourEntry)
		if he.max > curMax {
			curMax = he.max
		}
	}
	ds.cachedMax = curMax
	ds.validMax = true
}

// fillAvgPct caches the average, 95th percentile of the DayStore.
func (ds *DayStore) fillAvgPct() {
	ds.validAvgPct = true

	// If no past Hourly data has been flushed to the window,
	// return the average and 95th percentile of the past hour.
	if ds.size == 0 {
		ds.cachedAverage, _ = ds.Hour.Average()
		ds.cachedNinetyFifth, _ = ds.Hour.Percentile(0.95)
		return
	}
	// Otherwise, ignore the past one hour and use the window values

	// generate a slice of the window
	day := ds.window.Slice()

	// calculate the average value of the hourly averages
	// also create a sortable slice of float64
	var sum uint64
	var nf []float64
	for _, elem := range day {
		he := elem.(hourEntry)
		sum += he.average
		nf = append(nf, float64(he.ninetyFifth))
	}
	ds.cachedAverage = sum / uint64(ds.size)

	// sort and calculate the 95th percentile
	sort.Float64s(nf)
	pcIdx := int(math.Trunc(0.95 * float64(ds.size+1)))
	if pcIdx >= len(nf) {
		pcIdx = len(nf) - 1
	}
	ds.cachedNinetyFifth = uint64(nf[pcIdx])
}

// Average returns the average value of the hourly averages in the past day.
func (ds *DayStore) Average() (uint64, error) {
	ds.Lock()
	defer ds.Unlock()

	if ds.Hour.IsEmpty() {
		return uint64(0), fmt.Errorf("the DayStore is empty")
	}

	if !ds.validAvgPct {
		ds.fillAvgPct()
	}

	return ds.cachedAverage, nil
}

// Max returns the maximum value of the hourly maxes in the past day.
func (ds *DayStore) Max() (uint64, error) {
	ds.Lock()
	defer ds.Unlock()

	if ds.Hour.IsEmpty() {
		return uint64(0), fmt.Errorf("the DayStore is empty")
	}

	if !ds.validMax {
		ds.fillMax()
	}

	return ds.cachedMax, nil
}

// NinetyFifth returns the 95th percentile of the hourly 95th percentiles in the past day.
func (ds *DayStore) NinetyFifth() (uint64, error) {
	ds.Lock()
	defer ds.Unlock()

	if ds.Hour.IsEmpty() {
		return uint64(0), fmt.Errorf("the DayStore is empty")
	}

	if !ds.validAvgPct {
		ds.fillAvgPct()
	}

	return ds.cachedNinetyFifth, nil
}
