// Copyright 2014 Google Inc. All Rights Reserved.
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

package utils

import (
	"sort"
	"time"
)

type timedStoreDataSlice []timedStoreData

func (t timedStoreDataSlice) Less(i, j int) bool {
	return t[i].timestamp.Before(t[j].timestamp)
}

func (t timedStoreDataSlice) Len() int {
	return len(t)
}

func (t timedStoreDataSlice) Swap(i, j int) {
	t[i], t[j] = t[j], t[i]
}

// A time-based buffer for ContainerStats.
// Holds information for a specific time period and/or a max number of items.
type TimedStore struct {
	buffer   timedStoreDataSlice
	age      time.Duration
	maxItems int
}

type timedStoreData struct {
	timestamp time.Time
	data      interface{}
}

// Returns a new thread-compatible TimedStore.
// A maxItems value of -1 means no limit.
func NewTimedStore(age time.Duration, maxItems int) *TimedStore {
	return &TimedStore{
		buffer:   make(timedStoreDataSlice, 0),
		age:      age,
		maxItems: maxItems,
	}
}

// Adds an element to the start of the buffer (removing one from the end if necessary).
func (s *TimedStore) Add(timestamp time.Time, item interface{}) {
	data := timedStoreData{
		timestamp: timestamp,
		data:      item,
	}
	// Common case: data is added in order.
	if len(s.buffer) == 0 || !timestamp.Before(s.buffer[len(s.buffer)-1].timestamp) {
		s.buffer = append(s.buffer, data)
	} else {
		// Data is out of order; insert it in the correct position.
		index := sort.Search(len(s.buffer), func(index int) bool {
			return s.buffer[index].timestamp.After(timestamp)
		})
		s.buffer = append(s.buffer, timedStoreData{}) // Make room to shift the elements
		copy(s.buffer[index+1:], s.buffer[index:])    // Shift the elements over
		s.buffer[index] = data
	}

	// Remove any elements before eviction time.
	// TODO(rjnagal): This is assuming that the added entry has timestamp close to now.
	evictTime := timestamp.Add(-s.age)
	index := sort.Search(len(s.buffer), func(index int) bool {
		return s.buffer[index].timestamp.After(evictTime)
	})
	if index < len(s.buffer) {
		s.buffer = s.buffer[index:]
	}

	// Remove any elements if over our max size.
	if s.maxItems >= 0 && len(s.buffer) > s.maxItems {
		startIndex := len(s.buffer) - s.maxItems
		s.buffer = s.buffer[startIndex:]
	}
}

// Returns up to maxResult elements in the specified time period (inclusive).
// Results are from first to last. maxResults of -1 means no limit.
func (s *TimedStore) InTimeRange(start, end time.Time, maxResults int) []interface{} {
	// No stats, return empty.
	if len(s.buffer) == 0 {
		return []interface{}{}
	}

	var startIndex int
	if start.IsZero() {
		// None specified, start at the beginning.
		startIndex = len(s.buffer) - 1
	} else {
		// Start is the index before the elements smaller than it. We do this by
		// finding the first element smaller than start and taking the index
		// before that element
		startIndex = sort.Search(len(s.buffer), func(index int) bool {
			// buffer[index] < start
			return s.getData(index).timestamp.Before(start)
		}) - 1
		// Check if start is after all the data we have.
		if startIndex < 0 {
			return []interface{}{}
		}
	}

	var endIndex int
	if end.IsZero() {
		// None specified, end with the latest stats.
		endIndex = 0
	} else {
		// End is the first index smaller than or equal to it (so, not larger).
		endIndex = sort.Search(len(s.buffer), func(index int) bool {
			// buffer[index] <= t -> !(buffer[index] > t)
			return !s.getData(index).timestamp.After(end)
		})
		// Check if end is before all the data we have.
		if endIndex == len(s.buffer) {
			return []interface{}{}
		}
	}

	// Trim to maxResults size.
	numResults := startIndex - endIndex + 1
	if maxResults != -1 && numResults > maxResults {
		startIndex -= numResults - maxResults
		numResults = maxResults
	}

	// Return in sorted timestamp order so from the "back" to "front".
	result := make([]interface{}, numResults)
	for i := 0; i < numResults; i++ {
		result[i] = s.Get(startIndex - i)
	}
	return result
}

// Gets the element at the specified index. Note that elements are output in LIFO order.
func (s *TimedStore) Get(index int) interface{} {
	return s.getData(index).data
}

// Gets the data at the specified index. Note that elements are output in LIFO order.
func (s *TimedStore) getData(index int) timedStoreData {
	return s.buffer[len(s.buffer)-index-1]
}

func (s *TimedStore) Size() int {
	return len(s.buffer)
}
