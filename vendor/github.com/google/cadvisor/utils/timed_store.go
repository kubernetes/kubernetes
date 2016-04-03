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
func (self *TimedStore) Add(timestamp time.Time, item interface{}) {
	// Remove any elements if over our max size.
	if self.maxItems >= 0 && (len(self.buffer)+1) > self.maxItems {
		startIndex := len(self.buffer) + 1 - self.maxItems
		self.buffer = self.buffer[startIndex:]
	}
	// Add the new element first and sort. We can then remove an expired element, if required.
	copied := item
	self.buffer = append(self.buffer, timedStoreData{
		timestamp: timestamp,
		data:      copied,
	})

	sort.Sort(self.buffer)
	// Remove any elements before eviction time.
	// TODO(rjnagal): This is assuming that the added entry has timestamp close to now.
	evictTime := timestamp.Add(-self.age)
	index := sort.Search(len(self.buffer), func(index int) bool {
		return self.buffer[index].timestamp.After(evictTime)
	})
	if index < len(self.buffer) {
		self.buffer = self.buffer[index:]
	}

}

// Returns up to maxResult elements in the specified time period (inclusive).
// Results are from first to last. maxResults of -1 means no limit.
func (self *TimedStore) InTimeRange(start, end time.Time, maxResults int) []interface{} {
	// No stats, return empty.
	if len(self.buffer) == 0 {
		return []interface{}{}
	}

	var startIndex int
	if start.IsZero() {
		// None specified, start at the beginning.
		startIndex = len(self.buffer) - 1
	} else {
		// Start is the index before the elements smaller than it. We do this by
		// finding the first element smaller than start and taking the index
		// before that element
		startIndex = sort.Search(len(self.buffer), func(index int) bool {
			// buffer[index] < start
			return self.getData(index).timestamp.Before(start)
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
		endIndex = sort.Search(len(self.buffer), func(index int) bool {
			// buffer[index] <= t -> !(buffer[index] > t)
			return !self.getData(index).timestamp.After(end)
		})
		// Check if end is before all the data we have.
		if endIndex == len(self.buffer) {
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
		result[i] = self.Get(startIndex - i)
	}
	return result
}

// Gets the element at the specified index. Note that elements are output in LIFO order.
func (self *TimedStore) Get(index int) interface{} {
	return self.getData(index).data
}

// Gets the data at the specified index. Note that elements are output in LIFO order.
func (self *TimedStore) getData(index int) timedStoreData {
	return self.buffer[len(self.buffer)-index-1]
}

func (self *TimedStore) Size() int {
	return len(self.buffer)
}
