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

package store

import (
	"container/list"
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
)

// TODO: Consider using cadvisor's in memory storage instead.
type timeStore struct {
	// A list that will contain all the timeStore entries.
	// This list is in reverse chronological order.
	// E.x. [4, 3, 1]
	// list.Front refers to the most recent entry and list.Back refers to the oldest entry.
	buffer *list.List
	rwLock sync.RWMutex
}

func (ts *timeStore) Put(tp TimePoint) error {
	if tp.Value == nil {
		return fmt.Errorf("cannot store TimePoint with nil data")
	}
	if (tp.Timestamp == time.Time{}) {
		return fmt.Errorf("cannot store TimePoint with zero timestamp")
	}
	ts.rwLock.Lock()
	defer ts.rwLock.Unlock()
	if ts.buffer.Len() == 0 {
		glog.V(5).Infof("put pushfront: %v, %v", tp.Timestamp, tp.Value)
		ts.buffer.PushFront(tp)
		return nil
	}
	for elem := ts.buffer.Front(); elem != nil; elem = elem.Next() {
		if tp.Timestamp.After(elem.Value.(TimePoint).Timestamp) {
			glog.V(5).Infof("put insert before: %v, %v, %v", elem, tp.Timestamp, tp.Value)
			ts.buffer.InsertBefore(tp, elem)
			return nil
		}
	}
	glog.V(5).Infof("put pushback: %v, %v", tp.Timestamp, tp.Value)
	ts.buffer.PushBack(tp)
	return nil
}

// Returns true if 't1' is equal to or before 't2'
func timeEqualOrBefore(t1, t2 time.Time) bool {
	if t1.Equal(t2) || t1.Before(t2) {
		return true
	}
	return false
}

// Returns true if 't1' is equal to or after 't2'
func timeEqualOrAfter(t1, t2 time.Time) bool {
	if t1.Equal(t2) || t1.After(t2) {
		return true
	}
	return false
}

func (ts *timeStore) Get(start, end time.Time) []TimePoint {
	ts.rwLock.RLock()
	defer ts.rwLock.RUnlock()
	if ts.buffer.Len() == 0 {
		return nil
	}
	zeroTime := time.Time{}
	result := []TimePoint{}
	for elem := ts.buffer.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(TimePoint)
		// Break the loop if we encounter a timestamp that is before 'start'
		if entry.Timestamp.Before(start) {
			break
		}
		// Add all entries whose timestamp is before end.
		if end != zeroTime && entry.Timestamp.After(end) {
			continue
		}
		result = append(result, entry)
	}
	return result
}

func (ts *timeStore) Delete(start, end time.Time) error {
	ts.rwLock.Lock()
	defer ts.rwLock.Unlock()
	if ts.buffer.Len() == 0 {
		return nil
	}
	if (end != time.Time{}) && !end.After(start) {
		return fmt.Errorf("end time %v is not after start time %v", end, start)
	}
	// Assuming that deletes will happen more frequently for older data.
	elem := ts.buffer.Back()
	for elem != nil {
		entry := elem.Value.(TimePoint)
		if (end != time.Time{}) && entry.Timestamp.After(end) {
			// If we have reached an entry which is more recent than 'end' stop iterating.
			break
		}
		oldElem := elem
		elem = elem.Prev()

		// Skip entried which are before start.
		if !entry.Timestamp.Before(start) {
			ts.buffer.Remove(oldElem)
		}
	}
	return nil
}

func NewTimeStore() TimeStore {
	return &timeStore{
		rwLock: sync.RWMutex{},
		buffer: list.New(),
	}
}
