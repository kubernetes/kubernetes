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

// cmaStore maintains the cumulative moving average of values at a specific Timestamp.
// cmaStore is an implementation of TimeStore where the values are casted as uint64.
// TODO(alex): avoid duplication with in_memory.
type cmaStore struct {
	// buffer is a list of tpContainers that is sequenced in a time-descending order.
	buffer *list.List
	// rwLock protects all operations on the buffer.
	rwLock sync.RWMutex
}

// tpContainer is the actual struct that is being stored in the buffer that implements cmaStore.
// tpContainer contains a TimePoint and the count of TimePoints with the same timestamp that
// have been averaged to a single tpContainer. The TimePoint.Value field contains the average
// of all these TimePoints.
type tpContainer struct {
	TimePoint
	count int
}

func (ts *cmaStore) Put(tp TimePoint) error {
	if tp.Value == nil {
		return fmt.Errorf("cannot store TimePoint with nil data")
	}
	if (tp.Timestamp == time.Time{}) {
		return fmt.Errorf("cannot store TimePoint with zero timestamp")
	}
	ts.rwLock.Lock()
	defer ts.rwLock.Unlock()
	newTPC := tpContainer{
		TimePoint: tp,
		count:     1,
	}
	if ts.buffer.Len() == 0 {
		glog.V(5).Infof("put pushfront: %v, %v", tp.Timestamp, tp.Value)
		ts.buffer.PushFront(newTPC)
		return nil
	}
	for elem := ts.buffer.Front(); elem != nil; elem = elem.Next() {
		curr := elem.Value.(tpContainer)
		if tp.Timestamp.Equal(curr.Timestamp) {
			// If an element with that timestamp exists, update its average and count
			newVal := tp.Value.(uint64)
			n := uint64(curr.count)
			oldAvg := curr.Value.(uint64)
			curr.Value = uint64((newVal + (n * oldAvg)) / (n + 1))
			curr.count = curr.count + 1
			elem.Value = curr
			return nil
		} else if tp.Timestamp.After(curr.Timestamp) {
			glog.V(5).Infof("put insert before: %v, %v, %v", elem, tp.Timestamp, tp.Value)
			ts.buffer.InsertBefore(newTPC, elem)
			return nil
		}
	}
	glog.V(5).Infof("put pushback: %v, %v", tp.Timestamp, tp.Value)
	ts.buffer.PushBack(newTPC)
	return nil
}

func (ts *cmaStore) Get(start, end time.Time) []TimePoint {
	ts.rwLock.RLock()
	defer ts.rwLock.RUnlock()
	if ts.buffer.Len() == 0 {
		return nil
	}
	zeroTime := time.Time{}
	result := []TimePoint{}
	for elem := ts.buffer.Front(); elem != nil; elem = elem.Next() {
		tpc := elem.Value.(tpContainer)
		entry := tpc.TimePoint
		// Skip entries until the first one after start
		if !entry.Timestamp.After(start) {
			continue
		}
		// Add all entries whose timestamp is not after end.
		if end != zeroTime && entry.Timestamp.After(end) {
			continue
		}
		result = append(result, entry)
	}
	return result
}

func (ts *cmaStore) Delete(start, end time.Time) error {
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
		tpc := elem.Value.(tpContainer)
		if (end != time.Time{}) && tpc.Timestamp.After(end) {
			// If we have reached an entry which is more recent than 'end' stop iterating.
			break
		}
		oldElem := elem
		elem = elem.Prev()

		// Skip entries before the start time.
		if !tpc.Timestamp.Before(start) {
			ts.buffer.Remove(oldElem)
		}
	}
	return nil
}

func NewCMAStore() TimeStore {
	return &cmaStore{
		buffer: list.New(),
	}
}
