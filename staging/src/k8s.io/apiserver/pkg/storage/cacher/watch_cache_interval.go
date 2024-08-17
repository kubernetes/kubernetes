/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cacher

import (
	"fmt"
	"sort"
	"sync"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
)

// watchCacheInterval serves as an abstraction over a source
// of watchCacheEvents. It maintains a window of events over
// an underlying source and these events can be served using
// the exposed Next() API. The main intent for doing things
// this way is to introduce an upper bound of memory usage
// for starting a watch and reduce the maximum possible time
// interval for which the lock would be held while events are
// copied over.
//
// The source of events for the interval is typically either
// the watchCache circular buffer, if events being retrieved
// need to be for resource versions > 0 or the underlying
// implementation of Store, if resource version = 0.
//
// Furthermore, an interval can be either valid or invalid at
// any given point of time. The notion of validity makes sense
// only in cases where the window of events in the underlying
// source can change over time - i.e. for watchCache circular
// buffer. When the circular buffer is full and an event needs
// to be popped off, watchCache::startIndex is incremented. In
// this case, an interval tracking that popped event is valid
// only if it has already been copied to its internal buffer.
// However, for efficiency we perform that lazily and we mark
// an interval as invalid iff we need to copy events from the
// watchCache and we end up needing events that have already
// been popped off. This translates to the following condition:
//
//	watchCacheInterval::startIndex >= watchCache::startIndex.
//
// When this condition becomes false, the interval is no longer
// valid and should not be used to retrieve and serve elements
// from the underlying source.
type watchCacheInterval struct {
	// startIndex denotes the starting point of the interval
	// being considered. The value is the index in the actual
	// source of watchCacheEvents. If the source of events is
	// the watchCache, then this must be used modulo capacity.
	startIndex int

	// endIndex denotes the ending point of the interval being
	// considered. The value is the index in the actual source
	// of events. If the source of the events is the watchCache,
	// then this should be used modulo capacity.
	endIndex int

	// indexer is meant to inject behaviour for how an event must
	// be retrieved from the underlying source given an index.
	indexer indexerFunc

	// indexValidator is used to check if a given index is still
	// valid perspective. If it is deemed that the index is not
	// valid, then this interval can no longer be used to serve
	// events. Use of indexValidator is warranted only in cases
	// where the window of events in the underlying source can
	// change over time. Furthermore, an interval is invalid if
	// its startIndex no longer coincides with the startIndex of
	// underlying source.
	indexValidator indexValidator

	// buffer holds watchCacheEvents that this interval returns on
	// a call to Next(). This exists mainly to reduce acquiring the
	// lock on each invocation of Next().
	buffer *watchCacheIntervalBuffer

	// lock effectively protects access to the underlying source
	// of events through - indexer and indexValidator.
	//
	// Given that indexer and indexValidator only read state, if
	// possible, Locker obtained through RLocker() is provided.
	lock sync.Locker
}

type attrFunc func(runtime.Object) (labels.Set, fields.Set, error)
type indexerFunc func(int) *watchCacheEvent
type indexValidator func(int) bool

func newCacheInterval(startIndex, endIndex int, indexer indexerFunc, indexValidator indexValidator, locker sync.Locker) *watchCacheInterval {
	return &watchCacheInterval{
		startIndex:     startIndex,
		endIndex:       endIndex,
		indexer:        indexer,
		indexValidator: indexValidator,
		buffer:         &watchCacheIntervalBuffer{buffer: make([]*watchCacheEvent, bufferSize)},
		lock:           locker,
	}
}

type sortableWatchCacheEvents []*watchCacheEvent

func (s sortableWatchCacheEvents) Len() int {
	return len(s)
}

func (s sortableWatchCacheEvents) Less(i, j int) bool {
	return s[i].Key < s[j].Key
}

func (s sortableWatchCacheEvents) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// newCacheIntervalFromStore is meant to handle the case of rv=0, such that the events
// returned by Next() need to be events from a List() done on the underlying store of
// the watch cache.
// The items returned in the interval will be sorted by Key.
func newCacheIntervalFromStore(resourceVersion uint64, store cache.Store, getAttrsFunc attrFunc, key string, matchesSingle bool) (*watchCacheInterval, error) {
	buffer := &watchCacheIntervalBuffer{}
	var allItems []interface{}

	if matchesSingle {
		item, exists, err := store.GetByKey(key)
		if err != nil {
			return nil, err
		}

		if exists {
			allItems = append(allItems, item)
		}
	} else {
		allItems = store.List()
	}
	buffer.buffer = make([]*watchCacheEvent, len(allItems))
	for i, item := range allItems {
		elem, ok := item.(*storeElement)
		if !ok {
			return nil, fmt.Errorf("not a storeElement: %v", elem)
		}
		objLabels, objFields, err := getAttrsFunc(elem.Object)
		if err != nil {
			return nil, err
		}
		buffer.buffer[i] = &watchCacheEvent{
			Type:            watch.Added,
			Object:          elem.Object,
			ObjLabels:       objLabels,
			ObjFields:       objFields,
			Key:             elem.Key,
			ResourceVersion: resourceVersion,
		}
		buffer.endIndex++
	}
	sort.Sort(sortableWatchCacheEvents(buffer.buffer))
	ci := &watchCacheInterval{
		startIndex: 0,
		// Simulate that we already have all the events we're looking for.
		endIndex: 0,
		buffer:   buffer,
	}

	return ci, nil
}

// Next returns the next item in the cache interval provided the cache
// interval is still valid. An error is returned if the interval is
// invalidated.
func (wci *watchCacheInterval) Next() (*watchCacheEvent, error) {
	// if there are items in the buffer to return, return from
	// the buffer.
	if event, exists := wci.buffer.next(); exists {
		return event, nil
	}
	// check if there are still other events in this interval
	// that can be processed.
	if wci.startIndex >= wci.endIndex {
		return nil, nil
	}
	wci.lock.Lock()
	defer wci.lock.Unlock()

	if valid := wci.indexValidator(wci.startIndex); !valid {
		return nil, fmt.Errorf("cache interval invalidated, interval startIndex: %d", wci.startIndex)
	}

	wci.fillBuffer()
	if event, exists := wci.buffer.next(); exists {
		return event, nil
	}
	return nil, nil
}

func (wci *watchCacheInterval) fillBuffer() {
	wci.buffer.startIndex = 0
	wci.buffer.endIndex = 0
	for wci.startIndex < wci.endIndex && !wci.buffer.isFull() {
		event := wci.indexer(wci.startIndex)
		if event == nil {
			break
		}
		wci.buffer.buffer[wci.buffer.endIndex] = event
		wci.buffer.endIndex++
		wci.startIndex++
	}
}

const bufferSize = 100

// watchCacheIntervalBuffer is used to reduce acquiring
// the lock on each invocation of watchCacheInterval.Next().
type watchCacheIntervalBuffer struct {
	// buffer is used to hold watchCacheEvents that
	// the interval returns on a call to Next().
	buffer []*watchCacheEvent
	// The first element of buffer is defined by startIndex,
	// its last element is defined by endIndex.
	startIndex int
	endIndex   int
}

// next returns the next event present in the interval buffer provided
// it is not empty.
func (wcib *watchCacheIntervalBuffer) next() (*watchCacheEvent, bool) {
	if wcib.isEmpty() {
		return nil, false
	}
	next := wcib.buffer[wcib.startIndex]
	wcib.startIndex++
	return next, true
}

func (wcib *watchCacheIntervalBuffer) isFull() bool {
	return wcib.endIndex >= bufferSize
}

func (wcib *watchCacheIntervalBuffer) isEmpty() bool {
	return wcib.startIndex == wcib.endIndex
}
