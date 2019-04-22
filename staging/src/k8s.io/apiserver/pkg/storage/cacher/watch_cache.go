/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
	utiltrace "k8s.io/utils/trace"
)

const (
	// blockTimeout determines how long we're willing to block the request
	// to wait for a given resource version to be propagated to cache,
	// before terminating request and returning Timeout error with retry
	// after suggestion.
	blockTimeout = 3 * time.Second
)

// watchCacheEvent is a single "watch event" that is send to users of
// watchCache. Additionally to a typical "watch.Event" it contains
// the previous value of the object to enable proper filtering in the
// upper layers.
type watchCacheEvent struct {
	Type            watch.EventType
	Object          runtime.Object
	ObjLabels       labels.Set
	ObjFields       fields.Set
	PrevObject      runtime.Object
	PrevObjLabels   labels.Set
	PrevObjFields   fields.Set
	Key             string
	ResourceVersion uint64
}

// Computing a key of an object is generally non-trivial (it performs
// e.g. validation underneath). Similarly computing object fields and
// labels. To avoid computing them multiple times (to serve the event
// in different List/Watch requests), in the underlying store we are
// keeping structs (key, object, labels, fields).
type storeElement struct {
	Key    string
	Object runtime.Object
	Labels labels.Set
	Fields fields.Set
}

func storeElementKey(obj interface{}) (string, error) {
	elem, ok := obj.(*storeElement)
	if !ok {
		return "", fmt.Errorf("not a storeElement: %v", obj)
	}
	return elem.Key, nil
}

// watchCacheEventSlot is a slot of 'watchCacheEvent'. It is designed to be const to
// serve multiple readers, never reuse this structure to reduce GC pressure。
type watchCacheEventSlot struct {
	endIndex int
	events   []*watchCacheEvent
}

func newWatchCacheSlot(capacity int) *watchCacheEventSlot {
	return &watchCacheEventSlot{
		endIndex: 0,
		events:   make([]*watchCacheEvent, capacity),
	}
}

func (w *watchCacheEventSlot) isEmpty() bool {
	return w.endIndex == 0
}

func (w *watchCacheEventSlot) isFull() bool {
	return w.endIndex == len(w.events)
}

func (w *watchCacheEventSlot) add(event *watchCacheEvent) {
	w.events[w.endIndex] = event
	w.endIndex++
}

func (w *watchCacheEventSlot) size() int {
	return w.endIndex
}

// Caller ensures that the array is not empty
func (w *watchCacheEventSlot) firstResourceVersion() uint64 {
	return w.events[0].ResourceVersion
}

// Caller ensures that the array is not empty
func (w *watchCacheEventSlot) lastResourceVersion() uint64 {
	return w.events[w.endIndex-1].ResourceVersion
}

// watchCacheCyclicBuffer its first element (with the smallest
// resourceVersion) is defined by startIndex, its last element is defined
// by endIndex (if buffer is full it will be startIndex + 1 modulo capacity).
// Both startIndex and endIndex will be within than buffer capacity -
// you should always apply modulo capacity to calculate the index of this array.
// Each slot saves a fixed size events events, which will be read concurrently -
// we should never try to reuse watchCacheEventSlot(creating a new one when it is full).
type watchCacheCyclicBuffer struct {
	// ResourceVersion of the last list result (populated via Replace() method).
	listResourceVersion uint64
	// Buffer can save len(eventsSlots) * slotSize events at most,
	// (len(eventsSlots) - 1) * slotSize + 1 at least.
	slotSize    int
	eventsSlots []*watchCacheEventSlot
	// The start and end(included) index of eventsSlots.
	startIndex int
	endIndex   int
}

// Split a one-dimensional array into two-dimensional arrays, the second saves up to
// 256 elements - at lease two slots are required to form a circular buffer.
// We will make sure the capacity of watchCacheCyclicBuffer is larger than or equal to
// the original capacity.
func newWatchCacheCyclicBuffer(capacity int) *watchCacheCyclicBuffer {
	slotNum, slotSize := 2, capacity
	if capacity >= 256 {
		slotNum = (capacity+255)/256 + 1
		if slotNum <= 1 {
			slotNum = 2
		}
	}

	cache := &watchCacheCyclicBuffer{
		slotSize:    slotSize,
		eventsSlots: make([]*watchCacheEventSlot, slotNum),
	}
	cache.reset(0)
	return cache
}

func (w *watchCacheCyclicBuffer) reset(listResourceVersion uint64) {
	w.listResourceVersion = listResourceVersion
	w.startIndex = 0
	w.endIndex = 0
	w.eventsSlots[0] = newWatchCacheSlot(w.slotSize)
	// Reset the following slot to let gc recycle them
	for i := 1; i < len(w.eventsSlots); i++ {
		w.eventsSlots[i] = nil
	}
}

func (w *watchCacheCyclicBuffer) add(event *watchCacheEvent) {
	if w.eventsSlots[w.endIndex].isFull() {
		w.endIndex = (w.endIndex + 1) % len(w.eventsSlots)
		if w.endIndex == w.startIndex {
			// Cache is full - remove the oldest element
			w.startIndex = (w.startIndex + 1) % len(w.eventsSlots)
		}
		// create a new block of events to prevent reuse the old one
		w.eventsSlots[w.endIndex] = newWatchCacheSlot(w.slotSize)
	}

	w.eventsSlots[w.endIndex].add(event)
}

// returns an iterator pointing to events since resourceVersion(excluded), the content pointed
// to by the iterator is thread-safe and will not be flushed by subsequent modifications.
func (w *watchCacheCyclicBuffer) getEventsSince(resourceVersion uint64) (WatchCacheEventsIterator, error) {
	var oldest uint64
	switch {
	case (w.endIndex+1)%len(w.eventsSlots) == w.startIndex:
		// Once the watch event buffer is full, the oldest watch event we can deliver
		// is the first one in the buffer.
		oldest = w.eventsSlots[w.startIndex].firstResourceVersion()
	case w.listResourceVersion > 0:
		// If the watch event buffer isn't full, the oldest watch event we can deliver
		// is one greater than the resource version of the last full list.
		oldest = w.listResourceVersion + 1
	case !w.eventsSlots[w.startIndex].isEmpty():
		// If we've never completed a list, use the resourceVersion of the oldest event
		// in the buffer.
		// This should only happen in unit tests that populate the buffer without
		// performing list/replace operations.
		oldest = w.eventsSlots[w.startIndex].firstResourceVersion()
	default:
		return nil, fmt.Errorf("watch events isn't correctly initialized")
	}

	if resourceVersion < oldest-1 {
		return nil, errors.NewGone(fmt.Sprintf("too old resource version: %d (%d)", resourceVersion, oldest-1))
	}

	// Empty events list, this happens after watch_cache.Replace.
	if w.eventsSlots[w.startIndex].isEmpty() {
		return newCacheEventsIterator(nil), nil
	}

	capacity := len(w.eventsSlots)
	// Binary search the smallest index at which last resourceVersion is greater than the given one.
	f := func(i int) bool {
		return w.eventsSlots[(w.startIndex+i)%capacity].lastResourceVersion() > resourceVersion
	}
	size := (w.endIndex-w.startIndex+capacity)%capacity + 1
	first := sort.Search(size, f)
	result := make([]*watchCacheEventSlot, size-first)
	for i := 0; i < size-first; i++ {
		result[i] = w.eventsSlots[(w.startIndex+first+i)%capacity]
	}

	currIndexOfSlot := 0
	lastIndexOfSlot := 0
	if len(result) > 0 {
		// Binary search the smallest index at which resourceVersion is greater than the given one.
		f = func(i int) bool {
			return result[0].events[i].ResourceVersion > resourceVersion
		}
		currIndexOfSlot = sort.Search(result[0].size(), f)
		lastIndexOfSlot = result[len(result)-1].endIndex
	}

	return &cacheEventsIterator{
		slotIndex:       0,
		currIndexOfSlot: currIndexOfSlot,
		lastIndexOfSlot: lastIndexOfSlot,
		eventsSlots:     result,
	}, nil
}

type WatchCacheEventsIterator interface {
	Next() (*watchCacheEvent, bool)
}

// wrap a two-dimensional array as a WatchCacheEventsIterator pattern.
type cacheEventsIterator struct {
	slotIndex       int
	currIndexOfSlot int
	lastIndexOfSlot int
	eventsSlots     []*watchCacheEventSlot
}

func (v *cacheEventsIterator) Next() (*watchCacheEvent, bool) {
	if v.slotIndex == len(v.eventsSlots) || (v.slotIndex == len(v.eventsSlots)-1 && v.currIndexOfSlot >= v.lastIndexOfSlot) {
		return nil, false
	}

	event := v.eventsSlots[v.slotIndex].events[v.currIndexOfSlot]
	v.currIndexOfSlot++
	if v.currIndexOfSlot == len(v.eventsSlots[v.slotIndex].events) {
		v.slotIndex++
		v.currIndexOfSlot = 0
	}
	return event, true
}

// wrap an array as a WatchCacheEventsIterator pattern.
type cacheEventsArrayIterator struct {
	currIndex   int
	cacheEvents []*watchCacheEvent
}

func newCacheEventsIterator(cacheEvents []*watchCacheEvent) WatchCacheEventsIterator {
	return &cacheEventsArrayIterator{
		currIndex:   0,
		cacheEvents: cacheEvents,
	}
}

func (v *cacheEventsArrayIterator) Next() (*watchCacheEvent, bool) {
	if v.currIndex >= len(v.cacheEvents) {
		return nil, false
	}
	event := v.cacheEvents[v.currIndex]
	v.currIndex++
	return event, true
}

// watchCache implements a Store interface.
// However, it depends on the elements implementing runtime.Object interface.
//
// watchCache is a "sliding window" (with a limited capacity) of objects
// observed from a watch.
type watchCache struct {
	sync.RWMutex

	// Condition on which lists are waiting for the fresh enough
	// resource version.
	cond *sync.Cond

	// Maximum size of history window.
	capacity int

	// keyFunc is used to get a key in the underlying storage for a given object.
	keyFunc func(runtime.Object) (string, error)

	// getAttrsFunc is used to get labels and fields of an object.
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, error)

	// recentEventsCache is used a cyclic buffer to save most recent events.
	recentEventsCache *watchCacheCyclicBuffer

	// store will effectively support LIST operation from the "end of events
	// history" i.e. from the moment just after the newest cached watched event.
	// It is necessary to effectively allow clients to start watching at now.
	// NOTE: We assume that <store> is thread-safe.
	store cache.Store

	// ResourceVersion up to which the watchCache is propagated.
	resourceVersion uint64

	// This handler is run at the end of every successful Replace() method.
	onReplace func()

	// This handler is run at the end of every Add/Update/Delete method
	// and additionally gets the previous value of the object.
	onEvent func(*watchCacheEvent)

	// for testing timeouts.
	clock clock.Clock

	// An underlying storage.Versioner.
	versioner storage.Versioner
}

func newWatchCache(
	capacity int,
	keyFunc func(runtime.Object) (string, error),
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, error),
	versioner storage.Versioner) *watchCache {
	wc := &watchCache{
		capacity:          capacity,
		keyFunc:           keyFunc,
		getAttrsFunc:      getAttrsFunc,
		recentEventsCache: newWatchCacheCyclicBuffer(capacity),
		store:             cache.NewStore(storeElementKey),
		resourceVersion:   0,
		clock:             clock.RealClock{},
		versioner:         versioner,
	}
	wc.cond = sync.NewCond(wc.RLocker())
	return wc
}

// Add takes runtime.Object as an argument.
func (w *watchCache) Add(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Added, Object: object}

	f := func(elem *storeElement) error { return w.store.Add(elem) }
	return w.processEvent(event, resourceVersion, f)
}

// Update takes runtime.Object as an argument.
func (w *watchCache) Update(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Modified, Object: object}

	f := func(elem *storeElement) error { return w.store.Update(elem) }
	return w.processEvent(event, resourceVersion, f)
}

// Delete takes runtime.Object as an argument.
func (w *watchCache) Delete(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Deleted, Object: object}

	f := func(elem *storeElement) error { return w.store.Delete(elem) }
	return w.processEvent(event, resourceVersion, f)
}

func (w *watchCache) objectToVersionedRuntimeObject(obj interface{}) (runtime.Object, uint64, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, 0, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	resourceVersion, err := w.versioner.ObjectResourceVersion(object)
	if err != nil {
		return nil, 0, err
	}
	return object, resourceVersion, nil
}

func (w *watchCache) processEvent(event watch.Event, resourceVersion uint64, updateFunc func(*storeElement) error) error {
	key, err := w.keyFunc(event.Object)
	if err != nil {
		return fmt.Errorf("couldn't compute key: %v", err)
	}
	elem := &storeElement{Key: key, Object: event.Object}
	elem.Labels, elem.Fields, err = w.getAttrsFunc(event.Object)
	if err != nil {
		return err
	}

	watchCacheEvent := &watchCacheEvent{
		Type:            event.Type,
		Object:          elem.Object,
		ObjLabels:       elem.Labels,
		ObjFields:       elem.Fields,
		Key:             key,
		ResourceVersion: resourceVersion,
	}

	// TODO: We should consider moving this lock below after the watchCacheEvent
	// is created. In such situation, the only problematic scenario is Replace(
	// happening after getting object from store and before acquiring a lock.
	// Maybe introduce another lock for this purpose.
	w.Lock()
	defer w.Unlock()
	previous, exists, err := w.store.Get(elem)
	if err != nil {
		return err
	}
	if exists {
		previousElem := previous.(*storeElement)
		watchCacheEvent.PrevObject = previousElem.Object
		watchCacheEvent.PrevObjLabels = previousElem.Labels
		watchCacheEvent.PrevObjFields = previousElem.Fields
	}
	w.recentEventsCache.add(watchCacheEvent)
	w.resourceVersion = resourceVersion

	if w.onEvent != nil {
		w.onEvent(watchCacheEvent)
	}
	w.cond.Broadcast()
	return updateFunc(elem)
}

// List returns list of pointers to <storeElement> objects.
func (w *watchCache) List() []interface{} {
	return w.store.List()
}

// waitUntilFreshAndBlock waits until cache is at least as fresh as given <resourceVersion>.
// NOTE: This function acquired lock and doesn't release it.
// You HAVE TO explicitly call w.RUnlock() after this function.
func (w *watchCache) waitUntilFreshAndBlock(resourceVersion uint64, trace *utiltrace.Trace) error {
	startTime := w.clock.Now()
	go func() {
		// Wake us up when the time limit has expired.  The docs
		// promise that time.After (well, NewTimer, which it calls)
		// will wait *at least* the duration given. Since this go
		// routine starts sometime after we record the start time, and
		// it will wake up the loop below sometime after the broadcast,
		// we don't need to worry about waking it up before the time
		// has expired accidentally.
		<-w.clock.After(blockTimeout)
		w.cond.Broadcast()
	}()

	w.RLock()
	if trace != nil {
		trace.Step("watchCache locked acquired")
	}
	for w.resourceVersion < resourceVersion {
		if w.clock.Since(startTime) >= blockTimeout {
			// Timeout with retry after 1 second.
			return errors.NewTimeoutError(fmt.Sprintf("Too large resource version: %v, current: %v", resourceVersion, w.resourceVersion), 1)
		}
		w.cond.Wait()
	}
	if trace != nil {
		trace.Step("watchCache fresh enough")
	}
	return nil
}

// WaitUntilFreshAndList returns list of pointers to <storeElement> objects.
func (w *watchCache) WaitUntilFreshAndList(resourceVersion uint64, trace *utiltrace.Trace) ([]interface{}, uint64, error) {
	err := w.waitUntilFreshAndBlock(resourceVersion, trace)
	defer w.RUnlock()
	if err != nil {
		return nil, 0, err
	}
	return w.store.List(), w.resourceVersion, nil
}

// WaitUntilFreshAndGet returns a pointers to <storeElement> object.
func (w *watchCache) WaitUntilFreshAndGet(resourceVersion uint64, key string, trace *utiltrace.Trace) (interface{}, bool, uint64, error) {
	err := w.waitUntilFreshAndBlock(resourceVersion, trace)
	defer w.RUnlock()
	if err != nil {
		return nil, false, 0, err
	}
	value, exists, err := w.store.GetByKey(key)
	return value, exists, w.resourceVersion, err
}

func (w *watchCache) ListKeys() []string {
	return w.store.ListKeys()
}

// Get takes runtime.Object as a parameter. However, it returns
// pointer to <storeElement>.
func (w *watchCache) Get(obj interface{}) (interface{}, bool, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, false, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	key, err := w.keyFunc(object)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't compute key: %v", err)
	}

	return w.store.Get(&storeElement{Key: key, Object: object})
}

// GetByKey returns pointer to <storeElement>.
func (w *watchCache) GetByKey(key string) (interface{}, bool, error) {
	return w.store.GetByKey(key)
}

// Replace takes slice of runtime.Object as a parameter.
func (w *watchCache) Replace(objs []interface{}, resourceVersion string) error {
	version, err := w.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	toReplace := make([]interface{}, 0, len(objs))
	for _, obj := range objs {
		object, ok := obj.(runtime.Object)
		if !ok {
			return fmt.Errorf("didn't get runtime.Object for replace: %#v", obj)
		}
		key, err := w.keyFunc(object)
		if err != nil {
			return fmt.Errorf("couldn't compute key: %v", err)
		}
		objLabels, objFields, err := w.getAttrsFunc(object)
		if err != nil {
			return err
		}
		toReplace = append(toReplace, &storeElement{
			Key:    key,
			Object: object,
			Labels: objLabels,
			Fields: objFields,
		})
	}

	w.Lock()
	defer w.Unlock()

	w.recentEventsCache.reset(version)
	if err := w.store.Replace(toReplace, resourceVersion); err != nil {
		return err
	}
	w.resourceVersion = version
	if w.onReplace != nil {
		w.onReplace()
	}
	w.cond.Broadcast()
	klog.V(3).Infof("Replace watchCache (rev: %v) ", resourceVersion)
	return nil
}

func (w *watchCache) SetOnReplace(onReplace func()) {
	w.Lock()
	defer w.Unlock()
	w.onReplace = onReplace
}

func (w *watchCache) SetOnEvent(onEvent func(*watchCacheEvent)) {
	w.Lock()
	defer w.Unlock()
	w.onEvent = onEvent
}

func (w *watchCache) GetAllEventsSinceThreadUnsafe(resourceVersion uint64) (WatchCacheEventsIterator, error) {
	if resourceVersion == 0 {
		// resourceVersion = 0 means that we don't require any specific starting point
		// and we would like to start watching from ~now.
		// However, to keep backward compatibility, we additionally need to return the
		// current state and only then start watching from that point.
		//
		// TODO: In v2 api, we should stop returning the current state - #13969.
		allItems := w.store.List()
		result := make([]*watchCacheEvent, len(allItems))
		for i, item := range allItems {
			elem, ok := item.(*storeElement)
			if !ok {
				return nil, fmt.Errorf("not a storeElement: %v", elem)
			}
			objLabels, objFields, err := w.getAttrsFunc(elem.Object)
			if err != nil {
				return nil, err
			}
			result[i] = &watchCacheEvent{
				Type:            watch.Added,
				Object:          elem.Object,
				ObjLabels:       objLabels,
				ObjFields:       objFields,
				Key:             elem.Key,
				ResourceVersion: w.resourceVersion,
			}
		}
		return newCacheEventsIterator(result), nil
	}

	return w.recentEventsCache.getEventsSince(resourceVersion)
}

func (w *watchCache) GetAllEventsSince(resourceVersion uint64) (WatchCacheEventsIterator, error) {
	w.RLock()
	defer w.RUnlock()
	return w.GetAllEventsSinceThreadUnsafe(resourceVersion)
}

func (w *watchCache) Resync() error {
	// Nothing to do
	return nil
}
