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

package storage

import (
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/watch"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	"k8s.io/client-go/tools/cache"
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
	Type                 watch.EventType
	Object               runtime.Object
	ObjLabels            labels.Set
	ObjFields            fields.Set
	ObjUninitialized     bool
	PrevObject           runtime.Object
	PrevObjLabels        labels.Set
	PrevObjFields        fields.Set
	PrevObjUninitialized bool
	Key                  string
	ResourceVersion      uint64
}

// Computing a key of an object is generally non-trivial (it performs
// e.g. validation underneath). To avoid computing it multiple times
// (to serve the event in different List/Watch requests), in the
// underlying store we are keeping pair (key, object).
type storeElement struct {
	Key    string
	Object runtime.Object
}

func storeElementKey(obj interface{}) (string, error) {
	elem, ok := obj.(*storeElement)
	if !ok {
		return "", fmt.Errorf("not a storeElement: %v", obj)
	}
	return elem.Key, nil
}

// watchCacheElement is a single "watch event" stored in a cache.
// It contains the resource version of the object and the object
// itself.
type watchCacheElement struct {
	resourceVersion uint64
	watchCacheEvent *watchCacheEvent
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
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, bool, error)

	// cache is used a cyclic buffer - its first element (with the smallest
	// resourceVersion) is defined by startIndex, its last element is defined
	// by endIndex (if cache is full it will be startIndex + capacity).
	// Both startIndex and endIndex can be greater than buffer capacity -
	// you should always apply modulo capacity to get an index in cache array.
	cache      []watchCacheElement
	startIndex int
	endIndex   int

	// store will effectively support LIST operation from the "end of cache
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
}

func newWatchCache(
	capacity int,
	keyFunc func(runtime.Object) (string, error),
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, bool, error)) *watchCache {
	wc := &watchCache{
		capacity:        capacity,
		keyFunc:         keyFunc,
		getAttrsFunc:    getAttrsFunc,
		cache:           make([]watchCacheElement, capacity),
		startIndex:      0,
		endIndex:        0,
		store:           cache.NewStore(storeElementKey),
		resourceVersion: 0,
		clock:           clock.RealClock{},
	}
	wc.cond = sync.NewCond(wc.RLocker())
	return wc
}

// Add takes runtime.Object as an argument.
func (w *watchCache) Add(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Added, Object: object}

	f := func(elem *storeElement) error { return w.store.Add(elem) }
	return w.processEvent(event, resourceVersion, f)
}

// Update takes runtime.Object as an argument.
func (w *watchCache) Update(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Modified, Object: object}

	f := func(elem *storeElement) error { return w.store.Update(elem) }
	return w.processEvent(event, resourceVersion, f)
}

// Delete takes runtime.Object as an argument.
func (w *watchCache) Delete(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Deleted, Object: object}

	f := func(elem *storeElement) error { return w.store.Delete(elem) }
	return w.processEvent(event, resourceVersion, f)
}

func objectToVersionedRuntimeObject(obj interface{}) (runtime.Object, uint64, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, 0, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	meta, err := meta.Accessor(object)
	if err != nil {
		return nil, 0, err
	}
	resourceVersion, err := parseResourceVersion(meta.GetResourceVersion())
	if err != nil {
		return nil, 0, err
	}
	return object, resourceVersion, nil
}

func parseResourceVersion(resourceVersion string) (uint64, error) {
	if resourceVersion == "" {
		return 0, nil
	}
	// Use bitsize being the size of int on the machine.
	return strconv.ParseUint(resourceVersion, 10, 0)
}

func (w *watchCache) processEvent(event watch.Event, resourceVersion uint64, updateFunc func(*storeElement) error) error {
	key, err := w.keyFunc(event.Object)
	if err != nil {
		return fmt.Errorf("couldn't compute key: %v", err)
	}
	elem := &storeElement{Key: key, Object: event.Object}

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
	objLabels, objFields, objUninitialized, err := w.getAttrsFunc(event.Object)
	if err != nil {
		return err
	}
	var prevObject runtime.Object
	var prevObjLabels labels.Set
	var prevObjFields fields.Set
	var prevObjUninitialized bool
	if exists {
		prevObject = previous.(*storeElement).Object
		prevObjLabels, prevObjFields, prevObjUninitialized, err = w.getAttrsFunc(prevObject)
		if err != nil {
			return err
		}
	}
	watchCacheEvent := &watchCacheEvent{
		Type:                 event.Type,
		Object:               event.Object,
		ObjLabels:            objLabels,
		ObjFields:            objFields,
		ObjUninitialized:     objUninitialized,
		PrevObject:           prevObject,
		PrevObjLabels:        prevObjLabels,
		PrevObjFields:        prevObjFields,
		PrevObjUninitialized: prevObjUninitialized,
		Key:                  key,
		ResourceVersion:      resourceVersion,
	}
	if w.onEvent != nil {
		w.onEvent(watchCacheEvent)
	}
	w.updateCache(resourceVersion, watchCacheEvent)
	w.resourceVersion = resourceVersion
	w.cond.Broadcast()
	return updateFunc(elem)
}

// Assumes that lock is already held for write.
func (w *watchCache) updateCache(resourceVersion uint64, event *watchCacheEvent) {
	if w.endIndex == w.startIndex+w.capacity {
		// Cache is full - remove the oldest element.
		w.startIndex++
	}
	w.cache[w.endIndex%w.capacity] = watchCacheElement{resourceVersion, event}
	w.endIndex++
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

// Replace takes slice of runtime.Object as a paramater.
func (w *watchCache) Replace(objs []interface{}, resourceVersion string) error {
	version, err := parseResourceVersion(resourceVersion)
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
		toReplace = append(toReplace, &storeElement{Key: key, Object: object})
	}

	w.Lock()
	defer w.Unlock()

	w.startIndex = 0
	w.endIndex = 0
	if err := w.store.Replace(toReplace, resourceVersion); err != nil {
		return err
	}
	w.resourceVersion = version
	if w.onReplace != nil {
		w.onReplace()
	}
	w.cond.Broadcast()
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

func (w *watchCache) GetAllEventsSinceThreadUnsafe(resourceVersion uint64) ([]*watchCacheEvent, error) {
	size := w.endIndex - w.startIndex
	// if we have no watch events in our cache, the oldest one we can successfully deliver to a watcher
	// is the *next* event we'll receive, which will be at least one greater than our current resourceVersion
	oldest := w.resourceVersion + 1
	if size > 0 {
		oldest = w.cache[w.startIndex%w.capacity].resourceVersion
	}
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
			objLabels, objFields, objUninitialized, err := w.getAttrsFunc(elem.Object)
			if err != nil {
				return nil, err
			}
			result[i] = &watchCacheEvent{
				Type:             watch.Added,
				Object:           elem.Object,
				ObjLabels:        objLabels,
				ObjFields:        objFields,
				ObjUninitialized: objUninitialized,
				Key:              elem.Key,
				ResourceVersion:  w.resourceVersion,
			}
		}
		return result, nil
	}
	if resourceVersion < oldest-1 {
		return nil, errors.NewGone(fmt.Sprintf("too old resource version: %d (%d)", resourceVersion, oldest-1))
	}

	// Binary search the smallest index at which resourceVersion is greater than the given one.
	f := func(i int) bool {
		return w.cache[(w.startIndex+i)%w.capacity].resourceVersion > resourceVersion
	}
	first := sort.Search(size, f)
	result := make([]*watchCacheEvent, size-first)
	for i := 0; i < size-first; i++ {
		result[i] = w.cache[(w.startIndex+first+i)%w.capacity].watchCacheEvent
	}
	return result, nil
}

func (w *watchCache) GetAllEventsSince(resourceVersion uint64) ([]*watchCacheEvent, error) {
	w.RLock()
	defer w.RUnlock()
	return w.GetAllEventsSinceThreadUnsafe(resourceVersion)
}

func (w *watchCache) Resync() error {
	// Nothing to do
	return nil
}
