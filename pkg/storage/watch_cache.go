/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// watchCacheEvent is a single "watch event" that is send to users of
// watchCache. Additionally to a typical "watch.Event" it contains
// the previous value of the object to enable proper filtering in the
// upper layers.
type watchCacheEvent struct {
	Type       watch.EventType
	Object     runtime.Object
	PrevObject runtime.Object
}

// watchCacheElement is a single "watch event" stored in a cache.
// It contains the resource version of the object and the object
// itself.
type watchCacheElement struct {
	resourceVersion uint64
	watchCacheEvent watchCacheEvent
}

// watchCache implements a Store interface.
// However, it depends on the elements implementing runtime.Object interface.
//
// watchCache is a "sliding window" (with a limitted capacity) of objects
// observed from a watch.
type watchCache struct {
	sync.RWMutex

	// Maximum size of history window.
	capacity int

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
	store cache.Store

	// ResourceVersion up to which the watchCache is propagated.
	resourceVersion uint64

	// This handler is run at the end of every successful Replace() method.
	onReplace func()

	// This handler is run at the end of every Add/Update/Delete method
	// and additionally gets the previous value of the object.
	onEvent func(watchCacheEvent)
}

func newWatchCache(capacity int) *watchCache {
	return &watchCache{
		capacity:        capacity,
		cache:           make([]watchCacheElement, capacity),
		startIndex:      0,
		endIndex:        0,
		store:           cache.NewStore(cache.MetaNamespaceKeyFunc),
		resourceVersion: 0,
	}
}

func (w *watchCache) Add(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Added, Object: object}

	f := func(obj runtime.Object) error { return w.store.Add(obj) }
	return w.processEvent(event, resourceVersion, f)
}

func (w *watchCache) Update(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Modified, Object: object}

	f := func(obj runtime.Object) error { return w.store.Update(obj) }
	return w.processEvent(event, resourceVersion, f)
}

func (w *watchCache) Delete(obj interface{}) error {
	object, resourceVersion, err := objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Deleted, Object: object}

	f := func(obj runtime.Object) error { return w.store.Delete(obj) }
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
	resourceVersion, err := parseResourceVersion(meta.ResourceVersion())
	if err != nil {
		return nil, 0, err
	}
	return object, resourceVersion, nil
}

func parseResourceVersion(resourceVersion string) (uint64, error) {
	if resourceVersion == "" {
		return 0, nil
	}
	return strconv.ParseUint(resourceVersion, 10, 64)
}

func (w *watchCache) processEvent(event watch.Event, resourceVersion uint64, updateFunc func(runtime.Object) error) error {
	w.Lock()
	defer w.Unlock()
	previous, exists, err := w.store.Get(event.Object)
	if err != nil {
		return err
	}
	var prevObject runtime.Object
	if exists {
		prevObject = previous.(runtime.Object)
	} else {
		prevObject = nil
	}
	watchCacheEvent := watchCacheEvent{event.Type, event.Object, prevObject}
	if w.onEvent != nil {
		w.onEvent(watchCacheEvent)
	}
	w.updateCache(resourceVersion, watchCacheEvent)
	w.resourceVersion = resourceVersion
	return updateFunc(event.Object)
}

// Assumes that lock is already held for write.
func (w *watchCache) updateCache(resourceVersion uint64, event watchCacheEvent) {
	if w.endIndex == w.startIndex+w.capacity {
		// Cache is full - remove the oldest element.
		w.startIndex++
	}
	w.cache[w.endIndex%w.capacity] = watchCacheElement{resourceVersion, event}
	w.endIndex++
}

func (w *watchCache) List() []interface{} {
	w.RLock()
	defer w.RUnlock()
	return w.store.List()
}

func (w *watchCache) ListWithVersion() ([]interface{}, uint64) {
	w.RLock()
	defer w.RUnlock()
	return w.store.List(), w.resourceVersion
}

func (w *watchCache) ListKeys() []string {
	w.RLock()
	defer w.RUnlock()
	return w.store.ListKeys()
}

func (w *watchCache) Get(obj interface{}) (interface{}, bool, error) {
	w.RLock()
	defer w.RUnlock()
	return w.store.Get(obj)
}

func (w *watchCache) GetByKey(key string) (interface{}, bool, error) {
	w.RLock()
	defer w.RUnlock()
	return w.store.GetByKey(key)
}

func (w *watchCache) Replace(objs []interface{}, resourceVersion string) error {
	version, err := parseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	w.Lock()
	defer w.Unlock()

	w.startIndex = 0
	w.endIndex = 0
	if err := w.store.Replace(objs, resourceVersion); err != nil {
		return err
	}
	w.resourceVersion = version
	if w.onReplace != nil {
		w.onReplace()
	}
	return nil
}

func (w *watchCache) SetOnReplace(onReplace func()) {
	w.Lock()
	defer w.Unlock()
	w.onReplace = onReplace
}

func (w *watchCache) SetOnEvent(onEvent func(watchCacheEvent)) {
	w.Lock()
	defer w.Unlock()
	w.onEvent = onEvent
}

func (w *watchCache) GetAllEventsSinceThreadUnsafe(resourceVersion uint64) ([]watchCacheEvent, error) {
	size := w.endIndex - w.startIndex
	oldest := w.resourceVersion
	if size > 0 {
		oldest = w.cache[w.startIndex%w.capacity].resourceVersion
	}
	if resourceVersion < oldest {
		return nil, fmt.Errorf("too old resource version: %d (%d)", resourceVersion, oldest)
	}

	// Binary seatch the smallest index at which resourceVersion is not smaller than
	// the given one.
	f := func(i int) bool {
		return w.cache[(w.startIndex+i)%w.capacity].resourceVersion >= resourceVersion
	}
	first := sort.Search(size, f)
	result := make([]watchCacheEvent, size-first)
	for i := 0; i < size-first; i++ {
		result[i] = w.cache[(w.startIndex+first+i)%w.capacity].watchCacheEvent
	}
	return result, nil
}

func (w *watchCache) GetAllEventsSince(resourceVersion uint64) ([]watchCacheEvent, error) {
	w.RLock()
	defer w.RUnlock()
	return w.GetAllEventsSinceThreadUnsafe(resourceVersion)
}
