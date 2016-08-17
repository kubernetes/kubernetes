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

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/clock"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// MaximumListWait determines how long we're willing to wait for a
	// list if a client specified a resource version in the future.
	MaximumListWait = 60 * time.Second
)

// watchCacheEvent is a single "watch event" that is send to users of
// watchCache. Additionally to a typical "watch.Event" it contains
// the previous value of the object to enable proper filtering in the
// upper layers.
type watchCacheEvent struct {
	Type            watch.EventType
	Object          runtime.Object
	PrevObject      runtime.Object
	ResourceVersion uint64
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
// watchCache is a "sliding window" (with a limited capacity) of objects
// observed from a watch.
type watchCache struct {
	sync.RWMutex

	// Condition on which lists are waiting for the fresh enough
	// resource version.
	cond *sync.Cond

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

	// for testing timeouts.
	clock clock.Clock
}

func newWatchCache(capacity int) *watchCache {
	wc := &watchCache{
		capacity:        capacity,
		cache:           make([]watchCacheElement, capacity),
		startIndex:      0,
		endIndex:        0,
		store:           cache.NewStore(cache.MetaNamespaceKeyFunc),
		resourceVersion: 0,
		clock:           clock.RealClock{},
	}
	wc.cond = sync.NewCond(wc.RLocker())
	return wc
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
	}
	watchCacheEvent := watchCacheEvent{event.Type, event.Object, prevObject, resourceVersion}
	if w.onEvent != nil {
		w.onEvent(watchCacheEvent)
	}
	w.updateCache(resourceVersion, watchCacheEvent)
	w.resourceVersion = resourceVersion
	w.cond.Broadcast()
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

func (w *watchCache) WaitUntilFreshAndList(resourceVersion uint64) ([]interface{}, uint64, error) {
	startTime := w.clock.Now()
	go func() {
		// Wake us up when the time limit has expired.  The docs
		// promise that time.After (well, NewTimer, which it calls)
		// will wait *at least* the duration given. Since this go
		// routine starts sometime after we record the start time, and
		// it will wake up the loop below sometime after the broadcast,
		// we don't need to worry about waking it up before the time
		// has expired accidentally.
		<-w.clock.After(MaximumListWait)
		w.cond.Broadcast()
	}()

	w.RLock()
	defer w.RUnlock()
	for w.resourceVersion < resourceVersion {
		if w.clock.Since(startTime) >= MaximumListWait {
			return nil, 0, fmt.Errorf("time limit exceeded while waiting for resource version %v (current value: %v)", resourceVersion, w.resourceVersion)
		}
		w.cond.Wait()
	}
	return w.store.List(), w.resourceVersion, nil
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
	w.cond.Broadcast()
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
	if resourceVersion == 0 {
		// resourceVersion = 0 means that we don't require any specific starting point
		// and we would like to start watching from ~now.
		// However, to keep backward compatibility, we additionally need to return the
		// current state and only then start watching from that point.
		//
		// TODO: In v2 api, we should stop returning the current state - #13969.
		allItems := w.store.List()
		result := make([]watchCacheEvent, len(allItems))
		for i, item := range allItems {
			result[i] = watchCacheEvent{Type: watch.Added, Object: item.(runtime.Object)}
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

func (w *watchCache) Resync() error {
	// Nothing to do
	return nil
}
