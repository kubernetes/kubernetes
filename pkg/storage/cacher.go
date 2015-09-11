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
	"reflect"
	"strconv"
	"strings"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// CacherConfig contains the configuration for a given Cache.
type CacherConfig struct {
	// Maximum size of the history cached in memory.
	CacheCapacity int

	// An underlying storage.Interface.
	Storage Interface

	// An underlying storage.Versioner.
	Versioner Versioner

	// The Cache will be caching objects of a given Type and assumes that they
	// are all stored under ResourcePrefix directory in the underlying database.
	Type           interface{}
	ResourcePrefix string

	// KeyFunc is used to get a key in the underyling storage for a given object.
	KeyFunc func(runtime.Object) (string, error)

	// NewList is a function that creates new empty object storing a list of
	// objects of type Type.
	NewListFunc func() runtime.Object

	// Cacher will be stopped when the StopChannel will be closed.
	StopChannel <-chan struct{}
}

// Cacher is responsible for serving WATCH and LIST requests for a given
// resource from its internal cache and updating its cache in the background
// based on the underlying storage contents.
// Cacher implements storage.Interface (although most of the calls are just
// delegated to the underlying storage).
type Cacher struct {
	sync.RWMutex

	// Each user-facing method that is not simply redirected to the underlying
	// storage has to read-lock on this mutex before starting any processing.
	// This is necessary to prevent users from accessing structures that are
	// uninitialized or are being repopulated right now.
	// NOTE: We cannot easily reuse the main mutex for it due to multi-threaded
	// interactions of Cacher with the underlying WatchCache. Since Cacher is
	// caling WatchCache directly and WatchCache is calling Cacher methods
	// via its OnEvent and OnReplace hooks, we explicitly assume that if mutexes
	// of both structures are held, the one from WatchCache is acquired first
	// to avoid deadlocks. Unfortunately, forcing this rule in startCaching
	// would be very difficult and introducing one more mutex seems to be much
	// easier.
	usable sync.RWMutex

	// Underlying storage.Interface.
	storage Interface

	// "sliding window" of recent changes of objects and the current state.
	watchCache *watchCache
	reflector  *cache.Reflector

	// Registered watchers.
	watcherIdx int
	watchers   map[int]*cacheWatcher

	// Versioner is used to handle resource versions.
	versioner Versioner

	// keyFunc is used to get a key in the underyling storage for a given object.
	keyFunc func(runtime.Object) (string, error)
}

// Create a new Cacher responsible from service WATCH and LIST requests from its
// internal cache and updating its cache in the background based on the given
// configuration.
func NewCacher(config CacherConfig) *Cacher {
	watchCache := newWatchCache(config.CacheCapacity)
	listerWatcher := newCacherListerWatcher(config.Storage, config.ResourcePrefix, config.NewListFunc)

	cacher := &Cacher{
		usable:     sync.RWMutex{},
		storage:    config.Storage,
		watchCache: watchCache,
		reflector:  cache.NewReflector(listerWatcher, config.Type, watchCache, 0),
		watcherIdx: 0,
		watchers:   make(map[int]*cacheWatcher),
		versioner:  config.Versioner,
		keyFunc:    config.KeyFunc,
	}
	cacher.usable.Lock()
	// See startCaching method for why explanation on it.
	watchCache.SetOnReplace(func() { cacher.usable.Unlock() })
	watchCache.SetOnEvent(cacher.processEvent)

	stopCh := config.StopChannel
	go util.Until(func() { cacher.startCaching(stopCh) }, 0, stopCh)
	return cacher
}

func (c *Cacher) startCaching(stopChannel <-chan struct{}) {
	// Whenever we enter startCaching method, usable mutex is held.
	// We explicitly do NOT Unlock it in this method, because we do
	// not want to allow any Watch/List methods not explicitly redirected
	// to the underlying storage when the cache is being initialized.
	// Once the underlying cache is propagated, onReplace handler will
	// be called, which will do the usable.Unlock() as configured in
	// NewCacher().
	// Note: the same behavior is also triggered every time we fall out of
	// backend storage watch event window.
	defer c.usable.Lock()

	c.terminateAllWatchers()
	// Note that since onReplace may be not called due to errors, we explicitly
	// need to retry it on errors under lock.
	for {
		if err := c.reflector.ListAndWatch(stopChannel); err != nil {
			glog.Errorf("unexpected ListAndWatch error: %v", err)
		} else {
			break
		}
	}
}

// Implements storage.Interface.
func (c *Cacher) Backends() []string {
	return c.storage.Backends()
}

// Implements storage.Interface.
func (c *Cacher) Versioner() Versioner {
	return c.storage.Versioner()
}

// Implements storage.Interface.
func (c *Cacher) Create(key string, obj, out runtime.Object, ttl uint64) error {
	return c.storage.Create(key, obj, out, ttl)
}

// Implements storage.Interface.
func (c *Cacher) Set(key string, obj, out runtime.Object, ttl uint64) error {
	return c.storage.Set(key, obj, out, ttl)
}

// Implements storage.Interface.
func (c *Cacher) Delete(key string, out runtime.Object) error {
	return c.storage.Delete(key, out)
}

// Implements storage.Interface.
func (c *Cacher) Watch(key string, resourceVersion uint64, filter FilterFunc) (watch.Interface, error) {
	// Do NOT allow Watch to start when the underlying structures are not propagated.
	c.usable.RLock()
	defer c.usable.RUnlock()

	// We explicitly use thread unsafe version and do locking ourself to ensure that
	// no new events will be processed in the meantime. The watchCache will be unlocked
	// on return from this function.
	// Note that we cannot do it under Cacher lock, to avoid a deadlock, since the
	// underlying watchCache is calling processEvent under its lock.
	c.watchCache.RLock()
	defer c.watchCache.RUnlock()
	initEvents, err := c.watchCache.GetAllEventsSinceThreadUnsafe(resourceVersion)
	if err != nil {
		return nil, err
	}

	c.Lock()
	defer c.Unlock()
	watcher := newCacheWatcher(initEvents, filterFunction(key, c.keyFunc, filter), forgetWatcher(c, c.watcherIdx))
	c.watchers[c.watcherIdx] = watcher
	c.watcherIdx++
	return watcher, nil
}

// Implements storage.Interface.
func (c *Cacher) WatchList(key string, resourceVersion uint64, filter FilterFunc) (watch.Interface, error) {
	return c.Watch(key, resourceVersion, filter)
}

// Implements storage.Interface.
func (c *Cacher) Get(key string, objPtr runtime.Object, ignoreNotFound bool) error {
	return c.storage.Get(key, objPtr, ignoreNotFound)
}

// Implements storage.Interface.
func (c *Cacher) GetToList(key string, listObj runtime.Object) error {
	return c.storage.GetToList(key, listObj)
}

// Implements storage.Interface.
func (c *Cacher) List(key string, listObj runtime.Object) error {
	return c.storage.List(key, listObj)
}

// ListFromMemory implements list operation (the same signature as List method)
// but it serves the contents from memory.
// Current we cannot use ListFromMemory() instead of List(), because it only
// guarantees eventual consistency (e.g. it's possible for Get called right after
// Create to return not-exist, before the change is propagate).
// TODO: We may consider changing to use ListFromMemory in the future, but this
// requires wider discussion as an "api semantic change".
func (c *Cacher) ListFromMemory(key string, listObj runtime.Object) error {
	// Do NOT allow Watch to start when the underlying structures are not propagated.
	c.usable.RLock()
	defer c.usable.RUnlock()

	listPtr, err := runtime.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}
	filter := filterFunction(key, c.keyFunc, Everything)

	objs, resourceVersion := c.watchCache.ListWithVersion()
	for _, obj := range objs {
		object, ok := obj.(runtime.Object)
		if !ok {
			return fmt.Errorf("non runtime.Object returned from storage: %v", obj)
		}
		if filter(object) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(object).Elem()))
		}
	}
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, resourceVersion); err != nil {
			return err
		}
	}
	return nil
}

// Implements storage.Interface.
func (c *Cacher) GuaranteedUpdate(key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate UpdateFunc) error {
	return c.storage.GuaranteedUpdate(key, ptrToType, ignoreNotFound, tryUpdate)
}

// Implements storage.Interface.
func (c *Cacher) Codec() runtime.Codec {
	return c.storage.Codec()
}

func (c *Cacher) processEvent(event watchCacheEvent) {
	c.Lock()
	defer c.Unlock()
	for _, watcher := range c.watchers {
		watcher.add(event)
	}
}

func (c *Cacher) terminateAllWatchers() {
	c.Lock()
	defer c.Unlock()
	for key, watcher := range c.watchers {
		delete(c.watchers, key)
		watcher.stop()
	}
}

func forgetWatcher(c *Cacher, index int) func() {
	return func() {
		c.Lock()
		defer c.Unlock()
		// It's possible that the watcher is already not in the map (e.g. in case of
		// simulaneous Stop() and terminateAllWatchers(), but it doesn't break anything.
		delete(c.watchers, index)
	}
}

func filterFunction(key string, keyFunc func(runtime.Object) (string, error), filter FilterFunc) FilterFunc {
	return func(obj runtime.Object) bool {
		objKey, err := keyFunc(obj)
		if err != nil {
			glog.Errorf("Invalid object for filter: %v", obj)
			return false
		}
		if !strings.HasPrefix(objKey, key) {
			return false
		}
		return filter(obj)
	}
}

// Returns resource version to which the underlying cache is synced.
func (c *Cacher) LastSyncResourceVersion() (uint64, error) {
	c.RLock()
	defer c.RUnlock()

	resourceVersion := c.reflector.LastSyncResourceVersion()
	if resourceVersion == "" {
		return 0, nil
	}
	return strconv.ParseUint(resourceVersion, 10, 64)
}

// cacherListerWatcher opaques storage.Interface to expose cache.ListerWatcher.
type cacherListerWatcher struct {
	storage        Interface
	resourcePrefix string
	newListFunc    func() runtime.Object
}

func newCacherListerWatcher(storage Interface, resourcePrefix string, newListFunc func() runtime.Object) cache.ListerWatcher {
	return &cacherListerWatcher{
		storage:        storage,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
	}
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) List() (runtime.Object, error) {
	list := lw.newListFunc()
	if err := lw.storage.List(lw.resourcePrefix, list); err != nil {
		return nil, err
	}
	return list, nil
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) Watch(resourceVersion string) (watch.Interface, error) {
	version, err := ParseWatchResourceVersion(resourceVersion, lw.resourcePrefix)
	if err != nil {
		return nil, err
	}
	return lw.storage.WatchList(lw.resourcePrefix, version, Everything)
}

// cacherWatch implements watch.Interface
type cacheWatcher struct {
	sync.Mutex
	input   chan watchCacheEvent
	result  chan watch.Event
	filter  FilterFunc
	stopped bool
	forget  func()
}

func newCacheWatcher(initEvents []watchCacheEvent, filter FilterFunc, forget func()) *cacheWatcher {
	watcher := &cacheWatcher{
		input:   make(chan watchCacheEvent, 10),
		result:  make(chan watch.Event, 10),
		filter:  filter,
		stopped: false,
		forget:  forget,
	}
	go watcher.process(initEvents)
	return watcher
}

// Implements watch.Interface.
func (c *cacheWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *cacheWatcher) Stop() {
	c.forget()
	c.stop()
}

func (c *cacheWatcher) stop() {
	c.Lock()
	defer c.Unlock()
	if !c.stopped {
		c.stopped = true
		close(c.input)
	}
}

func (c *cacheWatcher) add(event watchCacheEvent) {
	c.input <- event
}

func (c *cacheWatcher) sendWatchCacheEvent(event watchCacheEvent) {
	curObjPasses := event.Type != watch.Deleted && c.filter(event.Object)
	oldObjPasses := false
	if event.PrevObject != nil {
		oldObjPasses = c.filter(event.PrevObject)
	}
	if !curObjPasses && !oldObjPasses {
		// Watcher is not interested in that object.
		return
	}

	object, err := api.Scheme.Copy(event.Object)
	if err != nil {
		glog.Errorf("unexpected copy error: %v", err)
		return
	}
	switch {
	case curObjPasses && !oldObjPasses:
		c.result <- watch.Event{Type: watch.Added, Object: object}
	case curObjPasses && oldObjPasses:
		c.result <- watch.Event{Type: watch.Modified, Object: object}
	case !curObjPasses && oldObjPasses:
		c.result <- watch.Event{Type: watch.Deleted, Object: object}
	}
}

func (c *cacheWatcher) process(initEvents []watchCacheEvent) {
	for _, event := range initEvents {
		c.sendWatchCacheEvent(event)
	}
	defer close(c.result)
	defer c.Stop()
	for {
		event, ok := <-c.input
		if !ok {
			return
		}
		c.sendWatchCacheEvent(event)
	}
}
