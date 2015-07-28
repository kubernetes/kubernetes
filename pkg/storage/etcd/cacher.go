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

package etcd

import (
	"fmt"
	"reflect"
	"sync"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// CacherConfig contains the configuration for a given etcdCache.
type CacherConfig struct {
	// Maximum size of the history cached in memory.
	CacheCapacity int

	// An underlying storage.Interface.
	Storage storage.Interface

	// The Cache will be caching objects of a given Type and assumes that they
	// are all stored under ResourcePrefix directory in the underlying database.
	Type           interface{}
	ResourcePrefix string

	// NewList is a function that creates new empty object storing a list of
	// objects of type Type.
	NewListFunc func() runtime.Object

	// StopChannel will effectively stop updating the underlying cache.
	StopChannel chan struct{}
}

// Cacher is responsible for serving WATCH and LIST requests for a given
// resource from its internal cache and updating its cache in the background
// based on the underlying storage contents.
type Cacher struct {
	sync.RWMutex

	// Whether Cacher is initialized.
	initialized sync.WaitGroup
	initOnce    sync.Once

	// "sliding window" of recent changes of objects and the current state.
	watchCache *cache.WatchCache
	reflector  *cache.Reflector

	// Registered watchers.
	watcherIdx int
	watchers   map[int]*cacheWatcher

	// Versioner is used to handle resource versions.
	versioner storage.Versioner
}

func NewCacher(config CacherConfig) storage.ListerAndWatcher {
	watchCache := cache.NewWatchCache(config.CacheCapacity)
	listerWatcher := newCacherListerWatcher(config.Storage, config.ResourcePrefix, config.NewListFunc)

	cacher := &Cacher{
		initialized: sync.WaitGroup{},
		watchCache:  watchCache,
		reflector:   cache.NewReflector(listerWatcher, config.Type, watchCache, 0),
		watcherIdx:  0,
		watchers:    make(map[int]*cacheWatcher),
		versioner:   APIObjectVersioner{},
	}
	cacher.initialized.Add(1)
	// See startCaching method for why explanation on it.
	watchCache.SetOnReplace(func() { cacher.initOnce.Do(func() { cacher.initialized.Done() }); cacher.Unlock() })
	watchCache.SetOnEvent(cacher.processEvent)

	go util.Until(func() { cacher.startCaching(config.StopChannel) }, 0, config.StopChannel)
	cacher.initialized.Wait()
	return cacher
}

func (c *Cacher) startCaching(stopChannel <-chan struct{}) {
	c.Lock()
	c.terminateAllWatchers()
	// We explicitly do NOT Unlock() in this method.
	// This is because we do not want to allow any WATCH/LIST methods before
	// the cache is initialized. Once the underlying cache is propagated,
	// onReplace handler will be called, which will do the Unlock() as
	// configured in NewCacher().
	c.reflector.ListAndWatch(stopChannel)
}

func (c *Cacher) Watch(key string, resourceVersion uint64, filter storage.FilterFunc) (watch.Interface, error) {
	c.Lock()
	defer c.Unlock()

	watcher := newCacheWatcher(filterFunction(key, filter), forgetWatcher(c, c.watcherIdx))
	c.initWatcher(watcher, resourceVersion)
	c.watchers[c.watcherIdx] = watcher
	c.watcherIdx++
	return watcher, nil
}

func (c *Cacher) WatchList(key string, resourceVersion uint64, filter storage.FilterFunc) (watch.Interface, error) {
	c.Lock()
	defer c.Unlock()

	watcher := newCacheWatcher(filterFunction(key, filter), forgetWatcher(c, c.watcherIdx))
	c.initWatcher(watcher, resourceVersion)
	c.watchers[c.watcherIdx] = watcher
	c.watcherIdx++
	return watcher, nil
}

func (c *Cacher) List(key string, listObj runtime.Object) error {
	listPtr, err := runtime.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}
	filter := filterFunction(key, storage.Everything)

	objs, resourceVersion := c.watchCache.ListWithVersion()
	for _, obj := range(objs) {
		object, ok := obj.(runtime.Object)
		if !ok {
			return fmt.Errorf("non runtime.Object returned from storage: %v", obj)
		}
		if filter(object) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(object)))
		}
	}
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, resourceVersion); err != nil {
			return err
		}
	}
	return nil
}

func (c *Cacher) processEvent(event watch.Event) {
	c.Lock()
	defer c.Unlock()
	for _, watcher := range c.watchers {
		watcher.add(event)
	}
}

// Send all events since resourceVersion up-to now to the watcher.
func (c *Cacher) initWatcher(watcher *cacheWatcher, resourceVersion uint64) {
	events := c.watchCache.GetAllEventsSince(resourceVersion)
	for _, event := range events {
		watcher.add(event)
	}
}

func (c *Cacher) terminateAllWatchers() {
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
		// simulanuous Stop() and terminateAllWatchers(), but it doesn't break anything.
		delete(c.watchers, index)
	}
}

func filterFunction(key string, filter storage.FilterFunc) storage.FilterFunc {
	// TODO: This method is in fact revert-engineering of the key.
	// We should change the api of WATCH and LIST methods to take a combination of
	// resource kind, namespace and (possibly empty) name instead.
	items := strings.Split(key, "/")
	namespace := ""
	name := ""
	if len(items) >= 2 {
		namespace = items[1]
	}
	if len(items) >= 3 {
		name = items[2]
	}
	return func(obj runtime.Object) bool {
		meta, err := meta.Accessor(obj)
		if err != nil {
			glog.Errorf("Invalid object for filter: %v", obj)
			return false
		}
		if namespace != "" && meta.Namespace() != namespace {
			return false
		}
		if name != "" && meta.Name() != name {
			return false
		}
		return filter(obj)
	}
}

// cacherListerWatcher opaques storage.Interface to expose cache.ListerWatcher.
type cacherListerWatcher struct {
	storage        storage.Interface
	resourcePrefix string
	newListFunc    func() runtime.Object
}

func newCacherListerWatcher(storage storage.Interface, resourcePrefix string, newListFunc func() runtime.Object) cache.ListerWatcher {
	return &cacherListerWatcher{
		storage:        storage,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
	}
}

func (lw *cacherListerWatcher) List() (runtime.Object, error) {
	list := lw.newListFunc()
	if err := lw.storage.GetToList(lw.resourcePrefix, list); err != nil {
		return nil, err
	}
	return list, nil
}

func (lw *cacherListerWatcher) Watch(resourceVersion string) (watch.Interface, error) {
	version, err := storage.ParseWatchResourceVersion(resourceVersion, lw.resourcePrefix)
	if err != nil {
		return nil, err
	}
	return lw.storage.WatchList(lw.resourcePrefix, version, storage.Everything)
}

// cacherWatch implements watch.Interface
type cacheWatcher struct {
	sync.Mutex
	input   chan watch.Event
	result  chan watch.Event
	filter  storage.FilterFunc
	stopped bool
	forget  func()
}

func newCacheWatcher(filter storage.FilterFunc, forget func()) *cacheWatcher {
	watcher := &cacheWatcher{
		input:   make(chan watch.Event, 10),
		result:  make(chan watch.Event, 10),
		filter:  filter,
		stopped: false,
		forget:  forget,
	}
	go watcher.process()
	return watcher
}

func (c *cacheWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

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

func (c *cacheWatcher) add(event watch.Event) {
	c.input <- event
}

func (c *cacheWatcher) process() {
	defer close(c.result)
	defer c.Stop()
	for {
		event, ok := <-c.input
		if !ok {
			return
		}
		if c.filter(event.Object) {
			c.result <- event
		}
	}
}
