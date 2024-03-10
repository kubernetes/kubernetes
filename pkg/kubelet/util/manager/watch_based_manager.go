/*
Copyright 2018 The Kubernetes Authors.

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

package manager

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/utils/clock"
)

type listObjectFunc func(string, metav1.ListOptions) (runtime.Object, error)
type watchObjectFunc func(string, metav1.ListOptions) (watch.Interface, error)
type newObjectFunc func() runtime.Object
type isImmutableFunc func(runtime.Object) bool

// objectCacheItem is a single item stored in objectCache.
type objectCacheItem struct {
	refMap    map[types.UID]int
	store     *cacheStore
	reflector *cache.Reflector

	hasSynced func() (bool, error)

	// waitGroup is used to ensure that there won't be two concurrent calls to reflector.Run
	waitGroup sync.WaitGroup

	// lock is to ensure the access and modify of lastAccessTime, stopped, and immutable are thread safety,
	// and protecting from closing stopCh multiple times.
	lock           sync.Mutex
	lastAccessTime time.Time
	stopped        bool
	immutable      bool
	stopCh         chan struct{}
}

func (i *objectCacheItem) stop() bool {
	i.lock.Lock()
	defer i.lock.Unlock()
	return i.stopThreadUnsafe()
}

func (i *objectCacheItem) stopThreadUnsafe() bool {
	if i.stopped {
		return false
	}
	i.stopped = true
	close(i.stopCh)
	if !i.immutable {
		i.store.unsetInitialized()
	}
	return true
}

func (i *objectCacheItem) setLastAccessTime(time time.Time) {
	i.lock.Lock()
	defer i.lock.Unlock()
	i.lastAccessTime = time
}

func (i *objectCacheItem) setImmutable() {
	i.lock.Lock()
	defer i.lock.Unlock()
	i.immutable = true
}

func (i *objectCacheItem) stopIfIdle(now time.Time, maxIdleTime time.Duration) bool {
	i.lock.Lock()
	defer i.lock.Unlock()
	// Ensure that we don't try to stop not yet initialized reflector.
	// In case of overloaded kube-apiserver, if the list request is
	// already being processed, all the work would lost and would have
	// to be retried.
	if !i.stopped && i.store.hasSynced() && now.After(i.lastAccessTime.Add(maxIdleTime)) {
		return i.stopThreadUnsafe()
	}
	return false
}

func (i *objectCacheItem) restartReflectorIfNeeded() {
	i.lock.Lock()
	defer i.lock.Unlock()
	if i.immutable || !i.stopped {
		return
	}
	i.stopCh = make(chan struct{})
	i.stopped = false
	go i.startReflector()
}

func (i *objectCacheItem) startReflector() {
	i.waitGroup.Wait()
	i.waitGroup.Add(1)
	defer i.waitGroup.Done()
	i.reflector.Run(i.stopCh)
}

// cacheStore is in order to rewrite Replace function to mark initialized flag
type cacheStore struct {
	cache.Store
	lock        sync.Mutex
	initialized bool
}

func (c *cacheStore) Replace(list []interface{}, resourceVersion string) error {
	c.lock.Lock()
	defer c.lock.Unlock()
	err := c.Store.Replace(list, resourceVersion)
	if err != nil {
		return err
	}
	c.initialized = true
	return nil
}

func (c *cacheStore) hasSynced() bool {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.initialized
}

func (c *cacheStore) unsetInitialized() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.initialized = false
}

// objectCache is a local cache of objects propagated via
// individual watches.
type objectCache struct {
	listObject    listObjectFunc
	watchObject   watchObjectFunc
	newObject     newObjectFunc
	isImmutable   isImmutableFunc
	groupResource schema.GroupResource
	clock         clock.Clock
	maxIdleTime   time.Duration

	lock    sync.RWMutex
	items   map[objectKey]*objectCacheItem
	stopped bool
}

const minIdleTime = 1 * time.Minute

// NewObjectCache returns a new watch-based instance of Store interface.
func NewObjectCache(
	listObject listObjectFunc,
	watchObject watchObjectFunc,
	newObject newObjectFunc,
	isImmutable isImmutableFunc,
	groupResource schema.GroupResource,
	clock clock.Clock,
	maxIdleTime time.Duration,
	stopCh <-chan struct{}) Store {

	if maxIdleTime < minIdleTime {
		maxIdleTime = minIdleTime
	}

	store := &objectCache{
		listObject:    listObject,
		watchObject:   watchObject,
		newObject:     newObject,
		isImmutable:   isImmutable,
		groupResource: groupResource,
		clock:         clock,
		maxIdleTime:   maxIdleTime,
		items:         make(map[objectKey]*objectCacheItem),
	}

	go wait.Until(store.startRecycleIdleWatch, time.Minute, stopCh)
	go store.shutdownWhenStopped(stopCh)
	return store
}

func (c *objectCache) newStore() *cacheStore {
	// TODO: We may consider created a dedicated store keeping just a single
	// item, instead of using a generic store implementation for this purpose.
	// However, simple benchmarks show that memory overhead in that case is
	// decrease from ~600B to ~300B per object. So we are not optimizing it
	// until we will see a good reason for that.
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	return &cacheStore{store, sync.Mutex{}, false}
}

func (c *objectCache) newReflectorLocked(namespace, name string) *objectCacheItem {
	fieldSelector := fields.Set{"metadata.name": name}.AsSelector().String()
	listFunc := func(options metav1.ListOptions) (runtime.Object, error) {
		options.FieldSelector = fieldSelector
		return c.listObject(namespace, options)
	}
	watchFunc := func(options metav1.ListOptions) (watch.Interface, error) {
		options.FieldSelector = fieldSelector
		return c.watchObject(namespace, options)
	}
	store := c.newStore()
	reflector := cache.NewReflectorWithOptions(
		&cache.ListWatch{ListFunc: listFunc, WatchFunc: watchFunc},
		c.newObject(),
		store,
		cache.ReflectorOptions{
			Name: fmt.Sprintf("object-%q/%q", namespace, name),
			// Bump default 5m MinWatchTimeout to avoid recreating
			// watches too often.
			MinWatchTimeout: 30 * time.Minute,
			EnableMetrics:   false,
		},
	)
	item := &objectCacheItem{
		refMap:    make(map[types.UID]int),
		store:     store,
		reflector: reflector,
		hasSynced: func() (bool, error) { return store.hasSynced(), nil },
		stopCh:    make(chan struct{}),
	}

	// Don't start reflector if Kubelet is already shutting down.
	if !c.stopped {
		go item.startReflector()
	}
	return item
}

func (c *objectCache) AddReference(namespace, name string, referencedFrom types.UID) {
	key := objectKey{namespace: namespace, name: name}

	// AddReference is called from RegisterPod thus it needs to be efficient.
	// Thus, it is only increasing refCount and in case of first registration
	// of a given object it starts corresponding reflector.
	// It's responsibility of the first Get operation to wait until the
	// reflector propagated the store.
	c.lock.Lock()
	defer c.lock.Unlock()
	item, exists := c.items[key]
	if !exists {
		item = c.newReflectorLocked(namespace, name)
		c.items[key] = item
	}
	item.refMap[referencedFrom]++
}

func (c *objectCache) DeleteReference(namespace, name string, referencedFrom types.UID) {
	key := objectKey{namespace: namespace, name: name}

	c.lock.Lock()
	defer c.lock.Unlock()
	if item, ok := c.items[key]; ok {
		item.refMap[referencedFrom]--
		if item.refMap[referencedFrom] == 0 {
			delete(item.refMap, referencedFrom)
		}
		if len(item.refMap) == 0 {
			// Stop the underlying reflector.
			item.stop()
			delete(c.items, key)
		}
	}
}

// key returns key of an object with a given name and namespace.
// This has to be in-sync with cache.MetaNamespaceKeyFunc.
func (c *objectCache) key(namespace, name string) string {
	if len(namespace) > 0 {
		return namespace + "/" + name
	}
	return name
}

func (c *objectCache) isStopped() bool {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.stopped
}

func (c *objectCache) Get(namespace, name string) (runtime.Object, error) {
	key := objectKey{namespace: namespace, name: name}

	c.lock.RLock()
	item, exists := c.items[key]
	c.lock.RUnlock()

	if !exists {
		return nil, fmt.Errorf("object %q/%q not registered", namespace, name)
	}
	// Record last access time independently if it succeeded or not.
	// This protects from premature (racy) reflector closure.
	item.setLastAccessTime(c.clock.Now())

	// Don't restart reflector if Kubelet is already shutting down.
	if !c.isStopped() {
		item.restartReflectorIfNeeded()
	}
	if err := wait.PollImmediate(10*time.Millisecond, time.Second, item.hasSynced); err != nil {
		return nil, fmt.Errorf("failed to sync %s cache: %v", c.groupResource.String(), err)
	}
	obj, exists, err := item.store.GetByKey(c.key(namespace, name))
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, apierrors.NewNotFound(c.groupResource, name)
	}
	if object, ok := obj.(runtime.Object); ok {
		// If the returned object is immutable, stop the reflector.
		//
		// NOTE: we may potentially not even start the reflector if the object is
		// already immutable. However, given that:
		// - we want to also handle the case when object is marked as immutable later
		// - Secrets and ConfigMaps are periodically fetched by volumemanager anyway
		// - doing that wouldn't provide visible scalability/performance gain - we
		//   already have it from here
		// - doing that would require significant refactoring to reflector
		// we limit ourselves to just quickly stop the reflector here.
		if c.isImmutable(object) {
			item.setImmutable()
			if item.stop() {
				klog.V(4).InfoS("Stopped watching for changes - object is immutable", "obj", klog.KRef(namespace, name))
			}
		}
		return object, nil
	}
	return nil, fmt.Errorf("unexpected object type: %v", obj)
}

func (c *objectCache) startRecycleIdleWatch() {
	c.lock.Lock()
	defer c.lock.Unlock()

	for key, item := range c.items {
		if item.stopIfIdle(c.clock.Now(), c.maxIdleTime) {
			klog.V(4).InfoS("Not acquired for long time, Stopped watching for changes", "objectKey", key, "maxIdleTime", c.maxIdleTime)
		}
	}
}

func (c *objectCache) shutdownWhenStopped(stopCh <-chan struct{}) {
	<-stopCh

	c.lock.Lock()
	defer c.lock.Unlock()

	c.stopped = true
	for _, item := range c.items {
		item.stop()
	}
}

// NewWatchBasedManager creates a manager that keeps a cache of all objects
// necessary for registered pods.
// It implements the following logic:
//   - whenever a pod is created or updated, we start individual watches for all
//     referenced objects that aren't referenced from other registered pods
//   - every GetObject() returns a value from local cache propagated via watches
func NewWatchBasedManager(
	listObject listObjectFunc,
	watchObject watchObjectFunc,
	newObject newObjectFunc,
	isImmutable isImmutableFunc,
	groupResource schema.GroupResource,
	resyncInterval time.Duration,
	getReferencedObjects func(*v1.Pod) sets.Set[string]) Manager {

	// If a configmap/secret is used as a volume, the volumeManager will visit the objectCacheItem every resyncInterval cycle,
	// We just want to stop the objectCacheItem referenced by environment variables,
	// So, maxIdleTime is set to an integer multiple of resyncInterval,
	// We currently set it to 5 times.
	maxIdleTime := resyncInterval * 5

	// TODO propagate stopCh from the higher level.
	objectStore := NewObjectCache(listObject, watchObject, newObject, isImmutable, groupResource, clock.RealClock{}, maxIdleTime, wait.NeverStop)
	return NewCacheBasedManager(objectStore, getReferencedObjects)
}
