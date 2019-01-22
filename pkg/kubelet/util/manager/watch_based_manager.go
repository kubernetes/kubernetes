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

// TODO: We did some scalability tests and using watchBasedManager
// seems to help with apiserver performance at scale visibly.
// No issues we also observed at the scale of ~200k watchers with a
// single apiserver.
// However, we need to perform more extensive testing before we
// enable this in production setups.

package manager

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
)

type listObjectFunc func(string, metav1.ListOptions) (runtime.Object, error)
type watchObjectFunc func(string, metav1.ListOptions) (watch.Interface, error)
type newObjectFunc func() runtime.Object

// objectCacheItem is a single item stored in objectCache.
type objectCacheItem struct {
	refCount  int
	store     cache.Store
	hasSynced func() (bool, error)
	stopCh    chan struct{}
}

// objectCache is a local cache of objects propagated via
// individual watches.
type objectCache struct {
	listObject    listObjectFunc
	watchObject   watchObjectFunc
	newObject     newObjectFunc
	groupResource schema.GroupResource

	lock  sync.Mutex
	items map[objectKey]*objectCacheItem
}

// NewObjectCache returns a new watch-based instance of Store interface.
func NewObjectCache(listObject listObjectFunc, watchObject watchObjectFunc, newObject newObjectFunc, groupResource schema.GroupResource) Store {
	return &objectCache{
		listObject:    listObject,
		watchObject:   watchObject,
		newObject:     newObject,
		groupResource: groupResource,
		items:         make(map[objectKey]*objectCacheItem),
	}
}

func (c *objectCache) newStore() cache.Store {
	// TODO: We may consider created a dedicated store keeping just a single
	// item, instead of using a generic store implementation for this purpose.
	// However, simple benchmarks show that memory overhead in that case is
	// decrease from ~600B to ~300B per object. So we are not optimizing it
	// until we will see a good reason for that.
	return cache.NewStore(cache.MetaNamespaceKeyFunc)
}

func (c *objectCache) newReflector(namespace, name string) *objectCacheItem {
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
	reflector := cache.NewNamedReflector(
		fmt.Sprintf("object-%q/%q", namespace, name),
		&cache.ListWatch{ListFunc: listFunc, WatchFunc: watchFunc},
		c.newObject(),
		store,
		0,
	)
	stopCh := make(chan struct{})
	go reflector.Run(stopCh)
	return &objectCacheItem{
		refCount:  0,
		store:     store,
		hasSynced: func() (bool, error) { return reflector.LastSyncResourceVersion() != "", nil },
		stopCh:    stopCh,
	}
}

func (c *objectCache) AddReference(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	// AddReference is called from RegisterPod thus it needs to be efficient.
	// Thus, it is only increaisng refCount and in case of first registration
	// of a given object it starts corresponding reflector.
	// It's responsibility of the first Get operation to wait until the
	// reflector propagated the store.
	c.lock.Lock()
	defer c.lock.Unlock()
	item, exists := c.items[key]
	if !exists {
		item = c.newReflector(namespace, name)
		c.items[key] = item
	}
	item.refCount++
}

func (c *objectCache) DeleteReference(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	c.lock.Lock()
	defer c.lock.Unlock()
	if item, ok := c.items[key]; ok {
		item.refCount--
		if item.refCount == 0 {
			// Stop the underlying reflector.
			close(item.stopCh)
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

func (c *objectCache) Get(namespace, name string) (runtime.Object, error) {
	key := objectKey{namespace: namespace, name: name}

	c.lock.Lock()
	item, exists := c.items[key]
	c.lock.Unlock()

	if !exists {
		return nil, fmt.Errorf("object %q/%q not registered", namespace, name)
	}
	if err := wait.PollImmediate(10*time.Millisecond, time.Second, item.hasSynced); err != nil {
		return nil, fmt.Errorf("couldn't propagate object cache: %v", err)
	}

	obj, exists, err := item.store.GetByKey(c.key(namespace, name))
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, apierrors.NewNotFound(c.groupResource, name)
	}
	if object, ok := obj.(runtime.Object); ok {
		return object, nil
	}
	return nil, fmt.Errorf("unexpected object type: %v", obj)
}

// NewWatchBasedManager creates a manager that keeps a cache of all objects
// necessary for registered pods.
// It implements the following logic:
// - whenever a pod is created or updated, we start individual watches for all
//   referenced objects that aren't referenced from other registered pods
// - every GetObject() returns a value from local cache propagated via watches
func NewWatchBasedManager(listObject listObjectFunc, watchObject watchObjectFunc, newObject newObjectFunc, groupResource schema.GroupResource, getReferencedObjects func(*v1.Pod) sets.String) Manager {
	objectStore := NewObjectCache(listObject, watchObject, newObject, groupResource)
	return NewCacheBasedManager(objectStore, getReferencedObjects)
}
