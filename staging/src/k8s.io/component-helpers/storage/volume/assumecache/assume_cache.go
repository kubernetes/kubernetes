/*
Copyright 2025 The Kubernetes Authors.

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

// Package assumecache provides a cache layered on top of a shared informer that
// lets a client observe its own writes ("assumptions") before the informer
// delivers them, while always deferring to the informer as the source of truth.
package assumecache

import (
	"fmt"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// Informer is the subset of [cache.SharedInformer] that NewAssumeCache depends upon.
type Informer interface {
	AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error)
	GetIndexer() cache.Indexer
}

// AssumeCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object.
//
// An informer update always overrides the assumed object.
//
// This is different from pkg/scheduler/util/assumecache in:
//   - this does not dispatch events
//   - this is always up-to-date with the informer
type AssumeCache[T v1.Object] struct {
	// The logger that was chosen when setting up the cache.
	// Will be used for all operations.
	logger klog.Logger
	gr     schema.GroupResource

	// Synchronizes updates to all fields below.
	// Although [store] have its own lock, we still need to hold our lock
	// before reading from either [store] or [assumed] if we compare
	// ResourceVersion between them.
	rwMutex sync.RWMutex

	// Objects from informer
	store   cache.Indexer
	assumed map[string]assumedObject[T]
}

// assumedObject is an object assumed into the cache together with how it should
// be reconciled against the informer.
type assumedObject[T v1.Object] struct {
	object T
	// Whether object was already written to the apiserver.
	persisted bool
}

// preferOver reports whether the assumed object should be served instead of the
// informer object with the given resource version, i.e. the assumption has not
// yet been superseded by the informer.
func (a assumedObject[T]) preferOver(storeRV string) bool {
	assumedRV := a.object.GetResourceVersion()
	if !a.persisted {
		// Keep the optimistic object, skip resync.
		return assumedRV == storeRV
	}
	cmp, err := resourceversion.CompareResourceVersion(assumedRV, storeRV)
	if err != nil {
		// Non-conformant apiserver. Drop assumption to avoid leaks.
		return false
	}
	// Keep the written object until the informer reaches its version.
	return cmp > 0
}

// NewAssumeCache creates an assume cache for objects of type T.
func NewAssumeCache[T v1.Object](logger klog.Logger, informer Informer, gr schema.GroupResource) (*AssumeCache[T], error) {
	c := &AssumeCache[T]{
		logger:  logger,
		gr:      gr,
		store:   informer.GetIndexer(),
		assumed: make(map[string]assumedObject[T]),
	}

	_, err := informer.AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.add,
			UpdateFunc: c.update,
			DeleteFunc: c.delete,
		},
	)
	return c, err
}

// Receives events from informer. May expire the assumed object once the
// informer has caught up to (or past) it.
func (c *AssumeCache[T]) mayExpire(key string) {
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[key]
	if !ok {
		return
	}

	// Get the latest version to avoid overwriting newer object from [Assume]
	obj, exists, err := c.store.GetByKey(key)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "mayExpire get", "key", key)
		return
	}

	if exists {
		newMeta, err := meta.Accessor(obj)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "mayExpire meta", "key", key)
			return
		}

		if assumed.preferOver(newMeta.GetResourceVersion()) {
			c.logger.V(10).Info("keeping assumed object", "key", key,
				"version", assumed.object.GetResourceVersion(), "informerVersion", newMeta.GetResourceVersion())
			return
		}
		c.logger.V(4).Info("assumed object expired", "key", key,
			"version", assumed.object.GetResourceVersion(), "newVersion", newMeta.GetResourceVersion())
	} else {
		c.logger.V(4).Info("assumed object expired", "key", key, "version", assumed.object.GetResourceVersion())
	}
	delete(c.assumed, key)
}

func (c *AssumeCache[T]) add(obj any) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "Add object get key")
		return
	}
	c.mayExpire(key)
}

func (c *AssumeCache[T]) update(_, obj any) {
	c.add(obj)
}

func (c *AssumeCache[T]) delete(obj any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "Delete object get key")
		return
	}
	c.mayExpire(key)
}

// ByIndex returns the stored objects whose set of indexed values for the named index includes the given indexed value
//
// The index is evaluated on the object from store. Objects from [Assume] will present in the result but will not affect the index.
func (c *AssumeCache[T]) ByIndex(indexName, indexedValue string) ([]T, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objs, err := c.store.ByIndex(indexName, indexedValue)
	if err != nil {
		return nil, err
	}
	return c.replaceAssumed(objs), nil
}

// Get the object by its key.
func (c *AssumeCache[T]) Get(key string) (T, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	obj, err := c.GetAPIObj(key)
	if err != nil {
		return obj, err
	}

	assumed, ok := c.assumed[key]
	if !ok || !assumed.preferOver(obj.GetResourceVersion()) {
		// not assumed, or the informer object is preferred
		return obj, nil
	}
	return assumed.object, nil
}

// GetAPIObj gets the informer cache's version by its key.
func (c *AssumeCache[T]) GetAPIObj(key string) (T, error) {
	obj, ok, err := c.store.GetByKey(key)
	var zero T
	if err != nil {
		return zero, err
	}
	if !ok {
		return zero, apierrors.NewNotFound(c.gr, key)
	}
	v, ok := obj.(T)
	if !ok {
		return zero, fmt.Errorf("object is not of type %T", zero)
	}
	return v, nil
}

func keyOf[T v1.Object](obj T) string {
	return cache.MetaObjectToName(obj).String()
}

func (c *AssumeCache[T]) replaceAssumed(objs []any) []T {
	allObjs := make([]T, 0, len(objs))
	for _, obj := range objs {
		v, ok := obj.(T)
		if !ok {
			utilruntime.HandleErrorWithLogger(c.logger, nil, "listed object has wrong type", "type", fmt.Sprintf("%T", obj))
			continue
		}
		if assumed, ok := c.assumed[keyOf(v)]; ok && assumed.preferOver(v.GetResourceVersion()) {
			// the assumed object is newer than (or not yet in) the informer
			v = assumed.object
		}
		allObjs = append(allObjs, v)
	}
	return allObjs
}

// Assume optimistically records an in-memory change to an object before it is
// written to the apiserver. The object's ResourceVersion must equal the
// informer's current version, otherwise an error is returned. The assumption is
// dropped as soon as the informer observes any newer version.
//
// For an object that has already been written to the apiserver, use
// [AssumeCache.AssumeWritten] instead.
func (c *AssumeCache[T]) Assume(obj T) error {
	key := keyOf(obj)

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	stored, err := c.GetAPIObj(key)
	if err != nil {
		return err
	}

	if stored.GetResourceVersion() != obj.GetResourceVersion() {
		return fmt.Errorf("%q is out of sync (stored: %s, assume: %s)", key, stored.GetResourceVersion(), obj.GetResourceVersion())
	}
	c.assumed[key] = assumedObject[T]{object: obj, persisted: false}
	c.logger.V(4).Info("Assumed object", "key", key, "version", obj.GetResourceVersion())
	return nil
}

// AssumeWritten records an object that has already been written to the
// apiserver, i.e. obj carries the server-assigned ResourceVersion returned by
// the write. Reads observe obj until the informer delivers that version or a
// newer one, so a sequence of writes (e.g. spec then status) is not transiently
// reverted to an intermediate version while the informer catches up.
//
// It is a no-op (returning nil) when:
//   - obj is not present in the informer — either a creation whose event has
//     not been delivered yet, or an object that has already been deleted;
//     leaving these to the informer is what prevents a deleted object from
//     being resurrected; or
//   - the informer already holds the same or a newer version, or the
//     ResourceVersions are not comparable (a non-conformant apiserver).
func (c *AssumeCache[T]) AssumeWritten(obj T) error {
	key := keyOf(obj)

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	stored, exists, err := c.store.GetByKey(key)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	storedMeta, err := meta.Accessor(stored)
	if err != nil {
		return err
	}
	assumed := assumedObject[T]{object: obj, persisted: true}
	if assumed.preferOver(storedMeta.GetResourceVersion()) {
		c.assumed[key] = assumed
		c.logger.V(4).Info("Assumed written object", "key", key, "version", obj.GetResourceVersion())
	} else {
		c.logger.V(4).Info("Reject expired assumption",
			"key", key, "version", obj.GetResourceVersion(), "informerVersion", storedMeta.GetResourceVersion())
	}
	return nil
}

// Restore the informer cache's version of the object.
func (c *AssumeCache[T]) Restore(obj T) {
	key := keyOf(obj)

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[key]
	if ok && assumed.object.GetResourceVersion() == obj.GetResourceVersion() {
		delete(c.assumed, key)
		c.logger.V(4).Info("Restored object", "key", key, "version", obj.GetResourceVersion())
	}
}
