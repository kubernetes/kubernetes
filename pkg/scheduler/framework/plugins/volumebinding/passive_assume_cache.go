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

package volumebinding

import (
	"fmt"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// informer is the subset of [cache.SharedInformer] that newAssumeCache depends upon.
type informer interface {
	AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error)
	GetIndexer() cache.Indexer
}

// passiveAssumeCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object.
//
// An informer update always overrides the assumed object.
//
// This is different from pkg/scheduler/util/assumecache in:
//   - this does not dispatch events
//   - this is always up-to-date with the informer
//   - this only allow assuming objects yet to be sent to the apiserver,
//     not the ones returned from the apiserver
type passiveAssumeCache[T v1.Object] struct {
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
	assumed map[string]T
}

// newAssumeCache creates an assume cache for objects of type T.
func newAssumeCache[T v1.Object](logger klog.Logger, informer informer, gr schema.GroupResource) (*passiveAssumeCache[T], error) {
	c := &passiveAssumeCache[T]{
		logger:  logger,
		gr:      gr,
		store:   informer.GetIndexer(),
		assumed: make(map[string]T),
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

// Receives events from informer. May expire the assumed object if it is older.
func (c *passiveAssumeCache[T]) mayExpire(key string) {
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

	expire := true
	if exists {
		newMeta, err := meta.Accessor(obj)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "mayExpire meta", "key", key)
			return
		}

		// Only overwrite assumed object if version is newer (not resync).
		if assumed.GetResourceVersion() == newMeta.GetResourceVersion() {
			c.logger.V(10).Info("ignoring resync of assumed object", "key", key, "version", assumed.GetResourceVersion())
			expire = false
		} else {
			c.logger.V(4).Info("assumed object expired", "newVersion", newMeta.GetResourceVersion(),
				"key", key, "version", assumed.GetResourceVersion())
		}
	} else {
		c.logger.V(4).Info("assumed object expired", "key", key, "version", assumed.GetResourceVersion())
	}
	if expire {
		delete(c.assumed, key)
	}
}

func (c *passiveAssumeCache[T]) add(obj any) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "Add object get key")
		return
	}
	c.mayExpire(key)
}

func (c *passiveAssumeCache[T]) update(_, obj any) {
	c.add(obj)
}

func (c *passiveAssumeCache[T]) delete(obj any) {
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
func (c *passiveAssumeCache[T]) ByIndex(indexName, indexedValue string) ([]T, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objs, err := c.store.ByIndex(indexName, indexedValue)
	if err != nil {
		return nil, err
	}
	return c.replaceAssumed(objs), nil
}

// Get the object by its key.
func (c *passiveAssumeCache[T]) Get(key string) (T, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	obj, err := c.GetAPIObj(key)
	if err != nil {
		return obj, err
	}

	assumed, ok := c.assumed[key]
	if !ok || assumed.GetResourceVersion() != obj.GetResourceVersion() { // not assumed or Informer object is newer
		return obj, nil
	}
	return assumed, nil
}

// GetAPIObj gets the informer cache's version by its key.
func (c *passiveAssumeCache[T]) GetAPIObj(key string) (T, error) {
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

func (c *passiveAssumeCache[T]) replaceAssumed(objs []any) []T {
	allObjs := make([]T, 0, len(objs))
	for _, obj := range objs {
		v, ok := obj.(T)
		if !ok {
			utilruntime.HandleErrorWithLogger(c.logger, nil, "listed object has wrong type", "type", fmt.Sprintf("%T", obj))
			continue
		}
		assumed, ok := c.assumed[keyOf(v)]
		if ok && assumed.GetResourceVersion() == v.GetResourceVersion() {
			// assumed object is not in informer yet
			v = assumed
		}
		allObjs = append(allObjs, v)
	}
	return allObjs
}

// Assume updates the object in-memory only.
//
// The version of the object must be equal to
// the current object, otherwise an error is returned.
// If an update is received via the informer while such an
// object is assumed, it gets dropped in favor of the
// newer object from the apiserver.
func (c *passiveAssumeCache[T]) Assume(obj T) error {
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
	c.assumed[key] = obj
	c.logger.V(4).Info("Assumed object", "key", key, "version", obj.GetResourceVersion())
	return nil
}

// Restore the informer cache's version of the object.
func (c *passiveAssumeCache[T]) Restore(obj T) {
	key := keyOf(obj)

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[key]
	if ok && assumed.GetResourceVersion() == obj.GetResourceVersion() {
		delete(c.assumed, key)
		c.logger.V(4).Info("Restored object", "key", key, "version", obj.GetResourceVersion())
	}
}
