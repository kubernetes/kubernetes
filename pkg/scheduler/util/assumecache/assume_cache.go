/*
Copyright 2017 The Kubernetes Authors.

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

package assumecache

import (
	"errors"
	"fmt"
	"strconv"
	"sync"

	"k8s.io/apimachinery/pkg/api/meta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/utils/buffer"
)

// Informer is the subset of [cache.SharedInformer] that NewAssumeCache depends upon.
type Informer interface {
	AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error)
	GetIndexer() cache.Indexer
}

// AddTestObject adds an object to the assume cache.
// Only use this for unit testing!
func AddTestObject(cache *AssumeCache, obj interface{}) {
	cache.add(obj)
}

// UpdateTestObject updates an object in the assume cache.
// Only use this for unit testing!
func UpdateTestObject(cache *AssumeCache, obj interface{}) {
	cache.update(nil, obj)
}

// DeleteTestObject deletes object in the assume cache.
// Only use this for unit testing!
func DeleteTestObject(cache *AssumeCache, obj interface{}) {
	cache.delete(obj)
}

// Sentinel errors that can be checked for with errors.Is.
var (
	ErrWrongType  = errors.New("object has wrong type")
	ErrNotFound   = errors.New("object not found")
	ErrObjectName = errors.New("cannot determine object name")
)

type WrongTypeError struct {
	TypeName string
	Object   interface{}
}

func (e WrongTypeError) Error() string {
	return fmt.Sprintf("could not convert object to type %v: %+v", e.TypeName, e.Object)
}

func (e WrongTypeError) Is(err error) bool {
	return err == ErrWrongType
}

type NotFoundError struct {
	TypeName  string
	ObjectKey string
}

func (e NotFoundError) Error() string {
	return fmt.Sprintf("could not find %v %q", e.TypeName, e.ObjectKey)
}

func (e NotFoundError) Is(err error) bool {
	return err == ErrNotFound
}

type ObjectMetaError struct {
	DetailedErr error
}

func (e ObjectMetaError) Error() string {
	return fmt.Sprintf("failed to get object metadata: %v", e.DetailedErr)
}

func (e ObjectMetaError) Is(err error) bool {
	return err == ErrObjectName
}

// AssumeCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object. Objects are assumed to be
// Kubernetes API objects that are supported by [meta.Accessor].
//
// Objects can referenced via their key, with [cache.MetaNamespaceKeyFunc]
// as key function.
//
// AssumeCache stores two pointers to represent a single object:
//   - The pointer to the informer object.
//   - The pointer to the latest object, which could be the same as
//     the informer object, or an in-memory object.
//
// An informer update always overrides the latest object pointer.
//
// Assume() only updates the latest object pointer.
// Restore() sets the latest object pointer back to the informer object.
// Get/List() always returns the latest object pointer.
type AssumeCache struct {
	// The logger that was chosen when setting up the cache.
	// Will be used for all operations.
	logger klog.Logger

	// Synchronizes updates to all fields below.
	rwMutex sync.RWMutex

	// All registered event handlers.
	eventHandlers       []cache.ResourceEventHandler
	handlerRegistration cache.ResourceEventHandlerRegistration

	// The eventQueue contains functions which deliver an event to one
	// event handler.
	//
	// These functions must be invoked while *not locking* rwMutex because
	// the event handlers are allowed to access the assume cache. Holding
	// rwMutex then would cause a deadlock.
	//
	// New functions get added as part of processing a cache update while
	// the rwMutex is locked. Each function which adds something to the queue
	// also drains the queue before returning, therefore it is guaranteed
	// that all event handlers get notified immediately (useful for unit
	// testing).
	//
	// A channel cannot be used here because it cannot have an unbounded
	// capacity. This could lead to a deadlock (writer holds rwMutex,
	// gets blocked because capacity is exhausted, reader is in a handler
	// which tries to lock the rwMutex). Writing into such a channel
	// while not holding the rwMutex doesn't work because in-order delivery
	// of events would no longer be guaranteed.
	eventQueue buffer.Ring[func()]

	// describes the object stored
	description string

	// Objects from informer
	store   cache.Indexer
	assumed map[string]v1.Object

	indexName string
}

// NewAssumeCache creates an assume cache for general objects.
func NewAssumeCache(logger klog.Logger, informer Informer, description, indexName string, indexFunc cache.IndexFunc) *AssumeCache {
	c := &AssumeCache{
		logger:      logger,
		description: description,
		store:       informer.GetIndexer(),
		assumed:     make(map[string]v1.Object),
		indexName:   indexName,
		eventQueue:  *buffer.NewRing[func()](buffer.RingOptions{InitialSize: 0, NormalSize: 4}),
	}
	if indexName != "" && indexFunc != nil {
		utilruntime.Must(c.store.AddIndexers(cache.Indexers{indexName: indexFunc}))
	}

	var err error
	c.handlerRegistration, err = informer.AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.add,
			UpdateFunc: c.update,
			DeleteFunc: c.delete,
		},
	)
	utilruntime.Must(err)

	return c
}

func (c *AssumeCache) add(obj interface{}) {
	c.update(nil, obj)
}

// Receives the new object from informer. May expire the assumed object if it is older.
func (c *AssumeCache) update(oldObj interface{}, obj interface{}) {
	if obj == nil {
		return
	}

	newMeta, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, ObjectMetaError{DetailedErr: err}, "Add failed")
		return
	}
	name := cache.MetaObjectToName(newMeta).String()

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[name]
	if ok {
		// Object is assumed, check if the informer object is newer.
		cmp, err := compareRV(newMeta, assumed)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, ObjectMetaError{DetailedErr: err}, "Add failed")
			return
		}

		// Only update object if version is newer.
		// This is so we don't override assumed objects due to informer resync.
		if cmp <= 0 {
			c.logger.V(10).Info("Skip adding object to assume cache because version is not newer than assumedVersion",
				"description", c.description, "cacheKey", name, "newVersion", newMeta.GetResourceVersion(), "assumedVersion", assumed.GetResourceVersion())
			return
		}
		oldObj = assumed
		delete(c.assumed, name)
		c.logger.V(10).Info("assumed object expired", "description", c.description, "cacheKey", name, "assumeCache", obj)
	}

	c.pushEvent(oldObj, obj)
}

func (c *AssumeCache) delete(obj interface{}) {
	if obj == nil {
		return
	}

	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = d.Obj
	}
	metadata, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, ObjectMetaError{DetailedErr: err}, "Failed to delete")
		return
	}
	name := cache.MetaObjectToName(metadata).String()

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[name]
	if ok {
		obj = assumed
		delete(c.assumed, name)
	}

	c.pushEvent(obj, nil)
}

// pushEvent gets called while the mutex is locked for writing.
// It ensures that all currently registered event handlers get
// notified about a change when the caller starts delivering
// those with emitEvents.
//
// For a delete event, newObj is nil. For an add, oldObj is nil.
// An update has both as non-nil.
func (c *AssumeCache) pushEvent(oldObj, newObj interface{}) {
	for _, handler := range c.eventHandlers {
		handler := handler
		if oldObj == nil {
			c.eventQueue.WriteOne(func() {
				handler.OnAdd(newObj, false)
			})
		} else if newObj == nil {
			c.eventQueue.WriteOne(func() {
				handler.OnDelete(oldObj)
			})
		} else {
			c.eventQueue.WriteOne(func() {
				handler.OnUpdate(oldObj, newObj)
			})
		}
	}
}

func parseRV(rv string) (int64, error) {
	return strconv.ParseInt(rv, 10, 64)
}

func compareRV(a, b v1.Object) (int, error) {
	av, err := parseRV(a.GetResourceVersion())
	if err != nil {
		return 0, fmt.Errorf("failed to parse resource version for %q: %v", a.GetName(), err)
	}
	bv, err := parseRV(b.GetResourceVersion())
	if err != nil {
		return 0, fmt.Errorf("failed to parse resource version for %q: %v", b.GetName(), err)
	}
	if av < bv {
		return -1, nil
	} else if av > bv {
		return 1, nil
	}
	return 0, nil
}

func (c *AssumeCache) get(key string) (v1.Object, error) {
	obj, ok, err := c.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, &NotFoundError{TypeName: c.description, ObjectKey: key}
	}

	metadata, ok := obj.(v1.Object)
	if !ok {
		return nil, &WrongTypeError{TypeName: "v1.Object", Object: obj}
	}
	return metadata, nil
}

// Get the object by its key.
func (c *AssumeCache) Get(key string) (v1.Object, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	obj, err := c.get(key)
	if err != nil {
		return nil, err
	}

	assumed, ok := c.assumed[cache.MetaObjectToName(obj).String()]
	if !ok { // not assumed
		return obj, nil
	}
	cmp, err := compareRV(obj, assumed)
	if err != nil {
		return nil, err
	}
	if cmp > 0 { // Informer object is newer
		return obj, nil
	}
	return assumed, nil
}

// GetAPIObj gets the informer cache's version by its key.
func (c *AssumeCache) GetAPIObj(key string) (v1.Object, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	return c.get(key)
}

// List all the objects in the cache.
func (c *AssumeCache) List() []interface{} {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	return c.listLocked()
}

// ByIndex returns the stored objects whose set of indexed values
// for the named index includes the given indexed value
//
// Assumed objects will not be returned
func (c *AssumeCache) ByIndex(indexedValue string) []interface{} {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objs, err := c.store.ByIndex(c.indexName, indexedValue)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "List index error")
		return nil
	}
	return c.filterList(objs, false)
}

func (c *AssumeCache) filterList(objs []interface{}, includeAssumed bool) []interface{} {
	allObjs := make([]interface{}, 0, len(objs))
	for _, obj := range objs {
		metadata, err := meta.Accessor(obj)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "List error")
			continue
		}
		key := cache.MetaObjectToName(metadata).String()

		assumed, ok := c.assumed[key]
		if ok {
			cmp, err := compareRV(metadata, assumed)
			if err != nil {
				utilruntime.HandleErrorWithLogger(c.logger, err, "List error")
				continue
			}
			if cmp <= 0 { // assumed object is not in informer yet
				if includeAssumed {
					allObjs = append(allObjs, assumed)
				}
				continue
			}
		}
		allObjs = append(allObjs, obj)
	}
	return allObjs
}

func (c *AssumeCache) listLocked() []interface{} {
	objs := c.store.List()
	return c.filterList(objs, true)
}

// Assume updates the object in-memory only.
//
// The version of the object must be greater or equal to
// the current object, otherwise an error is returned.
//
// Storing an object with the same version is supported
// by the assume cache, but suffers from a race: if an
// update is received via the informer while such an
// object is assumed, it gets dropped in favor of the
// newer object from the apiserver.
//
// Only assuming objects that were returned by an apiserver
// operation (Update, Patch) is safe.
func (c *AssumeCache) Assume(obj v1.Object) error {
	key := cache.MetaObjectToName(obj).String()

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	stored, err := c.get(key)
	if err != nil {
		return err
	}

	cmp, err := compareRV(stored, obj)
	if err != nil {
		return err
	}

	if cmp > 0 {
		return fmt.Errorf("%v %q is out of sync (stored: %s, assume: %s)", c.description, key, stored.GetResourceVersion(), obj.GetResourceVersion())
	}

	c.assumed[key] = obj
	c.pushEvent(stored, obj)
	c.logger.V(4).Info("Assumed object", "description", c.description, "cacheKey", key, "version", obj.GetResourceVersion())
	return nil
}

// Restore the informer cache's version of the object.
func (c *AssumeCache) Restore(objName string) {
	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	assumed, ok := c.assumed[objName]
	if !ok {
		c.logger.V(5).Info("No need to restore object, not assumed", "description", c.description, "cacheKey", objName)
		return
	}

	delete(c.assumed, objName)
	obj, exists, err := c.store.GetByKey(objName)

	if err != nil || !exists {
		// This could be expected if object got deleted
		c.logger.V(5).Info("Restore object", "description", c.description, "cacheKey", objName, "err", err)
	} else {
		c.pushEvent(assumed, obj)
		c.logger.V(4).Info("Restored object", "description", c.description, "cacheKey", objName)
	}
}

// AddEventHandler adds an event handler to the cache. Events to a
// single handler are delivered sequentially, but there is no
// coordination between different handlers. A handler may use the
// cache.
//
// The return value can be used to wait for cache synchronization.
func (c *AssumeCache) AddEventHandler(handler cache.ResourceEventHandler) cache.ResourceEventHandlerRegistration {
	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	c.eventHandlers = append(c.eventHandlers, handler)
	allObjs := c.listLocked()
	for _, obj := range allObjs {
		c.eventQueue.WriteOne(func() {
			handler.OnAdd(obj, true)
		})
	}

	if c.handlerRegistration == nil {
		// No informer, so immediately synced.
		return syncedHandlerRegistration{}
	}

	return c.handlerRegistration
}

// emitEvents delivers all pending events that are in the queue, in the order
// in which they were stored there (FIFO).
func (c *AssumeCache) emitEvents() {
	for {
		c.rwMutex.Lock()
		deliver, ok := c.eventQueue.ReadOne()
		c.rwMutex.Unlock()

		if !ok {
			return
		}
		func() {
			defer utilruntime.HandleCrash()
			deliver()
		}()
	}
}

// syncedHandlerRegistration is an implementation of ResourceEventHandlerRegistration
// which always returns true.
type syncedHandlerRegistration struct{}

func (syncedHandlerRegistration) HasSynced() bool { return true }
