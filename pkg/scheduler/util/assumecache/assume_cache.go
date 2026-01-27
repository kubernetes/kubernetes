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

	"k8s.io/klog/v2"
	"k8s.io/utils/buffer"

	"k8s.io/apimachinery/pkg/api/meta"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
)

// Informer is the subset of [cache.SharedInformer] that NewAssumeCache depends upon.
type Informer interface {
	AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error)
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

type ObjectNameError struct {
	DetailedErr error
}

func (e ObjectNameError) Error() string {
	return fmt.Sprintf("failed to get object name: %v", e.DetailedErr)
}

func (e ObjectNameError) Is(err error) bool {
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

	// cond is used by emitEvents.
	cond *sync.Cond

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

	// emittingEvents is true while one emitEvents call is actively emitting events.
	emittingEvents bool

	// describes the object stored
	description string

	// Stores objInfo pointers
	store cache.Indexer

	// Index function for object
	indexFunc cache.IndexFunc
	indexName string
}

type objInfo struct {
	// name of the object
	name string

	// Latest version of object could be cached-only or from informer
	latestObj interface{}

	// Latest object from informer
	apiObj interface{}
}

func objInfoKeyFunc(obj interface{}) (string, error) {
	objInfo, ok := obj.(*objInfo)
	if !ok {
		return "", &WrongTypeError{TypeName: "objInfo", Object: obj}
	}
	return objInfo.name, nil
}

func (c *AssumeCache) objInfoIndexFunc(obj interface{}) ([]string, error) {
	objInfo, ok := obj.(*objInfo)
	if !ok {
		return []string{""}, &WrongTypeError{TypeName: "objInfo", Object: obj}
	}
	return c.indexFunc(objInfo.latestObj)
}

// NewAssumeCache creates an assume cache for general objects.
func NewAssumeCache(logger klog.Logger, informer Informer, description, indexName string, indexFunc cache.IndexFunc) *AssumeCache {
	c := &AssumeCache{
		logger:      logger,
		description: description,
		indexFunc:   indexFunc,
		indexName:   indexName,
		eventQueue:  *buffer.NewRing[func()](buffer.RingOptions{InitialSize: 0, NormalSize: 4}),
	}
	c.cond = sync.NewCond(&c.rwMutex)
	indexers := cache.Indexers{}
	if indexName != "" && indexFunc != nil {
		indexers[indexName] = c.objInfoIndexFunc
	}
	c.store = cache.NewIndexer(objInfoKeyFunc, indexers)

	// Unit tests don't use informers
	if informer != nil {
		// Cannot fail in practice?! No-one bothers checking the error.
		c.handlerRegistration, _ = informer.AddEventHandler(
			cache.ResourceEventHandlerFuncs{
				AddFunc:    c.add,
				UpdateFunc: c.update,
				DeleteFunc: c.delete,
			},
		)
	}
	return c
}

func (c *AssumeCache) add(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, &ObjectNameError{err}, "Add failed")
		return
	}

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	var oldObj interface{}
	if objInfo, _ := c.getObjInfo(name); objInfo != nil {
		newVersion, err := c.getObjVersion(name, obj)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "Add failed: couldn't get object version")
			return
		}

		storedVersion, err := c.getObjVersion(name, objInfo.latestObj)
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "Add failed: couldn't get stored object version")
			return
		}

		// Only update object if version is newer.
		// This is so we don't override assumed objects due to informer resync.
		if newVersion <= storedVersion {
			c.logger.V(10).Info("Skip adding object to assume cache because version is not newer than storedVersion", "description", c.description, "cacheKey", name, "newVersion", newVersion, "storedVersion", storedVersion)
			return
		}
		oldObj = objInfo.latestObj
	}

	objInfo := &objInfo{name: name, latestObj: obj, apiObj: obj}
	if err = c.store.Update(objInfo); err != nil {
		c.logger.Info("Error occurred while updating stored object", "err", err)
	} else {
		c.logger.V(10).Info("Adding object to assume cache", "description", c.description, "cacheKey", name, "assumeCache", obj)
		c.pushEvent(oldObj, obj)
	}
}

func (c *AssumeCache) update(oldObj interface{}, newObj interface{}) {
	c.add(newObj)
}

func (c *AssumeCache) delete(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, &ObjectNameError{err}, "Failed to delete")
		return
	}

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	var oldObj interface{}
	if len(c.eventHandlers) > 0 {
		if objInfo, _ := c.getObjInfo(name); objInfo != nil {
			oldObj = objInfo.latestObj
		}
	}

	objInfo := &objInfo{name: name}
	err = c.store.Delete(objInfo)
	if err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "Failed to delete", "description", c.description, "cacheKey", name)
	}

	c.pushEvent(oldObj, nil)
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

func (c *AssumeCache) getObjVersion(name string, obj interface{}) (int64, error) {
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return -1, err
	}

	objResourceVersion, err := strconv.ParseInt(objAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		//nolint:errorlint // Intentionally not wrapping the error, the underlying error is an implementation detail.
		return -1, fmt.Errorf("error parsing ResourceVersion %q for %v %q: %v", objAccessor.GetResourceVersion(), c.description, name, err)
	}
	return objResourceVersion, nil
}

func (c *AssumeCache) getObjInfo(key string) (*objInfo, error) {
	obj, ok, err := c.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, &NotFoundError{TypeName: c.description, ObjectKey: key}
	}

	objInfo, ok := obj.(*objInfo)
	if !ok {
		return nil, &WrongTypeError{"objInfo", obj}
	}
	return objInfo, nil
}

// Get the object by its key.
func (c *AssumeCache) Get(key string) (interface{}, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(key)
	if err != nil {
		return nil, err
	}
	return objInfo.latestObj, nil
}

// GetAPIObj gets the informer cache's version by its key.
func (c *AssumeCache) GetAPIObj(key string) (interface{}, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(key)
	if err != nil {
		return nil, err
	}
	return objInfo.apiObj, nil
}

// List all the objects in the cache.
func (c *AssumeCache) List(indexObj interface{}) []interface{} {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	return c.listLocked(indexObj)
}

func (c *AssumeCache) listLocked(indexObj interface{}) []interface{} {
	allObjs := []interface{}{}
	var objs []interface{}
	if c.indexName != "" {
		o, err := c.store.Index(c.indexName, &objInfo{latestObj: indexObj})
		if err != nil {
			utilruntime.HandleErrorWithLogger(c.logger, err, "List index error")
			return nil
		}
		objs = o
	} else {
		objs = c.store.List()
	}

	for _, obj := range objs {
		objInfo, ok := obj.(*objInfo)
		if !ok {
			utilruntime.HandleErrorWithLogger(c.logger, &WrongTypeError{TypeName: "objInfo", Object: obj}, "List error")
			continue
		}
		allObjs = append(allObjs, objInfo.latestObj)
	}
	return allObjs
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
func (c *AssumeCache) Assume(obj interface{}) error {
	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		return &ObjectNameError{err}
	}

	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo, err := c.getObjInfo(name)
	if err != nil {
		return err
	}

	newVersion, err := c.getObjVersion(name, obj)
	if err != nil {
		return err
	}

	storedVersion, err := c.getObjVersion(name, objInfo.latestObj)
	if err != nil {
		return err
	}

	if newVersion < storedVersion {
		return fmt.Errorf("%v %q is out of sync (stored: %d, assume: %d)", c.description, name, storedVersion, newVersion)
	}

	c.pushEvent(objInfo.latestObj, obj)

	// Only update the cached object
	objInfo.latestObj = obj
	c.logger.V(4).Info("Assumed object", "description", c.description, "cacheKey", name, "version", newVersion)
	return nil
}

// Restore the informer cache's version of the object.
func (c *AssumeCache) Restore(objName string) {
	defer c.emitEvents()
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		// This could be expected if object got deleted
		c.logger.V(5).Info("Restore object", "description", c.description, "cacheKey", objName, "err", err)
	} else {
		if objInfo.latestObj != objInfo.apiObj {
			c.pushEvent(objInfo.latestObj, objInfo.apiObj)
			objInfo.latestObj = objInfo.apiObj
		}
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
	allObjs := c.listLocked(nil)
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
// in which they were stored there (FIFO). Only one goroutine at a time is
// delivering events, to ensure correct order.
func (c *AssumeCache) emitEvents() {
	c.rwMutex.Lock()
	for c.emittingEvents {
		// Wait for the active caller of emitEvents to finish.
		// When it is done, it may or may not have drained
		// the events pushed by our caller.
		// We'll check below ourselves.
		c.cond.Wait()
	}
	c.emittingEvents = true
	c.rwMutex.Unlock()

	defer func() {
		c.rwMutex.Lock()
		c.emittingEvents = false
		// Hand over the batton to one other goroutine, if there is one.
		// We don't need to wake up more than one because only one of
		// them would be able to grab the "emittingEvents" responsibility.
		c.cond.Signal()
		c.rwMutex.Unlock()
	}()

	// When we get here, this instance of emitEvents is the active one.
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
