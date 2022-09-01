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

package cache

import (
	"fmt"
	"strconv"
	"sync"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// AssumeCacheInformer is the subset of a SharedInformer required by NewAssumeCache.
type AssumeCacheInformer interface {
	// AddEventHandler adds an event handler to the shared informer using the shared informer's resync
	// period. Events to a single handler are delivered sequentially, but there is no coordination
	// between different handlers.
	AddEventHandler(handler ResourceEventHandler) (ResourceEventHandlerRegistration, error)
}

// AssumeCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object.  Objects are assumed to be
// Kubernetes API objects that implement meta.Interface.
//
// Methods with log output support contextual logging by taking an explicit
// logger parameter. The log level at which info messages get emitted
// is 4 or higher.
type AssumeCache[T any] interface {
	// Assume updates the object in-memory only
	Assume(logger klog.Logger, obj T) error

	// Restore the informer cache's version of the object
	Restore(logger klog.Logger, objName string)

	// Get the object by name. apierrors.IsNotFound will return true
	// for the error if no such object exists.
	Get(objName string) (T, error)

	// Get the API object by name. apierrors.IsNotFound will return true
	// for the error if no such object exists.
	GetAPIObj(objName string) (T, error)

	// List all the objects in the cache
	List(indexObj T) []T

	// Event handlers get invoked when the AssumeCache is called by the
	// underlying informer after the AssumeCache has updated its own state.
	// This allows work queues to react to changes in the apiserver without
	// racing with the AssumeCache updating its state.
	//
	// Assume and Restore do not invoke event handlers.
	AddEventHandler(handler ResourceEventHandler)
}

// Typecast does a checked typecast into some target type.
// It returns the result plus an error if the cast did not
// succeed. The error describes the intended type and
// the actual object (including its content).
func Typecast[T any](obj interface{}) (t T, err error) {
	ok := false
	t, ok = obj.(T)
	if !ok {
		err = &errWrongType{out: t, in: obj}
	}
	return
}

type errWrongType struct {
	in, out interface{}
}

func (e *errWrongType) Error() string {
	return fmt.Sprintf("could not convert object to type %T: %+v", e.out, e.in)
}

type errObjectName struct {
	detailedErr error
}

func (e *errObjectName) Error() string {
	return fmt.Sprintf("failed to get object name: %v", e.detailedErr)
}

// assumeCache stores two pointers to represent a single object:
//   - The pointer to the informer object.
//   - The pointer to the latest object, which could be the same as
//     the informer object, or an in-memory object.
//
// An informer update always overrides the latest object pointer.
//
// Assume() only updates the latest object pointer.
// Restore() sets the latest object pointer back to the informer object.
// Get/List() always returns the latest object pointer.
type assumeCache[T any] struct {
	logger klog.Logger

	// Synchronizes updates to store and eventHandlers
	rwMutex sync.RWMutex

	// describes the object stored
	description string

	// Stores objInfo pointers
	store Indexer

	// Index function for object
	indexFunc IndexFunc
	indexName string

	eventHandlers []ResourceEventHandler
}

type objInfo struct {
	// name of the object
	name string

	// Latest version of object could be cached-only or from informer
	latestObj interface{}

	// Latest object from informer
	apiObj interface{}
}

func objInfoKeyFunc[T any](obj interface{}) (string, error) {
	objInfo, err := Typecast[*objInfo](obj)
	if err != nil {
		return "", err
	}
	return objInfo.name, nil
}

func (c *assumeCache[T]) objInfoIndexFunc(obj interface{}) ([]string, error) {
	objInfo, err := Typecast[*objInfo](obj)
	if err != nil {
		// []string{""} is from the original implementation of the assumeCache, but seems odd.
		return []string{""}, err
	}
	return c.indexFunc(objInfo.latestObj)
}

// NewAssumeCache creates an assume cache for general objects.
// The logger is used for callbacks from the informer.
func NewAssumeCache[T any](logger klog.Logger, informer AssumeCacheInformer, description, indexName string, indexFunc IndexFunc) AssumeCache[T] {
	c := &assumeCache[T]{
		logger:      logger,
		description: description,
		indexFunc:   indexFunc,
		indexName:   indexName,
	}
	indexers := Indexers{}
	if indexName != "" && indexFunc != nil {
		indexers[indexName] = c.objInfoIndexFunc
	}
	c.store = NewIndexer(objInfoKeyFunc[T], indexers)

	// Unit tests don't use informers
	if informer != nil {
		informer.AddEventHandler(
			ResourceEventHandlerFuncs{
				AddFunc:    c.add,
				UpdateFunc: c.update,
				DeleteFunc: c.delete,
			},
		)
	}
	return c
}

func (c *assumeCache[T]) AddEventHandler(handler ResourceEventHandler) {
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	c.eventHandlers = append(c.eventHandlers, handler)
}

func (c *assumeCache[T]) add(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := MetaNamespaceKeyFunc(obj)
	if err != nil {
		c.logger.Error(&errObjectName{err}, "Add failed")
		return
	}

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnAdd(obj)
	}
	defer c.rwMutex.Unlock()

	if objInfo := c.getObjInfoNoError(name); objInfo != nil {
		newVersion, err := c.getObjVersion(name, obj)
		if err != nil {
			c.logger.Error(err, "Add failed: couldn't get object version")
			return
		}

		storedVersion, err := c.getObjVersion(name, objInfo.latestObj)
		if err != nil {
			c.logger.Error(err, "Add failed: couldn't get stored object version")
			return
		}

		// Only update object if version is newer.
		// This is so we don't override assumed objects due to informer resync.
		if newVersion <= storedVersion {
			c.logger.V(10).Info("Skip adding object to assume cache because version is not newer than storedVersion", "description", c.description, "cacheKey", name, "newVersion", newVersion, "storedVersion", storedVersion)
			return
		}
	}

	objInfo := &objInfo{name: name, latestObj: obj, apiObj: obj}
	if err = c.store.Update(objInfo); err != nil {
		c.logger.Info("Error occurred while updating stored object", "err", err)
	} else {
		c.logger.V(10).Info("Adding object to assume cache", "description", c.description, "cacheKey", name, "assumeCache", obj)
	}
}

func (c *assumeCache[T]) update(oldObj interface{}, newObj interface{}) {
	if newObj == nil {
		return
	}

	name, err := MetaNamespaceKeyFunc(newObj)
	if err != nil {
		c.logger.Error(&errObjectName{err}, "Update failed")
		return
	}

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnUpdate(oldObj, newObj)
	}
	defer c.rwMutex.Unlock()

	if objInfo, _ := c.getObjInfo(name); objInfo != nil {
		newVersion, err := c.getObjVersion(name, newObj)
		if err != nil {
			klog.ErrorS(err, "Add failed: couldn't get object version")
			return
		}

		storedVersion, err := c.getObjVersion(name, objInfo.latestObj)
		if err != nil {
			klog.ErrorS(err, "Add failed: couldn't get stored object version")
			return
		}

		// Only update object if version is newer.
		// This is so we don't override assumed objects due to informer resync.
		if newVersion <= storedVersion {
			c.logger.V(10).Info("Skip adding object to assume cache because version is not newer than storedVersion", "description", c.description, "cacheKey", name, "newVersion", newVersion, "storedVersion", storedVersion)
			return
		}
	}

	objInfo := &objInfo{name: name, latestObj: newObj, apiObj: newObj}
	if err = c.store.Update(objInfo); err != nil {
		c.logger.Info("Error occurred while updating stored object", "err", err)
	} else {
		c.logger.V(10).Info("Adding object to assume cache", "description", c.description, "cacheKey", name, "assumeCache", newObj)
	}
}

func (c *assumeCache[T]) delete(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		c.logger.Error(&errObjectName{err}, "Delete failed")
		return
	}

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnDelete(obj)
	}
	defer c.rwMutex.Unlock()

	objInfo := &objInfo{name: name}
	err = c.store.Delete(objInfo)
	if err != nil {
		c.logger.Error(err, "Failed to delete", "description", c.description, "cacheKey", name)
	}
}

func (c *assumeCache[T]) getObjVersion(name string, obj interface{}) (int64, error) {
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return -1, err
	}

	objResourceVersion, err := strconv.ParseInt(objAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return -1, fmt.Errorf("error parsing ResourceVersion %q for %v %q: %s", objAccessor.GetResourceVersion(), c.description, name, err)
	}
	return objResourceVersion, nil
}

func (c *assumeCache[T]) getObjInfo(name string) (*objInfo, error) {
	obj, ok, err := c.store.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: c.description}, name)
	}

	return Typecast[*objInfo](obj)
}

// getObjInfoNoError avoids allocating an error that add above would
// just discard again.
func (c *assumeCache[T]) getObjInfoNoError(name string) *objInfo {
	obj, ok, err := c.store.GetByKey(name)
	if err != nil || !ok {
		return nil
	}
	if info, err := Typecast[*objInfo](obj); err == nil {
		return info
	}
	return nil
}

func (c *assumeCache[T]) Get(objName string) (obj T, finalErr error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		finalErr = err
		return
	}
	return Typecast[T](objInfo.latestObj)
}

func (c *assumeCache[T]) GetAPIObj(objName string) (obj T, finalErr error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		finalErr = err
		return
	}
	return Typecast[T](objInfo.apiObj)
}

func (c *assumeCache[T]) List(indexObj T) []T {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	allObjs := []T{}
	objs, err := c.store.Index(c.indexName, &objInfo{latestObj: indexObj})
	if err != nil {
		c.logger.Error(err, "List index error")
		return nil
	}

	for _, obj := range objs {
		objInfo, err := Typecast[*objInfo](obj)
		if err != nil {
			c.logger.Error(err, "List error")
			continue
		}
		t, err := Typecast[T](objInfo.latestObj)
		if err != nil {
			c.logger.Error(err, "List error")
			continue
		}
		allObjs = append(allObjs, t)
	}
	return allObjs
}

func (c *assumeCache[T]) Assume(logger klog.Logger, obj T) error {
	name, err := MetaNamespaceKeyFunc(obj)
	if err != nil {
		return &errObjectName{err}
	}

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

	// Only update the cached object
	objInfo.latestObj = obj
	logger.V(4).Info("Assumed object", "description", c.description, "cacheKey", name, "version", newVersion)
	return nil
}

func (c *assumeCache[T]) Restore(logger klog.Logger, objName string) {
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		// This could be expected if object got deleted
		logger.V(5).Info("Restore object", "description", c.description, "cacheKey", objName, "err", err)
	} else {
		objInfo.latestObj = objInfo.apiObj
		logger.V(4).Info("Restored object", "description", c.description, "cacheKey", objName)
	}
}
