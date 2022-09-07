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
	"sync"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
// objects outside of informer events and also restoring the informer cache's
// version of the object. Objects must be Kubernetes API objects that implement
// metav1.Object.
//
// AssumeCache cannot guarantee that it always keeps the newest object because
// it cannot determine which two objects with different ResourceVersion is more
// revent. If in doubt, the object sent by the apiserver through the informer
// wins.
//
// Example where AssumeCache provides some benefit over a plain informer
// cache:
//   - AssumeCache receives object with version 1 via informer.
//   - User modifies that object, calls Update, receives object with
//     version 2 and stores that in the cache with Assume.
//   - AssumeCache.Get returns the object with version 2. An update
//     based on this version would succeed whereas an update base
//     on the object from the informer cache would fail.
//   - AssumeCache receives object with version 2 via informer ->
//     cache remains the same.
//
// Example where AssumeCache does not provide the newest object:
//   - AssumeCache receives object with version 1 from informer.
//   - Object gets replaced with version 2 in the apiserver.
//   - User modifies the local object with version 1, calls Patch,
//     receives object with version 3 and stores that in the cache
//     with Assume.
//   - AssumeCache receives object with version 2 and stores that
//     because it cannot determine that it is older than the assumed
//     object with version 3.
//   - AssumeCache.Get returns object with version 2 until the informer
//     receives version 3.
//
// Methods with log output support contextual logging by taking an explicit
// logger parameter. The log level at which info messages get emitted
// is 4 or higher.
type AssumeCache[T metav1.Object] interface {
	// Assume updates the object in-memory only. If the old object is
	// not the one that is currently in the cache, for example because
	// an update was received through the informer, then an error is
	// returned instead of replacing the potentially newer object
	// in the cache.
	Assume(logger klog.Logger, oldObj, newObj T) error

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
type assumeCache[T metav1.Object] struct {
	logger klog.Logger

	// Synchronizes updates to store and eventHandlers
	rwMutex sync.RWMutex

	// describes the object stored
	description string

	// Stores objInfo pointers
	store Indexer

	// Index function for object
	indexFunc func(obj T) ([]string, error)
	indexName string

	eventHandlers []ResourceEventHandler
}

type objInfo[T any] struct {
	// name of the object
	name string

	// Latest version of object could be cached-only or from informer
	latestObj T

	// Latest object from informer
	apiObj T
}

func objInfoKeyFunc[T any](obj interface{}) (string, error) {
	objInfo, err := Typecast[*objInfo[T]](obj)
	if err != nil {
		return "", err
	}
	return objInfo.name, nil
}

func (c *assumeCache[T]) objInfoIndexFunc(obj interface{}) ([]string, error) {
	objInfo, err := Typecast[*objInfo[T]](obj)
	if err != nil {
		// []string{""} is from the original implementation of the assumeCache, but seems odd.
		return []string{""}, err
	}
	return c.indexFunc(objInfo.latestObj)
}

// NewAssumeCache creates an assume cache for general objects.
// The logger is used for callbacks from the informer.
func NewAssumeCache[T metav1.Object](logger klog.Logger, informer AssumeCacheInformer, description, indexName string, indexFunc func(obj T) ([]string, error)) AssumeCache[T] {
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

	objT, err := Typecast[T](obj)
	if err != nil {
		c.logger.Error(err, "Add failed")
		return
	}

	name := getObjName(objT)

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnAdd(obj)
	}
	defer c.rwMutex.Unlock()

	// The informer always wins, replacing whatever was stored before.
	objInfo := &objInfo[T]{name: name, latestObj: objT, apiObj: objT}
	if err = c.store.Update(objInfo); err != nil {
		c.logger.Info("Error occurred while updating stored object", "err", err)
	} else {
		c.logger.V(10).Info("Adding object to assume cache", "description", c.description, "cacheKey", name, "assumeCache", objT)
	}
}

func (c *assumeCache[T]) update(oldObj interface{}, newObj interface{}) {
	if newObj == nil {
		return
	}

	objT, err := Typecast[T](newObj)
	if err != nil {
		c.logger.Error(err, "Update failed")
		return
	}

	name := getObjName(objT)

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnUpdate(oldObj, newObj)
	}
	defer c.rwMutex.Unlock()

	// The informer always wins, replacing whatever was stored before.
	objInfo := &objInfo[T]{name: name, latestObj: objT, apiObj: objT}
	if err = c.store.Update(objInfo); err != nil {
		c.logger.Info("Error occurred while updating stored object", "err", err)
	} else {
		c.logger.V(10).Info("Adding object to assume cache", "description", c.description, "cacheKey", name, "assumeCache", objT)
	}
}

func (c *assumeCache[T]) delete(obj interface{}) {
	if obj == nil {
		return
	}

	if d, ok := obj.(DeletedFinalStateUnknown); ok {
		obj = d.Obj
	}
	objT, err := Typecast[T](obj)
	if err != nil {
		c.logger.Error(err, "Delete failed")
		return
	}

	name := getObjName(objT)

	c.rwMutex.Lock()
	for _, handler := range c.eventHandlers {
		// This will get called after unlocking the mutex.
		defer handler.OnDelete(obj)
	}
	defer c.rwMutex.Unlock()

	objInfo := &objInfo[T]{name: name}
	err = c.store.Delete(objInfo)
	if err != nil {
		c.logger.Error(err, "Failed to delete", "description", c.description, "cacheKey", name)
	}
}

func getObjName(obj metav1.Object) string {
	namespace := obj.GetNamespace()
	name := obj.GetName()
	if namespace != "" {
		return namespace + "/" + name
	}
	return name
}

func (c *assumeCache[T]) getObjInfo(name string) (*objInfo[T], error) {
	obj, ok, err := c.store.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: c.description}, name)
	}

	return Typecast[*objInfo[T]](obj)
}

func (c *assumeCache[T]) Get(objName string) (obj T, finalErr error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		finalErr = err
		return
	}
	return objInfo.latestObj, nil
}

func (c *assumeCache[T]) GetAPIObj(objName string) (obj T, finalErr error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		finalErr = err
		return
	}
	return objInfo.apiObj, nil
}

func (c *assumeCache[T]) List(indexObj T) []T {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	allObjs := []T{}
	objs, err := c.store.Index(c.indexName, &objInfo[T]{latestObj: indexObj})
	if err != nil {
		c.logger.Error(err, "List index error")
		return nil
	}

	for _, obj := range objs {
		objInfo, err := Typecast[*objInfo[T]](obj)
		if err != nil {
			c.logger.Error(err, "List error")
			continue
		}
		allObjs = append(allObjs, objInfo.latestObj)
	}
	return allObjs
}

func (c *assumeCache[T]) Assume(logger klog.Logger, oldObj, newObj T) error {
	name, err := MetaNamespaceKeyFunc(newObj)
	if err != nil {
		return &errObjectName{err}
	}

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo, err := c.getObjInfo(name)
	if err != nil {
		return err
	}

	oldVersion := oldObj.GetResourceVersion()
	storedVersion := objInfo.latestObj.GetResourceVersion()
	newVersion := newObj.GetResourceVersion()

	// It's not an error to store an object that
	// has the same version as the stored one. We
	// could just return in that case and do nothing,
	// but traditionally the AssumeCache has stored
	// the object, so we continue to do that below.
	if oldVersion != storedVersion && newVersion != storedVersion {
		return fmt.Errorf("%v %q is out of sync (stored: %s, original: %s, new: %s)", c.description, name, storedVersion, oldVersion, newVersion)
	}

	// Only update the cached object
	objInfo.latestObj = newObj
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
