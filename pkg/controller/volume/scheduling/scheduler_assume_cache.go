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

package scheduling

import (
	"fmt"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"strconv"
	"sync"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/tools/cache"
)

// AssumeCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object.  Objects are assumed to be
// Kubernetes API objects that implement meta.Interface
type AssumeCache interface {
	// Assume updates the object in-memory only
	Assume(obj interface{}) error

	// Restore the informer cache's version of the object
	Restore(objName string)

	// Get the object by name
	Get(objName string) (interface{}, error)

	// Get the API object by name
	GetAPIObj(objName string) (interface{}, error)

	// List all the objects in the cache
	List(indexObj interface{}) []interface{}
}

type errWrongType struct {
	typeName string
	object   interface{}
}

func (e *errWrongType) Error() string {
	return fmt.Sprintf("could not convert object to type %v: %+v", e.typeName, e.object)
}

type errNotFound struct {
	typeName   string
	objectName string
}

func (e *errNotFound) Error() string {
	return fmt.Sprintf("could not find %v %q", e.typeName, e.objectName)
}

type errObjectName struct {
	detailedErr error
}

func (e *errObjectName) Error() string {
	return fmt.Sprintf("failed to get object name: %v", e.detailedErr)
}

// assumeCache stores two pointers to represent a single object:
// * The pointer to the informer object.
// * The pointer to the latest object, which could be the same as
//   the informer object, or an in-memory object.
//
// An informer update always overrides the latest object pointer.
//
// Assume() only updates the latest object pointer.
// Restore() sets the latest object pointer back to the informer object.
// Get/List() always returns the latest object pointer.
type assumeCache struct {
	// Synchronizes updates to store
	rwMutex sync.RWMutex

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
		return "", &errWrongType{"objInfo", obj}
	}
	return objInfo.name, nil
}

func (c *assumeCache) objInfoIndexFunc(obj interface{}) ([]string, error) {
	objInfo, ok := obj.(*objInfo)
	if !ok {
		return []string{""}, &errWrongType{"objInfo", obj}
	}
	return c.indexFunc(objInfo.latestObj)
}

// NewAssumeCache creates an assume cache for general objects.
func NewAssumeCache(informer cache.SharedIndexInformer, description, indexName string, indexFunc cache.IndexFunc) AssumeCache {
	c := &assumeCache{
		description: description,
		indexFunc:   indexFunc,
		indexName:   indexName,
	}
	indexers := cache.Indexers{}
	if indexName != "" && indexFunc != nil {
		indexers[indexName] = c.objInfoIndexFunc
	}
	c.store = cache.NewIndexer(objInfoKeyFunc, indexers)

	// Unit tests don't use informers
	if informer != nil {
		informer.AddEventHandler(
			cache.ResourceEventHandlerFuncs{
				AddFunc:    c.add,
				UpdateFunc: c.update,
				DeleteFunc: c.delete,
			},
		)
	}
	return c
}

func (c *assumeCache) add(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		klog.Errorf("add failed: %v", &errObjectName{err})
		return
	}

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	if objInfo, _ := c.getObjInfo(name); objInfo != nil {
		newVersion, err := c.getObjVersion(name, obj)
		if err != nil {
			klog.Errorf("add: couldn't get object version: %v", err)
			return
		}

		storedVersion, err := c.getObjVersion(name, objInfo.latestObj)
		if err != nil {
			klog.Errorf("add: couldn't get stored object version: %v", err)
			return
		}

		// Only update object if version is newer.
		// This is so we don't override assumed objects due to informer resync.
		if newVersion <= storedVersion {
			klog.V(10).Infof("Skip adding %v %v to assume cache because version %v is not newer than %v", c.description, name, newVersion, storedVersion)
			return
		}
	}

	objInfo := &objInfo{name: name, latestObj: obj, apiObj: obj}
	if err = c.store.Update(objInfo); err != nil {
		klog.Warningf("got error when updating stored object : %v", err)
	} else {
		klog.V(10).Infof("Adding %v %v to assume cache: %+v ", c.description, name, obj)
	}
}

func (c *assumeCache) update(oldObj interface{}, newObj interface{}) {
	c.add(newObj)
}

func (c *assumeCache) delete(obj interface{}) {
	if obj == nil {
		return
	}

	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		klog.Errorf("delete failed: %v", &errObjectName{err})
		return
	}

	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo := &objInfo{name: name}
	err = c.store.Delete(objInfo)
	if err != nil {
		klog.Errorf("delete: failed to delete %v %v: %v", c.description, name, err)
	}
}

func (c *assumeCache) getObjVersion(name string, obj interface{}) (int64, error) {
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

func (c *assumeCache) getObjInfo(name string) (*objInfo, error) {
	obj, ok, err := c.store.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, &errNotFound{c.description, name}
	}

	objInfo, ok := obj.(*objInfo)
	if !ok {
		return nil, &errWrongType{"objInfo", obj}
	}
	return objInfo, nil
}

func (c *assumeCache) Get(objName string) (interface{}, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		return nil, err
	}
	return objInfo.latestObj, nil
}

func (c *assumeCache) GetAPIObj(objName string) (interface{}, error) {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		return nil, err
	}
	return objInfo.apiObj, nil
}

func (c *assumeCache) List(indexObj interface{}) []interface{} {
	c.rwMutex.RLock()
	defer c.rwMutex.RUnlock()

	allObjs := []interface{}{}
	objs, err := c.store.Index(c.indexName, &objInfo{latestObj: indexObj})
	if err != nil {
		klog.Errorf("list index error: %v", err)
		return nil
	}

	for _, obj := range objs {
		objInfo, ok := obj.(*objInfo)
		if !ok {
			klog.Errorf("list error: %v", &errWrongType{"objInfo", obj})
			continue
		}
		allObjs = append(allObjs, objInfo.latestObj)
	}
	return allObjs
}

func (c *assumeCache) Assume(obj interface{}) error {
	name, err := cache.MetaNamespaceKeyFunc(obj)
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
	klog.V(4).Infof("Assumed %v %q, version %v", c.description, name, newVersion)
	return nil
}

func (c *assumeCache) Restore(objName string) {
	c.rwMutex.Lock()
	defer c.rwMutex.Unlock()

	objInfo, err := c.getObjInfo(objName)
	if err != nil {
		// This could be expected if object got deleted
		klog.V(5).Infof("Restore %v %q warning: %v", c.description, objName, err)
	} else {
		objInfo.latestObj = objInfo.apiObj
		klog.V(4).Infof("Restored %v %q", c.description, objName)
	}
}

// PVAssumeCache is a AssumeCache for PersistentVolume objects
type PVAssumeCache interface {
	AssumeCache

	GetPV(pvName string) (*v1.PersistentVolume, error)
	GetAPIPV(pvName string) (*v1.PersistentVolume, error)
	ListPVs(storageClassName string) []*v1.PersistentVolume
}

type pvAssumeCache struct {
	AssumeCache
}

func pvStorageClassIndexFunc(obj interface{}) ([]string, error) {
	if pv, ok := obj.(*v1.PersistentVolume); ok {
		return []string{storagehelpers.GetPersistentVolumeClass(pv)}, nil
	}
	return []string{""}, fmt.Errorf("object is not a v1.PersistentVolume: %v", obj)
}

// NewPVAssumeCache creates a PV assume cache.
func NewPVAssumeCache(informer cache.SharedIndexInformer) PVAssumeCache {
	return &pvAssumeCache{NewAssumeCache(informer, "v1.PersistentVolume", "storageclass", pvStorageClassIndexFunc)}
}

func (c *pvAssumeCache) GetPV(pvName string) (*v1.PersistentVolume, error) {
	obj, err := c.Get(pvName)
	if err != nil {
		return nil, err
	}

	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		return nil, &errWrongType{"v1.PersistentVolume", obj}
	}
	return pv, nil
}

func (c *pvAssumeCache) GetAPIPV(pvName string) (*v1.PersistentVolume, error) {
	obj, err := c.GetAPIObj(pvName)
	if err != nil {
		return nil, err
	}
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		return nil, &errWrongType{"v1.PersistentVolume", obj}
	}
	return pv, nil
}

func (c *pvAssumeCache) ListPVs(storageClassName string) []*v1.PersistentVolume {
	objs := c.List(&v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: storageClassName,
		},
	})
	pvs := []*v1.PersistentVolume{}
	for _, obj := range objs {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			klog.Errorf("ListPVs: %v", &errWrongType{"v1.PersistentVolume", obj})
			continue
		}
		pvs = append(pvs, pv)
	}
	return pvs
}

// PVCAssumeCache is a AssumeCache for PersistentVolumeClaim objects
type PVCAssumeCache interface {
	AssumeCache

	// GetPVC returns the PVC from the cache with given pvcKey.
	// pvcKey is the result of MetaNamespaceKeyFunc on PVC obj
	GetPVC(pvcKey string) (*v1.PersistentVolumeClaim, error)
	GetAPIPVC(pvcKey string) (*v1.PersistentVolumeClaim, error)
}

type pvcAssumeCache struct {
	AssumeCache
}

// NewPVCAssumeCache creates a PVC assume cache.
func NewPVCAssumeCache(informer cache.SharedIndexInformer) PVCAssumeCache {
	return &pvcAssumeCache{NewAssumeCache(informer, "v1.PersistentVolumeClaim", "", nil)}
}

func (c *pvcAssumeCache) GetPVC(pvcKey string) (*v1.PersistentVolumeClaim, error) {
	obj, err := c.Get(pvcKey)
	if err != nil {
		return nil, err
	}

	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, &errWrongType{"v1.PersistentVolumeClaim", obj}
	}
	return pvc, nil
}

func (c *pvcAssumeCache) GetAPIPVC(pvcKey string) (*v1.PersistentVolumeClaim, error) {
	obj, err := c.GetAPIObj(pvcKey)
	if err != nil {
		return nil, err
	}
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, &errWrongType{"v1.PersistentVolumeClaim", obj}
	}
	return pvc, nil
}
