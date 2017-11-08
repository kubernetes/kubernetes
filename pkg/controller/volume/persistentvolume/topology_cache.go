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

package persistentvolume

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/tools/cache"
)

// TmpCache is a cache on top of the informer that allows for updating
// objects outside of informer events and also restoring the informer
// cache's version of the object.  Objects are assumed to be
// Kubernetes API objects that implement meta.Interface
type TmpCache interface {
	// TmpUpdate updates the object in-memory only
	TmpUpdate(obj interface{}) error

	// Restore the informer cache's version of the object
	Restore(objName string)
}

type tmpCache struct {
	mutex sync.Mutex

	// describes the object stored
	description string

	// Stores objInfo pointers
	store cache.Store
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
		return "", fmt.Errorf("Could not convert object to type objInfo")
	}
	return objInfo.name, nil
}

func newTmpCache(informer cache.SharedIndexInformer, description string) *tmpCache {
	// TODO: index by storageclass
	c := &tmpCache{store: cache.NewStore(objInfoKeyFunc), description: description}

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

func (c *tmpCache) add(obj interface{}) {
	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		glog.Errorf("add: failed to get name: %v", err)
		return
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	objInfo := &objInfo{name: name, latestObj: obj, apiObj: obj}
	c.store.Update(objInfo)
}

func (c *tmpCache) update(oldObj interface{}, newObj interface{}) {
	c.add(newObj)
}

func (c *tmpCache) delete(obj interface{}) {
	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		glog.Errorf("delete: failed to get name: %v", err)
		return
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	objInfo := &objInfo{name: name}
	err = c.store.Delete(objInfo)
	if err != nil {
		glog.Errorf("delete: failed to delete %v %v: %v", c.description, name, err)
	}
}

func (c *tmpCache) getObjVersion(name string, obj interface{}) (int64, error) {
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return -1, err
	}

	objResourceVersion, err := strconv.ParseInt(objAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return -1, fmt.Errorf("Error parsing ResourceVersion %q for %v %q: %s", objAccessor.GetResourceVersion(), c.description, name, err)
	}
	return objResourceVersion, nil
}

func (c *tmpCache) getObjInfo(name string) *objInfo {
	obj, ok, err := c.store.GetByKey(name)
	if err != nil {
		glog.Errorf("failed to get objInfo for %v %q: %v", c.description, name, err)
		return nil
	}
	if !ok {
		return nil
	}

	objInfo, ok := obj.(*objInfo)
	if !ok {
		glog.Errorf("object is not of type objInfo for %v %q", c.description, name)
		return nil
	}
	return objInfo
}

func (c *tmpCache) get(objName string) interface{} {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if objInfo := c.getObjInfo(objName); objInfo != nil {
		return objInfo.latestObj
	}
	return nil
}

func (c *tmpCache) list() []interface{} {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	allObjs := []interface{}{}
	for _, obj := range c.store.List() {
		objInfo, ok := obj.(*objInfo)
		if !ok {
			glog.Errorf("List: object is not of type objInfo for %v", c.description)
			continue
		}
		allObjs = append(allObjs, objInfo.latestObj)
	}
	return allObjs
}

func (c *tmpCache) TmpUpdate(obj interface{}) error {
	name, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		return fmt.Errorf("TmpUpdate: failed to get name: %v", err)
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	objInfo := c.getObjInfo(name)
	if objInfo == nil {
		return fmt.Errorf("TmpUpdate: %v %q not found in cache", c.description, name)
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
		return fmt.Errorf("TmpUpdate: %v %q is out of sync", c.description, name)
	}

	// Only update the cached PV
	objInfo.latestObj = obj
	glog.V(4).Infof("TmpUpdate %v %q, version %v", c.description, name, newVersion)
	return nil
}

func (c *tmpCache) Restore(objName string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	glog.V(5).Infof("Entering restore %v %q", c.description, objName)

	if objInfo := c.getObjInfo(objName); objInfo != nil {
		objInfo.latestObj = objInfo.apiObj
		glog.V(4).Infof("Restore %v %q", c.description, objName)
	}
}

// PVTmpCache is a TmpCache for PersistentVolume objects
type PVTmpCache interface {
	TmpCache

	GetPV(pvName string) *v1.PersistentVolume
	ListPVs() []*v1.PersistentVolume
}

type pvTmpCache struct {
	*tmpCache
}

func newPVTmpCache(informer cache.SharedIndexInformer) PVTmpCache {
	return &pvTmpCache{tmpCache: newTmpCache(informer, "PV")}
}

func (c *pvTmpCache) GetPV(pvName string) *v1.PersistentVolume {
	obj := c.get(pvName)
	if obj == nil {
		return nil
	}

	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		glog.Errorf("GetPV: object %q is not of type v1.PersistentVolume", pvName)
	}
	return pv
}

func (c *pvTmpCache) ListPVs() []*v1.PersistentVolume {
	objs := c.list()
	pvs := []*v1.PersistentVolume{}
	for _, obj := range objs {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("ListPVs: object is not of type v1.PersistentVolume")
		}
		pvs = append(pvs, pv)
	}
	return pvs
}
