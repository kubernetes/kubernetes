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

package volumebinding

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// PVAssumeCache is a AssumeCache for PersistentVolume objects
type PVAssumeCache struct {
	*assumecache.AssumeCache
	logger klog.Logger
}

func pvStorageClassIndexFunc(obj interface{}) ([]string, error) {
	if pv, ok := obj.(*v1.PersistentVolume); ok {
		return []string{storagehelpers.GetPersistentVolumeClass(pv)}, nil
	}
	return []string{""}, fmt.Errorf("object is not a v1.PersistentVolume: %v", obj)
}

// NewPVAssumeCache creates a PV assume cache.
func NewPVAssumeCache(logger klog.Logger, informer assumecache.Informer) *PVAssumeCache {
	logger = klog.LoggerWithName(logger, "PV Cache")
	return &PVAssumeCache{
		AssumeCache: assumecache.NewAssumeCache(logger, informer, "v1.PersistentVolume", "storageclass", pvStorageClassIndexFunc),
		logger:      logger,
	}
}

func (c *PVAssumeCache) GetPV(pvName string) (*v1.PersistentVolume, error) {
	obj, err := c.Get(pvName)
	if err != nil {
		return nil, err
	}

	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		return nil, &assumecache.WrongTypeError{TypeName: "v1.PersistentVolume", Object: obj}
	}
	return pv, nil
}

func (c *PVAssumeCache) GetAPIPV(pvName string) (*v1.PersistentVolume, error) {
	obj, err := c.GetAPIObj(pvName)
	if err != nil {
		return nil, err
	}
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		return nil, &assumecache.WrongTypeError{TypeName: "v1.PersistentVolume", Object: obj}
	}
	return pv, nil
}

func (c *PVAssumeCache) ListPVs(storageClassName string) []*v1.PersistentVolume {
	objs := c.List(&v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: storageClassName,
		},
	})
	pvs := []*v1.PersistentVolume{}
	for _, obj := range objs {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			c.logger.Error(&assumecache.WrongTypeError{TypeName: "v1.PersistentVolume", Object: obj}, "ListPVs")
			continue
		}
		pvs = append(pvs, pv)
	}
	return pvs
}

// PVCAssumeCache is a AssumeCache for PersistentVolumeClaim objects
type PVCAssumeCache struct {
	*assumecache.AssumeCache
	logger klog.Logger
}

// NewPVCAssumeCache creates a PVC assume cache.
func NewPVCAssumeCache(logger klog.Logger, informer assumecache.Informer) *PVCAssumeCache {
	logger = klog.LoggerWithName(logger, "PVC Cache")
	return &PVCAssumeCache{
		AssumeCache: assumecache.NewAssumeCache(logger, informer, "v1.PersistentVolumeClaim", "", nil),
		logger:      logger,
	}
}

func (c *PVCAssumeCache) GetPVC(pvcKey string) (*v1.PersistentVolumeClaim, error) {
	obj, err := c.Get(pvcKey)
	if err != nil {
		return nil, err
	}

	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, &assumecache.WrongTypeError{TypeName: "v1.PersistentVolumeClaim", Object: obj}
	}
	return pvc, nil
}

func (c *PVCAssumeCache) GetAPIPVC(pvcKey string) (*v1.PersistentVolumeClaim, error) {
	obj, err := c.GetAPIObj(pvcKey)
	if err != nil {
		return nil, err
	}
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, &assumecache.WrongTypeError{TypeName: "v1.PersistentVolumeClaim", Object: obj}
	}
	return pvc, nil
}
