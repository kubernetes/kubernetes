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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
)

// PVAssumeCache is a AssumeCache for PersistentVolume objects
type PVAssumeCache struct {
	*passiveAssumeCache[*v1.PersistentVolume]
}

func pvStorageClassIndexFunc(obj interface{}) ([]string, error) {
	if pv, ok := obj.(*v1.PersistentVolume); ok {
		return []string{storagehelpers.GetPersistentVolumeClass(pv)}, nil
	}
	return nil, fmt.Errorf("object is not a v1.PersistentVolume: %v", obj)
}

const storageClassIndex = "storageclass"

// NewPVAssumeCache creates a PV assume cache.
func NewPVAssumeCache(logger klog.Logger, informer informer) (PVAssumeCache, error) {
	logger = klog.LoggerWithName(logger, "pv-cache")
	err := informer.GetIndexer().AddIndexers(map[string]cache.IndexFunc{
		storageClassIndex: pvStorageClassIndexFunc,
	})
	if err != nil {
		// Ignore the error if the index already exists. This can happen if
		// the same informer is shared among multiple PVAssumeCache, maybe created from multiple profiles.
		if informer.GetIndexer().GetIndexers()[storageClassIndex] == nil {
			return PVAssumeCache{}, err
		}
	}
	cache, err := newAssumeCache[*v1.PersistentVolume](logger, informer, schema.GroupResource{Resource: "persistentvolumes"})
	return PVAssumeCache{cache}, err
}

func (c PVAssumeCache) ListPVs(storageClassName string) ([]*v1.PersistentVolume, error) {
	// This works because we will never change the storage class in scheduler
	// Assumed PVs needs to be included here to ensure the same PVC will not be bound to another PV in the next scheduling cycle.
	return c.ByIndex(storageClassIndex, storageClassName)
}

// PVCAssumeCache is a AssumeCache for PersistentVolumeClaim objects
type PVCAssumeCache struct {
	*passiveAssumeCache[*v1.PersistentVolumeClaim]
}

// NewPVCAssumeCache creates a PVC assume cache.
func NewPVCAssumeCache(logger klog.Logger, informer informer) (PVCAssumeCache, error) {
	logger = klog.LoggerWithName(logger, "pvc-cache")
	cache, err := newAssumeCache[*v1.PersistentVolumeClaim](logger, informer, schema.GroupResource{Resource: "persistentvolumeclaims"})
	return PVCAssumeCache{cache}, err
}
