/*
Copyright 2024 The Kubernetes Authors.

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

package dynamicresources

import (
	"sync"

	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/ptr"
)

// foreachAllocatedDevice invokes the provided callback for each
// device in the claim's allocation result which was allocated
// exclusively for the claim.
//
// Devices allocated with admin access can be shared with other
// claims and are skipped without invoking the callback.
//
// foreachAllocatedDevice does nothing if the claim is not allocated.
func foreachAllocatedDevice(claim *resourceapi.ResourceClaim, cb func(deviceID structured.DeviceID)) {
	if claim.Status.Allocation == nil {
		return
	}
	for _, result := range claim.Status.Allocation.Devices.Results {
		// Kubernetes 1.31 did not set this, 1.32 always does.
		// Supporting 1.31 is not worth the additional code that
		// would have to be written (= looking up in request) because
		// it is extremely unlikely that there really is a result
		// that still exists in a cluster from 1.31 where this matters.
		if ptr.Deref(result.AdminAccess, false) {
			// Is not considered as allocated.
			continue
		}
		deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)

		// None of the users of this helper need to abort iterating,
		// therefore it's not supported as it only would add overhead.
		cb(deviceID)
	}
}

// allocatedDevices reacts to events in a cache and maintains a set of all allocated devices.
// This is cheaper than repeatedly calling List, making strings unique, and building the set
// each time PreFilter is called.
//
// All methods are thread-safe. Get returns a cloned set.
type allocatedDevices struct {
	logger klog.Logger

	mutex sync.RWMutex
	ids   sets.Set[structured.DeviceID]
}

func newAllocatedDevices(logger klog.Logger) *allocatedDevices {
	return &allocatedDevices{
		logger: logger,
		ids:    sets.New[structured.DeviceID](),
	}
}

func (a *allocatedDevices) Get() sets.Set[structured.DeviceID] {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	return a.ids.Clone()
}

func (a *allocatedDevices) handlers() cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		AddFunc:    a.onAdd,
		UpdateFunc: a.onUpdate,
		DeleteFunc: a.onDelete,
	}
}

func (a *allocatedDevices) onAdd(obj any) {
	claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onAdd")
		return
	}

	if claim.Status.Allocation != nil {
		a.addDevices(claim)
	}
}

func (a *allocatedDevices) onUpdate(oldObj, newObj any) {
	originalClaim, modifiedClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onUpdate")
		return
	}

	switch {
	case originalClaim.Status.Allocation == nil && modifiedClaim.Status.Allocation != nil:
		a.addDevices(modifiedClaim)
	case originalClaim.Status.Allocation != nil && modifiedClaim.Status.Allocation == nil:
		a.removeDevices(originalClaim)
	default:
		// Nothing to do. Either both nil or both non-nil, in which case the content
		// also must be the same (immutable!).
	}
}

func (a *allocatedDevices) onDelete(obj any) {
	claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onDelete")
		return
	}

	a.removeDevices(claim)
}

func (a *allocatedDevices) addDevices(claim *resourceapi.ResourceClaim) {
	if claim.Status.Allocation == nil {
		return
	}
	// Locking of the mutex gets minimized by pre-computing what needs to be done
	// without holding the lock.
	deviceIDs := make([]structured.DeviceID, 0, 20)
	foreachAllocatedDevice(claim, func(deviceID structured.DeviceID) {
		a.logger.V(6).Info("Observed device allocation", "device", deviceID, "claim", klog.KObj(claim))
		deviceIDs = append(deviceIDs, deviceID)
	})

	a.mutex.Lock()
	defer a.mutex.Unlock()
	for _, deviceID := range deviceIDs {
		a.ids.Insert(deviceID)
	}
}

func (a *allocatedDevices) removeDevices(claim *resourceapi.ResourceClaim) {
	if claim.Status.Allocation == nil {
		return
	}

	// Locking of the mutex gets minimized by pre-computing what needs to be done
	// without holding the lock.
	deviceIDs := make([]structured.DeviceID, 0, 20)
	foreachAllocatedDevice(claim, func(deviceID structured.DeviceID) {
		a.logger.V(6).Info("Observed device deallocation", "device", deviceID, "claim", klog.KObj(claim))
		deviceIDs = append(deviceIDs, deviceID)
	})

	a.mutex.Lock()
	defer a.mutex.Unlock()
	for _, deviceID := range deviceIDs {
		a.ids.Delete(deviceID)
	}
}
