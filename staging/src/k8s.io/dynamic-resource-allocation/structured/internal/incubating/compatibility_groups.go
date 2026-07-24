/*
Copyright The Kubernetes Authors.

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

package incubating

import (
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/structured/internal"
)

// groupedCounterSetsForPool returns the counter sets of the pool on which an
// already-allocated device declares at least one compatibility group, read
// directly from the pool's ResourceSlices. A device that declares no groups
// contributes nothing. Used only by the version-skew skip while the
// DRADeviceCompatibilityGroups feature is disabled.
//
// The result is computed lazily the first time a pool is touched and then
// cached, mirroring availableCounters: pools whose devices never reach the
// skip - including every pool in a cluster that does not use compatibility
// groups - never pay for the walk.
func (alloc *allocator) groupedCounterSetsForPool(pool *Pool) sets.Set[draapi.UniqueString] {
	poolID := pool.PoolID

	alloc.mutex.RLock()
	grouped, found := alloc.groupedCounterSets[poolID]
	alloc.mutex.RUnlock()
	if found {
		return grouped
	}

	// Computed without holding the lock; concurrent goroutines may duplicate
	// the work, but the input is the same for all of them, so the result is,
	// too.
	grouped = sets.New[draapi.UniqueString]()
	for _, resourceSlices := range [][]*draapi.ResourceSlice{pool.DeviceSlicesTargetingNode, pool.DeviceSlicesNotTargetingNode} {
		for _, slice := range resourceSlices {
			for _, device := range slice.Spec.Devices {
				deviceID := DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
				if !internal.IsDeviceAllocated(deviceID, &alloc.allocatedState) {
					continue
				}
				for _, deviceCounterConsumption := range device.ConsumesCounters {
					if len(deviceCounterConsumption.CompatibilityGroups) > 0 {
						grouped.Insert(deviceCounterConsumption.CounterSet)
					}
				}
			}
		}
	}

	alloc.mutex.Lock()
	alloc.groupedCounterSets[poolID] = grouped
	alloc.mutex.Unlock()
	return grouped
}
