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

package stable

import (
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured/internal"
)

// groupedCounterSetsFromSlices returns, per pool, the counter sets on which an
// already-allocated device declares at least one compatibility group, read
// directly from the source ResourceSlices. A device that declares no groups
// contributes nothing. Used only by the version-skew skip while the
// DRADeviceCompatibilityGroups feature is disabled.
func groupedCounterSetsFromSlices(slices []*resourceapi.ResourceSlice, state AllocatedState) map[PoolID]sets.Set[string] {
	grouped := make(map[PoolID]sets.Set[string])
	for _, slice := range slices {
		for _, device := range slice.Spec.Devices {
			deviceID := internal.MakeDeviceID(slice.Spec.Driver, slice.Spec.Pool.Name, device.Name)
			if !internal.IsDeviceAllocated(deviceID, &state) {
				continue
			}
			for _, deviceCounterConsumption := range device.ConsumesCounters {
				if len(deviceCounterConsumption.CompatibilityGroups) == 0 {
					continue
				}
				poolID := PoolID{Driver: deviceID.Driver, Pool: deviceID.Pool}
				cs := grouped[poolID]
				if cs == nil {
					cs = sets.New[string]()
					grouped[poolID] = cs
				}
				cs.Insert(deviceCounterConsumption.CounterSet)
			}
		}
	}
	return grouped
}
