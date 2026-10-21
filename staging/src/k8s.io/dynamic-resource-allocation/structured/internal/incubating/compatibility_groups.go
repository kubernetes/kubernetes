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
	resourceapi "k8s.io/api/resource/v1"
)

// sliceUsesCompatibilityGroups reports whether any device in the slice declares
// compatibility groups on one of its counter consumptions.
//
// While the DRADeviceCompatibilityGroups feature is disabled, GatherPools
// ignores such slices entirely: this allocator cannot validate co-allocation
// against compatibility groups, so those devices must not be allocated, and
// ignoring the slice makes the pool incomplete, which prevents allocating the
// pool's other devices, too. Whether any of the devices are already allocated
// is deliberately irrelevant, keeping the outcome independent of allocation
// order. This lets the feature be enabled later without deleting pods.
func sliceUsesCompatibilityGroups(slice *resourceapi.ResourceSlice) bool {
	for _, device := range slice.Spec.Devices {
		for _, deviceCounterConsumption := range device.ConsumesCounters {
			if len(deviceCounterConsumption.CompatibilityGroups) > 0 {
				return true
			}
		}
	}
	return false
}

// slicesWithoutCompatibilityGroups returns the slices which do not match
// sliceUsesCompatibilityGroups, preserving their order.
func slicesWithoutCompatibilityGroups(slices []*resourceapi.ResourceSlice) []*resourceapi.ResourceSlice {
	filtered := make([]*resourceapi.ResourceSlice, 0, len(slices))
	for _, slice := range slices {
		if sliceUsesCompatibilityGroups(slice) {
			continue
		}
		filtered = append(filtered, slice)
	}
	return filtered
}
