/*
Copyright 2025 The Kubernetes Authors.

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

package resourceclaimspec

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledFields removes disabled fields from the spec unless they were in
// use there before.
//
// Theoretically some features could be in use in an old ResourceClaim status
// and not in use in the spec. This can only occur in a spec update, which is
// currently prevented because the entire spec is immutable. Even if it was
// allowed, preventing adding disabled fields to the spec is the right thing to
// do regardless of what may have ended up in the status earlier.
func DropDisabledFields(new, old *resource.ResourceClaimSpec) {
	dropDisabledDRAPrioritizedListFields(new, old)
	dropDisabledDRADeviceTaintsFields(new, old) // Intentionally after dropDisabledDRAPrioritizedListFields to avoid iterating over FirstAvailable slice which needs to be dropped.
	dropDisabledDRAAdminAccessFields(new, old)
	dropDisabledDRAResourceClaimConsumableCapacityFields(new, old)
}

func dropDisabledDRADeviceTaintsFields(new, old *resource.ResourceClaimSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaints) ||
		draDeviceTaintsInUse(old) {
		return
	}

	for i, req := range new.Devices.Requests {
		if exactly := req.Exactly; exactly != nil {
			exactly.Tolerations = nil
		}
		for e := range req.FirstAvailable {
			new.Devices.Requests[i].FirstAvailable[e].Tolerations = nil
		}
	}
}

func draDeviceTaintsInUse(spec *resource.ResourceClaimSpec) bool {
	if spec == nil {
		return false
	}

	for _, req := range spec.Devices.Requests {
		if exactly := req.Exactly; exactly != nil && len(exactly.Tolerations) > 0 {
			return true
		}
		for _, sub := range req.FirstAvailable {
			if len(sub.Tolerations) > 0 {
				return true
			}
		}
	}

	return false
}

func dropDisabledDRAPrioritizedListFields(new, old *resource.ResourceClaimSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList) {
		return
	}
	if draPrioritizedListFeatureInUse(old) {
		return
	}

	for i := range new.Devices.Requests {
		new.Devices.Requests[i].FirstAvailable = nil
	}
}

func draPrioritizedListFeatureInUse(spec *resource.ResourceClaimSpec) bool {
	if spec == nil {
		return false
	}

	for _, request := range spec.Devices.Requests {
		if len(request.FirstAvailable) > 0 {
			return true
		}
	}

	return false
}

func dropDisabledDRAAdminAccessFields(new, old *resource.ResourceClaimSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess) ||
		DRAAdminAccessFeatureInUse(old) {
		// No need to drop anything.
		return
	}

	for i := range new.Devices.Requests {
		if new.Devices.Requests[i].Exactly != nil {
			new.Devices.Requests[i].Exactly.AdminAccess = nil
		}
	}
}

// DRAAdminAccessFeatureInUse checks whether the feature is in use in the spec.
func DRAAdminAccessFeatureInUse(spec *resource.ResourceClaimSpec) bool {
	if spec == nil {
		return false
	}

	for _, request := range spec.Devices.Requests {
		if request.Exactly != nil && request.Exactly.AdminAccess != nil {
			return true
		}
	}

	return false
}

func dropDisabledDRAResourceClaimConsumableCapacityFields(new, old *resource.ResourceClaimSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity) ||
		DRAConsumableCapacityFeatureInUse(old) {
		// No need to drop anything.
		return
	}

	for i := range new.Devices.Constraints {
		new.Devices.Constraints[i].DistinctAttribute = nil
	}

	for i := range new.Devices.Requests {
		if new.Devices.Requests[i].Exactly != nil {
			new.Devices.Requests[i].Exactly.Capacity = nil
		}
		request := new.Devices.Requests[i]
		for j := range request.FirstAvailable {
			new.Devices.Requests[i].FirstAvailable[j].Capacity = nil
		}
	}
}

// DRAConsumableCapacityFeatureInUse checks whether the feature is in use in the spec.
func DRAConsumableCapacityFeatureInUse(spec *resource.ResourceClaimSpec) bool {
	if spec == nil {
		return false
	}

	for _, constaint := range spec.Devices.Constraints {
		if constaint.DistinctAttribute != nil {
			return true
		}
	}

	for _, request := range spec.Devices.Requests {
		if request.Exactly != nil && request.Exactly.Capacity != nil {
			return true
		}
		for _, subRequest := range request.FirstAvailable {
			if subRequest.Capacity != nil {
				return true
			}
		}
	}

	return false
}
