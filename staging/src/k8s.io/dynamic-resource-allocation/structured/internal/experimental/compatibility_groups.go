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

package experimental

import (
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/structured/internal"
)

// compatibilityGroupIntersection is the rolling state of the
// DRADeviceCompatibilityGroups constraint for a single counter set. It captures
// the intersection of the compatibilityGroups declared by every device placed
// on that counter set so far - both already-allocated peers (their groups read
// live from the slice) and devices being allocated in the current attempt.
//
// The KEP nil-semantics are encoded explicitly: a device that declares no
// groups is only co-allocatable with sibling devices that likewise declare no
// groups. That cannot be represented by an ordinary set intersection (the empty
// set would wrongly reject a second no-group device), so the "all members had
// no groups" state is tracked separately via allEmpty.
type compatibilityGroupIntersection struct {
	// hasMembers is false until the first device is placed on the counter set.
	// While false any candidate is admissible.
	hasMembers bool
	// allEmpty is true while every device placed so far declared no groups. In
	// that state only further no-group devices are admissible. Only meaningful
	// when hasMembers is true.
	allEmpty bool
	// groups is the running intersection of the (non-empty) group lists of the
	// devices placed so far. Only meaningful when hasMembers && !allEmpty.
	groups sets.Set[string]
}

// admits reports whether a device declaring the given groups may be co-allocated
// on the counter set, given the devices already folded into the intersection.
func (i compatibilityGroupIntersection) admits(groups sets.Set[string]) bool {
	if !i.hasMembers {
		return true
	}
	if groups.Len() == 0 {
		// A device with no groups only co-allocates with other no-group devices.
		return i.allEmpty
	}
	if i.allEmpty {
		// Existing members declared no groups; a grouped device is incompatible.
		return false
	}
	return i.groups.Intersection(groups).Len() > 0
}

// add folds a device's groups into the intersection and returns the updated
// value. It never mutates the receiver's set in place (it only ever assigns a
// freshly created set), so a value copy taken before add is a safe snapshot for
// backtracking. The caller is expected to have verified admits(groups) first;
// add stays defensive for the inconsistent states that slice mutation could
// otherwise produce.
func (i compatibilityGroupIntersection) add(groups sets.Set[string]) compatibilityGroupIntersection {
	if !i.hasMembers {
		i.hasMembers = true
		if groups.Len() == 0 {
			i.allEmpty = true
			i.groups = nil
		} else {
			i.allEmpty = false
			i.groups = groups.Clone()
		}
		return i
	}
	if i.allEmpty || groups.Len() == 0 {
		// Either existing members or the new device declared no groups; admits()
		// guarantees the other side matched, so the state is unchanged.
		return i
	}
	i.groups = i.groups.Intersection(groups)
	return i
}

// compatibilityGroupSet builds a set from a device's declared groups. A nil or
// empty list yields a nil set, which the intersection logic treats as "no
// groups".
func compatibilityGroupSet(groups []string) sets.Set[string] {
	if len(groups) == 0 {
		return nil
	}
	return sets.New[string](groups...)
}

// groupedCounterSetsForPool returns the counter sets of the pool on which an
// already-allocated device declares at least one compatibility group, read
// directly from the pool's ResourceSlices. A device that declares no groups
// contributes nothing. Membership is read live, exactly like
// checkAvailableCounters does for counters. Used only by the version-skew skip
// while the feature is disabled; enforcement (feature on) uses the richer
// compatibilityGroupsBaselineForPool instead.
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
