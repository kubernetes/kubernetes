/*
Copyright 2026 The Kubernetes Authors.

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
// on that counter set so far - both already-allocated peers (read from the
// claim-status snapshot) and devices being allocated in the current attempt.
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

// compatibilityGroupsBaselineForPool returns, per counter set in the pool, the
// compatibility-group intersection contributed by the devices that are already
// allocated (i.e. recorded in allocatedState). Counter-set membership is read
// from the slice, exactly like checkAvailableCounters does for counters, but the
// group values themselves come from the claim-status snapshot recorded in
// allocatedState.AllocatedCompatibilityGroups so that enforcement is unaffected
// by later mutations of the source ResourceSlice. A device that is allocated but
// has no snapshot entry for a counter set is treated as declaring no groups.
//
// The result is computed once per pool and cached, mirroring availableCounters.
func (alloc *allocator) compatibilityGroupsBaselineForPool(pool *Pool) map[string]compatibilityGroupIntersection {
	poolName := pool.PoolID.Pool

	alloc.mutex.RLock()
	baseline, found := alloc.compatibilityGroupsBaseline[poolName]
	alloc.mutex.RUnlock()
	if found {
		return baseline
	}

	baseline = make(map[string]compatibilityGroupIntersection)
	for _, resourceSlices := range [][]*draapi.ResourceSlice{pool.DeviceSlicesTargetingNode, pool.DeviceSlicesNotTargetingNode} {
		for _, slice := range resourceSlices {
			for _, device := range slice.Spec.Devices {
				deviceID := DeviceID{
					Driver: slice.Spec.Driver,
					Pool:   slice.Spec.Pool.Name,
					Device: device.Name,
				}
				if !internal.IsDeviceAllocated(deviceID, &alloc.allocatedState) {
					continue
				}
				snapshot := alloc.allocatedState.AllocatedCompatibilityGroups[deviceID]
				for _, deviceCounterConsumption := range device.ConsumesCounters {
					counterSetName := deviceCounterConsumption.CounterSet.String()
					cur := baseline[counterSetName]
					baseline[counterSetName] = cur.add(compatibilityGroupSet(snapshot[counterSetName]))
				}
			}
		}
	}

	alloc.mutex.Lock()
	alloc.compatibilityGroupsBaseline[poolName] = baseline
	alloc.mutex.Unlock()
	return baseline
}

// checkAndConsumeCompatibilityGroups verifies that the candidate device can be
// co-allocated, for every counter set it consumes, with the devices already
// placed on that counter set (allocated peers plus in-flight allocations of the
// current attempt). If so it folds the candidate's groups into the rolling
// intersection and returns a rollback function that restores the previous state
// when the allocator backtracks. If not, it leaves the running state unchanged
// and returns false.
func (alloc *allocator) checkAndConsumeCompatibilityGroups(device deviceWithID) (bool, func()) {
	poolName := device.pool.PoolID.Pool
	running, found := alloc.consumedCompatibilityGroups[poolName]
	if !found {
		running = make(map[string]compatibilityGroupIntersection)
		alloc.consumedCompatibilityGroups[poolName] = running
	}
	baseline := alloc.compatibilityGroupsBaselineForPool(device.pool)

	type savedEntry struct {
		counterSet string
		prev       compatibilityGroupIntersection
		existed    bool
	}
	var saved []savedEntry
	rollback := func() {
		// Restore in reverse order so repeated entries for the same counter set
		// (a device cannot have them, but be defensive) unwind correctly.
		for idx := len(saved) - 1; idx >= 0; idx-- {
			s := saved[idx]
			if s.existed {
				running[s.counterSet] = s.prev
			} else {
				delete(running, s.counterSet)
			}
		}
	}

	for _, deviceCounterConsumption := range device.ConsumesCounters {
		counterSetName := deviceCounterConsumption.CounterSet.String()
		cur, existed := running[counterSetName]
		if !existed {
			// Seed from the already-allocated peers for this counter set.
			cur = baseline[counterSetName]
		}
		groups := compatibilityGroupSet(deviceCounterConsumption.CompatibilityGroups)
		if !cur.admits(groups) {
			rollback()
			return false, nil
		}
		saved = append(saved, savedEntry{counterSet: counterSetName, prev: cur, existed: existed})
		running[counterSetName] = cur.add(groups)
	}

	return true, rollback
}

// skipForDisabledCompatibilityGroups reports whether a device must be skipped
// while the DRADeviceCompatibilityGroups feature is disabled in this scheduler
// but compatibility groups are still present in the cluster (served by the
// apiserver during a version skew). To avoid allocations it cannot validate, the
// scheduler ignores any device that declares compatibility groups, and any
// device drawing from a counter set on which an already-allocated device has
// declared groups. This lets the feature be enabled later without having to
// delete pods.
func (alloc *allocator) skipForDisabledCompatibilityGroups(device deviceWithID) bool {
	// The device itself declares groups (the common skip reason).
	for _, deviceCounterConsumption := range device.ConsumesCounters {
		if len(deviceCounterConsumption.CompatibilityGroups) > 0 {
			return true
		}
	}
	// Otherwise the device only needs to be skipped if it would join a counter
	// set on which an already-allocated device declared groups. If nothing in the
	// cluster declares groups there is nothing to protect, so the (cached) per-pool
	// baseline does not need to be computed at all.
	if len(alloc.allocatedState.AllocatedCompatibilityGroups) == 0 {
		return false
	}
	baseline := alloc.compatibilityGroupsBaselineForPool(device.pool)
	for _, deviceCounterConsumption := range device.ConsumesCounters {
		if cur, ok := baseline[deviceCounterConsumption.CounterSet.String()]; ok && cur.hasMembers && !cur.allEmpty {
			return true
		}
	}
	return false
}

// snapshotCompatibilityGroups returns the per-counter-set compatibility groups
// to record on the device's allocation result. Counter sets for which the device
// declares no groups are omitted, and the result is nil when the device declares
// no groups on any counter set (so the status field stays unset in the common
// case).
func snapshotCompatibilityGroups(device deviceWithID) map[string][]string {
	var snapshot map[string][]string
	for _, deviceCounterConsumption := range device.ConsumesCounters {
		if len(deviceCounterConsumption.CompatibilityGroups) == 0 {
			continue
		}
		if snapshot == nil {
			snapshot = make(map[string][]string)
		}
		groups := make([]string, len(deviceCounterConsumption.CompatibilityGroups))
		copy(groups, deviceCounterConsumption.CompatibilityGroups)
		snapshot[deviceCounterConsumption.CounterSet.String()] = groups
	}
	return snapshot
}
