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

package dynamicresources

import (
	"context"
	"iter"
	"maps"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/utils/ptr"
)

// This file implements PreFilterExtensions for DRA preemption (KEP-5690,
// gated by DRAPreemption), simulating device release when candidate victims
// are removed without mutating the shared DRAManager.
//
// DefaultPreemption simulates preemption (DryRunPreemption) on a clone of
// CycleState per candidate node. Read-only data from PreFilter lives on the
// shared stateData.preemption, while per-clone changes from RemovePod and
// AddPod live in a separate preemptionSimulationState CycleState entry that
// Clone copies.

var _ fwk.PreFilterExtensions = &DynamicResources{}

const preemptionSimulationStateKey fwk.StateKey = Name + "/simulation"

// preemptionData holds the read-only PreFilter inputs shared across the
// CycleState clones of preemption simulations. Only slices mutates after
// PreFilter, via its own synchronization.
type preemptionData struct {
	// canPreemptDevices reports whether preempting pods can free devices for
	// the pod being scheduled. It is false when the pod has no claims to
	// allocate, does not own all claims to allocate (see
	// ownsClaimsToAllocate), disables preemption, uses DRA-backed extended
	// resources (unsupported by KEP-5690), or belongs to a PodGroup (handled
	// by workload-aware preemption).
	canPreemptDevices bool
	// allocatedState and slices are the PreFilter inputs of state.allocator,
	// which can't be changed in place. allocatorForNode builds allocators
	// from them for nodes whose state the simulation changed. allocatedState
	// is nil when the pod has no claims to allocate.
	allocatedState *structured.AllocatedState
	slices         nodeSlices
}

func (pl *DynamicResources) newPreemptionData(pod *v1.Pod, claims *claimStore, allocatedState *structured.AllocatedState, resourceSlices []*draapi.ResourceSlice) *preemptionData {
	data := &preemptionData{
		allocatedState: allocatedState,
		canPreemptDevices: allocatedState != nil &&
			claims.extendedResourceClaim() == nil &&
			!pl.inPodGroup(pod) &&
			ptr.Deref(pod.Spec.PreemptionPolicy, v1.PreemptLowerPriority) != v1.PreemptNever &&
			ownsClaimsToAllocate(pod, claims),
	}
	data.slices.slices = resourceSlices
	return data
}

// ownsClaimsToAllocate reports whether pod owns every claim it needs to
// allocate (as with ResourceClaimTemplate-generated claims). Alpha only
// preempts for such pods; a claim referenced by name may be shared by
// pending pods that could each preempt for it.
func ownsClaimsToAllocate(pod *v1.Pod, claims *claimStore) bool {
	for _, claim := range claims.toAllocate() {
		if !metav1.IsControlledBy(claim, pod) {
			return false
		}
	}
	return true
}

// canReleaseOnNode reports whether all non-admin devices of allocation are
// local to nodeName and at least one of them is allocated exclusively, so that
// releasing allocation frees a device on the node. Alpha doesn't handle devices
// made available via a node selector or allNodes; they aren't local (see
// localDevicesByNode), so a claim with such a device isn't released.
func (data *preemptionData) canReleaseOnNode(allocation *resourceapi.AllocationResult, nodeName string) bool {
	localDevices := data.slices.localDevices(nodeName)
	found := false
	for _, result := range allocation.Devices.Results {
		// Admin-access allocations do not block normal allocations, so
		// releasing them frees nothing for the preemptor.
		if ptr.Deref(result.AdminAccess, false) {
			continue
		}
		if !localDevices.Has(structured.MakeDeviceID(result.Driver, result.Pool, result.Device)) {
			// Reject the whole claim so releasing it does not also free non-local devices.
			return false
		}
		if result.ShareID != nil {
			// Shares stay counted in alpha, so they free nothing.
			continue
		}
		found = true
	}
	return found
}

// allocatedStateForSlices returns PreFilter's AllocatedState with
// AllocatedDevices reduced to the devices of resourceSlices, avoiding a
// cluster-wide copy per simulated Filter call. Simulations only modify
// AllocatedDevices, because alpha only frees and holds exclusively allocated
// devices. All other fields are shared, which is safe because allocators only
// read them.
func (data *preemptionData) allocatedStateForSlices(resourceSlices []*draapi.ResourceSlice) *structured.AllocatedState {
	result := *data.allocatedState
	result.AllocatedDevices = sets.New[structured.DeviceID]()
	for _, slice := range resourceSlices {
		for _, device := range slice.Spec.Devices {
			id := structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
			if data.allocatedState.AllocatedDevices.Has(id) {
				result.AllocatedDevices.Insert(id)
			}
		}
	}
	return &result
}

// nodeSlices lazily indexes PreFilter's ResourceSlices by node on first use,
// avoiding the indexing cost when preemption never runs. Each index is built
// once and only read afterwards, so concurrent Filter calls need no locking.
type nodeSlices struct {
	slices []*draapi.ResourceSlice

	indexOnce sync.Once
	// byPool has all slices of each pool, of every generation. nodePools
	// has the pools with a slice for a node via spec.nodeName, sharedPools
	// those with a slice without spec.nodeName.
	byPool      map[poolID][]*draapi.ResourceSlice
	nodePools   map[string]sets.Set[poolID]
	sharedPools sets.Set[poolID]

	// devicesOnce is separate from indexOnce because Filter on a node with a
	// hold only needs forNode, and building the devices of every node there
	// would be wasted.
	devicesOnce sync.Once
	// localDevicesByNode holds the IDs of the devices restricted to each
	// node, either via slice-level spec.nodeName or device-level nodeName
	// under PerDeviceNodeSelection, in the latest generation of each pool.
	localDevicesByNode map[string]sets.Set[structured.DeviceID]
}

type poolID struct {
	driver, pool draapi.UniqueString
}

func (s *nodeSlices) buildIndex() {
	s.indexOnce.Do(func() {
		s.byPool = make(map[poolID][]*draapi.ResourceSlice)
		s.nodePools = make(map[string]sets.Set[poolID])
		s.sharedPools = sets.New[poolID]()
		for _, slice := range s.slices {
			id := poolID{driver: slice.Spec.Driver, pool: slice.Spec.Pool.Name}
			s.byPool[id] = append(s.byPool[id], slice)
			nodeName := ptr.Deref(slice.Spec.NodeName, "")
			if nodeName == "" {
				s.sharedPools.Insert(id)
				continue
			}
			if s.nodePools[nodeName] == nil {
				s.nodePools[nodeName] = sets.New[poolID]()
			}
			s.nodePools[nodeName].Insert(id)
		}
	})
}

// forNode returns all slices of the pools which have a slice for nodeName or a
// slice without spec.nodeName. Pools are returned whole because the allocator
// checks the completeness and the generation of a pool against all of its
// slices. Leaving out the pools of other nodes keeps per-node allocators small.
func (s *nodeSlices) forNode(nodeName string) []*draapi.ResourceSlice {
	s.buildIndex()
	var result []*draapi.ResourceSlice
	for id := range s.nodePools[nodeName].Union(s.sharedPools) {
		result = append(result, s.byPool[id]...)
	}
	return result
}

// TODO(https://github.com/kubernetes/enhancements/issues/5690): Before beta,
// build the local devices only of the nodes being checked (nodePools plus
// PerDeviceNodeSelection slices) instead of every node.
func (s *nodeSlices) buildLocalDevices() {
	s.devicesOnce.Do(func() {
		// The allocator only uses each pool's latest generation, so ignore
		// older slices.
		latest := make(map[poolID]int64)
		for _, slice := range s.slices {
			id := poolID{driver: slice.Spec.Driver, pool: slice.Spec.Pool.Name}
			latest[id] = max(latest[id], slice.Spec.Pool.Generation)
		}
		s.localDevicesByNode = make(map[string]sets.Set[structured.DeviceID])
		for _, slice := range s.slices {
			if slice.Spec.Pool.Generation < latest[poolID{driver: slice.Spec.Driver, pool: slice.Spec.Pool.Name}] {
				continue
			}
			sliceNodeName := ptr.Deref(slice.Spec.NodeName, "")
			if sliceNodeName == "" && !ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
				continue
			}
			for _, device := range slice.Spec.Devices {
				nodeName := sliceNodeName
				if nodeName == "" {
					nodeName = ptr.Deref(device.NodeName, "")
				}
				if nodeName == "" {
					continue
				}
				devices := s.localDevicesByNode[nodeName]
				if devices == nil {
					devices = sets.New[structured.DeviceID]()
					s.localDevicesByNode[nodeName] = devices
				}
				devices.Insert(structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name})
			}
		}
	})
}

// localDevices returns the IDs of devices restricted to nodeName (either via
// slice-level spec.nodeName or device-level nodeName under
// PerDeviceNodeSelection), nil if there are none. The returned set is shared;
// callers must not mutate it.
func (s *nodeSlices) localDevices(nodeName string) sets.Set[structured.DeviceID] {
	s.buildLocalDevices()
	return s.localDevicesByNode[nodeName]
}

// preemptionSimulationState tracks the pod removals simulated in a single
// CycleState clone.
type preemptionSimulationState struct {
	// released maps claim UIDs to claims that will be deallocated once their
	// removed owning pods are gone.
	released map[types.UID]*releasedClaim
}

// releasedClaim describes a claim whose devices become free once its removed
// owning pod is gone. Instances are immutable because CycleState clones
// share them.
type releasedClaim struct {
	claim    types.NamespacedName
	uid      types.UID
	nodeName string
	// allocatedDevices are from the claim's committed or in-flight
	// allocation. Only they are kept: PreBind mutates other fields of an
	// in-flight AllocationResult, which the claim tracker shares.
	allocatedDevices *resourceapi.DeviceAllocationResult
	// podUID is the UID of the owning pod.
	podUID types.UID
}

// getPreemptionSimulationState returns nil when DRAPreemption is disabled or
// the pod has no claims.
func getPreemptionSimulationState(cs fwk.CycleState) *preemptionSimulationState {
	state, err := cs.Read(preemptionSimulationStateKey)
	if err != nil {
		return nil
	}
	sim, _ := state.(*preemptionSimulationState)
	return sim
}

// Clone shallow-copies the map; *releasedClaim values are immutable and safe
// to share across clones.
func (sim *preemptionSimulationState) Clone() fwk.StateData {
	return &preemptionSimulationState{
		released: maps.Clone(sim.released),
	}
}

// unrelease drops claims released for podUID and reports whether any were dropped.
func (sim *preemptionSimulationState) unrelease(podUID types.UID) bool {
	numReleased := len(sim.released)
	maps.DeleteFunc(sim.released, func(_ types.UID, released *releasedClaim) bool {
		return released.podUID == podUID
	})
	return len(sim.released) < numReleased
}

func (sim *preemptionSimulationState) changesNode(nodeName string) bool {
	for _, released := range sim.released {
		if released.nodeName == nodeName {
			return true
		}
	}
	return false
}

// applyTo frees sim's released claims in allocatedState. In alpha, only
// exclusively allocated devices, including their fixed counter consumption
// from ResourceSlices, are freed; shares, AggregatedCapacity, and
// request-dependent ConsumedCounters stay counted, which can only make the
// allocator more pessimistic.
func (sim *preemptionSimulationState) applyTo(allocatedState *structured.AllocatedState) {
	for _, released := range sim.released {
		for id := range exclusiveDevices(released.allocatedDevices) {
			allocatedState.AllocatedDevices.Delete(id)
		}
	}
}

// exclusiveDevices yields exclusively allocated device IDs, the only devices
// preemption frees and holds cover in alpha. Admin-access results block
// nothing and are skipped. Shares are skipped even when DRAConsumableCapacity
// is off because an existing claim allocated while the gate was on may still
// share the device with other claims.
func exclusiveDevices(devices *resourceapi.DeviceAllocationResult) iter.Seq[structured.DeviceID] {
	return func(yield func(structured.DeviceID) bool) {
		for _, result := range devices.Results {
			if ptr.Deref(result.AdminAccess, false) || result.ShareID != nil {
				continue
			}
			if !yield(structured.MakeDeviceID(result.Driver, result.Pool, result.Device)) {
				return
			}
		}
	}
}

// PreFilterExtensions returns the plugin itself when DRAPreemption is enabled.
func (pl *DynamicResources) PreFilterExtensions() fwk.PreFilterExtensions {
	if !pl.fts.EnableDRAPreemption {
		return nil
	}
	return pl
}

// RemovePod simulates removing podInfoToRemove from nodeInfo, marking its
// node-local claims that will be deallocated as released in cs.
func (pl *DynamicResources) RemovePod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	data, sim := state.preemption, getPreemptionSimulationState(cs)
	if data == nil || sim == nil || !data.canPreemptDevices {
		// Nothing to simulate: DRAPreemption is off, the pod being
		// scheduled has no claims, or it can't use devices of victims.
		return nil
	}
	pod := podInfoToRemove.GetPod()
	if len(pod.Spec.ResourceClaims) == 0 || nodeInfo.Node() == nil {
		// Nothing to release: the removed pod has no claims, or its node is
		// gone.
		return nil
	}
	if pl.inPodGroup(pod) {
		// PodGroup victims release nothing in alpha (KEP-5690), so that the
		// simulation for a node never releases devices on other nodes.
		return nil
	}
	logger := klog.FromContext(ctx)
	nodeName := nodeInfo.Node().Name
	if sim.released == nil {
		sim.released = make(map[types.UID]*releasedClaim)
	}
	// TODO(https://github.com/kubernetes/enhancements/issues/5690): Before
	// beta, read claims from a snapshot consistent with data.allocatedState
	// instead of the live claim tracker. If a victim claim is deallocated
	// mid-simulation, a failed reprieve's second RemovePod sees it as
	// unallocated and won't re-add it to sim.released even though
	// allocatedState still counts its devices, causing remaining reprieve
	// checks on the node to fail.
	for _, claim := range pl.claimsOfPod(pod) {
		released, reason := pl.releasable(state, pod, claim, nodeName)
		if released == nil {
			logger.V(5).Info("Removing pod does not release claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "reason", reason)
			continue
		}
		logger.V(5).Info("Removing pod releases claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "node", nodeName)
		sim.released[claim.UID] = released
	}
	return nil
}

// AddPod undoes a prior RemovePod in cs, restoring any claims that the pod
// would keep allocated. It is a no-op if the pod was not removed.
func (pl *DynamicResources) AddPod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	sim := getPreemptionSimulationState(cs)
	if sim == nil {
		// DRAPreemption is off or the pod being scheduled has no claims.
		return nil
	}
	// Undo RemovePod, as when the simulation reprieves a victim: its claims
	// are in use again.
	sim.unrelease(podInfoToAdd.GetPod().UID)
	return nil
}

// claimsOfPod resolves the existing ResourceClaims referenced by pod, typically
// a victim rather than the pod being scheduled. Unlike podResourceClaims, it
// skips unresolvable claims instead of failing and keeps claims being deleted,
// since they still hold devices until the pod terminates.
func (pl *DynamicResources) claimsOfPod(pod *v1.Pod) []*resourceapi.ResourceClaim {
	var claims []*resourceapi.ResourceClaim
	for _, podClaim := range pod.Spec.ResourceClaims {
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &podClaim)
		if err != nil || claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		if err != nil {
			continue
		}
		if mustCheckOwner && resourceclaim.IsForPod(pod, claim, pl.fts.EnableDRAWorkloadResourceClaims) != nil {
			continue
		}
		claims = append(claims, claim)
	}
	return claims
}

// releasable checks whether removing pod allows claim to be deallocated and its
// devices reused on nodeName. If not, it returns nil and a log reason.
//
// Alpha only releases claims owned by pod and reserved by no other consumer
// (as with ResourceClaimTemplate-generated claims), so deallocation depends
// on pod alone, independent of other consumers and RemovePod/AddPod call order.
func (pl *DynamicResources) releasable(state *stateData, pod *v1.Pod, claim *resourceapi.ResourceClaim, nodeName string) (*releasedClaim, string) {
	allocation := pl.allocationOrPending(claim)
	switch {
	case allocation == nil:
		return nil, "not allocated"
	case !metav1.IsControlledBy(claim, pod):
		return nil, "not owned by the pod"
	case reservedForOthers(claim, pod.UID):
		return nil, "also reserved for other consumers"
	case state.claims.lookupByUID(claim.UID) != nil:
		return nil, "used by the pod being scheduled"
	case !state.preemption.canReleaseOnNode(allocation, nodeName):
		return nil, "has devices which are not local to the node, or only shares"
	}
	return &releasedClaim{
		claim:            types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
		uid:              claim.UID,
		nodeName:         nodeName,
		allocatedDevices: &allocation.Devices,
		podUID:           pod.UID,
	}, ""
}

// allocationOrPending returns claim's committed allocation, falling back to
// any in-flight allocation for a pod in Permit or PreBind.
func (pl *DynamicResources) allocationOrPending(claim *resourceapi.ResourceClaim) *resourceapi.AllocationResult {
	if claim.Status.Allocation != nil {
		return claim.Status.Allocation
	}
	return pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
}

// reservedForOthers reports whether any consumer other than podUID reserves claim.
func reservedForOthers(claim *resourceapi.ResourceClaim, podUID types.UID) bool {
	return len(claim.Status.ReservedFor) > 1 ||
		len(claim.Status.ReservedFor) == 1 && claim.Status.ReservedFor[0].UID != podUID
}

// inPodGroup reports whether pod belongs to a PodGroup, which workload-aware
// preemption handles as a unit.
func (pl *DynamicResources) inPodGroup(pod *v1.Pod) bool {
	return pl.fts.EnableGenericWorkload && pod.Spec.SchedulingGroup != nil
}

// allocatorForNode returns state.allocator unless sim released devices on
// nodeName, in which case it builds a node-scoped allocator with those devices
// freed.
func (pl *DynamicResources) allocatorForNode(ctx context.Context, state *stateData, sim *preemptionSimulationState, nodeName string) (structured.Allocator, error) {
	data := state.preemption
	if data == nil || sim == nil || !sim.changesNode(nodeName) {
		return state.allocator, nil
	}
	// structured.NewAllocator precomputes consumed counters from
	// AllocatedState, so state.allocator cannot be mutated in place. Scope
	// the new allocator to slices relevant to nodeName to keep simulations
	// fast.
	resourceSlices := data.slices.forNode(nodeName)
	allocatedState := data.allocatedStateForSlices(resourceSlices)
	sim.applyTo(allocatedState)
	return structured.NewAllocator(ctx, AllocatorFeatures(pl.fts), *allocatedState, pl.draManager.DeviceClasses(), resourceSlices, pl.celCache)
}
