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
	"maps"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
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
// Preemption dry runs clone CycleState per candidate node. Read-only data from
// PreFilter lives on the shared stateData.preemption, while per-clone changes
// from RemovePod and AddPod live in a separate simulationState CycleState entry
// that Clone copies.

var _ fwk.PreFilterExtensions = &DynamicResources{}

const simulationStateKey fwk.StateKey = Name + "/simulation"

// preemptionData holds the read-only PreFilter inputs shared across CycleState
// clones during preemption dry runs. Only slices mutates after PreFilter, via
// its own synchronization.
type preemptionData struct {
	// canRelease reports whether the pod being scheduled can benefit from
	// releasing victim devices. It is false when the pod has no claims to
	// allocate, disables preemption, uses DRA-backed extended resources
	// (unsupported by KEP-5690), or belongs to a PodGroup (handled by
	// workload-aware preemption).
	canRelease bool
	// podClaims holds the UIDs of claims referenced by the pod being scheduled,
	// so removing a victim sharing one of them never treats it as deallocated.
	podClaims sets.Set[types.UID]
	// allocatedState is the PreFilter allocation snapshot, or nil when the pod
	// has no claims to allocate.
	allocatedState *structured.AllocatedState
	slices         nodeSlices
}

// nodeSlices lazily indexes PreFilter's ResourceSlices by node on first use,
// avoiding the indexing cost when preemption never runs.
type nodeSlices struct {
	slices []*draapi.ResourceSlice

	indexOnce sync.Once
	// byNode groups slices that set spec.nodeName; other holds the rest.
	byNode map[string][]*draapi.ResourceSlice
	other  []*draapi.ResourceSlice
	// perDeviceNodeDevices groups devices from PerDeviceNodeSelection slices by
	// their device-level nodeName.
	perDeviceNodeDevices map[string][]structured.DeviceID

	// mutex guards localDevicesByNode.
	mutex              sync.Mutex
	localDevicesByNode map[string]sets.Set[structured.DeviceID]
}

// simulationState tracks the pod removals simulated in a single CycleState
// clone.
type simulationState struct {
	// removedPods contains UIDs of pods with claims currently removed in this clone.
	removedPods sets.Set[types.UID]
	// released are the claims that would be deallocated once removedPods
	// are gone, keyed by claim UID.
	released map[types.UID]*releasedClaim
}

// Clone shallow-copies the maps; *releasedClaim values are immutable and safe
// to share across clones.
func (s *simulationState) Clone() fwk.StateData {
	return &simulationState{
		removedPods: s.removedPods.Clone(),
		released:    maps.Clone(s.released),
	}
}

// releasedClaim describes a claim whose devices become free once the removed
// pods using it are gone. Instances are immutable because CycleState clones
// share them; updates replace the map entry instead of mutating in place.
type releasedClaim struct {
	claim    types.NamespacedName
	uid      types.UID
	nodeName string
	// allocation is the claim's committed or in-flight AllocationResult.
	// Callers must only read allocation.Devices: an in-flight AllocationResult
	// is shared with the claim tracker, where PreBind concurrently mutates
	// AllocationTimestamp.
	allocation *resourceapi.AllocationResult
	// pods are the removed pods that use this claim. Re-adding any of them via
	// AddPod keeps the claim in use and un-releases it.
	pods map[types.UID]types.NamespacedName
}

// getSimulationState returns nil when DRAPreemption is disabled or the pod has
// no claims.
func getSimulationState(cs fwk.CycleState) *simulationState {
	state, err := cs.Read(simulationStateKey)
	if err != nil {
		return nil
	}
	sim, _ := state.(*simulationState)
	return sim
}

func (pl *DynamicResources) newPreemptionData(pod *v1.Pod, claims *claimStore, allocatedState *structured.AllocatedState, resourceSlices []*draapi.ResourceSlice) *preemptionData {
	data := &preemptionData{
		podClaims:      sets.New[types.UID](),
		allocatedState: allocatedState,
		canRelease: allocatedState != nil &&
			claims.extendedResourceClaim() == nil &&
			(!pl.fts.EnableGenericWorkload || pod.Spec.SchedulingGroup == nil) &&
			ptr.Deref(pod.Spec.PreemptionPolicy, v1.PreemptLowerPriority) != v1.PreemptNever,
	}
	data.slices.slices = resourceSlices
	for _, claim := range claims.all() {
		data.podClaims.Insert(claim.UID)
	}
	return data
}

// PreFilterExtensions returns the plugin itself when DRAPreemption is enabled.
func (pl *DynamicResources) PreFilterExtensions() fwk.PreFilterExtensions {
	if !pl.fts.EnableDRAPreemption {
		return nil
	}
	return pl
}

// RemovePod simulates removing podInfoToRemove from nodeInfo, marking any of
// its node-local claims that would be deallocated once all removed pods are
// gone as released in cs.
func (pl *DynamicResources) RemovePod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	data, sim := state.preemption, getSimulationState(cs)
	if data == nil || sim == nil || !data.canRelease {
		return nil
	}
	pod := podInfoToRemove.GetPod()
	if len(pod.Spec.ResourceClaims) == 0 || sim.removedPods.Has(pod.UID) || nodeInfo.Node() == nil {
		return nil
	}
	logger := klog.FromContext(ctx)
	nodeName := nodeInfo.Node().Name
	if sim.removedPods == nil {
		sim.removedPods = sets.New[types.UID]()
	}
	if sim.released == nil {
		sim.released = make(map[types.UID]*releasedClaim)
	}
	sim.removedPods.Insert(pod.UID)
	podName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	for _, claim := range pl.claimsOfPod(pod) {
		if released := sim.released[claim.UID]; released != nil {
			// Another pod referencing this claim (for example, before PreBind
			// adds it to ReservedFor) was already removed; record this pod too
			// so AddPod for either pod restores the claim. Copy instead of
			// mutating because CycleState clones share *releasedClaim values.
			updated := *released
			updated.pods = maps.Clone(released.pods)
			updated.pods[pod.UID] = podName
			sim.released[claim.UID] = &updated
			continue
		}
		allocation, pods, reason := pl.releasable(data, sim, claim, pod, nodeName)
		if allocation == nil {
			logger.V(5).Info("Removing pod does not release claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "reason", reason)
			continue
		}
		logger.V(5).Info("Removing pod releases claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "node", nodeName)
		sim.released[claim.UID] = &releasedClaim{
			claim:      types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			uid:        claim.UID,
			nodeName:   nodeName,
			allocation: allocation,
			pods:       pods,
		}
	}
	return nil
}

// AddPod undoes a prior RemovePod in cs, restoring any claims that the pod
// would keep allocated. It is a no-op if the pod was not removed.
func (pl *DynamicResources) AddPod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	sim := getSimulationState(cs)
	if sim == nil {
		return nil
	}
	pod := podInfoToAdd.GetPod()
	if !sim.removedPods.Has(pod.UID) {
		return nil
	}
	sim.removedPods.Delete(pod.UID)
	for uid, released := range sim.released {
		if _, ok := released.pods[pod.UID]; ok {
			delete(sim.released, uid)
		}
	}
	return nil
}

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

// releasable checks whether removing pod (along with previously removed pods
// in sim) allows claim to be deallocated and its devices reused on nodeName.
// On success it returns the claim's allocation and the set of removed pods
// holding it; otherwise it returns nil and a log reason.
func (pl *DynamicResources) releasable(data *preemptionData, sim *simulationState, claim *resourceapi.ResourceClaim, pod *v1.Pod, nodeName string) (*resourceapi.AllocationResult, map[types.UID]types.NamespacedName, string) {
	allocation := pl.allocationOrPending(claim)
	pending := claim.Status.Allocation == nil
	switch {
	case allocation == nil:
		return nil, nil, "not allocated"
	case data.podClaims.Has(claim.UID):
		return nil, nil, "used by the pod being scheduled"
	case !data.devicesOnNode(allocation, nodeName, pl.fts.EnableDRAConsumableCapacity):
		// KEP-5690 scopes preemption to devices local to the candidate node.
		return nil, nil, "has devices which are not local to the node"
	}

	pods := map[types.UID]types.NamespacedName{pod.UID: {Namespace: pod.Namespace, Name: pod.Name}}
	if len(claim.Status.ReservedFor) == 0 {
		switch {
		case pending && pl.fts.EnableGenericWorkload && pod.Spec.SchedulingGroup != nil:
			// Other pods in the PodGroup may share this in-flight allocation
			// without appearing in ReservedFor yet.
			return nil, nil, "pending allocation may be shared within the PodGroup"
		case pending:
			// Ungrouped in-flight allocations belong to a single pod in Permit
			// or PreBind and are discarded if that pod is preempted.
			return allocation, pods, ""
		case metav1.IsControlledBy(claim, pod):
			// Claim was generated for pod (e.g. from a ResourceClaimTemplate)
			// and will be garbage-collected with it.
			return allocation, pods, ""
		default:
			// Standalone allocated claims with an empty ReservedFor are not
			// automatically deallocated when the pod is deleted.
			return nil, nil, "allocated, not reserved and not owned by the pod"
		}
	}
	for _, consumer := range claim.Status.ReservedFor {
		if isNonPodConsumer(consumer) {
			return nil, nil, "reserved for a PodGroup or some other non-pod consumer"
		}
		if sim.removedPods.Has(consumer.UID) {
			pods[consumer.UID] = types.NamespacedName{Namespace: claim.Namespace, Name: consumer.Name}
			continue
		}
		if !pl.podStoppedUsingClaims(claim.Namespace, consumer) {
			return nil, nil, "also reserved for other pods"
		}
	}
	return allocation, pods, ""
}

// allocationOrPending returns claim's committed allocation, falling back to
// any in-flight allocation for a pod in Permit or PreBind.
func (pl *DynamicResources) allocationOrPending(claim *resourceapi.ResourceClaim) *resourceapi.AllocationResult {
	if claim.Status.Allocation != nil {
		return claim.Status.Allocation
	}
	return pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
}

func isNonPodConsumer(consumer resourceapi.ResourceClaimConsumerReference) bool {
	return consumer.APIGroup != "" || consumer.Resource != "pods"
}

// podStoppedUsingClaims reports whether consumer no longer holds its claims
// (deleted, recreated with a new UID, terminating, or in a terminal phase)
// even if the ResourceClaim controller has not yet removed it from ReservedFor.
func (pl *DynamicResources) podStoppedUsingClaims(namespace string, consumer resourceapi.ResourceClaimConsumerReference) bool {
	pod, err := pl.podLister.Pods(namespace).Get(consumer.Name)
	if err != nil {
		return apierrors.IsNotFound(err)
	}
	return pod.UID != consumer.UID ||
		pod.DeletionTimestamp != nil ||
		pod.Status.Phase == v1.PodSucceeded ||
		pod.Status.Phase == v1.PodFailed
}

// devicesOnNode reports whether allocation frees at least one non-admin device
// and all of those devices are local to nodeName.
func (data *preemptionData) devicesOnNode(allocation *resourceapi.AllocationResult, nodeName string, consumableCapacity bool) bool {
	localDevices := data.slices.localDevices(nodeName)
	found := false
	for _, result := range allocation.Devices.Results {
		// Admin-access allocations do not block normal allocations, so
		// releasing them frees nothing for the preemptor.
		if ptr.Deref(result.AdminAccess, false) {
			continue
		}
		if result.ShareID != nil && !consumableCapacity {
			// Without DRAConsumableCapacity, AllocatedState does not track
			// share counts, so we cannot tell whether other claims still share
			// this device.
			return false
		}
		if !localDevices.Has(structured.MakeDeviceID(result.Driver, result.Pool, result.Device)) {
			return false
		}
		found = true
	}
	return found
}

func (s *nodeSlices) buildIndex() {
	s.indexOnce.Do(func() {
		s.byNode = make(map[string][]*draapi.ResourceSlice)
		s.perDeviceNodeDevices = make(map[string][]structured.DeviceID)
		for _, slice := range s.slices {
			if nodeName := ptr.Deref(slice.Spec.NodeName, ""); nodeName != "" {
				s.byNode[nodeName] = append(s.byNode[nodeName], slice)
				continue
			}
			s.other = append(s.other, slice)
			if !ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
				continue
			}
			for _, device := range slice.Spec.Devices {
				if nodeName := ptr.Deref(device.NodeName, ""); nodeName != "" {
					s.perDeviceNodeDevices[nodeName] = append(s.perDeviceNodeDevices[nodeName], structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name})
				}
			}
		}
	})
}

// forNode returns slices matching nodeName plus all slices without
// spec.nodeName. Omitting other nodes' spec.nodeName slices keeps per-node
// allocator setup cheap; if a single pool spans multiple nodes via
// spec.nodeName, its slice count will not match spec.pool.resourceSliceCount
// and the allocator will conservatively skip that pool.
func (s *nodeSlices) forNode(nodeName string) []*draapi.ResourceSlice {
	s.buildIndex()
	return slices.Concat(s.byNode[nodeName], s.other)
}

// localDevices returns the IDs of devices restricted to nodeName (either via
// slice-level spec.nodeName or device-level nodeName under
// PerDeviceNodeSelection). The returned set is cached; callers must not mutate
// it.
func (s *nodeSlices) localDevices(nodeName string) sets.Set[structured.DeviceID] {
	s.mutex.Lock()
	devices, ok := s.localDevicesByNode[nodeName]
	s.mutex.Unlock()
	if ok {
		return devices
	}

	s.buildIndex()
	devices = sets.New(s.perDeviceNodeDevices[nodeName]...)
	for _, slice := range s.byNode[nodeName] {
		for _, device := range slice.Spec.Devices {
			devices.Insert(structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name})
		}
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.localDevicesByNode == nil {
		s.localDevicesByNode = make(map[string]sets.Set[structured.DeviceID])
	}
	s.localDevicesByNode[nodeName] = devices
	return devices
}

func (sim *simulationState) changesNode(nodeName string) bool {
	for _, released := range sim.released {
		if released.nodeName == nodeName {
			return true
		}
	}
	return false
}

// allocatorForNode returns state.allocator unless sim released devices on
// nodeName, in which case it builds a node-scoped allocator with those devices
// freed. When DRAPreemption is enabled, it also returns the AllocatedState
// backing the allocator.
func (pl *DynamicResources) allocatorForNode(ctx context.Context, state *stateData, sim *simulationState, nodeName string) (structured.Allocator, *structured.AllocatedState, error) {
	data := state.preemption
	if data == nil || sim == nil {
		return state.allocator, nil, nil
	}
	if !sim.changesNode(nodeName) {
		return state.allocator, data.allocatedState, nil
	}
	// structured.NewAllocator precomputes consumed counters from
	// AllocatedState, so state.allocator cannot be mutated in place. Scope
	// the new allocator to slices relevant to nodeName to keep dry runs fast.
	resourceSlices := data.slices.forNode(nodeName)
	allocatedState := data.allocatedStateForSlices(resourceSlices)
	sim.applyTo(allocatedState, pl.fts.EnableDRAConsumableCapacity)
	allocator, err := structured.NewAllocator(ctx, AllocatorFeatures(pl.fts), *allocatedState, pl.draManager.DeviceClasses(), resourceSlices, pl.celCache)
	if err != nil {
		return nil, nil, err
	}
	return allocator, allocatedState, nil
}

// allocatedStateForSlices clones the subset of PreFilter's AllocatedState
// covering devices in resourceSlices, avoiding a full cluster-wide copy per
// simulated Filter call.
func (data *preemptionData) allocatedStateForSlices(resourceSlices []*draapi.ResourceSlice) *structured.AllocatedState {
	state := &structured.AllocatedState{
		AllocatedDevices:         sets.New[structured.DeviceID](),
		AllocatedSharedDeviceIDs: sets.New[structured.DeviceID](),
		AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
	}
	if data.allocatedState.AllocatedShareCounts != nil {
		state.AllocatedShareCounts = make(map[structured.DeviceID]int)
	}
	for _, slice := range resourceSlices {
		for _, device := range slice.Spec.Devices {
			id := structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
			if data.allocatedState.AllocatedDevices.Has(id) {
				state.AllocatedDevices.Insert(id)
			}
			if data.allocatedState.AllocatedSharedDeviceIDs.Has(id) {
				state.AllocatedSharedDeviceIDs.Insert(id)
			}
			if capacity, ok := data.allocatedState.AggregatedCapacity[id]; ok {
				state.AggregatedCapacity[id] = capacity.Clone()
			}
			if count, ok := data.allocatedState.AllocatedShareCounts[id]; ok {
				state.AllocatedShareCounts[id] = count
			}
		}
	}
	return state
}

// applyTo subtracts sim's released claims from state. Per-share consumed
// capacity is removed immediately, whereas a shared device (and any shared
// counters it consumes) remains in AllocatedSharedDeviceIDs until
// AllocatedShareCounts confirms that every active share has been released.
func (sim *simulationState) applyTo(state *structured.AllocatedState, consumableCapacity bool) {
	releasedShares := make(map[structured.DeviceID]int)
	for _, released := range sim.released {
		structured.ForEachAllocatedDevice(released.allocation,
			func(id structured.DeviceID) {
				state.AllocatedDevices.Delete(id)
			},
			consumableCapacity,
			func(id structured.SharedDeviceID) {
				releasedShares[id.GetDeviceID()]++
			},
			func(capacity structured.DeviceConsumedCapacity) {
				state.AggregatedCapacity.Remove(capacity)
			})
	}
	for id, numReleased := range releasedShares {
		if total, ok := state.AllocatedShareCounts[id]; ok && total <= numReleased {
			state.AllocatedSharedDeviceIDs.Delete(id)
		}
	}
}

// preemptionMightHelp reports whether Filter should return Unschedulable
// instead of UnschedulableAndUnresolvable after device allocation fails on
// nodeInfo, allowing DefaultPreemption to evaluate the node.
//
// As a fast check without looking up each pod's ResourceClaims, it requires a
// lower-priority pod with claims on the node and at least one node-local device
// allocated in allocatedState.
func preemptionMightHelp(state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocatedState *structured.AllocatedState) bool {
	data := state.preemption
	if data == nil || !data.canRelease || allocatedState == nil {
		return false
	}
	priority := corev1helpers.PodPriority(pod)
	if !slices.ContainsFunc(nodeInfo.GetPods(), func(podInfo fwk.PodInfo) bool {
		other := podInfo.GetPod()
		return len(other.Spec.ResourceClaims) > 0 && corev1helpers.PodPriority(other) < priority
	}) {
		return false
	}
	return data.hasAllocatedNodeLocalDevice(nodeInfo.Node().Name, allocatedState)
}

func (data *preemptionData) hasAllocatedNodeLocalDevice(nodeName string, state *structured.AllocatedState) bool {
	for id := range data.slices.localDevices(nodeName) {
		if structured.IsDeviceAllocated(id, state) {
			return true
		}
	}
	return false
}
