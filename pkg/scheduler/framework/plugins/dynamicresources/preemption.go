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
	"strings"
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
// are removed and applying device holds for nominated pods (see holds.go)
// without mutating the shared DRAManager.
//
// Read-only PreFilter inputs (stateData.preemption) and cached Filter results
// (stateData.dryRuns) live on stateData and are shared across CycleState
// clones. Per-clone changes from RemovePod and AddPod live in a separate
// simulationState CycleState entry that Clone copies.

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

// simulationState tracks the pod removals and nominated-pod holds simulated in
// a single CycleState clone.
type simulationState struct {
	// removedPods contains UIDs of pods with claims currently removed in this clone.
	removedPods sets.Set[types.UID]
	// released are the claims that would be deallocated once removedPods
	// are gone, keyed by claim UID.
	released map[types.UID]*releasedClaim
	// holds maps UIDs of nominated pods added via AddPod to their device holds.
	holds map[types.UID]*allocation
	// dryRun identifies the preemption dry run operating on this clone (0
	// until RemovePod removes a pod with claims). Clone preserves it so pass-1
	// clones in RunFilterPluginsWithNominatedPods share the parent dry run's ID.
	// Only the goroutine owning this clone accesses dryRun; dryRunResults.record
	// assigns it under r.mutex solely to allocate from r.numDryRuns.
	dryRun int
}

// Clone shallow-copies the maps; *releasedClaim and *allocation values are
// immutable and safe to share across clones.
func (s *simulationState) Clone() fwk.StateData {
	return &simulationState{
		removedPods: s.removedPods.Clone(),
		released:    maps.Clone(s.released),
		holds:       maps.Clone(s.holds),
		dryRun:      s.dryRun,
	}
}

type simulatedResultKey struct {
	nodeName string
	released string
}

func (sim *simulationState) resultKey(nodeName string) simulatedResultKey {
	return simulatedResultKey{nodeName: nodeName, released: sim.releasedKey(nodeName)}
}

type dryRunNode struct {
	dryRun   int
	nodeName string
}

// releasedKey returns a canonical key for the set of claims released on
// nodeName in sim ("" if none).
func (sim *simulationState) releasedKey(nodeName string) string {
	var uids []string
	for uid, released := range sim.released {
		if released.nodeName == nodeName {
			uids = append(uids, string(uid))
		}
	}
	slices.Sort(uids)
	return strings.Join(uids, ",")
}

// dryRunResults caches successful Filter allocations across CycleState clones
// within a single scheduling cycle for two purposes:
//
//  1. Hold-aware result reuse: RunFilterPluginsWithNominatedPods runs pass 1 on
//     a clone with holds (via AddPod) and pass 2 on the base state without
//     holds. Because holds only restrict availability, a Filter call with no
//     holds can reuse an earlier allocation for the same node and released-claim
//     set, ensuring pass 2 and Reserve pick the devices chosen in pass 1.
//  2. Final dry-run lookup: PrepareNomination needs the allocation from the
//     victim set that DefaultPreemption's dry run settled on for a node, which
//     need not be the last Filter call on that node (for example, if a plugin
//     after DynamicResources rejected a reprieved victim).
type dryRunResults struct {
	mutex sync.Mutex
	// results stores Filter results by node and released-claim key. Entries are
	// replaced, never mutated.
	results map[simulatedResultKey]*simulatedResult
	// lastSimulated records the latest releasedKey on (dryRun, nodeName) after
	// each RemovePod or AddPod. Keying by dryRun isolates a node's own dry run
	// from other nodes' dry runs that remove cross-node PodGroup members.
	lastSimulated map[dryRunNode]string
	// filteredDryRun records which dryRun last stored a Filter result for each
	// nodeName (0 for the normal non-preemption Filter pass).
	filteredDryRun map[string]int
	numDryRuns     int
}

type simulatedResult struct {
	// allocationResults holds Filter's newly computed allocations in
	// claimsToAllocate order (excluding PodGroup pending allocations, so
	// reusable can sanity-check the slice length).
	allocationResults []resourceapi.AllocationResult
	// claims and released capture the hold for PrepareNomination if the dry run
	// settles on this simulated state.
	claims   map[types.UID]heldClaim
	released []*releasedClaim
}

// record assigns sim a non-zero dryRun ID on first pod removal and updates
// the latest releasedKey for (sim.dryRun, nodeName).
func (r *dryRunResults) record(sim *simulationState, nodeName string) {
	key := sim.releasedKey(nodeName)
	r.mutex.Lock()
	defer r.mutex.Unlock()
	if sim.dryRun == 0 {
		r.numDryRuns++
		sim.dryRun = r.numDryRuns
	}
	if r.lastSimulated == nil {
		r.lastSimulated = make(map[dryRunNode]string)
	}
	r.lastSimulated[dryRunNode{dryRun: sim.dryRun, nodeName: nodeName}] = key
}

func (r *dryRunResults) reusable(key simulatedResultKey, numClaims int) ([]resourceapi.AllocationResult, bool) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	result := r.results[key]
	if result == nil || len(result.allocationResults) != numClaims {
		return nil, false
	}
	// Clone because Filter appends pendingResult to the returned slice.
	return slices.Clone(result.allocationResults), true
}

func (r *dryRunResults) store(nodeName string, sim *simulationState, result *simulatedResult) {
	key := sim.resultKey(nodeName)
	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.results == nil {
		r.results = make(map[simulatedResultKey]*simulatedResult)
	}
	if r.filteredDryRun == nil {
		r.filteredDryRun = make(map[string]int)
	}
	r.results[key] = result
	r.filteredDryRun[nodeName] = sim.dryRun
}

// final returns the Filter result matching the final victim set of nodeName's
// dry run, or nil if none. Callers must not mutate the returned value.
func (r *dryRunResults) final(nodeName string) *simulatedResult {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	key := dryRunNode{dryRun: r.filteredDryRun[nodeName], nodeName: nodeName}
	return r.results[simulatedResultKey{nodeName: nodeName, released: r.lastSimulated[key]}]
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

// RemovePod either drops a nominated pod's hold added by AddPod, or simulates
// removing podInfoToRemove from nodeInfo by marking any node-local claims that
// would be deallocated once all removed pods are gone as released in cs.
func (pl *DynamicResources) RemovePod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	data, sim := state.preemption, getSimulationState(cs)
	if data == nil || sim == nil || nodeInfo.Node() == nil {
		return nil
	}
	pod := podInfoToRemove.GetPod()
	if _, ok := sim.holds[pod.UID]; ok {
		delete(sim.holds, pod.UID)
		return nil
	}
	if !data.canRelease || len(pod.Spec.ResourceClaims) == 0 || sim.removedPods.Has(pod.UID) {
		return nil
	}
	logger := klog.FromContext(ctx)
	nodeName := nodeInfo.Node().Name
	defer state.dryRuns.record(sim, nodeName)
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

// AddPod either undoes a prior RemovePod in cs, or records the device hold of
// an unassigned pod nominated to nodeInfo so Filter reserves those devices.
// The framework only invokes AddPod for nominated pods with priority >=
// podToSchedule.
func (pl *DynamicResources) AddPod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	sim := getSimulationState(cs)
	if sim == nil || nodeInfo.Node() == nil {
		return nil
	}
	pod := podInfoToAdd.GetPod()
	nodeName := nodeInfo.Node().Name
	if sim.removedPods.Has(pod.UID) {
		sim.removedPods.Delete(pod.UID)
		for uid, released := range sim.released {
			if _, ok := released.pods[pod.UID]; ok {
				delete(sim.released, uid)
			}
		}
		state.dryRuns.record(sim, nodeName)
		return nil
	}
	if pl.store == nil || pod.Spec.NodeName != "" || pod.UID == podToSchedule.UID {
		return nil
	}
	hold := pl.store.nominatedAllocation(pod.UID)
	if hold == nil || hold.nodeName != nodeName {
		return nil
	}
	if sim.holds == nil {
		sim.holds = make(map[types.UID]*allocation)
	}
	sim.holds[pod.UID] = hold
	klog.FromContext(ctx).V(5).Info("Devices held for nominated pod", "pod", klog.KObj(podToSchedule), "nominatedPod", klog.KObj(pod), "node", nodeName, "numClaims", len(hold.claims))
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

// allocate runs device allocation on node, reusing a cached result from
// state.dryRuns when holds is nil and a matching result exists. The returned
// AllocatedState is non-nil only when allocatorForNode ran with DRAPreemption.
func (pl *DynamicResources) allocate(ctx context.Context, state *stateData, sim *simulationState, holds *nodeHolds, node *v1.Node, claims []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, *structured.AllocatedState, error) {
	if sim != nil && holds == nil {
		if results, ok := state.dryRuns.reusable(sim.resultKey(node.Name), len(claims)); ok {
			return results, nil, nil
		}
	}
	allocator, allocatedState, err := pl.allocatorForNode(ctx, state, sim, node.Name, holds)
	if err != nil {
		return nil, nil, err
	}
	results, err := allocator.Allocate(ctx, node, claims)
	return results, allocatedState, err
}

// allocatorForNode returns state.allocator unless sim released devices on
// nodeName or holds reserves devices there, in which case it builds a
// node-scoped allocator reflecting both. When DRAPreemption is enabled, it
// also returns the AllocatedState backing the allocator.
func (pl *DynamicResources) allocatorForNode(ctx context.Context, state *stateData, sim *simulationState, nodeName string, holds *nodeHolds) (structured.Allocator, *structured.AllocatedState, error) {
	data := state.preemption
	if data == nil || sim == nil {
		return state.allocator, nil, nil
	}
	if holds == nil && !sim.changesNode(nodeName) {
		return state.allocator, data.allocatedState, nil
	}
	// structured.NewAllocator precomputes consumed counters from
	// AllocatedState, so state.allocator cannot be mutated in place. Scope
	// the new allocator to slices relevant to nodeName to keep dry runs fast.
	resourceSlices := data.slices.forNode(nodeName)
	allocatedState := data.allocatedStateForSlices(resourceSlices)
	sim.applyTo(allocatedState, pl.fts.EnableDRAConsumableCapacity)
	if holds != nil {
		// Apply holds after sim.applyTo because a held device may be one of
		// the devices just freed from a removed victim.
		holds.applyTo(allocatedState, sim.released)
	}
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
//
// Nodes with victim claims still awaiting deallocation also return
// Unschedulable even after the victim pods are gone: if a nominated node were
// marked UnschedulableAndUnresolvable, DefaultPreemption would treat it as
// unusable and preempt on another node without consulting PodEligibleToPreempt.
func (pl *DynamicResources) preemptionMightHelp(state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocatedState *structured.AllocatedState) bool {
	data := state.preemption
	if data == nil || !data.canRelease || allocatedState == nil {
		return false
	}
	nodeName := nodeInfo.Node().Name
	priority := corev1helpers.PodPriority(pod)
	if !slices.ContainsFunc(nodeInfo.GetPods(), func(podInfo fwk.PodInfo) bool {
		other := podInfo.GetPod()
		return len(other.Spec.ResourceClaims) > 0 && corev1helpers.PodPriority(other) < priority
	}) && len(pl.settlingVictimClaims(nodeName, true)) == 0 {
		return false
	}
	return data.hasAllocatedNodeLocalDevice(nodeName, allocatedState)
}

func (data *preemptionData) hasAllocatedNodeLocalDevice(nodeName string, state *structured.AllocatedState) bool {
	for id := range data.slices.localDevices(nodeName) {
		if structured.IsDeviceAllocated(id, state) {
			return true
		}
	}
	return false
}
