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

// This file implements the DRA side of preemption (KEP-5690, DRAPreemption
// feature gate): PreFilterExtensions which simulate that devices of removed
// pods become available, without modifying the shared DRA manager.

var _ fwk.PreFilterExtensions = &DynamicResources{}

// preemptionState is part of stateData when DRAPreemption is enabled and the
// pod has claims.
type preemptionState struct {
	// root is shared by all copies of the CycleState.
	root *preemptionRoot
	// sim belongs to one copy of the CycleState. Clone copies it.
	sim simulation
}

// preemptionRoot holds what the simulation needs from PreFilter. Apart from
// the lazily computed indexes, it does not change after PreFilter.
type preemptionRoot struct {
	// canRelease is true if devices of removed pods may become available
	// for the pod. Pods in a PodGroup are handled by workload-aware
	// preemption, DRA extended resources are not supported, and pods which
	// don't preempt or have nothing to allocate don't need it.
	canRelease bool
	// podClaims are the UIDs of the pod's own claims. Removing a pod never
	// releases them, the pod is going to use them.
	podClaims sets.Set[types.UID]
	// allocatedState and slices are the input of the PreFilter allocator.
	// Both are nil if the pod has no claim to allocate.
	allocatedState *structured.AllocatedState
	slices         []*draapi.ResourceSlice
	// countShares is nil if the claim tracker cannot count the shares of
	// devices (for example, the one of Cluster Autoscaler).
	countShares func() map[structured.DeviceID]int

	indexOnce sync.Once
	// slicesByNode has the slices with spec.nodeName, otherSlices all others.
	slicesByNode map[string][]*draapi.ResourceSlice
	otherSlices  []*draapi.ResourceSlice
	// perDeviceNodeDevices has the devices with a node name from slices
	// with per-device node selection, by node name, so that looking up the
	// devices of a node does not have to go through all of them.
	perDeviceNodeDevices map[string][]structured.DeviceID

	// mutex protects the lazily computed fields below.
	mutex       sync.Mutex
	nodeDevices map[string]sets.Set[structured.DeviceID]
	shareCounts map[structured.DeviceID]int
	sharesKnown bool
}

// simulation records what RemovePod and AddPod changed in one copy of the
// CycleState.
type simulation struct {
	// removedPods are the pods with claims which were removed and not added back.
	removedPods sets.Set[types.UID]
	// released maps the UIDs of claims which removing pods released to
	// what got released. Entries are never modified, only replaced, because
	// copies of the simulation share them.
	released map[types.UID]*releasedClaim
}

// releasedClaim is a claim which gets deallocated once the removed pods are gone.
type releasedClaim struct {
	claim *resourceapi.ResourceClaim
	// allocation is the allocation of the claim, or its pending allocation
	// if the claim is reserved for a pod in Permit or PreBind.
	allocation *resourceapi.AllocationResult
	// nodeName is the node of the devices.
	nodeName string
	// consumers are the removed pods which use the claim. Adding back any
	// of them restores it.
	consumers sets.Set[types.UID]
}

func (p *preemptionState) clone() *preemptionState {
	return &preemptionState{
		root: p.root,
		sim: simulation{
			removedPods: maps.Clone(p.sim.removedPods),
			released:    maps.Clone(p.sim.released),
		},
	}
}

// newPreemptionState is called by PreFilter. allocatedState and
// resourceSlices are nil if the pod has no claim to allocate.
func (pl *DynamicResources) newPreemptionState(pod *v1.Pod, claims *claimStore, allocatedState *structured.AllocatedState, resourceSlices []*draapi.ResourceSlice) *preemptionState {
	root := &preemptionRoot{
		podClaims:      sets.New[types.UID](),
		allocatedState: allocatedState,
		slices:         resourceSlices,
		canRelease: allocatedState != nil &&
			claims.extendedResourceClaim() == nil &&
			(!pl.fts.EnableGenericWorkload || pod.Spec.SchedulingGroup == nil) &&
			ptr.Deref(pod.Spec.PreemptionPolicy, v1.PreemptLowerPriority) != v1.PreemptNever,
	}
	for _, claim := range claims.all() {
		root.podClaims.Insert(claim.UID)
	}
	if counter, ok := pl.draManager.ResourceClaims().(sharedDeviceCounter); ok {
		root.countShares = counter.countSharedDevices
	}
	return &preemptionState{root: root}
}

// PreFilterExtensions returns the plugin itself when DRAPreemption is enabled.
func (pl *DynamicResources) PreFilterExtensions() fwk.PreFilterExtensions {
	if !pl.fts.EnableDRAPreemption {
		return nil
	}
	return pl
}

// RemovePod simulates that the pod is gone: the devices of each of its claims
// become available if the claim gets deallocated once the pod and all other
// pods using it are gone.
func (pl *DynamicResources) RemovePod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	p := state.preemption
	if p == nil || !p.root.canRelease {
		return nil
	}
	pod := podInfoToRemove.GetPod()
	if len(pod.Spec.ResourceClaims) == 0 || p.sim.removedPods.Has(pod.UID) || nodeInfo.Node() == nil {
		return nil
	}
	logger := klog.FromContext(ctx)
	nodeName := nodeInfo.Node().Name
	if p.sim.removedPods == nil {
		p.sim.removedPods = sets.New[types.UID]()
	}
	if p.sim.released == nil {
		p.sim.released = make(map[types.UID]*releasedClaim)
	}
	p.sim.removedPods.Insert(pod.UID)
	for _, claim := range pl.claimsOfPod(pod) {
		if released := p.sim.released[claim.UID]; released != nil {
			// Released because of other removed pods, now this one
			// also stops using it.
			p.sim.released[claim.UID] = &releasedClaim{
				claim:      released.claim,
				allocation: released.allocation,
				nodeName:   released.nodeName,
				consumers:  released.consumers.Union(sets.New(pod.UID)),
			}
			continue
		}
		allocation, consumers, reason := pl.releasable(p, claim, pod, nodeName)
		if allocation == nil {
			logger.V(5).Info("Removing pod does not release claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "reason", reason)
			continue
		}
		logger.V(5).Info("Removing pod releases claim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim), "node", nodeName)
		p.sim.released[claim.UID] = &releasedClaim{
			claim:      claim,
			allocation: allocation,
			nodeName:   nodeName,
			consumers:  consumers,
		}
	}
	return nil
}

// AddPod reverts RemovePod. Pods which were not removed are ignored.
func (pl *DynamicResources) AddPod(ctx context.Context, cs fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	p := state.preemption
	if p == nil {
		return nil
	}
	pod := podInfoToAdd.GetPod()
	if !p.sim.removedPods.Has(pod.UID) {
		return nil
	}
	p.sim.removedPods.Delete(pod.UID)
	for uid, released := range p.sim.released {
		if released.consumers.Has(pod.UID) {
			delete(p.sim.released, uid)
		}
	}
	return nil
}

// claimsOfPod returns the existing claims referenced by the pod.
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

// releasable checks whether claim of the removed pod gets deallocated once
// the removed pods are gone, so that its devices can be used by the pod being
// scheduled. If yes, it returns the allocation and the removed pods which use
// the claim. If not, it returns nil and the reason.
func (pl *DynamicResources) releasable(p *preemptionState, claim *resourceapi.ResourceClaim, pod *v1.Pod, nodeName string) (*resourceapi.AllocationResult, sets.Set[types.UID], string) {
	allocation := claim.Status.Allocation
	pending := false
	if allocation == nil {
		// The pod is in Permit or PreBind.
		allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
		pending = true
	}
	switch {
	case allocation == nil:
		return nil, nil, "not allocated"
	case p.root.podClaims.Has(claim.UID):
		return nil, nil, "used by the pod being scheduled"
	case !p.root.devicesOnNode(allocation, nodeName, pl.fts.EnableDRAConsumableCapacity):
		// Devices which are not local to the node may be needed
		// elsewhere, so the KEP leaves them alone.
		return nil, nil, "has devices which are not local to the node"
	}

	consumers := sets.New(pod.UID)
	if len(claim.Status.ReservedFor) == 0 {
		switch {
		case pending && pl.fts.EnableGenericWorkload && pod.Spec.SchedulingGroup != nil:
			// Pods of the same PodGroup can share the pending allocation.
			return nil, nil, "pending allocation may be shared within the PodGroup"
		case pending:
			return allocation, consumers, ""
		case metav1.IsControlledBy(claim, pod):
			// Gets deleted together with the pod.
			return allocation, consumers, ""
		default:
			// Nothing deallocates a claim which nobody reserved.
			return nil, nil, "allocated, not reserved and not owned by the pod"
		}
	}
	for _, consumer := range claim.Status.ReservedFor {
		if consumer.APIGroup != "" || consumer.Resource != "pods" {
			return nil, nil, "reserved for a PodGroup or some other non-pod consumer"
		}
		if p.sim.removedPods.Has(consumer.UID) {
			consumers.Insert(consumer.UID)
			continue
		}
		if !pl.podStoppedUsingClaims(claim.Namespace, consumer) {
			return nil, nil, "also reserved for other pods"
		}
	}
	return allocation, consumers, ""
}

// podStoppedUsingClaims returns true if the consumer is gone, being deleted or
// done. The resourceclaim controller removes such pods from ReservedFor.
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

// devicesOnNode returns true if the allocation has at least one device which
// is not shared with admin access and all of them are local to the node.
func (r *preemptionRoot) devicesOnNode(allocation *resourceapi.AllocationResult, nodeName string, consumableCapacity bool) bool {
	localDevices := r.nodeLocalDevices(nodeName)
	found := false
	for _, result := range allocation.Devices.Results {
		if ptr.Deref(result.AdminAccess, false) {
			continue
		}
		if result.ShareID != nil && !consumableCapacity {
			// The claim tracker then treats the device as allocated
			// exclusively, so it is unknown whether other claims share it.
			return false
		}
		if !localDevices.Has(structured.MakeDeviceID(result.Driver, result.Pool, result.Device)) {
			return false
		}
		found = true
	}
	return found
}

func (r *preemptionRoot) buildIndex() {
	r.indexOnce.Do(func() {
		r.slicesByNode = make(map[string][]*draapi.ResourceSlice)
		r.perDeviceNodeDevices = make(map[string][]structured.DeviceID)
		for _, slice := range r.slices {
			if nodeName := ptr.Deref(slice.Spec.NodeName, ""); nodeName != "" {
				r.slicesByNode[nodeName] = append(r.slicesByNode[nodeName], slice)
				continue
			}
			r.otherSlices = append(r.otherSlices, slice)
			if !ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
				continue
			}
			for _, device := range slice.Spec.Devices {
				if nodeName := ptr.Deref(device.NodeName, ""); nodeName != "" {
					r.perDeviceNodeDevices[nodeName] = append(r.perDeviceNodeDevices[nodeName], structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name})
				}
			}
		}
	})
}

// nodeLocalDevices returns the devices which are only available on the node:
// those in slices for the node and those with that node name in slices with
// per-device node selection.
func (r *preemptionRoot) nodeLocalDevices(nodeName string) sets.Set[structured.DeviceID] {
	r.mutex.Lock()
	devices, ok := r.nodeDevices[nodeName]
	r.mutex.Unlock()
	if ok {
		return devices
	}

	r.buildIndex()
	devices = sets.New(r.perDeviceNodeDevices[nodeName]...)
	for _, slice := range r.slicesByNode[nodeName] {
		for _, device := range slice.Spec.Devices {
			devices.Insert(structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name})
		}
	}

	r.mutex.Lock()
	defer r.mutex.Unlock()
	if r.nodeDevices == nil {
		r.nodeDevices = make(map[string]sets.Set[structured.DeviceID])
	}
	r.nodeDevices[nodeName] = devices
	return devices
}

// sharesOfDevices returns how many shares of each device are allocated, nil
// if unknown. It gets computed once per scheduling cycle when needed.
func (r *preemptionRoot) sharesOfDevices() map[structured.DeviceID]int {
	if r.countShares == nil {
		return nil
	}
	r.mutex.Lock()
	defer r.mutex.Unlock()
	if !r.sharesKnown {
		r.shareCounts = r.countShares()
		r.sharesKnown = true
	}
	return r.shareCounts
}

// changesNode returns true if the simulation released devices on the node.
func (s *simulation) changesNode(nodeName string) bool {
	for _, released := range s.released {
		if released.nodeName == nodeName {
			return true
		}
	}
	return false
}

// allocatorForNode returns the allocator which Filter must use for the node
// and, if DRAPreemption is enabled, the allocated state it is based on.
func (pl *DynamicResources) allocatorForNode(ctx context.Context, state *stateData, nodeName string) (structured.Allocator, *structured.AllocatedState, error) {
	p := state.preemption
	if p == nil {
		return state.allocator, nil, nil
	}
	if !p.sim.changesNode(nodeName) {
		return state.allocator, p.root.allocatedState, nil
	}
	// The PreFilter allocator cannot be reused because allocators cache
	// counters derived from the allocated state. A new allocator for all
	// slices would be expensive, so it only gets the slices which can be
	// relevant for the node and the allocated devices in them.
	nodeSlices := p.root.slicesForNode(nodeName)
	allocatedState := p.root.allocatedStateForSlices(nodeSlices)
	p.sim.applyTo(allocatedState, pl.fts.EnableDRAConsumableCapacity, p.root.sharesOfDevices)
	allocator, err := structured.NewAllocator(ctx, AllocatorFeatures(pl.fts), *allocatedState, pl.draManager.DeviceClasses(), nodeSlices, pl.celCache)
	if err != nil {
		return nil, nil, err
	}
	return allocator, allocatedState, nil
}

// slicesForNode returns the slices for the node and all slices which are not
// tied to a single node by spec.nodeName. Pools which span several nodes
// through spec.nodeName then look incomplete and are not used, which is
// conservative.
func (r *preemptionRoot) slicesForNode(nodeName string) []*draapi.ResourceSlice {
	r.buildIndex()
	return slices.Concat(r.slicesByNode[nodeName], r.otherSlices)
}

// allocatedStateForSlices returns a copy of the PreFilter allocated state,
// limited to the devices in the slices.
func (r *preemptionRoot) allocatedStateForSlices(resourceSlices []*draapi.ResourceSlice) *structured.AllocatedState {
	state := &structured.AllocatedState{
		AllocatedDevices:         sets.New[structured.DeviceID](),
		AllocatedSharedDeviceIDs: sets.New[structured.DeviceID](),
		AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
	}
	for _, slice := range resourceSlices {
		for _, device := range slice.Spec.Devices {
			id := structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
			if r.allocatedState.AllocatedDevices.Has(id) {
				state.AllocatedDevices.Insert(id)
			}
			if r.allocatedState.AllocatedSharedDeviceIDs.Has(id) {
				state.AllocatedSharedDeviceIDs.Insert(id)
			}
			if capacity, ok := r.allocatedState.AggregatedCapacity[id]; ok {
				state.AggregatedCapacity[id] = capacity.Clone()
			}
		}
	}
	return state
}

// applyTo removes the released devices from the allocated state. Devices
// which are not in the state are ignored.
func (s *simulation) applyTo(state *structured.AllocatedState, consumableCapacity bool, sharesOfDevices func() map[structured.DeviceID]int) {
	releasedShares := make(map[structured.DeviceID]int)
	for _, released := range s.released {
		// Same handling of shared devices as in the claim tracker.
		foreachAllocatedDevice(&resourceapi.ResourceClaim{Status: resourceapi.ResourceClaimStatus{Allocation: released.allocation}},
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
	if len(releasedShares) == 0 {
		return
	}
	// A device stays shared unless all of its shares are known to be released.
	shares := sharesOfDevices()
	for id, numReleased := range releasedShares {
		if total, ok := shares[id]; ok && total <= numReleased {
			state.AllocatedSharedDeviceIDs.Delete(id)
		}
	}
}

// preemptionMightHelp is called by Filter when the allocator found no devices
// for the claims of the pod on the node, with the allocated state that the
// allocator used. Preemption only considers nodes where Filter returned
// Unschedulable, so this decides between that and UnschedulableAndUnresolvable.
//
// Removing pods might help if the pod can get devices of removed pods, some
// pod on the node with lower priority uses claims (only those can be
// preempted), and some device which is local to the node is allocated (only
// those get released).
func preemptionMightHelp(state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocatedState *structured.AllocatedState) bool {
	p := state.preemption
	if p == nil || !p.root.canRelease || allocatedState == nil {
		return false
	}
	priority := corev1helpers.PodPriority(pod)
	if !slices.ContainsFunc(nodeInfo.GetPods(), func(podInfo fwk.PodInfo) bool {
		other := podInfo.GetPod()
		return len(other.Spec.ResourceClaims) > 0 && corev1helpers.PodPriority(other) < priority
	}) {
		return false
	}
	return p.root.hasAllocatedNodeLocalDevice(nodeInfo.Node().Name, allocatedState)
}

// hasAllocatedNodeLocalDevice returns true if some device which is local to the
// node is allocated in the given state.
func (r *preemptionRoot) hasAllocatedNodeLocalDevice(nodeName string, state *structured.AllocatedState) bool {
	for id := range r.nodeLocalDevices(nodeName) {
		if structured.IsDeviceAllocated(id, state) {
			return true
		}
	}
	return false
}

// sharedDeviceCounter is implemented by claimTracker.
type sharedDeviceCounter interface {
	// countSharedDevices returns the number of allocated and in-flight
	// shares per device.
	countSharedDevices() map[structured.DeviceID]int
}

var _ sharedDeviceCounter = &claimTracker{}

func (c *claimTracker) countSharedDevices() map[structured.DeviceID]int {
	counts := c.allocatedDevices.countShares()
	for _, inFlight := range c.allInFlightAllocationsRLocked() {
		foreachAllocatedDevice(inFlight.claim, func(structured.DeviceID) {}, true,
			func(id structured.SharedDeviceID) {
				counts[id.GetDeviceID()]++
			},
			func(structured.DeviceConsumedCapacity) {})
	}
	return counts
}

func (a *allocatedDevices) countShares() map[structured.DeviceID]int {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	counts := make(map[structured.DeviceID]int)
	for id := range a.shareIDs {
		counts[id.GetDeviceID()]++
	}
	return counts
}
