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
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func preemptionPod(name string, claimNames ...string) *st.PodWrapper {
	pod := st.MakePod().Name(name).Namespace(namespace).UID(name)
	for _, claimName := range claimNames {
		pod = pod.PodResourceClaims(v1.PodResourceClaim{Name: claimName, ResourceClaimName: new(claimName)})
	}
	return pod
}

func fromPod(pod *v1.Pod) *st.PodWrapper {
	return &st.PodWrapper{Pod: *pod.DeepCopy()}
}

// preemptionClaim creates a ResourceClaim. Names of the form <pod>-claim<suffix>
// set pod <pod><suffix> (from preemptionPod) as controller owner, mimicking
// ResourceClaimTemplate-generated claims required by alpha preemption; other
// names create standalone claims.
func preemptionClaim(name string) *st.ResourceClaimWrapper {
	claim := st.MakeResourceClaim().Name(name).Namespace(namespace).Request(className)
	if prefix, suffix, ok := strings.Cut(name, "-claim"); ok {
		claim = claim.OwnerReference(prefix+suffix, prefix+suffix, podKind)
	}
	return claim
}

func preemptionAllocation(devices ...string) *resourceapi.AllocationResult {
	return poolAllocation(nodeName, devices...)
}

func poolAllocation(pool string, devices ...string) *resourceapi.AllocationResult {
	allocation := &resourceapi.AllocationResult{}
	for _, device := range devices {
		allocation.Devices.Results = append(allocation.Devices.Results, resourceapi.DeviceRequestAllocationResult{
			Driver: driver, Pool: pool, Device: device, Request: "req-1",
		})
	}
	return allocation
}

// withShares turns the results of allocation for devices into shares, as
// allocated with DRAConsumableCapacity.
func withShares(allocation *resourceapi.AllocationResult, devices ...string) *resourceapi.AllocationResult {
	for i := range allocation.Devices.Results {
		result := &allocation.Devices.Results[i]
		if slices.Contains(devices, result.Device) {
			result.ShareID = new(types.UID("share-" + result.Device))
		}
	}
	return allocation
}

// setupPreemption passes empty DynamicResourcesArgs so tests don't have to
// enable unrelated feature gates required by setup's default args.
func setupPreemption(tCtx ktesting.TContext, nodes []*v1.Node, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object, features feature.Features) *testContext {
	tCtx.Helper()
	return setup(tCtx, &config.DynamicResourcesArgs{}, nodes, claims, []*resourceapi.DeviceClass{deviceClass}, nil, objs, features, false /* failPatch */, nil)
}

func consumerRef(pod *v1.Pod) resourceapi.ResourceClaimConsumerReference {
	return resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: pod.Name, UID: pod.UID}
}

func TestPreemptionSimulation(t *testing.T) {
	testPreemptionSimulation(ktesting.Init(t))
}

func testPreemptionSimulation(tCtx ktesting.TContext) {
	twoDeviceSlice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	oneDeviceSlice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()

	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	preemptorClaim := preemptionClaim("preemptor-claim").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victimClaim := preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj()
	allocatedForVictim := reserve(victimClaim, victim)
	victimShare := reserve(preemptionClaim("victim-claim").Allocation(withShares(preemptionAllocation("instance-1"), "instance-1")).Obj(), victim)
	victimExclusiveAndShare := reserve(preemptionClaim("victim-claim").Allocation(withShares(preemptionAllocation("instance-1", "instance-2"), "instance-2")).Obj(), victim)
	var spanningPool []*resourceapi.ResourceSlice
	for _, s := range []struct{ node, device string }{{nodeName, "instance-1"}, {node2Name, "instance-2"}} {
		slice := st.MakeResourceSlice(s.node, driver).Device(s.device).ResourceSliceCount(2).Obj()
		slice.Name = "spanning-" + s.node
		slice.Spec.Pool.Name = "spanning"
		spanningPool = append(spanningPool, slice)
	}
	partitionableSlices := func() []*resourceapi.ResourceSlice {
		counters := st.MakeResourceSlice(nodeName, driver).ResourceSliceCount(2).Obj()
		counters.Name += "-counters"
		counters.Spec.SharedCounters = []resourceapi.CounterSet{{
			Name:     "gpu-0",
			Counters: map[string]resourceapi.Counter{"memory": {Value: apiresource.MustParse("8")}},
		}}
		devices := st.MakeResourceSlice(nodeName, driver).ResourceSliceCount(2).Obj()
		consumes := func(quantity string) []resourceapi.DeviceCounterConsumption {
			return []resourceapi.DeviceCounterConsumption{{
				CounterSet: "gpu-0",
				Counters:   map[string]resourceapi.CounterRequirement{"memory": {Value: new(apiresource.MustParse(quantity))}},
			}}
		}
		whole := func(whole bool) map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
			return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"whole": {BoolValue: new(whole)}}
		}
		devices.Spec.Devices = []resourceapi.Device{
			{Name: "gpu-0", ConsumesCounters: consumes("8"), Attributes: whole(true)},
			{Name: "gpu-0-part-a", ConsumesCounters: consumes("4"), Attributes: whole(false)},
			{Name: "gpu-0-part-b", ConsumesCounters: consumes("4"), Attributes: whole(false)},
		}
		return []*resourceapi.ResourceSlice{counters, devices}
	}()
	wholeDevicePreemptorClaim := preemptionClaim("preemptor-claim").Obj()
	wholeDevicePreemptorClaim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["` + driver + `"].whole`},
	}}

	type step struct {
		remove, add  *v1.Pod
		nodeInfo     int
		wantReleased []string
		wantFit      bool
	}
	testcases := map[string]struct {
		features  feature.Features
		gates     featuregatetesting.FeatureOverrides
		preemptor *v1.Pod
		nodes     []*v1.Node
		pods      []*v1.Pod
		claims    []*resourceapi.ResourceClaim
		pending   map[string]*resourceapi.AllocationResult
		slices    []*resourceapi.ResourceSlice
		wantFit   bool
		steps     []step
	}{
		"exclusive": {
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
				{add: victim},
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"preemption-policy-never": {
			preemptor: preemptionPod("preemptor", "preemptor-claim").PreemptionPolicy(v1.PreemptNever).Obj(),
			pods:      []*v1.Pod{victim},
			claims:    []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices:    []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"preemptor-in-pod-group": {
			features:  feature.Features{EnableGenericWorkload: true},
			preemptor: preemptionPod("preemptor", "preemptor-claim").PodGroupName(podGroupName).Obj(),
			pods:      []*v1.Pod{victim},
			claims:    []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices:    []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"preemptor-uses-victim-claim": {
			preemptor: preemptionPod("preemptor", "preemptor-claim", "victim-claim").Obj(),
			pods:      []*v1.Pod{victim, preemptionPod("other", "other-claim").Node(nodeName).Obj()},
			claims: []*resourceapi.ResourceClaim{
				preemptorClaim,
				allocatedForVictim,
				reserve(preemptionClaim("other-claim").Allocation(preemptionAllocation("instance-2")).Obj(), preemptionPod("other").Obj()),
			},
			slices: []*resourceapi.ResourceSlice{twoDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"shared-device": {
			features: feature.Features{EnableDRAConsumableCapacity: true},
			pods:     []*v1.Pod{victim},
			claims:   []*resourceapi.ResourceClaim{preemptorClaim, victimShare},
			slices:   []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"shared-device-without-consumable-capacity": {
			// The scheduler counts the share as an exclusively allocated
			// device, which other claims may still share.
			gates:  featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, victimShare},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"exclusive-and-shared-device": {
			features: feature.Features{EnableDRAConsumableCapacity: true},
			pods:     []*v1.Pod{victim},
			claims:   []*resourceapi.ResourceClaim{preemptorClaim, victimExclusiveAndShare},
			slices:   []*resourceapi.ResourceSlice{twoDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"exclusive-and-shared-device-both-needed": {
			// Releasing the claim frees instance-1, but its share keeps
			// instance-2 allocated.
			features: feature.Features{EnableDRAConsumableCapacity: true},
			pods:     []*v1.Pod{victim},
			claims:   []*resourceapi.ResourceClaim{preemptionClaim("preemptor-claim").Request(className).Obj(), victimExclusiveAndShare},
			slices:   []*resourceapi.ResourceSlice{twoDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}},
			},
		},
		"admin-access": {
			pods: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, reserve(preemptionClaim("victim-claim").Allocation(func() *resourceapi.AllocationResult {
				allocation := preemptionAllocation("instance-1")
				allocation.Devices.Results[0].AdminAccess = new(true)
				return allocation
			}()).Obj(), victim)},
			slices:  []*resourceapi.ResourceSlice{oneDeviceSlice},
			wantFit: true,
			steps: []step{
				{remove: victim, wantFit: true},
			},
		},
		"also-reserved-for-other-pod": {
			// A pod owns the claim, but another pod shares it by name.
			pods: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserveFor(victimClaim, consumerRef(victim), consumerRef(preemptionPod("other").Obj()))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"pending-allocation": {
			pods:    []*v1.Pod{victim},
			claims:  []*resourceapi.ResourceClaim{preemptorClaim, preemptionClaim("victim-claim").Obj()},
			pending: map[string]*resourceapi.AllocationResult{"victim-claim": preemptionAllocation("instance-1")},
			slices:  []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"victim-in-pod-group": {
			features: feature.Features{EnableGenericWorkload: true},
			pods:     []*v1.Pod{fromPod(victim).PodGroupName(podGroupName).Obj()},
			claims:   []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices:   []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: fromPod(victim).PodGroupName(podGroupName).Obj()},
			},
		},
		"victim-with-scheduling-group-without-generic-workload": {
			pods:   []*v1.Pod{fromPod(victim).PodGroupName(podGroupName).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: fromPod(victim).PodGroupName(podGroupName).Obj(), wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"allocated-unreserved": {
			// Garbage-collected with its owner.
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, victimClaim},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"standalone-claim": {
			pods: []*v1.Pod{preemptionPod("victim", "standalone").Node(nodeName).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("standalone").Allocation(preemptionAllocation("instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: preemptionPod("victim", "standalone").Node(nodeName).Obj()},
			},
		},
		"device-for-all-nodes": {
			pods: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation("network", "instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSliceWithAllNodes("network", driver).Device("instance-1").Obj()},
			steps: []step{
				{remove: victim},
			},
		},
		"device-with-node-name": {
			features: feature.Features{EnableDRAPartitionableDevices: true},
			pods:     []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation("per-device", "instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSliceWithPerDeviceNodeSelection("per-device", driver).Device("instance-1", st.NodeName(nodeName)).Obj()},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"device-with-other-node-name": {
			features: feature.Features{EnableDRAPartitionableDevices: true},
			pods:     []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation("per-device", "instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSliceWithPerDeviceNodeSelection("per-device", driver).Device("instance-1", st.NodeName(node2Name)).Obj()},
			steps: []step{
				{remove: victim},
			},
		},
		"victim-claim-missing": {
			pods:   []*v1.Pod{victim, preemptionPod("other", "other-claim").Node(nodeName).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, reserve(preemptionClaim("other-claim").Allocation(preemptionAllocation("instance-1")).Obj(), preemptionPod("other").Obj())},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"pool-spanning-nodes": {
			// The allocator for the node needs the slices of both nodes to
			// see the pool of the victim's device as complete.
			nodes:  []*v1.Node{workerNode, workerNode2},
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, reserve(preemptionClaim("victim-claim").Allocation(poolAllocation("spanning", "instance-1")).Obj(), victim)},
			slices: spanningPool,
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"victim-on-other-node": {
			nodes: []*v1.Node{workerNode, workerNode2},
			pods:  []*v1.Pod{fromPod(victim).Node(node2Name).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation(node2Name, "instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSlice(node2Name, driver).Device("instance-1").Obj()},
			steps: []step{
				// Releasing a claim on node 2 leaves Filter on node 1 failing.
				{remove: fromPod(victim).Node(node2Name).Obj(), nodeInfo: 1, wantReleased: []string{"victim-claim"}},
			},
		},
		"partitionable-devices": {
			// Releasing a partition frees the shared counters needed to allocate the whole device.
			features: feature.Features{EnableDRAPartitionableDevices: true},
			pods:     []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{
				wholeDevicePreemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0-part-a")).Obj(), victim),
			},
			slices: partitionableSlices,
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate, tc.gates)
			features := tc.features
			features.EnableDRAPreemption = true
			nodes := tc.nodes
			if nodes == nil {
				nodes = []*v1.Node{workerNode}
			}
			objs := []apiruntime.Object{}
			for _, slice := range tc.slices {
				objs = append(objs, slice)
			}
			for _, pod := range tc.pods {
				objs = append(objs, pod)
			}
			testCtx := setupPreemption(tCtx, nodes, tc.claims, objs, features)
			// Populate NodeInfo with non-terminal assigned pods (the scheduler's
			// pod informer drops Succeeded/Failed pods).
			for _, pod := range tc.pods {
				if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
					continue
				}
				podInfo, err := framework.NewPodInfo(pod)
				tCtx.ExpectNoError(err)
				for _, nodeInfo := range testCtx.nodeInfos {
					if nodeInfo.Node().Name == pod.Spec.NodeName {
						nodeInfo.AddPodInfo(podInfo)
					}
				}
			}
			for claimName, allocation := range tc.pending {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, claimName)
				tCtx.ExpectNoError(err, "get claim with pending allocation")
				claim = claim.DeepCopy()
				claim.Status.Allocation = allocation
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim), "add pending allocation")
			}
			pod := tc.preemptor
			if pod == nil {
				pod = preemptor
			}

			_, status := testCtx.p.PreFilter(tCtx, testCtx.state, pod, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			require.NotNil(tCtx, testCtx.p.PreFilterExtensions(), "PreFilterExtensions")
			fits := func(cs fwk.CycleState) bool {
				status := testCtx.p.Filter(tCtx, cs, pod, testCtx.nodeInfos[0])
				require.True(tCtx, status.IsSuccess() || status.IsRejected(), "Filter: %v", status)
				return status.IsSuccess()
			}
			require.Equal(tCtx, tc.wantFit, fits(testCtx.state), "Filter before preemption")

			cs := testCtx.state.Clone()
			for i, step := range tc.steps {
				nodeInfo := testCtx.nodeInfos[step.nodeInfo]
				if step.remove != nil {
					podInfo, err := framework.NewPodInfo(step.remove)
					tCtx.ExpectNoError(err)
					require.True(tCtx, testCtx.p.RemovePod(tCtx, cs, pod, podInfo, nodeInfo).IsSuccess(), "step %d: RemovePod", i)
				}
				if step.add != nil {
					podInfo, err := framework.NewPodInfo(step.add)
					tCtx.ExpectNoError(err)
					require.True(tCtx, testCtx.p.AddPod(tCtx, cs, pod, podInfo, nodeInfo).IsSuccess(), "step %d: AddPod", i)
				}
				var released []string
				for _, r := range getPreemptionSimulationState(cs).released {
					released = append(released, r.claim.Name)
				}
				slices.Sort(released)
				require.Equal(tCtx, step.wantReleased, released, "step %d: released claims", i)
				require.Equal(tCtx, step.wantFit, fits(cs), "step %d: Filter", i)
			}

			require.Equal(tCtx, tc.wantFit, fits(testCtx.state), "Filter of the original state after preemption")
		})
	}
}

// TestPreemptionSimulationStateClone verifies that cloning CycleState shares stateData
// while isolating preemptionSimulationState mutations.
func TestPreemptionSimulationStateClone(t *testing.T) {
	claim := &releasedClaim{uid: "claim", podUID: "victim"}
	original := &preemptionSimulationState{
		released: map[types.UID]*releasedClaim{"claim": claim},
	}
	state := &stateData{preemption: &preemptionData{}}
	cs := framework.NewCycleState()
	cs.Write(stateKey, state)
	cs.Write(preemptionSimulationStateKey, original)

	copied := cs.Clone()
	sim := getPreemptionSimulationState(copied)
	require.NotSame(t, original, sim, "simulation of the copy")
	sim.released["other"] = &releasedClaim{uid: "other"}
	delete(sim.released, "claim")
	require.Equal(t, map[types.UID]*releasedClaim{"claim": claim}, original.released, "released claims of the original")

	copiedState, err := getStateData(copied)
	require.NoError(t, err)
	require.Same(t, state, copiedState, "stateData of the copy")
}

// TestSimulationApplyTo verifies that released claims free only their
// exclusively allocated devices. Shares, their consumed capacity, and their
// consumed counters stay counted, even when DRAConsumableCapacity is off and
// shares are tracked in AllocatedDevices.
func TestSimulationApplyTo(t *testing.T) {
	quantity := apiresource.MustParse("60Gi")
	released := withShares(preemptionAllocation("gpu-0", "gpu-1"), "gpu-1")
	share := &released.Devices.Results[1]
	share.ConsumedCapacity = map[resourceapi.QualifiedName]apiresource.Quantity{"memory": quantity}
	share.ConsumedCounters = []resourceapi.CounterSetConsumption{{CounterSet: "gpu-1", Counters: map[string]apiresource.Quantity{"memory": quantity}}}
	remaining := &resourceapi.AllocationResult{}
	remaining.Devices.Results = []resourceapi.DeviceRequestAllocationResult{*share}
	// The released claim only has admin access to gpu-2, which another claim
	// has allocated exclusively.
	other := preemptionAllocation("gpu-2")
	admin := other.Devices.Results[0]
	admin.AdminAccess = new(true)
	released.Devices.Results = append(released.Devices.Results, admin)
	sim := &preemptionSimulationState{released: map[types.UID]*releasedClaim{"claim": {allocatedDevices: &released.Devices}}}

	for name, features := range map[string]structured.Features{
		"consumable-capacity":         {ConsumableCapacity: true, SharedConsumableCapacity: true},
		"without-consumable-capacity": {},
	} {
		t.Run(name, func(t *testing.T) {
			state, want := &structured.AllocatedState{}, &structured.AllocatedState{}
			structured.AddToAllocatedState(state, slices.Values([]*resourceapi.AllocationResult{released, other}), features)
			structured.AddToAllocatedState(want, slices.Values([]*resourceapi.AllocationResult{remaining, other}), features)
			sim.applyTo(state)
			require.Equal(t, want, state)
		})
	}
}

func TestCanPreemptDevices(t *testing.T) {
	testCanPreemptDevices(ktesting.Init(t))
}

func testCanPreemptDevices(tCtx ktesting.TContext) {
	allocatedState := &structured.AllocatedState{}
	for name, tc := range map[string]struct {
		features       feature.Features
		pod            *v1.Pod
		claims         []*resourceapi.ResourceClaim
		allocatedState *structured.AllocatedState
		extended       bool
		want           bool
	}{
		"regular": {
			pod:            preemptionPod("pod").Obj(),
			allocatedState: allocatedState,
			want:           true,
		},
		"standalone-claim-to-allocate": {
			pod:            preemptionPod("pod").Obj(),
			claims:         []*resourceapi.ResourceClaim{preemptionClaim("pod-claim").Obj(), preemptionClaim("standalone").Obj()},
			allocatedState: allocatedState,
		},
		"allocated-standalone-claim": {
			pod:            preemptionPod("pod").Obj(),
			claims:         []*resourceapi.ResourceClaim{preemptionClaim("pod-claim").Obj(), preemptionClaim("standalone").Allocation(preemptionAllocation("instance-1")).Obj()},
			allocatedState: allocatedState,
			want:           true,
		},
		"nothing-to-allocate": {
			pod: preemptionPod("pod").Obj(),
		},
		"preemption-policy-never": {
			pod:            preemptionPod("pod").PreemptionPolicy(v1.PreemptNever).Obj(),
			allocatedState: allocatedState,
		},
		"extended-resources": {
			pod:            preemptionPod("pod").Obj(),
			allocatedState: allocatedState,
			extended:       true,
		},
		"pod-group": {
			features:       feature.Features{EnableGenericWorkload: true},
			pod:            preemptionPod("pod").PodGroupName(podGroupName).Obj(),
			allocatedState: allocatedState,
		},
		"pod-group-disabled": {
			pod:            preemptionPod("pod").PodGroupName(podGroupName).Obj(),
			allocatedState: allocatedState,
			want:           true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupPreemption(tCtx, nil, nil, nil, tc.features)
			var extendedResourceClaim *resourceapi.ResourceClaim
			if tc.extended {
				extendedResourceClaim = &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: specialClaimInMemName, UID: "extended"}}
			}
			podClaims := tc.claims
			if podClaims == nil {
				podClaims = []*resourceapi.ResourceClaim{preemptionClaim("pod-claim").Obj()}
			}
			claims := newClaimStore(podClaims, extendedResourceClaim, nil)
			data := testCtx.p.newPreemptionData(tc.pod, &claims, tc.allocatedState, nil)
			require.Equal(tCtx, tc.want, data.canPreemptDevices)
		})
	}
}

func TestPreemptionGateDisabled(t *testing.T) {
	testPreemptionGateDisabled(ktesting.Init(t))
}

func testPreemptionGateDisabled(tCtx ktesting.TContext) {
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	objs := []apiruntime.Object{st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj(), victim}
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, objs, feature.Features{})

	preemptor := preemptionPod("preemptor", "preemptor-claim").Obj()
	_, status := testCtx.p.PreFilter(tCtx, testCtx.state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	require.Nil(tCtx, testCtx.p.PreFilterExtensions(), "PreFilterExtensions")
	state, err := getStateData(testCtx.state)
	tCtx.ExpectNoError(err)
	require.Same(tCtx, state, state.Clone(), "Clone")
	require.Nil(tCtx, getPreemptionSimulationState(testCtx.state), "simulation state")
	require.Equal(tCtx, fwk.UnschedulableAndUnresolvable, testCtx.p.Filter(tCtx, testCtx.state, preemptor, testCtx.nodeInfos[0]).Code(), "Filter")
}

// TestAllocatedStateForSlices verifies that allocatedStateForSlices reduces
// AllocatedDevices to the devices in the slices, in a set of its own, and
// shares all other fields with PreFilter's AllocatedState.
func TestAllocatedStateForSlices(t *testing.T) {
	exclusive := structured.MakeDeviceID(driver, nodeName, "gpu-0")
	shared := structured.MakeDeviceID(driver, nodeName, "gpu-1")
	removed := structured.MakeDeviceID(driver, nodeName, "gpu-2")
	elsewhere := structured.MakeDeviceID(driver, "other-pool", "gpu-0")
	capacities := func(ids ...structured.DeviceID) structured.ConsumedCapacityCollection {
		capacity := structured.NewConsumedCapacityCollection()
		for _, id := range ids {
			capacity.Insert(structured.NewDeviceConsumedCapacity(id, map[resourceapi.QualifiedName]apiresource.Quantity{"memory": apiresource.MustParse("1Gi")}))
		}
		return capacity
	}
	consumption := []schedulerapi.CounterConsumptionForAllocation{{{CounterSet: "gpu", Counters: map[string]apiresource.Quantity{"memory": apiresource.MustParse("1Gi")}}}}
	data := &preemptionData{allocatedState: &structured.AllocatedState{
		AllocatedDevices:         sets.New(exclusive, elsewhere),
		AllocatedSharedDeviceIDs: sets.New(shared, elsewhere),
		AggregatedCapacity:       capacities(shared, elsewhere),
		ConsumedCounters:         schedulerapi.CounterConsumptionByDevice{shared: consumption, removed: consumption, elsewhere: consumption},
	}}
	want := *data.allocatedState
	want.AllocatedDevices = sets.New(exclusive)
	resourceSlices := apiSlices(st.MakeResourceSlice(nodeName, driver).Devices("gpu-0", "gpu-1").Obj())
	got := data.allocatedStateForSlices(resourceSlices)
	require.Equal(t, &want, got)
	got.AllocatedDevices.Delete(exclusive)
	require.Equal(t, sets.New(exclusive, elsewhere), data.allocatedState.AllocatedDevices, "AllocatedDevices of PreFilter")
}

// TestNodeSlicesLocalDevices checks that a device listed only in an older
// generation of its pool is not treated as local, because the allocator ignores
// older generations.
func TestNodeSlicesLocalDevices(t *testing.T) {
	makeSlice := func(name string, generation int64, devices ...string) *resourceapi.ResourceSlice {
		slice := st.MakeResourceSlice(nodeName, driver).Devices(devices...).Obj()
		slice.Name = name
		slice.Spec.Pool.Generation = generation
		return slice
	}
	s := &nodeSlices{slices: apiSlices(
		makeSlice("old", 1, "gpu-0", "gpu-1"),
		makeSlice("new", 2, "gpu-0"),
	)}
	require.Equal(t, sets.New(structured.MakeDeviceID(driver, nodeName, "gpu-0")), s.localDevices(nodeName))
}

// TestNodeSlicesForNode checks that an allocator for a node gets all slices of
// the pools it may use, and none of the pools of other nodes.
func TestNodeSlicesForNode(t *testing.T) {
	makeSlice := func(name, pool, node string, generation int64) *resourceapi.ResourceSlice {
		wrapper := st.MakeResourceSliceWithAllNodes(pool, driver)
		if node != "" {
			wrapper = st.MakeResourceSlice(node, driver)
		}
		slice := wrapper.Obj()
		slice.Name = name
		slice.Spec.Pool.Name = pool
		slice.Spec.Pool.Generation = generation
		return slice
	}
	s := &nodeSlices{slices: apiSlices(
		makeSlice("node1-only", "node1-only", nodeName, 1),
		makeSlice("node2-only", "node2-only", node2Name, 1),
		makeSlice("spanning-1", "spanning", nodeName, 1),
		makeSlice("spanning-2", "spanning", node2Name, 1),
		makeSlice("shared-node2", "shared", node2Name, 1),
		makeSlice("shared-all-nodes", "shared", "", 1),
		// The allocator recognizes an obsolete generation by a newer
		// one, which may only have slices for other nodes.
		makeSlice("updated-old", "updated", nodeName, 1),
		makeSlice("updated-new", "updated", node2Name, 2),
	)}
	for node, want := range map[string][]string{
		nodeName:              {"node1-only", "shared-all-nodes", "shared-node2", "spanning-1", "spanning-2", "updated-new", "updated-old"},
		node2Name:             {"node2-only", "shared-all-nodes", "shared-node2", "spanning-1", "spanning-2", "updated-new", "updated-old"},
		"node-without-slices": {"shared-all-nodes", "shared-node2"},
	} {
		t.Run(node, func(t *testing.T) {
			var got []string
			for _, slice := range s.forNode(node) {
				got = append(got, slice.Name)
			}
			slices.Sort(got)
			require.Equal(t, want, got)
		})
	}
}
