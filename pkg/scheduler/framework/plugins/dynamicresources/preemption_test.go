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
	"testing"

	"github.com/google/go-cmp/cmp"
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
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// preemptionPod returns a pod which references claims by name.
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

func preemptionClaim(name string) *st.ResourceClaimWrapper {
	return st.MakeResourceClaim().Name(name).Namespace(namespace).Request(className)
}

// preemptionAllocation allocates devices from the pool of the worker node.
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

// setupPreemption avoids the default plugin arguments because they depend on
// features which the tests don't need.
func setupPreemption(tCtx ktesting.TContext, nodes []*v1.Node, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object, features feature.Features) *testContext {
	return setup(tCtx, &config.DynamicResourcesArgs{}, nodes, claims, []*resourceapi.DeviceClass{deviceClass}, nil, objs, features, false, nil)
}

func consumerRef(pod *v1.Pod) resourceapi.ResourceClaimConsumerReference {
	return resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: pod.Name, UID: pod.UID}
}

func TestPreemptionSimulation(t *testing.T) {
	testPreemptionSimulation(ktesting.Init(t))
}

// testPreemptionSimulation covers the release rules of RemovePod, AddPod
// and how Filter and its status code reflect them.
func testPreemptionSimulation(tCtx ktesting.TContext) {
	twoDeviceSlice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	oneDeviceSlice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()

	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	preemptorClaim := preemptionClaim("preemptor-claim").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim").Node(nodeName).Obj()
	running := preemptionPod("running", "victim-claim").Node(nodeName).Obj()
	victimClaim := preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj()
	allocatedForVictim := reserve(victimClaim, victim)
	sharedVictimClaim := func(consumers ...resourceapi.ResourceClaimConsumerReference) *resourceapi.ResourceClaim {
		return st.FromResourceClaim(victimClaim).ReservedFor(consumers...).Obj()
	}

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
		"shared-with-running-pod": {
			pods:   []*v1.Pod{victim, running},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, sharedVictimClaim(consumerRef(victim), consumerRef(running))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
			},
		},
		"shared-with-terminating-pod": {
			pods: []*v1.Pod{victim, fromPod(running).Terminating().Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				sharedVictimClaim(consumerRef(victim), consumerRef(running))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"shared-with-completed-pod": {
			pods: []*v1.Pod{victim, fromPod(running).Phase(v1.PodSucceeded).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				sharedVictimClaim(consumerRef(victim), consumerRef(running))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"shared-with-deleted-pod": {
			pods: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				sharedVictimClaim(consumerRef(victim), consumerRef(running))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"shared-by-two-victims": {
			pods: []*v1.Pod{victim, victim2},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				sharedVictimClaim(consumerRef(victim), consumerRef(victim2))},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
				{remove: victim2, wantReleased: []string{"victim-claim"}, wantFit: true},
				{add: victim},
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
				{add: victim2},
			},
		},
		"used-by-pod-in-prebind": {
			// victim2 is about to reserve the claim, so adding it back
			// also restores the claim.
			pods:   []*v1.Pod{victim, victim2},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, allocatedForVictim},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
				{remove: victim2, wantReleased: []string{"victim-claim"}, wantFit: true},
				{add: victim2},
			},
		},
		"shared-device-without-consumable-capacity": {
			gates: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			pods:  []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, reserve(preemptionClaim("victim-claim").Allocation(func() *resourceapi.AllocationResult {
				allocation := preemptionAllocation("instance-1")
				allocation.Devices.Results[0].ShareID = new(types.UID("share"))
				return allocation
			}()).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
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
		"reserved-for-pod-group": {
			pods: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				sharedVictimClaim(consumerRef(victim), podGroupReservation)},
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
		"pending-allocation-in-pod-group": {
			features: feature.Features{EnableGenericWorkload: true},
			pods:     []*v1.Pod{fromPod(victim).PodGroupName(podGroupName).Obj()},
			claims:   []*resourceapi.ResourceClaim{preemptorClaim, preemptionClaim("victim-claim").Obj()},
			pending:  map[string]*resourceapi.AllocationResult{"victim-claim": preemptionAllocation("instance-1")},
			slices:   []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: fromPod(victim).PodGroupName(podGroupName).Obj()},
			},
		},
		"allocated-unreserved-owned-by-victim": {
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, st.FromResourceClaim(victimClaim).OwnerReference(victim.Name, string(victim.UID), podKind).Obj()},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim, wantReleased: []string{"victim-claim"}, wantFit: true},
			},
		},
		"allocated-unreserved-not-owned": {
			pods:   []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{preemptorClaim, victimClaim},
			slices: []*resourceapi.ResourceSlice{oneDeviceSlice},
			steps: []step{
				{remove: victim},
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
		"victim-on-other-node": {
			nodes: []*v1.Node{workerNode, workerNode2},
			pods:  []*v1.Pod{fromPod(victim).Node(node2Name).Obj()},
			claims: []*resourceapi.ResourceClaim{preemptorClaim,
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation(node2Name, "instance-1")).Obj(), victim)},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSlice(node2Name, driver).Device("instance-1").Obj()},
			steps: []step{
				// Released on its own node, nothing changes on the first node.
				{remove: fromPod(victim).Node(node2Name).Obj(), nodeInfo: 1, wantReleased: []string{"victim-claim"}},
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
				state, err := getStateData(cs)
				tCtx.ExpectNoError(err)
				var released []string
				for _, r := range state.preemption.sim.released {
					released = append(released, r.claim.Name)
				}
				slices.Sort(released)
				require.Equal(tCtx, step.wantReleased, released, "step %d: released claims", i)
				require.Equal(tCtx, step.wantFit, fits(cs), "step %d: Filter", i)
			}

			// The copies did not modify the original state.
			require.Equal(tCtx, tc.wantFit, fits(testCtx.state), "Filter of the original state after preemption")
		})
	}
}

func TestPreemptionConsumableCapacity(t *testing.T) {
	testPreemptionConsumableCapacity(ktesting.Init(t))
}

// testPreemptionConsumableCapacity checks that releasing one share of a
// device frees its capacity.
func testPreemptionConsumableCapacity(tCtx ktesting.TContext) {
	const memory = resourceapi.QualifiedName("memory")
	slice := st.MakeResourceSlice(nodeName, driver).
		Device("gpu-0", map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{memory: {Value: apiresource.MustParse("80Gi")}}).
		Obj()
	slice.Spec.Devices[0].AllowMultipleAllocations = new(true)
	share := func(shareID types.UID, quantity string) *resourceapi.AllocationResult {
		allocation := preemptionAllocation("gpu-0")
		allocation.Devices.Results[0].ShareID = &shareID
		allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)}
		return allocation
	}
	withCapacity := func(claim *resourceapi.ResourceClaim, quantity string) *resourceapi.ResourceClaim {
		claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)},
		}
		return claim
	}

	preemptor := preemptionPod("preemptor", "preemptor-claim").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	running := preemptionPod("running", "running-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		withCapacity(preemptionClaim("preemptor-claim").Obj(), "40Gi"),
		reserve(withCapacity(preemptionClaim("victim-claim").Allocation(share("share-1", "60Gi")).Obj(), "60Gi"), victim),
		reserve(withCapacity(preemptionClaim("running-claim").Allocation(share("share-2", "20Gi")).Obj(), "20Gi"), running),
	}
	features := feature.Features{EnableDRAPreemption: true, EnableDRAConsumableCapacity: true}
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, []apiruntime.Object{slice, victim, running}, features)

	_, status := testCtx.p.PreFilter(tCtx, testCtx.state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	nodeInfo := testCtx.nodeInfos[0]
	require.True(tCtx, testCtx.p.Filter(tCtx, testCtx.state, preemptor, nodeInfo).IsRejected(), "Filter before preemption")

	cs := testCtx.state.Clone()
	podInfo, err := framework.NewPodInfo(victim)
	tCtx.ExpectNoError(err)
	require.True(tCtx, testCtx.p.RemovePod(tCtx, cs, preemptor, podInfo, nodeInfo).IsSuccess(), "RemovePod")
	require.True(tCtx, testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo).IsSuccess(), "Filter after removing the victim")
}

func TestPreemptionPartitionableDevices(t *testing.T) {
	testPreemptionPartitionableDevices(ktesting.Init(t))
}

// testPreemptionPartitionableDevices checks that releasing a partition frees
// the counters it consumed: the whole device cannot be allocated before.
func testPreemptionPartitionableDevices(tCtx ktesting.TContext) {
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
			Counters:   map[string]resourceapi.Counter{"memory": {Value: apiresource.MustParse(quantity)}},
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

	preemptorClaim := preemptionClaim("preemptor-claim").Obj()
	preemptorClaim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["` + driver + `"].whole`},
	}}
	preemptor := preemptionPod("preemptor", "preemptor-claim").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptorClaim,
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0-part-a")).Obj(), victim),
	}
	features := feature.Features{EnableDRAPreemption: true, EnableDRAPartitionableDevices: true}
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, []apiruntime.Object{counters, devices, victim}, features)

	_, status := testCtx.p.PreFilter(tCtx, testCtx.state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	nodeInfo := testCtx.nodeInfos[0]
	status = testCtx.p.Filter(tCtx, testCtx.state, preemptor, nodeInfo)
	require.True(tCtx, status.IsRejected(), "Filter before preemption: %v", status)

	cs := testCtx.state.Clone()
	podInfo, err := framework.NewPodInfo(victim)
	tCtx.ExpectNoError(err)
	require.True(tCtx, testCtx.p.RemovePod(tCtx, cs, preemptor, podInfo, nodeInfo).IsSuccess(), "RemovePod")
	status = testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo)
	require.True(tCtx, status.IsSuccess(), "Filter after removing the victim: %v", status)
}

// TestSimulationApplyTo checks the handling of shared devices: a device stays
// shared unless all of its shares are known to be released.
func TestSimulationApplyTo(t *testing.T) {
	const memory = resourceapi.QualifiedName("memory")
	device := structured.MakeDeviceID(driver, nodeName, "gpu-0")
	exclusive := structured.MakeDeviceID(driver, nodeName, "gpu-1")
	share := func(shareID types.UID, quantity string) *releasedClaim {
		allocation := preemptionAllocation("gpu-0")
		allocation.Devices.Results[0].ShareID = &shareID
		allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)}
		return &releasedClaim{allocation: allocation}
	}
	newState := func() *structured.AllocatedState {
		capacity := structured.NewConsumedCapacityCollection()
		capacity.Insert(structured.NewDeviceConsumedCapacity(device, map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse("80Gi")}))
		return &structured.AllocatedState{
			AllocatedDevices:         sets.New(exclusive),
			AllocatedSharedDeviceIDs: sets.New(device),
			AggregatedCapacity:       capacity,
		}
	}

	for name, tc := range map[string]struct {
		released     []*releasedClaim
		shares       map[structured.DeviceID]int
		wantExlusive bool
		wantShared   bool
		wantCapacity string
	}{
		"one-of-two-shares": {
			released:     []*releasedClaim{share("a", "60Gi")},
			shares:       map[structured.DeviceID]int{device: 2},
			wantExlusive: true,
			wantShared:   true,
			wantCapacity: "20Gi",
		},
		"all-shares": {
			released:     []*releasedClaim{share("a", "60Gi"), share("b", "20Gi")},
			shares:       map[structured.DeviceID]int{device: 2},
			wantExlusive: true,
		},
		"unknown-shares": {
			released:     []*releasedClaim{share("a", "60Gi"), share("b", "20Gi")},
			wantExlusive: true,
			wantShared:   true,
		},
		"exclusive": {
			released:     []*releasedClaim{{allocation: preemptionAllocation("gpu-1")}},
			wantShared:   true,
			wantCapacity: "80Gi",
		},
	} {
		t.Run(name, func(t *testing.T) {
			sim := simulation{released: make(map[types.UID]*releasedClaim)}
			for i, released := range tc.released {
				sim.released[types.UID(rune('a'+i))] = released
			}
			state := newState()
			sim.applyTo(state, true, func() map[structured.DeviceID]int { return tc.shares })
			if got := state.AllocatedDevices.Has(exclusive); got != tc.wantExlusive {
				t.Errorf("exclusive device allocated: got %v, want %v", got, tc.wantExlusive)
			}
			if got := state.AllocatedSharedDeviceIDs.Has(device); got != tc.wantShared {
				t.Errorf("device shared: got %v, want %v", got, tc.wantShared)
			}
			var gotCapacity string
			if capacity, ok := state.AggregatedCapacity[device]; ok {
				for _, quantity := range capacity {
					gotCapacity = quantity.String()
				}
			}
			if diff := cmp.Diff(tc.wantCapacity, gotCapacity); diff != "" {
				t.Errorf("unexpected remaining capacity (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestPreemptionCanRelease(t *testing.T) {
	testPreemptionCanRelease(ktesting.Init(t))
}

// testPreemptionCanRelease checks which pods get devices of removed pods.
func testPreemptionCanRelease(tCtx ktesting.TContext) {
	allocatedState := &structured.AllocatedState{}
	for name, tc := range map[string]struct {
		features       feature.Features
		pod            *v1.Pod
		allocatedState *structured.AllocatedState
		extended       bool
		want           bool
	}{
		"regular": {
			pod:            preemptionPod("pod").Obj(),
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
			claims := newClaimStore([]*resourceapi.ResourceClaim{preemptionClaim("claim").Obj()}, extendedResourceClaim, nil)
			p := testCtx.p.newPreemptionState(tc.pod, &claims, tc.allocatedState, nil)
			require.Equal(tCtx, tc.want, p.root.canRelease)
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
	require.Equal(tCtx, fwk.UnschedulableAndUnresolvable, testCtx.p.Filter(tCtx, testCtx.state, preemptor, testCtx.nodeInfos[0]).Code(), "Filter")
}

func TestPreemptionStatusCode(t *testing.T) {
	testPreemptionStatusCode(ktesting.Init(t))
}

// testPreemptionStatusCode checks which nodes Filter offers to preemption.
func testPreemptionStatusCode(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victimClaim := reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim)
	// Two slices of the same pool with the same device make the pool invalid.
	brokenSlices := func() []*resourceapi.ResourceSlice {
		var slices []*resourceapi.ResourceSlice
		for _, name := range []string{"broken-1", "broken-2"} {
			slice := st.MakeResourceSlice(nodeName, driver2).Device("duplicate").ResourceSliceCount(2).Obj()
			slice.Name = name
			slice.Spec.Pool.Name = "broken"
			slices = append(slices, slice)
		}
		return slices
	}()

	for name, tc := range map[string]struct {
		features   feature.Features
		preemptor  *v1.Pod
		podsOnNode []*v1.Pod
		claims     []*resourceapi.ResourceClaim
		slices     []*resourceapi.ResourceSlice
		want       *fwk.Status
	}{
		"lower-priority-pod": {
			podsOnNode: []*v1.Pod{victim},
			want:       fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"),
		},
		"equal-priority-pod": {
			podsOnNode: []*v1.Pod{fromPod(victim).Priority(100).Obj()},
			want:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"),
		},
		"lower-priority-pod-without-claims": {
			// The device is used by a pod that is not on the node (yet).
			podsOnNode: []*v1.Pod{preemptionPod("other").Node(nodeName).Obj()},
			want:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"),
		},
		"no-node-local-device": {
			podsOnNode: []*v1.Pod{victim},
			claims: []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				reserve(preemptionClaim("victim-claim").Allocation(poolAllocation("network", "instance-1")).Obj(), victim),
			},
			slices: []*resourceapi.ResourceSlice{st.MakeResourceSliceWithAllNodes("network", driver).Device("instance-1").Obj()},
			want:   fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"),
		},
		"preemption-policy-never": {
			preemptor:  fromPod(preemptor).PreemptionPolicy(v1.PreemptNever).Obj(),
			podsOnNode: []*v1.Pod{victim},
			want:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"),
		},
		"pod-group": {
			features:   feature.Features{EnableGenericWorkload: true},
			preemptor:  fromPod(preemptor).PodGroupName(podGroupName).Obj(),
			podsOnNode: []*v1.Pod{victim},
			want:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"),
		},
		"invalid-pool": {
			podsOnNode: []*v1.Pod{victim},
			slices:     append([]*resourceapi.ResourceSlice{slice}, brokenSlices...),
			want:       fwk.NewStatus(fwk.Unschedulable, "invalid resource pools were encountered"),
		},
		"invalid-pool-equal-priority": {
			podsOnNode: []*v1.Pod{fromPod(victim).Priority(100).Obj()},
			slices:     append([]*resourceapi.ResourceSlice{slice}, brokenSlices...),
			want:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "invalid resource pools were encountered"),
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			for _, enabled := range []bool{true, false} {
				features := tc.features
				features.EnableDRAPreemption = enabled
				claims := tc.claims
				if claims == nil {
					claims = []*resourceapi.ResourceClaim{preemptionClaim("preemptor-claim").Obj(), victimClaim}
				}
				objs := []apiruntime.Object{}
				resourceSlices := tc.slices
				if resourceSlices == nil {
					resourceSlices = []*resourceapi.ResourceSlice{slice}
				}
				for _, slice := range resourceSlices {
					objs = append(objs, slice)
				}
				testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, objs, features)
				nodeInfo := testCtx.nodeInfos[0].(*framework.NodeInfo)
				for _, pod := range tc.podsOnNode {
					nodeInfo.AddPod(pod)
				}
				pod := tc.preemptor
				if pod == nil {
					pod = preemptor
				}

				_, status := testCtx.p.PreFilter(tCtx, testCtx.state, pod, testCtx.nodeInfos)
				require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
				want := tc.want
				if !enabled {
					want = fwk.NewStatus(fwk.UnschedulableAndUnresolvable, want.Reasons()...)
				}
				require.Equal(tCtx, want, testCtx.p.Filter(tCtx, testCtx.state, pod, nodeInfo), "Filter with DRAPreemption=%v", enabled)
			}
		})
	}
}

func TestPreemptionPostFilter(t *testing.T) {
	testPreemptionPostFilter(ktesting.Init(t))
}

// testPreemptionPostFilter checks that preemption does not run after
// PostFilter deallocated a claim of the pod.
func testPreemptionPostFilter(tCtx ktesting.TContext) {
	allocation := preemptionAllocation("instance-1")
	allocation.NodeSelector = st.MakeNodeSelector().In("metadata.name", []string{node2Name}, st.NodeSelectorTypeMatchFields).Obj()
	pod := preemptionPod("pod", "claim").Obj()
	claims := []*resourceapi.ResourceClaim{preemptionClaim("claim").Allocation(allocation).Obj()}
	objs := []apiruntime.Object{st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()}

	for name, tc := range map[string]struct {
		features feature.Features
		want     fwk.Code
	}{
		"enabled":  {features: feature.Features{EnableDRAPreemption: true}, want: fwk.UnschedulableAndUnresolvable},
		"disabled": {want: fwk.Unschedulable},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, objs, tc.features)
			_, status := testCtx.p.PreFilter(tCtx, testCtx.state, pod, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			status = testCtx.p.Filter(tCtx, testCtx.state, pod, testCtx.nodeInfos[0])
			require.Equal(tCtx, fwk.UnschedulableAndUnresolvable, status.Code(), "Filter: %v", status)
			_, status = testCtx.p.PostFilter(tCtx, testCtx.state, pod, nil)
			require.Equal(tCtx, tc.want, status.Code(), "PostFilter: %v", status)
			require.Equal(tCtx, []string{"deallocation of ResourceClaim completed"}, status.Reasons(), "PostFilter reasons")
		})
	}
}
