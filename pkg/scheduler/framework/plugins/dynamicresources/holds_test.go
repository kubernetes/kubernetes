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
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

// filterWithNominated mimics RunFilterPluginsWithNominatedPods on workerNode:
// if any nominated pod has priority >= pod, it runs pass 1 on a cloned state
// with those pods added and asserts pass 2 on cs picks the same devices.
func filterWithNominated(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState, pod *v1.Pod, nominated ...*v1.Pod) *fwk.Status {
	tCtx.Helper()
	nodeInfo := testCtx.nodeInfos[0]
	passOne := cs.Clone()
	added := false
	for _, nominatedPod := range nominated {
		if nominatedPod.UID == pod.UID || corev1helpers.PodPriority(nominatedPod) < corev1helpers.PodPriority(pod) {
			continue
		}
		addPod(tCtx, testCtx, passOne, pod, nominatedPod, nodeInfo)
		added = true
	}
	if !added {
		return testCtx.p.Filter(tCtx, cs, pod, nodeInfo)
	}
	if status := testCtx.p.Filter(tCtx, passOne, pod, nodeInfo); !status.IsSuccess() {
		return status
	}
	devices := pickedDevices(tCtx, passOne)
	status := testCtx.p.Filter(tCtx, cs, pod, nodeInfo)
	require.True(tCtx, status.IsSuccess(), "pass 2 after pass 1 succeeded: %v", status)
	require.Equal(tCtx, devices, pickedDevices(tCtx, cs), "devices picked by pass 1 and pass 2")
	return status
}

func addPod(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState, pod, podToAdd *v1.Pod, nodeInfo fwk.NodeInfo) {
	tCtx.Helper()
	podInfo, err := framework.NewPodInfo(podToAdd)
	tCtx.ExpectNoError(err)
	require.True(tCtx, testCtx.p.AddPod(tCtx, cs, pod, podInfo, nodeInfo).IsSuccess(), "AddPod")
}

func removePod(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState, pod, podToRemove *v1.Pod, nodeInfo fwk.NodeInfo) {
	tCtx.Helper()
	podInfo, err := framework.NewPodInfo(podToRemove)
	tCtx.ExpectNoError(err)
	require.True(tCtx, testCtx.p.RemovePod(tCtx, cs, pod, podInfo, nodeInfo).IsSuccess(), "RemovePod")
}

func pickedDevices(tCtx ktesting.TContext, cs fwk.CycleState) []string {
	tCtx.Helper()
	state, err := getStateData(cs)
	tCtx.ExpectNoError(err)
	state.mutex.Lock()
	defer state.mutex.Unlock()
	var devices []string
	for _, allocation := range state.nodeAllocations[nodeName].allocationResults {
		for _, result := range allocation.Devices.Results {
			devices = append(devices, result.Device)
		}
	}
	return devices
}

func heldDevices(testCtx *testContext, pod *v1.Pod) []string {
	hold := testCtx.p.store.nominatedAllocation(pod.UID)
	if hold == nil {
		return nil
	}
	var devices []string
	for _, held := range hold.claims {
		for _, result := range held.allocation.Devices.Results {
			devices = append(devices, result.Device)
		}
	}
	slices.Sort(devices)
	return devices
}

func removeVictims(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, victims ...*v1.Pod) fwk.CycleState {
	tCtx.Helper()
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	cs := state.Clone()
	for _, victim := range victims {
		removePod(tCtx, testCtx, cs, pod, victim, testCtx.nodeInfos[0])
	}
	return cs
}

func nominate(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nominated []*v1.Pod, victims ...*v1.Pod) framework.NominationChangeResult {
	tCtx.Helper()
	state := dryRun(tCtx, testCtx, preemptor, nominated, victims...)
	assumeVictims(tCtx, testCtx, victims...)
	return addNomination(tCtx, testCtx, state, preemptor, nodeName)
}

func addNomination(tCtx ktesting.TContext, testCtx *testContext, state fwk.CycleState, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	prepared := testCtx.p.PrepareNomination(tCtx, state, pod, nodeName)
	return testCtx.p.AddNomination(tCtx, pod, nodeName, prepared)
}

func dryRun(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nominated []*v1.Pod, victims ...*v1.Pod) fwk.CycleState {
	tCtx.Helper()
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	cs := state.Clone()
	for _, victim := range victims {
		removePod(tCtx, testCtx, cs, preemptor, victim, testCtx.nodeInfos[0])
	}
	status = filterWithNominated(tCtx, testCtx, cs, preemptor, nominated...)
	require.True(tCtx, status.IsSuccess(), "dry run of %s: %v", preemptor.Name, status)
	return state
}

func endNomination(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) framework.NominationChangeResult {
	return testCtx.p.RemoveNomination(tCtx, pod, nodeName)
}

func TestHolds(t *testing.T) {
	testHolds(ktesting.Init(t))
}

func testHolds(tCtx ktesting.TContext) {
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
	p2 := preemptionPod("p2", "p2-claim").Priority(100).Obj()
	low := preemptionPod("low", "low-claim").Priority(50).Obj()
	high := preemptionPod("high", "high-claim").Priority(200).Obj()
	never := preemptionPod("never", "never-claim").Priority(50).PreemptionPolicy(v1.PreemptNever).Obj()
	sharer := preemptionPod("sharer", "p1-claim").Priority(50).Obj()
	setup := func(tCtx ktesting.TContext, numDevices int, victimDevices ...string) *testContext {
		slice := st.MakeResourceSlice(nodeName, driver)
		for i := range numDevices {
			slice = slice.Device(fmt.Sprintf("gpu-%d", i))
		}
		victimClaim := preemptionClaim("victim-claim")
		for range victimDevices[1:] {
			victimClaim = victimClaim.Request(className)
		}
		claims := []*resourceapi.ResourceClaim{
			preemptionClaim("p1-claim").Obj(),
			preemptionClaim("p2-claim").Obj(),
			preemptionClaim("low-claim").Obj(),
			preemptionClaim("high-claim").Obj(),
			preemptionClaim("never-claim").Obj(),
			reserve(victimClaim.Allocation(preemptionAllocation(victimDevices...)).Obj(), victim),
		}
		return setupVictims(tCtx, claims, []apiruntime.Object{slice.Obj(), victim, p1, p2, low, high, never, sharer})
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, nominated ...*v1.Pod) []string {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		status = filterWithNominated(tCtx, testCtx, state, pod, nominated...)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		return pickedDevices(tCtx, state)
	}

	tCtx.Run("shared-terminating-victim", func(tCtx ktesting.TContext) {
		// Two equal-priority preemptors preempt the same two-device victim; p1's
		// hold on gpu-0 steers p2 onto gpu-1.
		testCtx := setup(tCtx, 2, "gpu-0", "gpu-1")
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))

		cs := removeVictims(tCtx, testCtx, p2, victim)
		status := filterWithNominated(tCtx, testCtx, cs, p2, p1)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		require.Equal(tCtx, []string{"gpu-1"}, pickedDevices(tCtx, cs))
		nominate(tCtx, testCtx, p2, []*v1.Pod{p1}, victim)
		require.Equal(tCtx, []string{"gpu-1"}, heldDevices(testCtx, p2))
	})

	tCtx.Run("surplus-of-victim", func(tCtx ktesting.TContext) {
		// Victim's unheld second device (gpu-1) stays unavailable until
		// victim-claim is deallocated.
		testCtx := setup(tCtx, 3, "gpu-0", "gpu-1")
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		require.Equal(tCtx, []string{"gpu-2"}, filter(tCtx, testCtx, low, p1))
	})

	tCtx.Run("holds-that-apply", func(tCtx ktesting.TContext) {
		// Holds block pods with priority <= holder (including PreemptNever pods),
		// but not higher-priority pods, the holder itself, or pods sharing the
		// held claim.
		testCtx := setup(tCtx, 2, "gpu-0")
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		for name, tc := range map[string]struct {
			pod  *v1.Pod
			want string
		}{
			"lower-priority":    {pod: low, want: "gpu-1"},
			"equal-priority":    {pod: p2, want: "gpu-1"},
			"higher-priority":   {pod: high, want: "gpu-0"},
			"holder":            {pod: p1, want: "gpu-0"},
			"cannot-preempt":    {pod: never, want: "gpu-1"},
			"shares-held-claim": {pod: sharer, want: "gpu-0"},
		} {
			tCtx.Run(name, func(tCtx ktesting.TContext) {
				require.Equal(tCtx, []string{tc.want}, filter(tCtx, testCtx, tc.pod, p1))
			})
		}
	})

	tCtx.Run("add-pod", func(tCtx ktesting.TContext) {
		// AddPod records holds only for other unassigned pods nominated to the
		// target node, and RemovePod undoes it.
		testCtx := setup(tCtx, 2, "gpu-0")
		nominate(tCtx, testCtx, p1, nil, victim)
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, low, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		holders := func(cs fwk.CycleState) []types.UID {
			var uids []types.UID
			for uid := range getSimulationState(cs).holds {
				uids = append(uids, uid)
			}
			return uids
		}

		cs := state.Clone()
		addPod(tCtx, testCtx, cs, low, p1, testCtx.nodeInfos[1])
		require.Empty(tCtx, holders(cs), "hold for another node")
		addPod(tCtx, testCtx, cs, low, fromPod(p1).Node(nodeName).Obj(), testCtx.nodeInfos[0])
		require.Empty(tCtx, holders(cs), "assigned pod")
		addPod(tCtx, testCtx, cs, p1, p1, testCtx.nodeInfos[0])
		require.Empty(tCtx, holders(cs), "the pod itself")
		addPod(tCtx, testCtx, cs, low, p1, testCtx.nodeInfos[0])
		require.Equal(tCtx, []types.UID{p1.UID}, holders(cs), "hold of the nominated pod")
		require.Empty(tCtx, holders(state), "original state")
		removePod(tCtx, testCtx, cs, low, p1, testCtx.nodeInfos[0])
		require.Empty(tCtx, holders(cs), "after RemovePod")
		addPod(tCtx, testCtx, cs, low, p1, testCtx.nodeInfos[0])
		require.Equal(tCtx, []types.UID{p1.UID}, holders(cs), "after AddPod again")
	})

	tCtx.Run("renominated", func(tCtx ktesting.TContext) {
		// Re-preempting onto a different victim replaces the hold and emits a
		// release event for the old device.
		victim2 := preemptionPod("victim2", "victim2-claim").Node(nodeName).Obj()
		testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("p1-claim").Obj(),
			reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
			reserve(preemptionClaim("victim2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), victim2),
		}, []apiruntime.Object{st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj(), victim, victim2, p1})
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))

		require.Len(tCtx, nominate(tCtx, testCtx, p1, nil, victim2).Events, 1, "events for the old hold")
		require.Equal(tCtx, []string{"gpu-1"}, heldDevices(testCtx, p1), "new hold")
		hold := testCtx.p.store.nominatedAllocation(p1.UID)
		var released []string
		for _, claim := range hold.released {
			released = append(released, claim.claim.Name)
		}
		require.Equal(tCtx, []string{"victim2-claim"}, released, "released claims of the new hold")
	})

	tCtx.Run("held-claim-allocated", func(tCtx ktesting.TContext) {
		// Once p1-claim is already allocated (on gpu-1), p1's stale hold on
		// gpu-0 no longer blocks low.
		testCtx := setup(tCtx, 2, "gpu-0")
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		updateClaim(tCtx, testCtx, "p1-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.Allocation = preemptionAllocation("gpu-1")
		})
		require.Equal(tCtx, []string{"gpu-0"}, filter(tCtx, testCtx, low, p1))
	})

	tCtx.Run("held-claim-recreated", func(tCtx ktesting.TContext) {
		// Recreating p1-claim with a new UID invalidates the hold recorded for
		// the old UID.
		testCtx := setup(tCtx, 2, "gpu-0")
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		old, err := testCtx.draManager.ResourceClaims().Get(namespace, "p1-claim")
		tCtx.ExpectNoError(err)
		tCtx.ExpectNoError(testCtx.client.ResourceV1().ResourceClaims(namespace).Delete(tCtx, "p1-claim", metav1.DeleteOptions{}))
		_, err = testCtx.client.ResourceV1().ResourceClaims(namespace).Create(tCtx, preemptionClaim("p1-claim").Obj(), metav1.CreateOptions{})
		tCtx.ExpectNoError(err)
		eventually(tCtx, "claim created again", func() bool {
			claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "p1-claim")
			return err == nil && claim.UID != old.UID
		})
		require.Equal(tCtx, []string{"gpu-0"}, filter(tCtx, testCtx, low, p1))
	})
}

func TestHoldsConsumableCapacity(t *testing.T) {
	testHoldsConsumableCapacity(ktesting.Init(t))
}

// testHoldsConsumableCapacity verifies max(before, before - victims + held)
// capacity accounting when victim claims are still settling.
func testHoldsConsumableCapacity(tCtx ktesting.TContext) {
	const memory = resourceapi.QualifiedName("memory")
	slice := func(size string) *resourceapi.ResourceSlice {
		slice := st.MakeResourceSlice(nodeName, driver).
			Device("gpu-0", map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{memory: {Value: apiresource.MustParse(size)}}).
			Obj()
		slice.Spec.Devices[0].AllowMultipleAllocations = new(true)
		return slice
	}
	share := func(shareID types.UID, quantity string) *resourceapi.AllocationResult {
		allocation := preemptionAllocation("gpu-0")
		allocation.Devices.Results[0].ShareID = &shareID
		allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)}
		return allocation
	}
	request := func(name, quantity string) *resourceapi.ResourceClaim {
		claim := preemptionClaim(name).Obj()
		claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)},
		}
		return claim
	}
	setup := func(tCtx ktesting.TContext, claims []*resourceapi.ResourceClaim, objs ...apiruntime.Object) *testContext {
		featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.DRAPreemption:         true,
			features.DRAConsumableCapacity: true,
		})
		testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, objs, feature.Features{EnableDRAPreemption: true, EnableDRAConsumableCapacity: true})
		require.NotNil(tCtx, testCtx.p.store, "preemption store")
		return testCtx
	}
	fits := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, nominated []*v1.Pod, victims ...*v1.Pod) bool {
		cs := removeVictims(tCtx, testCtx, pod, victims...)
		status := filterWithNominated(tCtx, testCtx, cs, pod, nominated...)
		require.True(tCtx, status.IsSuccess() || status.IsRejected(), "Filter: %v", status)
		return status.IsSuccess()
	}

	tCtx.Run("victim-gone-claim-allocated", func(tCtx ktesting.TContext) {
		// v1 is deleted while v1-claim is still allocated; p1's 40Gi hold
		// offsets v1-claim's 40Gi rather than double-counting it, allowing p2
		// to fit after preempting v2.
		v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Obj()
		v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Obj()
		p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
		p2 := preemptionPod("p2", "p2-claim").Priority(100).Obj()
		testCtx := setup(tCtx, []*resourceapi.ResourceClaim{
			request("p1-claim", "40Gi"),
			request("p2-claim", "40Gi"),
			reserve(st.FromResourceClaim(request("v1-claim", "40Gi")).Allocation(share("v1", "40Gi")).Obj(), v1Pod),
			reserve(st.FromResourceClaim(request("v2-claim", "40Gi")).Allocation(share("v2", "40Gi")).Obj(), v2Pod),
		}, slice("80Gi"), v1Pod, v2Pod, p1, p2)
		nominate(tCtx, testCtx, p1, nil, v1Pod)
		require.Len(tCtx, heldDevices(testCtx, p1), 1, "held for p1")
		deletePod(tCtx, testCtx, v1Pod)

		require.True(tCtx, fits(tCtx, testCtx, p2, []*v1.Pod{p1}, v2Pod), "p2 after removing v2")
		require.False(tCtx, fits(tCtx, testCtx, p2, []*v1.Pod{p1}), "p2 without removing v2")
	})

	tCtx.Run("holds-share-victim-claim", func(tCtx ktesting.TContext) {
		// Two 30Gi holds backed by the same 40Gi victim claim deduct
		// victim-claim only once (reserving 60Gi total, leaving 40Gi < 50Gi).
		victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
		p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
		p2 := preemptionPod("p2", "p2-claim").Priority(100).Obj()
		low := preemptionPod("low", "low-claim").Priority(50).Obj()
		testCtx := setup(tCtx, []*resourceapi.ResourceClaim{
			request("p1-claim", "30Gi"),
			request("p2-claim", "30Gi"),
			request("low-claim", "50Gi"),
			reserve(st.FromResourceClaim(request("victim-claim", "40Gi")).Allocation(share("v", "40Gi")).Obj(), victim),
		}, slice("100Gi"), victim, p1, p2, low)
		nominate(tCtx, testCtx, p1, nil, victim)
		nominate(tCtx, testCtx, p2, []*v1.Pod{p1}, victim)
		require.Len(tCtx, heldDevices(testCtx, p2), 1, "held for p2")

		require.False(tCtx, fits(tCtx, testCtx, low, []*v1.Pod{p1, p2}), "low priority pod")
	})

	tCtx.Run("victim-of-both", func(tCtx ktesting.TContext) {
		// When p2's dry run already released victim-claim via sim.applyTo,
		// nodeHolds must not subtract victim-claim a second time.
		victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
		other := preemptionPod("other", "other-claim").Node(nodeName).Priority(1000).Obj()
		p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
		p2 := preemptionPod("p2", "p2-claim").Priority(100).Obj()
		testCtx := setup(tCtx, []*resourceapi.ResourceClaim{
			request("p1-claim", "30Gi"),
			request("p2-claim", "40Gi"),
			reserve(st.FromResourceClaim(request("other-claim", "20Gi")).Allocation(share("o", "20Gi")).Obj(), other),
			reserve(st.FromResourceClaim(request("victim-claim", "40Gi")).Allocation(share("v", "40Gi")).Obj(), victim),
		}, slice("80Gi"), victim, other, p1, p2)
		nominate(tCtx, testCtx, p1, nil, victim)
		require.Len(tCtx, heldDevices(testCtx, p1), 1, "held for p1")

		require.True(tCtx, fits(tCtx, testCtx, p2, nil, victim), "p2 without the hold")
		// 20Gi (other) + 30Gi (held for p1) leaves only 30Gi < 40Gi.
		require.False(tCtx, fits(tCtx, testCtx, p2, []*v1.Pod{p1}, victim), "p2 with the hold")
	})

	tCtx.Run("other-victim-claim", func(tCtx ktesting.TContext) {
		// An unrelated settling victim claim (vx-claim) must not offset p's
		// hold after p's own victim claim (v-claim) was deallocated.
		vx := preemptionPod("vx", "vx-claim").Node(nodeName).Obj()
		v := preemptionPod("v", "v-claim").Node(nodeName).Obj()
		px := preemptionPod("px", "px-claim").Priority(100).Obj()
		p := preemptionPod("p", "p-claim").Priority(100).Obj()
		low := preemptionPod("low", "low-claim").Priority(50).Obj()
		testCtx := setup(tCtx, []*resourceapi.ResourceClaim{
			request("px-claim", "40Gi"),
			request("p-claim", "40Gi"),
			request("low-claim", "40Gi"),
			reserve(st.FromResourceClaim(request("vx-claim", "40Gi")).Allocation(share("vx", "40Gi")).Obj(), vx),
			reserve(st.FromResourceClaim(request("v-claim", "40Gi")).Allocation(share("v", "40Gi")).Obj(), v),
		}, slice("80Gi"), vx, v, px, p, low)
		nominate(tCtx, testCtx, px, nil, vx)
		endNomination(tCtx, testCtx, px)
		nominate(tCtx, testCtx, p, nil, v)
		updateClaim(tCtx, testCtx, "v-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})

		require.False(tCtx, fits(tCtx, testCtx, low, []*v1.Pod{p}), "low priority pod")
		endNomination(tCtx, testCtx, p)
		require.True(tCtx, fits(tCtx, testCtx, low, []*v1.Pod{p}), "without the hold")
	})

	tCtx.Run("own-victim-claim", func(tCtx ktesting.TContext) {
		// Settling capacity of p's own victim claim (v-claim) does not offset
		// h's hold when p runs normal Filter without removing v.
		vh := preemptionPod("vh", "vh-claim").Node(nodeName).Obj()
		v := preemptionPod("v", "v-claim").Node(nodeName).Obj()
		h := preemptionPod("h", "h-claim").Priority(100).Obj()
		p := preemptionPod("p", "p-claim").Priority(100).Obj()
		testCtx := setup(tCtx, []*resourceapi.ResourceClaim{
			request("h-claim", "30Gi"),
			request("p-claim", "40Gi"),
			reserve(st.FromResourceClaim(request("vh-claim", "30Gi")).Allocation(share("vh", "30Gi")).Obj(), vh),
			reserve(st.FromResourceClaim(request("v-claim", "40Gi")).Allocation(share("v", "40Gi")).Obj(), v),
		}, slice("100Gi"), vh, v, h, p)
		nominate(tCtx, testCtx, h, nil, vh)
		updateClaim(tCtx, testCtx, "vh-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		nominate(tCtx, testCtx, p, []*v1.Pod{h}, v)
		require.Len(tCtx, heldDevices(testCtx, p), 1, "held for p")

		// 40Gi (v-claim) + 30Gi (held for h) leaves only 30Gi < 40Gi.
		require.False(tCtx, fits(tCtx, testCtx, p, []*v1.Pod{h, p}), "p before the claim of its victim is deallocated")
		updateClaim(tCtx, testCtx, "v-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		require.True(tCtx, fits(tCtx, testCtx, p, []*v1.Pod{h, p}), "p after the claim of its victim is deallocated")
	})
}

// TestSimulationStateCloneHolds verifies that Clone isolates holds mutations
// while preserving dryRun.
func TestSimulationStateCloneHolds(t *testing.T) {
	hold := &allocation{nodeName: nodeName}
	original := &simulationState{holds: map[types.UID]*allocation{"holder": hold}, dryRun: 1}
	copied := original.Clone().(*simulationState)
	require.Equal(t, 1, copied.dryRun, "dry run of the copy")
	copied.holds["other"] = &allocation{nodeName: nodeName}
	delete(copied.holds, "holder")
	require.Equal(t, map[types.UID]*allocation{"holder": hold}, original.holds, "holds of the original")
}

func TestJoinVictimClaim(t *testing.T) {
	testJoinVictimClaim(ktesting.Init(t))
}

// testJoinVictimClaim verifies that Filter rejects <= priority pods from
// reserving a victim claim held for a nominated preemptor.
func testJoinVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	replacement := preemptionPod("replacement", "victim-claim").Priority(10).Obj()
	important := preemptionPod("important", "victim-claim").Priority(200).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
	}
	setup := func(tCtx ktesting.TContext) *testContext {
		testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, replacement, important})
		nominate(tCtx, testCtx, preemptor, nil, victim)
		// Simulate victim termination leaving victim-claim allocated with empty ReservedFor.
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
		})
		return testCtx
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) (fwk.CycleState, *fwk.Status) {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		return state, filterWithNominated(tCtx, testCtx, state, pod, preemptor)
	}

	tCtx.Run("rejected", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		state, status := filter(tCtx, testCtx, replacement)
		require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated"), status, "Filter")
		_, status = testCtx.p.PostFilter(tCtx, state, replacement, nil)
		require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "deallocation of ResourceClaim completed"), status, "PostFilter")
	})

	tCtx.Run("higher-priority", func(tCtx ktesting.TContext) {
		// Higher-priority pods do not receive preemptor's hold in AddPod and may use the claim.
		testCtx := setup(tCtx)
		_, status := filter(tCtx, testCtx, important)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	})

	tCtx.Run("nomination-ended", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		endNomination(tCtx, testCtx, preemptor)
		_, status := filter(tCtx, testCtx, replacement)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	})

	tCtx.Run("victim-not-assumed-yet", func(tCtx ktesting.TContext) {
		// During async preemption, AddNomination runs before AssumeVictim;
		// hold.released already protects victim-claim.
		testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, replacement, important})
		state := dryRun(tCtx, testCtx, preemptor, nil, victim)
		addNomination(tCtx, testCtx, state, preemptor, nodeName)
		_, status := filter(tCtx, testCtx, replacement)
		require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated"), status, "Filter")
	})
}

func TestFilterResultReuse(t *testing.T) {
	testFilterResultReuse(ktesting.Init(t))
}

// testFilterResultReuse verifies that cached Filter allocations are reused only
// when holds is nil and the set of released claims on the node matches.
func testFilterResultReuse(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Device("gpu-2").Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	holder := preemptionPod("holder", "holder-claim").Priority(200).Obj()
	setup := func(tCtx ktesting.TContext) *testContext {
		return setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			preemptionClaim("holder-claim").Obj(),
			reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
			reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod),
		}, []apiruntime.Object{slice, v1Pod, v2Pod, preemptor, holder})
	}
	fits := func(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState) bool {
		status := testCtx.p.Filter(tCtx, cs, preemptor, testCtx.nodeInfos[0])
		require.True(tCtx, status.IsSuccess() || status.IsRejected(), "Filter: %v", status)
		return status.IsSuccess()
	}

	tCtx.Run("reprieve", func(tCtx ktesting.TContext) {
		// gpu-2 is used by another pod which is not a victim.
		testCtx := setup(tCtx)
		updateClaim(tCtx, testCtx, "holder-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.Allocation = preemptionAllocation("gpu-2")
			claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(holder)}
		})
		nodeInfo := testCtx.nodeInfos[0]
		cs := removeVictims(tCtx, testCtx, preemptor, v1Pod, v2Pod)
		require.True(tCtx, fits(tCtx, testCtx, cs), "both victims removed")
		require.Equal(tCtx, []string{"gpu-0"}, pickedDevices(tCtx, cs))
		addPod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		require.True(tCtx, fits(tCtx, testCtx, cs), "v2 added back")
		addPod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)
		require.False(tCtx, fits(tCtx, testCtx, cs), "v1 added back, its device must not come from an earlier result")
	})

	tCtx.Run("pass-one-only", func(tCtx ktesting.TContext) {
		// Pass 1 succeeded with v1 removed, then a later plugin rejected it and
		// v1 was reprieved; the next Filter call has a different releasedKey and
		// must not reuse gpu-0.
		testCtx := setup(tCtx)
		nominate(tCtx, testCtx, holder, nil)
		require.Equal(tCtx, []string{"gpu-2"}, heldDevices(testCtx, holder))
		nodeInfo := testCtx.nodeInfos[0]
		cs := removeVictims(tCtx, testCtx, preemptor, v1Pod)
		passOne := cs.Clone()
		addPod(tCtx, testCtx, passOne, preemptor, holder, nodeInfo)
		require.True(tCtx, fits(tCtx, testCtx, passOne), "pass 1 with the hold")
		require.Equal(tCtx, []string{"gpu-0"}, pickedDevices(tCtx, passOne))
		addPod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)
		require.True(tCtx, fits(tCtx, testCtx, cs), "without the hold, gpu-2 is free")
		require.Equal(tCtx, []string{"gpu-2"}, pickedDevices(tCtx, cs))
	})

	tCtx.Run("pass-two", func(tCtx ktesting.TContext) {
		// Pass 2 runs without holds and must reuse pass 1's gpu-2 choice instead of taking held gpu-0.
		testCtx := setup(tCtx)
		updateClaim(tCtx, testCtx, "v1-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		nominate(tCtx, testCtx, holder, nil)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, holder))
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		status = filterWithNominated(tCtx, testCtx, state, preemptor, holder)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		require.Equal(tCtx, []string{"gpu-2"}, pickedDevices(tCtx, state))
	})

	tCtx.Run("holds-after-result-without-holds", func(tCtx ktesting.TContext) {
		// A cached result computed without holds must not be reused when holds are active.
		testCtx := setup(tCtx)
		updateClaim(tCtx, testCtx, "v1-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		nominate(tCtx, testCtx, holder, nil)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, holder))
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		require.True(tCtx, fits(tCtx, testCtx, state), "without the hold")
		require.Equal(tCtx, []string{"gpu-0"}, pickedDevices(tCtx, state))
		passOne := state.Clone()
		addPod(tCtx, testCtx, passOne, preemptor, holder, testCtx.nodeInfos[0])
		require.True(tCtx, fits(tCtx, testCtx, passOne), "with the hold")
		require.Equal(tCtx, []string{"gpu-2"}, pickedDevices(tCtx, passOne))
	})
}

func TestPrepareNomination(t *testing.T) {
	testPrepareNomination(ktesting.Init(t))
}

func testPrepareNomination(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	setup := func(tCtx ktesting.TContext, preemptor *v1.Pod) (*testContext, fwk.CycleState) {
		testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
			reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod),
		}, []apiruntime.Object{slice, v1Pod, v2Pod, preemptor})
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		return testCtx, state
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState) {
		status := testCtx.p.Filter(tCtx, cs, preemptor, testCtx.nodeInfos[0])
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	}
	prepare := func(tCtx ktesting.TContext, testCtx *testContext, state fwk.CycleState, pod *v1.Pod, nodeName string) *allocation {
		tCtx.Helper()
		prepared := testCtx.p.PrepareNomination(tCtx, state, pod, nodeName)
		if prepared == nil {
			return nil
		}
		hold, ok := prepared.(*allocation)
		require.True(tCtx, ok, "PrepareNomination returned %T", prepared)
		require.NotNil(tCtx, hold, "PrepareNomination returned a typed nil")
		require.Equal(tCtx, nodeName, hold.nodeName, "node of the hold")
		return hold
	}
	devicesOf := func(hold *allocation) []string {
		var devices []string
		for _, held := range hold.claims {
			for _, result := range held.allocation.Devices.Results {
				devices = append(devices, result.Device)
			}
		}
		return devices
	}
	releasedOf := func(hold *allocation) []string {
		var claims []string
		for _, released := range hold.released {
			claims = append(claims, released.claim.Name)
		}
		slices.Sort(claims)
		return claims
	}

	tCtx.Run("final-state", func(tCtx ktesting.TContext) {
		// Both victims are removed, then each reprieve attempt passes
		// DynamicResources (updating pickedDevices to gpu-1) before a later
		// plugin rejects and re-removes it; PrepareNomination must return the
		// allocation for both victims removed.
		testCtx, state := setup(tCtx, preemptor)
		nodeInfo := testCtx.nodeInfos[0]
		cs := state.Clone()
		removePod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)
		removePod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		filter(tCtx, testCtx, cs)
		addPod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		filter(tCtx, testCtx, cs)
		removePod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		addPod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)
		filter(tCtx, testCtx, cs)
		require.Equal(tCtx, []string{"gpu-1"}, pickedDevices(tCtx, cs), "last Filter result")
		removePod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)

		got := prepare(tCtx, testCtx, state, preemptor, nodeName)
		require.NotNil(tCtx, got, "hold")
		require.Equal(tCtx, []string{"gpu-0"}, devicesOf(got), "devices of the final state")
		require.Equal(tCtx, []string{"v1-claim", "v2-claim"}, releasedOf(got), "released claims of the final state")
	})

	tCtx.Run("last-step-reprieved", func(tCtx ktesting.TContext) {
		// v2's reprieve succeeds, so the final state has only v1 removed.
		testCtx, state := setup(tCtx, preemptor)
		nodeInfo := testCtx.nodeInfos[0]
		cs := state.Clone()
		removePod(tCtx, testCtx, cs, preemptor, v1Pod, nodeInfo)
		removePod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		filter(tCtx, testCtx, cs)
		addPod(tCtx, testCtx, cs, preemptor, v2Pod, nodeInfo)
		filter(tCtx, testCtx, cs)

		got := prepare(tCtx, testCtx, state, preemptor, nodeName)
		require.NotNil(tCtx, got, "hold")
		require.Equal(tCtx, []string{"gpu-0"}, devicesOf(got), "devices of the final state")
		require.Equal(tCtx, []string{"v1-claim"}, releasedOf(got), "released claims of the final state")
	})

	tCtx.Run("not-filtered", func(tCtx ktesting.TContext) {
		// RemovePod ran without a subsequent Filter call for that victim set.
		testCtx, state := setup(tCtx, preemptor)
		cs := state.Clone()
		removePod(tCtx, testCtx, cs, preemptor, v1Pod, testCtx.nodeInfos[0])
		require.Nil(tCtx, prepare(tCtx, testCtx, state, preemptor, nodeName), "hold")
	})

	tCtx.Run("no-claims", func(tCtx ktesting.TContext) {
		// Pod without claims skips PreFilter; PrepareNomination is a no-op.
		testCtx, _ := setup(tCtx, preemptor)
		noClaims := preemptionPod("no-claims").Priority(100).Obj()
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, noClaims, testCtx.nodeInfos)
		require.True(tCtx, status.IsSkip(), "PreFilter: %v", status)
		require.Nil(tCtx, prepare(tCtx, testCtx, state, noClaims, nodeName), "hold")
	})

	tCtx.Run("cannot-release", func(tCtx ktesting.TContext) {
		// PreemptNever pods never acquire holds even if Filter allocated a free device.
		never := fromPod(preemptor).PreemptionPolicy(v1.PreemptNever).Obj()
		testCtx, state := setup(tCtx, never)
		updateClaim(tCtx, testCtx, "v2-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		_, status := testCtx.p.PreFilter(tCtx, state, never, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		status = testCtx.p.Filter(tCtx, state, never, testCtx.nodeInfos[0])
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		require.Nil(tCtx, prepare(tCtx, testCtx, state, never, nodeName), "hold")
	})

	tCtx.Run("podgroup-victim-of-other-dry-run", func(tCtx ktesting.TContext) {
		// Removing cross-node PodGroup member podA during node's dry run must
		// not clobber node2's own dry-run final state.
		podA := preemptionPod("a", "a-claim").Node(node2Name).Priority(10).Obj()
		podG := preemptionPod("g", "g-claim").Node(nodeName).Priority(10).Obj()
		podB := preemptionPod("b", "b-claim").Node(node2Name).Priority(20).Obj()
		for _, otherFirst := range []bool{false, true} {
			tCtx.Run(fmt.Sprintf("other-first-%v", otherFirst), func(tCtx ktesting.TContext) {
				testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
					preemptionClaim("preemptor-claim").Obj(),
					reserve(preemptionClaim("a-claim").Allocation(poolAllocation(node2Name, "gpu-0")).Obj(), podA),
					reserve(preemptionClaim("b-claim").Allocation(poolAllocation(node2Name, "gpu-1")).Obj(), podB),
					reserve(preemptionClaim("g-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), podG),
				}, []apiruntime.Object{
					st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Obj(),
					st.MakeResourceSlice(node2Name, driver).Device("gpu-0").Device("gpu-1").Obj(),
					podA, podB, podG, preemptor,
				})
				nodeInfo, otherNodeInfo := testCtx.nodeInfos[1], testCtx.nodeInfos[0]
				state := framework.NewCycleState()
				_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
				require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
				fits := func(cs fwk.CycleState) bool {
					status := testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo)
					require.True(tCtx, status.IsSuccess() || status.IsRejected(), "Filter: %v", status)
					return status.IsSuccess()
				}
				otherDryRun := func() {
					cs := state.Clone()
					removePod(tCtx, testCtx, cs, preemptor, podG, otherNodeInfo)
					removePod(tCtx, testCtx, cs, preemptor, podA, nodeInfo)
				}
				if otherFirst {
					otherDryRun()
				}
				// On node2, reprieving b passes DynamicResources on a's gpu-0
				// before a later plugin rejects it; reprieving PodGroup {a, g}
				// then succeeds on b's gpu-1.
				cs := state.Clone()
				removePod(tCtx, testCtx, cs, preemptor, podA, nodeInfo)
				removePod(tCtx, testCtx, cs, preemptor, podG, otherNodeInfo)
				removePod(tCtx, testCtx, cs, preemptor, podB, nodeInfo)
				require.True(tCtx, fits(cs), "all victims removed")
				addPod(tCtx, testCtx, cs, preemptor, podB, nodeInfo)
				require.True(tCtx, fits(cs), "b added back")
				removePod(tCtx, testCtx, cs, preemptor, podB, nodeInfo)
				addPod(tCtx, testCtx, cs, preemptor, podA, nodeInfo)
				addPod(tCtx, testCtx, cs, preemptor, podG, otherNodeInfo)
				require.True(tCtx, fits(cs), "PodGroup added back")
				if !otherFirst {
					otherDryRun()
				}

				got := prepare(tCtx, testCtx, state, preemptor, node2Name)
				require.NotNil(tCtx, got, "hold")
				require.Equal(tCtx, []string{"gpu-1"}, devicesOf(got), "devices of the final state")
				require.Equal(tCtx, []string{"b-claim"}, releasedOf(got), "released claims of the final state")
			})
		}
	})
}

func TestHoldsWithFramework(t *testing.T) {
	testHoldsWithFramework(ktesting.Init(t))
}

// testHoldsWithFramework exercises RunFilterPluginsWithNominatedPods and
// Reserve end-to-end across two scheduler profiles sharing DefaultDRAManager,
// verifying pass 2 and Reserve preserve pass 1's non-held device.
func testHoldsWithFramework(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	low := preemptionPod("low", "low-claim").Priority(50).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		preemptionClaim("low-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
	}
	testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, low})
	nominate(tCtx, testCtx, preemptor, nil, victim)
	require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor))
	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
	})

	// PodNominator requires nominated pods to exist in the queue to look up their priority.
	nominator := internalqueue.NewTestQueueWithInformerFactory(tCtx, (&queuesort.PrioritySort{}).Less, testCtx.informerFactory)
	nominator.Add(tCtx, preemptor)
	preemptorInfo, err := framework.NewPodInfo(preemptor)
	tCtx.ExpectNoError(err)
	nominator.AddNominatedPod(tCtx.Logger(), preemptorInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName})
	newPlugin := func(ctx context.Context, _ apiruntime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return New(ctx, &config.DynamicResourcesArgs{}, fh, feature.Features{EnableDRAPreemption: true})
	}
	fh, err := tf.NewFramework(tCtx, []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPluginAsExtensions(Name, newPlugin, "PreFilter", "Filter", "Reserve"),
	}, "other-profile",
		runtime.WithClientSet(testCtx.client),
		runtime.WithInformerFactory(testCtx.informerFactory),
		runtime.WithEventRecorder(&events.FakeRecorder{}),
		runtime.WithSharedDRAManager(testCtx.draManager),
		runtime.WithPodGroupManager(testCtx.podGroupManager),
		runtime.WithPodNominator(nominator),
		runtime.WithSnapshotSharedLister(internalcache.NewSnapshot(nil, []*v1.Node{workerNode, workerNode2})),
	)
	tCtx.ExpectNoError(err, "create framework")
	tCtx.Cleanup(func() { runtime.WaitForShutdown(fh) })

	state := framework.NewCycleState()
	_, status, _ := fh.RunPreFilterPlugins(tCtx, state, low)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	status = fh.RunFilterPluginsWithNominatedPods(tCtx, state, low, testCtx.nodeInfos[0])
	require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	require.Equal(tCtx, []string{"gpu-1"}, pickedDevices(tCtx, state))
	status = fh.RunReservePluginsReserve(tCtx, state, low, nodeName)
	require.True(tCtx, status.IsSuccess(), "Reserve: %v", status)
	lowClaim, err := testCtx.draManager.ResourceClaims().Get(namespace, "low-claim")
	tCtx.ExpectNoError(err)
	pending := testCtx.draManager.ResourceClaims().GetPendingAllocation(lowClaim.UID)
	require.NotNil(tCtx, pending, "allocation reserved for the low priority pod")
	require.Equal(tCtx, "gpu-1", pending.Devices.Results[0].Device, "reserved device")
}

func TestHoldsConcurrency(t *testing.T) {
	testHoldsConcurrency(ktesting.Init(t))
}

// testHoldsConcurrency exercises concurrent two-pass Filter calls on shared
// CycleState clones alongside AssumeVictim, PodEligibleToPreempt, queueing
// hints, and nomination updates under the race detector.
func testHoldsConcurrency(tCtx ktesting.TContext) {
	const numNodes = 4
	var nodes []*v1.Node
	var claims []*resourceapi.ResourceClaim
	var objs []apiruntime.Object
	var preemptors, victims []*v1.Pod
	for i := range numNodes {
		node := st.MakeNode().Name(fmt.Sprintf("node-%d", i)).Obj()
		victim := preemptionPod(fmt.Sprintf("victim-%d", i), fmt.Sprintf("victim-claim-%d", i)).Node(node.Name).Obj()
		preemptor := preemptionPod(fmt.Sprintf("preemptor-%d", i), fmt.Sprintf("preemptor-claim-%d", i)).Priority(100).Obj()
		claims = append(claims,
			preemptionClaim(fmt.Sprintf("preemptor-claim-%d", i)).Obj(),
			reserve(preemptionClaim(fmt.Sprintf("victim-claim-%d", i)).Allocation(poolAllocation(node.Name, "gpu-0")).Obj(), victim))
		objs = append(objs, st.MakeResourceSlice(node.Name, driver).Device("gpu-0").Device("gpu-1").Obj(), victim, preemptor)
		nodes = append(nodes, node)
		victims = append(victims, victim)
		preemptors = append(preemptors, preemptor)
	}
	low := preemptionPod("low", "low-claim").Priority(50).Obj()
	claims = append(claims, preemptionClaim("low-claim").Obj())
	objs = append(objs, low)
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, nodes, claims, objs, feature.Features{EnableDRAPreemption: true})
	nominationStates := make([]fwk.CycleState, numNodes)
	for i := range numNodes {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, preemptors[i], testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		cs := state.Clone()
		removePod(tCtx, testCtx, cs, preemptors[i], victims[i], testCtx.nodeInfos[i])
		status = testCtx.p.Filter(tCtx, cs, preemptors[i], testCtx.nodeInfos[i])
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		addNomination(tCtx, testCtx, state, preemptors[i], nodes[i].Name)
		nominationStates[i] = state
	}
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, low, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	logger := klog.FromContext(tCtx)

	var wg sync.WaitGroup
	statuses := make([]*fwk.Status, numNodes)
	for i, nodeInfo := range testCtx.nodeInfos {
		wg.Go(func() {
			passOne := state.Clone()
			podInfo, err := framework.NewPodInfo(preemptors[i])
			if err != nil {
				statuses[i] = fwk.AsStatus(err)
				return
			}
			if status := testCtx.p.AddPod(tCtx, passOne, low, podInfo, nodeInfo); !status.IsSuccess() {
				statuses[i] = status
				return
			}
			if status := testCtx.p.Filter(tCtx, passOne, low, nodeInfo); !status.IsSuccess() {
				statuses[i] = status
				return
			}
			statuses[i] = testCtx.p.Filter(tCtx, state, low, nodeInfo)
		})
	}
	wg.Go(func() {
		for range 10 {
			_, _ = testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, preemptors[0], victims[0], nil)
			_, _ = testCtx.p.PodEligibleToPreempt(tCtx, preemptors[0], nodes[0].Name)
			testCtx.p.AssumeVictim(tCtx, victims[0])
		}
	})
	wg.Go(func() {
		for range 10 {
			testCtx.p.RemoveNomination(tCtx, preemptors[1], nodes[1].Name)
			addNomination(tCtx, testCtx, nominationStates[1], preemptors[1], nodes[1].Name)
		}
	})
	wg.Wait()
	for i, status := range statuses {
		require.True(tCtx, status.IsSuccess(), "Filter on %s: %v", nodes[i].Name, status)
	}
}

func TestDryRunConcurrency(t *testing.T) {
	testDryRunConcurrency(ktesting.Init(t))
}

// testDryRunConcurrency exercises parallel multi-node preemption dry runs with
// shared devices and holds alongside concurrent nomination changes, victim
// tracking, queueing hints, and informer claim deallocation.
func testDryRunConcurrency(tCtx ktesting.TContext) {
	const memory = resourceapi.QualifiedName("memory")
	const numNodes = 4
	share := func(pool string, shareID types.UID, quantity string) *resourceapi.AllocationResult {
		allocation := poolAllocation(pool, "gpu-0")
		allocation.Devices.Results[0].ShareID = &shareID
		allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)}
		return allocation
	}
	request := func(name, quantity string) *st.ResourceClaimWrapper {
		claim := preemptionClaim(name)
		claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]apiresource.Quantity{memory: apiresource.MustParse(quantity)},
		}
		return claim
	}
	var nodes []*v1.Node
	var claims []*resourceapi.ResourceClaim
	var objs []apiruntime.Object
	var holders, victims1, victims2 []*v1.Pod
	for i := range numNodes {
		node := st.MakeNode().Name(fmt.Sprintf("node-%d", i)).Obj()
		slice := st.MakeResourceSlice(node.Name, driver).
			Device("gpu-0", map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{memory: {Value: apiresource.MustParse("80Gi")}}).
			Obj()
		slice.Spec.Devices[0].AllowMultipleAllocations = new(true)
		victim1 := preemptionPod(fmt.Sprintf("victim1-%d", i), fmt.Sprintf("victim1-claim-%d", i)).Node(node.Name).Priority(10).Obj()
		victim2 := preemptionPod(fmt.Sprintf("victim2-%d", i), fmt.Sprintf("victim2-claim-%d", i)).Node(node.Name).Priority(20).Obj()
		holder := preemptionPod(fmt.Sprintf("holder-%d", i), fmt.Sprintf("holder-claim-%d", i)).Priority(100).Obj()
		claims = append(claims,
			request(fmt.Sprintf("holder-claim-%d", i), "30Gi").Obj(),
			reserve(request(fmt.Sprintf("victim1-claim-%d", i), "40Gi").Allocation(share(node.Name, types.UID(fmt.Sprintf("s1-%d", i)), "40Gi")).Obj(), victim1),
			reserve(request(fmt.Sprintf("victim2-claim-%d", i), "40Gi").Allocation(share(node.Name, types.UID(fmt.Sprintf("s2-%d", i)), "40Gi")).Obj(), victim2),
		)
		objs = append(objs, slice, victim1, victim2, holder)
		nodes = append(nodes, node)
		holders = append(holders, holder)
		victims1 = append(victims1, victim1)
		victims2 = append(victims2, victim2)
	}
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(50).Obj()
	claims = append(claims, request("preemptor-claim", "20Gi").Obj())
	objs = append(objs, preemptor)
	featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAPreemption:         true,
		features.DRAConsumableCapacity: true,
	})
	testCtx := setupPreemption(tCtx, nodes, claims, objs, feature.Features{EnableDRAPreemption: true, EnableDRAConsumableCapacity: true})

	// Each holder preempted victim1 on its node.
	holderStates := make([]fwk.CycleState, numNodes)
	for i := range numNodes {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, holders[i], testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		cs := state.Clone()
		removePod(tCtx, testCtx, cs, holders[i], victims1[i], testCtx.nodeInfos[i])
		status = testCtx.p.Filter(tCtx, cs, holders[i], testCtx.nodeInfos[i])
		require.True(tCtx, status.IsSuccess(), "Filter of %s: %v", holders[i].Name, status)
		addNomination(tCtx, testCtx, state, holders[i], nodes[i].Name)
		require.NotNil(tCtx, testCtx.p.store.nominatedAllocation(holders[i].UID), "hold of %s", holders[i].Name)
		holderStates[i] = state
	}

	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	logger := klog.FromContext(tCtx)
	podInfo := func(pod *v1.Pod) fwk.PodInfo {
		podInfo, err := framework.NewPodInfo(pod)
		tCtx.ExpectNoError(err)
		return podInfo
	}
	holderInfos := make([]fwk.PodInfo, numNodes)
	victim1Infos := make([]fwk.PodInfo, numNodes)
	victim2Infos := make([]fwk.PodInfo, numNodes)
	for i := range numNodes {
		holderInfos[i], victim1Infos[i], victim2Infos[i] = podInfo(holders[i]), podInfo(victims1[i]), podInfo(victims2[i])
	}

	var wg sync.WaitGroup
	errs := make([]error, numNodes+1)
	for i, nodeInfo := range testCtx.nodeInfos {
		wg.Go(func() {
			// Run DefaultPreemption's reprieve loop with holds in pass 1.
			cs := state.Clone()
			fits := func() bool {
				passOne := cs.Clone()
				if status := testCtx.p.AddPod(tCtx, passOne, preemptor, holderInfos[i], nodeInfo); !status.IsSuccess() {
					errs[i] = status.AsError()
					return false
				}
				if !testCtx.p.Filter(tCtx, passOne, preemptor, nodeInfo).IsSuccess() {
					return false
				}
				return testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo).IsSuccess()
			}
			testCtx.p.RemovePod(tCtx, cs, preemptor, victim1Infos[i], nodeInfo)
			testCtx.p.RemovePod(tCtx, cs, preemptor, victim2Infos[i], nodeInfo)
			// Concurrent claim/nomination updates may cause a simulated pass to fail; skip reprieve if so.
			if !fits() {
				return
			}
			for _, victim := range []fwk.PodInfo{victim2Infos[i], victim1Infos[i]} {
				testCtx.p.AddPod(tCtx, cs, preemptor, victim, nodeInfo)
				if !fits() {
					testCtx.p.RemovePod(tCtx, cs, preemptor, victim, nodeInfo)
				}
			}
		})
	}
	wg.Go(func() {
		for range 20 {
			testCtx.p.RemoveNomination(tCtx, holders[0], nodes[0].Name)
			addNomination(tCtx, testCtx, holderStates[0], holders[0], nodes[0].Name)
		}
	})
	wg.Go(func() {
		for range 20 {
			testCtx.p.AssumeVictim(tCtx, victims1[1])
			_, _ = testCtx.p.PodEligibleToPreempt(tCtx, holders[1], nodes[1].Name)
			_, _ = testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, holders[1], victims1[1], nil)
		}
	})
	wg.Go(func() {
		testCtx.p.AssumeVictim(tCtx, victims2[3])
		claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim2-claim-3", metav1.GetOptions{})
		if err != nil {
			errs[numNodes] = err
			return
		}
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
		if _, err := testCtx.client.ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{}); err != nil {
			errs[numNodes] = err
			return
		}
		testCtx.p.RemoveNomination(tCtx, holders[2], nodes[2].Name)
	})
	wg.Wait()
	for _, err := range errs {
		tCtx.ExpectNoError(err)
	}
	for _, node := range nodes {
		_ = testCtx.p.PrepareNomination(tCtx, state, preemptor, node.Name)
	}
	eventually(tCtx, "deallocated victim claim forgotten", func() bool {
		return len(testCtx.p.store.victimClaimsOnNode(nodes[3].Name)) == 0
	})
}
