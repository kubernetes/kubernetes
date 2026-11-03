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
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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

// filterBothPasses runs Filter on the worker node like
// RunFilterPluginsWithNominatedPods when it adds nominated pods: first on a
// copy of the state, then on the state itself. Holds don't depend on the
// pods which it adds, so both passes must agree.
func filterBothPasses(tCtx ktesting.TContext, testCtx *testContext, cs fwk.CycleState, pod *v1.Pod) *fwk.Status {
	tCtx.Helper()
	nodeInfo := testCtx.nodeInfos[0]
	passOne := cs.Clone()
	if status := testCtx.p.Filter(tCtx, passOne, pod, nodeInfo); !status.IsSuccess() {
		return status
	}
	devices := pickedDevices(tCtx, passOne)
	status := testCtx.p.Filter(tCtx, cs, pod, nodeInfo)
	require.True(tCtx, status.IsSuccess(), "pass 2 after pass 1 succeeded: %v", status)
	require.Equal(tCtx, devices, pickedDevices(tCtx, cs), "devices picked by pass 1 and pass 2")
	return status
}

// pickedDevices returns the devices which the last successful Filter on
// the worker node picked.
func pickedDevices(tCtx ktesting.TContext, cs fwk.CycleState) []string {
	tCtx.Helper()
	state, err := getStateData(cs)
	tCtx.ExpectNoError(err)
	var devices []string
	for _, allocation := range state.nodeAllocations[nodeName].allocationResults {
		for _, result := range allocation.Devices.Results {
			devices = append(devices, result.Device)
		}
	}
	return devices
}

func heldDevices(testCtx *testContext, pod *v1.Pod) []string {
	nomination := testCtx.p.nominations.nomination(pod.UID)
	if nomination == nil {
		return nil
	}
	var devices []string
	for _, held := range nomination.holds {
		for _, result := range held.allocation.Devices.Results {
			devices = append(devices, result.Device)
		}
	}
	return devices
}

// removeVictims runs PreFilter and returns a copy of the state with the
// victims removed, like the dry run of preemption.
func removeVictims(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, victims ...*v1.Pod) fwk.CycleState {
	tCtx.Helper()
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	cs := state.Clone()
	for _, victim := range victims {
		podInfo, err := framework.NewPodInfo(victim)
		tCtx.ExpectNoError(err)
		require.True(tCtx, testCtx.p.RemovePod(tCtx, cs, pod, podInfo, testCtx.nodeInfos[0]).IsSuccess(), "RemovePod")
	}
	return cs
}

func TestHolds(t *testing.T) {
	testHolds(ktesting.Init(t))
}

// testHolds covers devices held for nominated pods with exclusive devices.
func testHolds(tCtx ktesting.TContext) {
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
	p2 := preemptionPod("p2", "p2-claim").Priority(100).Obj()
	low := preemptionPod("low", "low-claim").Priority(50).Obj()
	high := preemptionPod("high", "high-claim").Priority(200).Obj()
	never := preemptionPod("never", "never-claim").Priority(50).PreemptionPolicy(v1.PreemptNever).Obj()
	sharer := preemptionPod("sharer", "p1-claim").Priority(50).Obj()
	// setup creates a node with numDevices devices and a victim claim
	// which has the victimDevices allocated.
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
		return setupNominations(tCtx, claims, []apiruntime.Object{slice.Obj(), victim, p1, p2, low, high, never, sharer})
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) []string {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		status = filterBothPasses(tCtx, testCtx, state, pod)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		return pickedDevices(tCtx, state)
	}

	tCtx.Run("shared-terminating-victim", func(tCtx ktesting.TContext) {
		// KEP example: both preemptors pick the same victim with two
		// devices, each gets one of them.
		testCtx := setup(tCtx, 2, "gpu-0", "gpu-1")
		nominate(tCtx, testCtx, p1, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))

		cs := removeVictims(tCtx, testCtx, p2, victim)
		status := filterBothPasses(tCtx, testCtx, cs, p2)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		require.Equal(tCtx, []string{"gpu-1"}, pickedDevices(tCtx, cs))
		nominate(tCtx, testCtx, p2, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-1"}, heldDevices(testCtx, p2))
	})

	tCtx.Run("surplus-of-victim", func(tCtx ktesting.TContext) {
		// The victim claim keeps gpu-1 allocated until it gets
		// deallocated, so Filter must not offer it.
		testCtx := setup(tCtx, 3, "gpu-0", "gpu-1")
		nominate(tCtx, testCtx, p1, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		require.Equal(tCtx, []string{"gpu-2"}, filter(tCtx, testCtx, low))
	})

	tCtx.Run("holds-that-apply", func(tCtx ktesting.TContext) {
		// Like the nominated pods which RunFilterPluginsWithNominatedPods
		// adds: those of other pods with at least the priority of the
		// pod, also for pods which cannot preempt. Without the hold, the
		// allocator picks gpu-0.
		testCtx := setup(tCtx, 2, "gpu-0")
		nominate(tCtx, testCtx, p1, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		for name, tc := range map[string]struct {
			pod  *v1.Pod
			want string
		}{
			"lower-priority":  {pod: low, want: "gpu-1"},
			"equal-priority":  {pod: p2, want: "gpu-1"},
			"higher-priority": {pod: high, want: "gpu-0"},
			// The held claim is its own claim, like for
			// shares-held-claim. That its victims don't count either
			// is covered by TestHoldsConsumableCapacity/own-victim-claim.
			"holder":         {pod: p1, want: "gpu-0"},
			"cannot-preempt": {pod: never, want: "gpu-1"},
			// The held claim is the claim of the pod, which allocates
			// it for both.
			"shares-held-claim": {pod: sharer, want: "gpu-0"},
		} {
			tCtx.Run(name, func(tCtx ktesting.TContext) {
				require.Equal(tCtx, []string{tc.want}, filter(tCtx, testCtx, tc.pod))
			})
		}
	})

	tCtx.Run("own-hold-ignored", func(tCtx ktesting.TContext) {
		// Preempting again for the same node replaces the old hold.
		testCtx := setup(tCtx, 3, "gpu-0", "gpu-1", "gpu-2")
		nominate(tCtx, testCtx, p1, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
		nominate(tCtx, testCtx, p1, nodeName, victim)
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
	})
}

func TestHoldsConsumableCapacity(t *testing.T) {
	testHoldsConsumableCapacity(ktesting.Init(t))
}

// testHoldsConsumableCapacity covers how held capacity combines with the
// capacity of victim claims which are still allocated.
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
		require.NotNil(tCtx, testCtx.p.nominations, "claimNominations")
		return testCtx
	}
	fits := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, victims ...*v1.Pod) bool {
		cs := removeVictims(tCtx, testCtx, pod, victims...)
		status := filterBothPasses(tCtx, testCtx, cs, pod)
		require.True(tCtx, status.IsSuccess() || status.IsRejected(), "Filter: %v", status)
		return status.IsSuccess()
	}

	tCtx.Run("victim-gone-claim-allocated", func(tCtx ktesting.TContext) {
		// KEP example: after v1 is gone, p2 preempts v2. The capacity
		// of v1 must not be counted together with the hold of p1.
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
		nominate(tCtx, testCtx, p1, nodeName, v1Pod)
		require.Len(tCtx, heldDevices(testCtx, p1), 1, "held for p1")
		deletePod(tCtx, testCtx, v1Pod)

		require.True(tCtx, fits(tCtx, testCtx, p2, v2Pod), "p2 after removing v2")
		require.False(tCtx, fits(tCtx, testCtx, p2), "p2 without removing v2")
	})

	tCtx.Run("holds-share-victim-claim", func(tCtx ktesting.TContext) {
		// Two holds offset by the same victim claim: it counts once.
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
		nominate(tCtx, testCtx, p1, nodeName, victim)
		nominate(tCtx, testCtx, p2, nodeName, victim)
		require.Len(tCtx, heldDevices(testCtx, p2), 1, "held for p2")

		// 60Gi are held, the remaining 40Gi are not enough.
		require.False(tCtx, fits(tCtx, testCtx, low), "low priority pod")
	})

	tCtx.Run("other-victim-claim", func(tCtx ktesting.TContext) {
		// A victim claim of a different preemption does not offset the
		// hold: the capacity freed for the holder stays with it.
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
		nominate(tCtx, testCtx, px, nodeName, vx)
		testCtx.p.RemoveNominatedPod(tCtx, px, "", false)
		nominate(tCtx, testCtx, p, nodeName, v)
		updateClaim(tCtx, testCtx, "v-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})

		require.False(tCtx, fits(tCtx, testCtx, low), "low priority pod")
		testCtx.p.RemoveNominatedPod(tCtx, p, "", false)
		require.True(tCtx, fits(tCtx, testCtx, low), "without the hold")
	})

	tCtx.Run("own-victim-claim", func(tCtx ktesting.TContext) {
		// The claim of the pod's own victim does not offset the hold of
		// another preemptor: while it is allocated, the pod does not get
		// the capacity held for the other one.
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
		nominate(tCtx, testCtx, h, nodeName, vh)
		updateClaim(tCtx, testCtx, "vh-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		nominate(tCtx, testCtx, p, nodeName, v)
		require.Len(tCtx, heldDevices(testCtx, p), 1, "held for p")

		// 40Gi of the victim plus 30Gi held for h leave 30Gi.
		require.False(tCtx, fits(tCtx, testCtx, p), "p before the claim of its victim is deallocated")
		updateClaim(tCtx, testCtx, "v-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
		require.True(tCtx, fits(tCtx, testCtx, p), "p after the claim of its victim is deallocated")
	})
}

func TestJoinVictimClaim(t *testing.T) {
	testJoinVictimClaim(ktesting.Init(t))
}

// testJoinVictimClaim checks that a pod cannot start to use a claim of a
// preempted pod while a nominated preemptor waits for its deallocation.
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
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, replacement, important})
		nominate(tCtx, testCtx, preemptor, nodeName, victim)
		// The victim is gone, its PreBind got cancelled or the claim
		// controller has not caught up yet.
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
		})
		return testCtx
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) (fwk.CycleState, *fwk.Status) {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		return state, filterBothPasses(tCtx, testCtx, state, pod)
	}

	tCtx.Run("rejected", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		state, status := filter(tCtx, testCtx, replacement)
		require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated"), status, "Filter")
		_, status = testCtx.p.PostFilter(tCtx, state, replacement, nil)
		require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "deallocation of ResourceClaim completed"), status, "PostFilter")
	})

	tCtx.Run("higher-priority", func(tCtx ktesting.TContext) {
		// The preemptor only holds against pods with at most its priority.
		testCtx := setup(tCtx)
		_, status := filter(tCtx, testCtx, important)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	})

	tCtx.Run("nomination-ended", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		testCtx.p.RemoveNominatedPod(tCtx, preemptor, "", false)
		_, status := filter(tCtx, testCtx, replacement)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	})
}

func TestHoldReleaseEvents(t *testing.T) {
	testHoldReleaseEvents(ktesting.Init(t))
}

func testHoldReleaseEvents(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim-2").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
		reserve(preemptionClaim("victim-claim-2").Allocation(preemptionAllocation("gpu-1")).Obj(), victim2),
	}
	type event struct {
		device   string
		holder   types.UID
		priority int32
	}

	for name, tc := range map[string]struct {
		change     func(tCtx ktesting.TContext, testCtx *testContext)
		wantEvents []event
	}{
		"deleted": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.RemoveNominatedPod(tCtx, preemptor, "", false)
			},
			wantEvents: []event{{device: "gpu-0", holder: preemptor.UID, priority: 100}},
		},
		"stale-node": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.RemoveNominatedPod(tCtx, preemptor, node2Name, false)
			},
		},
		"assumed-with-held-devices": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
				tCtx.ExpectNoError(err)
				claim = claim.DeepCopy()
				claim.Status.Allocation = preemptionAllocation("gpu-0")
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim))
				testCtx.p.RemoveNominatedPod(tCtx, fromPod(preemptor).Node(nodeName).Obj(), "", false)
			},
		},
		"assumed-with-other-devices": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
				tCtx.ExpectNoError(err)
				claim = claim.DeepCopy()
				claim.Status.Allocation = preemptionAllocation("gpu-1")
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim))
				testCtx.p.RemoveNominatedPod(tCtx, fromPod(preemptor).Node(nodeName).Obj(), "", false)
			},
			wantEvents: []event{{device: "gpu-0", holder: preemptor.UID, priority: 100}},
		},
		"same-hold-again": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				nominate(tCtx, testCtx, preemptor, nodeName, victim)
			},
		},
		"different-hold": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				nominate(tCtx, testCtx, preemptor, nodeName, victim2)
			},
			wantEvents: []event{{device: "gpu-0", holder: preemptor.UID, priority: 100}},
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, victim2, preemptor})
			var events []event
			testCtx.draManager.SetNominationReleaseHandler(func(claim *resourceapi.ResourceClaim, holderUID types.UID, priority int32) {
				events = append(events, event{device: claim.Status.Allocation.Devices.Results[0].Device, holder: holderUID, priority: priority})
			})
			nominate(tCtx, testCtx, preemptor, nodeName, victim)
			require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor))
			tc.change(tCtx, testCtx)
			require.Equal(tCtx, tc.wantEvents, events)
		})
	}
}

func TestNominationWithoutHold(t *testing.T) {
	testNominationWithoutHold(ktesting.Init(t))
}

// testNominationWithoutHold covers a preemptor whose claims cannot be
// allocated in the simulation, for example because the node changed since
// the dry run. Nothing gets held, but the claims of its victims are still
// protected and still defer preemption.
func testNominationWithoutHold(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	replacement := preemptionPod("replacement", "victim-claim").Priority(10).Obj()
	claims := []*resourceapi.ResourceClaim{
		// Two devices, the node only has one.
		preemptionClaim("preemptor-claim").Request(className).Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, replacement})
	var events int
	testCtx.draManager.SetNominationReleaseHandler(func(*resourceapi.ResourceClaim, types.UID, int32) { events++ })
	nominate(tCtx, testCtx, preemptor, nodeName, victim)
	require.NotNil(tCtx, testCtx.p.nominations.nomination(preemptor.UID), "nomination")
	require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices")

	eligible, _ := testCtx.p.PodEligibleToPreempt(tCtx, preemptor, nodeName)
	require.False(tCtx, eligible, "preemptor eligible to preempt again")
	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
	})
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, replacement, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	status = filterBothPasses(tCtx, testCtx, state, replacement)
	require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated"), status, "Filter of a pod which uses the claim of the victim")

	testCtx.p.RemoveNominatedPod(tCtx, preemptor, "", false)
	require.Zero(tCtx, events, "release events")
}

func TestHoldsWithFramework(t *testing.T) {
	testHoldsWithFramework(ktesting.Init(t))
}

// testHoldsWithFramework runs both passes through the framework, with a
// real nominator and a plugin instance of another profile than the one
// which recorded the nomination: pass 2, whose result gets stored, must not
// take the held device, although nothing uses it.
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
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, preemptor, low})
	nominate(tCtx, testCtx, preemptor, nodeName, victim)
	require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor))
	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
	})

	// The nominator makes the framework add the preemptor and therefore run
	// both passes. It only knows the priority of pods which are in the queue.
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
		tf.RegisterPluginAsExtensions(Name, newPlugin, "PreFilter", "Filter"),
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

	state := framework.NewCycleState()
	_, status, _ := fh.RunPreFilterPlugins(tCtx, state, low)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	status = fh.RunFilterPluginsWithNominatedPods(tCtx, state, low, testCtx.nodeInfos[0])
	require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
	require.Equal(tCtx, []string{"gpu-1"}, pickedDevices(tCtx, state))
}

func TestHoldsConcurrency(t *testing.T) {
	testHoldsConcurrency(ktesting.Init(t))
}

// testHoldsConcurrency is meant for the race detector. While devices are
// held, Filter runs in both passes for several nodes in parallel on the same
// CycleState, as in the scheduler, while queueing hints run and nominations
// change.
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
		nominationStates[i] = nominate(tCtx, testCtx, preemptors[i], nodes[i].Name, victims[i])
	}
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, low, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	logger := klog.FromContext(tCtx)

	var wg sync.WaitGroup
	statuses := make([]*fwk.Status, numNodes)
	for i, nodeInfo := range testCtx.nodeInfos {
		wg.Go(func() {
			if status := testCtx.p.Filter(tCtx, state.Clone(), low, nodeInfo); !status.IsSuccess() {
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
		}
	})
	wg.Go(func() {
		for range 10 {
			testCtx.p.RemoveNominatedPod(tCtx, preemptors[1], "", false)
			testCtx.p.AddNominatedPod(tCtx, nominationStates[1], preemptors[1], nodes[1].Name, []*v1.Pod{victims[1]})
		}
	})
	wg.Wait()
	for i, status := range statuses {
		require.True(tCtx, status.IsSuccess(), "Filter on %s: %v", nodes[i].Name, status)
	}
}
