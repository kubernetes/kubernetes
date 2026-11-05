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

// filterWithNominated mimics RunFilterPluginsWithNominatedPods on workerNode.
// If any nominated pod has priority >= pod, it runs pass 1 on a cloned state
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
	hold := testCtx.p.preemptionTracker.hold(pod.UID)
	if hold == nil {
		return nil
	}
	var devices []string
	for _, held := range hold.heldClaims {
		for _, result := range held.allocatedDevices.Results {
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
	state := simulatePreemption(tCtx, testCtx, preemptor, nominated, victims...)
	assumeVictims(tCtx, testCtx, victims...)
	return addNomination(tCtx, testCtx, state, preemptor, nodeName)
}

func addNomination(tCtx ktesting.TContext, testCtx *testContext, state fwk.CycleState, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	tCtx.Helper()
	prepared := testCtx.p.PrepareNomination(tCtx, state, pod, nodeName)
	return testCtx.p.AddNomination(tCtx, pod, nodeName, prepared)
}

func simulatePreemption(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nominated []*v1.Pod, victims ...*v1.Pod) fwk.CycleState {
	tCtx.Helper()
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	cs := state.Clone()
	for _, victim := range victims {
		removePod(tCtx, testCtx, cs, preemptor, victim, testCtx.nodeInfos[0])
	}
	status = filterWithNominated(tCtx, testCtx, cs, preemptor, nominated...)
	require.True(tCtx, status.IsSuccess(), "simulation of %s: %v", preemptor.Name, status)
	return state
}

func endNomination(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) framework.NominationChangeResult {
	tCtx.Helper()
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
	setup := func(tCtx ktesting.TContext, numDevices int, victimDevices ...string) *testContext {
		tCtx.Helper()
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
		return setupVictims(tCtx, claims, []apiruntime.Object{slice.Obj(), victim, p1, p2, low, high, never})
	}
	deallocateVictim := func(tCtx ktesting.TContext, testCtx *testContext) {
		tCtx.Helper()
		updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
		})
	}
	filter := func(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod, nominated ...*v1.Pod) []string {
		tCtx.Helper()
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		status = filterWithNominated(tCtx, testCtx, state, pod, nominated...)
		require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
		return pickedDevices(tCtx, state)
	}

	filterCases := map[string]struct {
		numDevices    int
		victimDevices []string
		keepVictim    bool
		mutate        func(tCtx ktesting.TContext, testCtx *testContext)
		pod           *v1.Pod
		wantDevice    string
	}{
		"surplus-of-victim": {
			// Victim's unheld second device (gpu-1) stays unavailable until
			// victim-claim is deallocated.
			numDevices:    3,
			victimDevices: []string{"gpu-0", "gpu-1"},
			keepVictim:    true,
			pod:           low,
			wantDevice:    "gpu-2",
		},
		// Holds block pods with priority <= holder (including PreemptNever pods),
		// but not higher-priority pods or the holder itself.
		"holds-that-apply/lower-priority":  {pod: low, wantDevice: "gpu-1"},
		"holds-that-apply/equal-priority":  {pod: p2, wantDevice: "gpu-1"},
		"holds-that-apply/higher-priority": {pod: high, wantDevice: "gpu-0"},
		"holds-that-apply/holder":          {pod: p1, wantDevice: "gpu-0"},
		"holds-that-apply/cannot-preempt":  {pod: never, wantDevice: "gpu-1"},
		"held-claim-allocated": {
			// Once p1-claim is already allocated (on gpu-1), p1's stale hold on
			// gpu-0 no longer blocks low.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "p1-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("gpu-1")
				})
			},
			pod:        low,
			wantDevice: "gpu-0",
		},
		"held-claim-pending": {
			// A pod sharing p1-claim waits in Permit with a pending allocation
			// on gpu-1, which AllocatedState counts, so p1's stale hold on gpu-0
			// no longer blocks low.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				pending, err := testCtx.draManager.ResourceClaims().Get(namespace, "p1-claim")
				tCtx.ExpectNoError(err)
				pending = pending.DeepCopy()
				pending.Status.Allocation = preemptionAllocation("gpu-1")
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(pending.UID, pending))
			},
			pod:        low,
			wantDevice: "gpu-0",
		},
		"held-claim-recreated": {
			// Recreating p1-claim with a new UID invalidates the hold recorded for
			// the old UID.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				old, err := testCtx.draManager.ResourceClaims().Get(namespace, "p1-claim")
				tCtx.ExpectNoError(err)
				tCtx.ExpectNoError(testCtx.client.ResourceV1().ResourceClaims(namespace).Delete(tCtx, "p1-claim", metav1.DeleteOptions{}))
				_, err = testCtx.client.ResourceV1().ResourceClaims(namespace).Create(tCtx, preemptionClaim("p1-claim").Obj(), metav1.CreateOptions{})
				tCtx.ExpectNoError(err)
				eventually(tCtx, "claim created again", func() bool {
					claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "p1-claim")
					return err == nil && claim.UID != old.UID
				})
			},
			pod:        low,
			wantDevice: "gpu-0",
		},
	}

	for name, tc := range filterCases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			numDevices := tc.numDevices
			if numDevices == 0 {
				numDevices = 2
			}
			victimDevices := tc.victimDevices
			if victimDevices == nil {
				victimDevices = []string{"gpu-0"}
			}
			testCtx := setup(tCtx, numDevices, victimDevices...)
			nominate(tCtx, testCtx, p1, nil, victim)
			require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, p1))
			if !tc.keepVictim {
				deallocateVictim(tCtx, testCtx)
			}
			if tc.mutate != nil {
				tc.mutate(tCtx, testCtx)
			}
			require.Equal(tCtx, []string{tc.wantDevice}, filter(tCtx, testCtx, tc.pod, p1))
		})
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
	})
}

func TestHoldsAddPod(t *testing.T) {
	testHoldsAddPod(ktesting.Init(t))
}

// testHoldsAddPod verifies that AddPod records holds only for other unassigned
// pods nominated to the target node, without mutating the original CycleState,
// and that RemovePod undoes it.
func testHoldsAddPod(tCtx ktesting.TContext) {
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	p1 := preemptionPod("p1", "p1-claim").Priority(100).Obj()
	low := preemptionPod("low", "low-claim").Priority(50).Obj()
	slice := st.MakeResourceSlice(nodeName, driver).Devices("gpu-0", "gpu-1").Obj()

	holders := func(cs fwk.CycleState) []types.UID {
		var uids []types.UID
		for uid := range getPreemptionSimulationState(cs).holds {
			uids = append(uids, uid)
		}
		slices.Sort(uids)
		return uids
	}

	type op struct {
		podToSchedule *v1.Pod
		pod           *v1.Pod
		nodeIdx       int
		remove        bool
		wantHolders   []types.UID
	}
	testcases := map[string]struct {
		ops []op
	}{
		"other-node": {
			ops: []op{{podToSchedule: low, pod: p1, nodeIdx: 1}},
		},
		"assigned-pod": {
			ops: []op{{podToSchedule: low, pod: fromPod(p1).Node(nodeName).Obj(), nodeIdx: 0}},
		},
		"self": {
			ops: []op{{podToSchedule: p1, pod: p1, nodeIdx: 0}},
		},
		"nominated-pod": {
			ops: []op{{podToSchedule: low, pod: p1, nodeIdx: 0, wantHolders: []types.UID{p1.UID}}},
		},
		"remove-and-readd": {
			ops: []op{
				{podToSchedule: low, pod: p1, nodeIdx: 0, wantHolders: []types.UID{p1.UID}},
				{podToSchedule: low, pod: p1, nodeIdx: 0, remove: true},
				{podToSchedule: low, pod: p1, nodeIdx: 0, wantHolders: []types.UID{p1.UID}},
			},
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
				preemptionClaim("p1-claim").Obj(),
				preemptionClaim("low-claim").Obj(),
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
			}, []apiruntime.Object{slice, victim, p1, low})
			nominate(tCtx, testCtx, p1, nil, victim)

			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, low, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			cs := state.Clone()

			for i, step := range tc.ops {
				nodeInfo := testCtx.nodeInfos[step.nodeIdx]
				if step.remove {
					removePod(tCtx, testCtx, cs, step.podToSchedule, step.pod, nodeInfo)
				} else {
					addPod(tCtx, testCtx, cs, step.podToSchedule, step.pod, nodeInfo)
				}
				require.Equal(tCtx, step.wantHolders, holders(cs), "step %d: holders", i)
				require.Empty(tCtx, holders(state), "step %d: original state mutated", i)
			}
		})
	}
}

func TestFilterResultReuse(t *testing.T) {
	testFilterResultReuse(ktesting.Init(t))
}

// testFilterResultReuse verifies that cached Filter allocations are reused only
// when holds is nil and the set of released claims on the node matches.
func testFilterResultReuse(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Devices("gpu-0", "gpu-1", "gpu-2").Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	holder := preemptionPod("holder", "holder-claim").Priority(200).Obj()

	type step struct {
		// add reprieves a victim pod on the base simulation state before Filter.
		add *v1.Pod
		// withHolder runs Filter on a pass-1 clone with holder added (no pass 2).
		withHolder bool
		// twoPassWithHolder runs filterWithNominated (pass 1 with holder, then
		// pass 2 on base state asserting both passes pick the same devices).
		twoPassWithHolder bool
		wantFit           bool
		wantDevices       []string
	}
	testcases := map[string]struct {
		mutate         func(tCtx ktesting.TContext, testCtx *testContext)
		nominateHolder bool
		wantHeld       []string
		removeVictims  []*v1.Pod
		steps          []step
	}{
		"reprieve": {
			// gpu-2 is used by a non-victim pod.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "holder-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("gpu-2")
					claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(holder)}
				})
			},
			removeVictims: []*v1.Pod{v1Pod, v2Pod},
			steps: []step{
				{wantFit: true, wantDevices: []string{"gpu-0"}},
				{add: v2Pod, wantFit: true},
				{add: v1Pod, wantFit: false},
			},
		},
		"pass-one-only": {
			// Pass 1 succeeded with v1 removed, then a later plugin rejected it and
			// v1 was reprieved; the next Filter call has a different
			// releasedClaimsKey and must not reuse gpu-0.
			nominateHolder: true,
			wantHeld:       []string{"gpu-2"},
			removeVictims:  []*v1.Pod{v1Pod},
			steps: []step{
				{withHolder: true, wantFit: true, wantDevices: []string{"gpu-0"}},
				{add: v1Pod, wantFit: true, wantDevices: []string{"gpu-2"}},
			},
		},
		"pass-two": {
			// Pass 2 runs without holds and must reuse pass 1's gpu-2 choice
			// instead of taking held gpu-0.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "v1-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
					claim.Status.Allocation = nil
				})
			},
			nominateHolder: true,
			wantHeld:       []string{"gpu-0"},
			steps: []step{
				{twoPassWithHolder: true, wantFit: true, wantDevices: []string{"gpu-2"}},
			},
		},
		"holds-after-result-without-holds": {
			// A cached result computed without holds must not be reused when holds are active.
			mutate: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "v1-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
					claim.Status.Allocation = nil
				})
			},
			nominateHolder: true,
			wantHeld:       []string{"gpu-0"},
			steps: []step{
				{wantFit: true, wantDevices: []string{"gpu-0"}},
				{withHolder: true, wantFit: true, wantDevices: []string{"gpu-2"}},
			},
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				preemptionClaim("holder-claim").Obj(),
				reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
				reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod),
			}, []apiruntime.Object{slice, v1Pod, v2Pod, preemptor, holder})
			if tc.mutate != nil {
				tc.mutate(tCtx, testCtx)
			}
			if tc.nominateHolder {
				nominate(tCtx, testCtx, holder, nil)
				require.Equal(tCtx, tc.wantHeld, heldDevices(testCtx, holder))
			}

			nodeInfo := testCtx.nodeInfos[0]
			cs := removeVictims(tCtx, testCtx, preemptor, tc.removeVictims...)
			for i, step := range tc.steps {
				if step.add != nil {
					addPod(tCtx, testCtx, cs, preemptor, step.add, nodeInfo)
				}
				var status *fwk.Status
				target := cs
				switch {
				case step.twoPassWithHolder:
					status = filterWithNominated(tCtx, testCtx, cs, preemptor, holder)
				case step.withHolder:
					target = cs.Clone()
					addPod(tCtx, testCtx, target, preemptor, holder, nodeInfo)
					status = testCtx.p.Filter(tCtx, target, preemptor, nodeInfo)
				default:
					status = testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo)
				}
				require.True(tCtx, status.IsSuccess() || status.IsRejected(), "step %d: Filter: %v", i, status)
				require.Equal(tCtx, step.wantFit, status.IsSuccess(), "step %d: Filter fit", i)
				if step.wantDevices != nil {
					require.Equal(tCtx, step.wantDevices, pickedDevices(tCtx, target), "step %d: picked devices", i)
				}
			}
		})
	}
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
	var hintErr error
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
			if _, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, preemptors[0], victims[0], nil); err != nil {
				hintErr = err
				return
			}
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
	tCtx.ExpectNoError(hintErr)
	for i, status := range statuses {
		require.True(tCtx, status.IsSuccess(), "Filter on %s: %v", nodes[i].Name, status)
	}
}

func TestPreemptionSimulationConcurrency(t *testing.T) {
	testPreemptionSimulationConcurrency(ktesting.Init(t))
}

// testPreemptionSimulationConcurrency exercises parallel multi-node preemption
// simulations with holds alongside concurrent nomination changes, victim
// tracking, queueing hints, and informer claim deallocation.
func testPreemptionSimulationConcurrency(tCtx ktesting.TContext) {
	const numNodes = 4
	var nodes []*v1.Node
	var claims []*resourceapi.ResourceClaim
	var objs []apiruntime.Object
	var holders, victims1, victims2 []*v1.Pod
	for i := range numNodes {
		node := st.MakeNode().Name(fmt.Sprintf("node-%d", i)).Obj()
		victim1 := preemptionPod(fmt.Sprintf("victim1-%d", i), fmt.Sprintf("victim1-claim-%d", i)).Node(node.Name).Priority(10).Obj()
		victim2 := preemptionPod(fmt.Sprintf("victim2-%d", i), fmt.Sprintf("victim2-claim-%d", i)).Node(node.Name).Priority(20).Obj()
		holder := preemptionPod(fmt.Sprintf("holder-%d", i), fmt.Sprintf("holder-claim-%d", i)).Priority(100).Obj()
		claims = append(claims,
			preemptionClaim(fmt.Sprintf("holder-claim-%d", i)).Obj(),
			reserve(preemptionClaim(fmt.Sprintf("victim1-claim-%d", i)).Allocation(poolAllocation(node.Name, "gpu-0")).Obj(), victim1),
			reserve(preemptionClaim(fmt.Sprintf("victim2-claim-%d", i)).Allocation(poolAllocation(node.Name, "gpu-1")).Obj(), victim2),
		)
		objs = append(objs, st.MakeResourceSlice(node.Name, driver).Device("gpu-0").Device("gpu-1").Obj(), victim1, victim2, holder)
		nodes = append(nodes, node)
		holders = append(holders, holder)
		victims1 = append(victims1, victim1)
		victims2 = append(victims2, victim2)
	}
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(50).Obj()
	claims = append(claims, preemptionClaim("preemptor-claim").Obj())
	objs = append(objs, preemptor)
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, nodes, claims, objs, feature.Features{EnableDRAPreemption: true})

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
		require.NotNil(tCtx, testCtx.p.preemptionTracker.hold(holders[i].UID), "hold of %s", holders[i].Name)
		holderStates[i] = state
	}

	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	logger := klog.FromContext(tCtx)
	podInfo := func(pod *v1.Pod) fwk.PodInfo {
		tCtx.Helper()
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
	errs := make([]error, numNodes+2)
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
				if status := testCtx.p.Filter(tCtx, passOne, preemptor, nodeInfo); !status.IsSuccess() {
					if !status.IsRejected() {
						errs[i] = status.AsError()
					}
					return false
				}
				status := testCtx.p.Filter(tCtx, cs, preemptor, nodeInfo)
				if !status.IsSuccess() && !status.IsRejected() {
					errs[i] = status.AsError()
				}
				return status.IsSuccess()
			}
			if status := testCtx.p.RemovePod(tCtx, cs, preemptor, victim1Infos[i], nodeInfo); !status.IsSuccess() {
				errs[i] = status.AsError()
				return
			}
			if status := testCtx.p.RemovePod(tCtx, cs, preemptor, victim2Infos[i], nodeInfo); !status.IsSuccess() {
				errs[i] = status.AsError()
				return
			}
			// Concurrent claim/nomination updates may cause a simulated pass to fail; skip reprieve if so.
			if !fits() {
				return
			}
			for _, victim := range []fwk.PodInfo{victim2Infos[i], victim1Infos[i]} {
				if status := testCtx.p.AddPod(tCtx, cs, preemptor, victim, nodeInfo); !status.IsSuccess() {
					errs[i] = status.AsError()
					return
				}
				if !fits() {
					if status := testCtx.p.RemovePod(tCtx, cs, preemptor, victim, nodeInfo); !status.IsSuccess() {
						errs[i] = status.AsError()
						return
					}
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
			if _, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, holders[1], victims1[1], nil); err != nil {
				errs[numNodes+1] = err
				return
			}
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
		return len(testCtx.p.preemptionTracker.victimClaimsOnNode(nodes[3].Name)) == 0
	})
}
