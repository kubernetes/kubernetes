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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestNominationExtensions(t *testing.T) {
	testNominationExtensions(ktesting.Init(t))
}

// testNominationExtensions verifies when AddNomination and RemoveNomination
// emit EventResourceClaimHoldReleased because held devices became free without
// a ResourceClaim event.
func testNominationExtensions(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	assumedPreemptor := fromPod(preemptor).Node(nodeName).Obj()
	setup := func(tCtx ktesting.TContext) *testContext {
		testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
		}, []apiruntime.Object{slice, victim, preemptor})
		requireEvents(tCtx, 0, nominate(tCtx, testCtx, preemptor, nil, victim))
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor), "held devices")
		return testCtx
	}
	signalPending := func(tCtx ktesting.TContext, testCtx *testContext, allocation *resourceapi.AllocationResult) {
		claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
		tCtx.ExpectNoError(err)
		claim = claim.DeepCopy()
		claim.Status.Allocation = allocation
		tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim))
	}

	tCtx.Run("nomination-ended", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		requireEvents(tCtx, 1, endNomination(tCtx, testCtx, preemptor))
		require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices")
	})

	tCtx.Run("renominated-same-devices", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		requireEvents(tCtx, 0, nominate(tCtx, testCtx, preemptor, nil, victim))
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor), "held devices")
	})

	tCtx.Run("renominated-other-devices", func(tCtx ktesting.TContext) {
		// Re-nominating without removing victim selects free gpu-1, releasing held gpu-0.
		testCtx := setup(tCtx)
		requireEvents(tCtx, 1, nominate(tCtx, testCtx, preemptor, nil))
		require.Equal(tCtx, []string{"gpu-1"}, heldDevices(testCtx, preemptor), "held devices")
	})

	tCtx.Run("renominated-other-node", func(tCtx ktesting.TContext) {
		// PodNominator calls RemoveNomination on the old node before AddNomination on the new node.
		testCtx := setup(tCtx)
		hold := &allocation{nodeName: node2Name, claims: testCtx.p.store.nominatedAllocation(preemptor.UID).claims}
		requireEvents(tCtx, 1, endNomination(tCtx, testCtx, preemptor))
		requireEvents(tCtx, 0, testCtx.p.AddNomination(tCtx, preemptor, node2Name, hold))
		require.Same(tCtx, hold, testCtx.p.store.nominatedAllocation(preemptor.UID), "hold on the new node")
	})

	tCtx.Run("renominated-claim-recreated", func(tCtx ktesting.TContext) {
		// Claim was recreated under the same name with a new UID; keeping the same device emits no release event.
		testCtx := setup(tCtx)
		recreated := make(map[types.UID]heldClaim)
		for uid, held := range testCtx.p.store.nominatedAllocation(preemptor.UID).claims {
			recreated[uid+"-recreated"] = held
		}
		requireEvents(tCtx, 0, testCtx.p.AddNomination(tCtx, preemptor, nodeName, &allocation{nodeName: nodeName, claims: recreated}))
	})

	tCtx.Run("replaced-by-other-node", func(tCtx ktesting.TContext) {
		// Moving the hold directly to another node releases the old node's hold even if device names match.
		testCtx := setup(tCtx)
		hold := &allocation{nodeName: node2Name, claims: testCtx.p.store.nominatedAllocation(preemptor.UID).claims}
		requireEvents(tCtx, 1, testCtx.p.AddNomination(tCtx, preemptor, node2Name, hold))
	})

	for name, prepare := range map[string]func(testCtx *testContext) any{
		"nothing-prepared": func(*testContext) any { return nil },
		"prepared-for-other-node": func(testCtx *testContext) any {
			return &allocation{nodeName: node2Name, claims: testCtx.p.store.nominatedAllocation(preemptor.UID).claims}
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			// Re-nominating with nil (or mismatched-node) prepared state drops
			// the existing hold and emits one event; repeating with no active
			// hold is a no-op.
			testCtx := setup(tCtx)
			prepared := prepare(testCtx)
			requireEvents(tCtx, 1, testCtx.p.AddNomination(tCtx, preemptor, nodeName, prepared))
			require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices")
			requireEvents(tCtx, 0, testCtx.p.AddNomination(tCtx, preemptor, nodeName, prepared))
		})
	}

	// twoClaims builds a two-claim hold so partial-release tests exercise both
	// map iteration orders.
	twoClaims := func(deviceA, deviceB string) map[types.UID]heldClaim {
		claims := map[types.UID]heldClaim{
			"claim-a": {claim: types.NamespacedName{Namespace: namespace, Name: "claim-a"}, allocation: preemptionAllocation(deviceA)},
		}
		if deviceB != "" {
			claims["claim-b"] = heldClaim{claim: types.NamespacedName{Namespace: namespace, Name: "claim-b"}, allocation: preemptionAllocation(deviceB)}
		}
		return claims
	}
	for name, newer := range map[string]map[types.UID]heldClaim{
		"renominated-partially-kept": twoClaims("gpu-0", "gpu-2"),
		"renominated-claim-missing":  twoClaims("gpu-0", ""),
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupVictims(tCtx, nil, []apiruntime.Object{slice, preemptor})
			for range 20 {
				testCtx.p.AddNomination(tCtx, preemptor, nodeName, &allocation{nodeName: nodeName, claims: twoClaims("gpu-0", "gpu-1")})
				result := testCtx.p.AddNomination(tCtx, preemptor, nodeName, &allocation{nodeName: nodeName, claims: newer})
				require.Len(tCtx, result.Events, 1, "events")
				require.Equal(tCtx, "claim-b", result.Events[0].OldObj.(*resourceapi.ResourceClaim).Name, "released claim")
			}
		})
	}

	tCtx.Run("other-node", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		requireEvents(tCtx, 0, testCtx.p.RemoveNomination(tCtx, preemptor, node2Name))
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor), "held devices")
	})

	tCtx.Run("not-nominated", func(tCtx ktesting.TContext) {
		testCtx := setupVictims(tCtx, nil, nil)
		requireEvents(tCtx, 0, endNomination(tCtx, testCtx, preemptor))
	})

	// Assumed pod's nomination ends after Reserve and Permit while its allocation is still pending.
	tCtx.Run("reserved-held-devices", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		signalPending(tCtx, testCtx, preemptionAllocation("gpu-0"))
		requireEvents(tCtx, 0, testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName))
		require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices")
	})

	tCtx.Run("reserved-held-devices-new-share", func(tCtx ktesting.TContext) {
		// Freshly generated ShareIDs do not prevent matching held devices.
		testCtx := setup(tCtx)
		allocation := preemptionAllocation("gpu-0")
		allocation.Devices.Results[0].ShareID = new(types.UID("share"))
		signalPending(tCtx, testCtx, allocation)
		requireEvents(tCtx, 0, testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName))
	})

	tCtx.Run("ended-during-own-cycle", func(tCtx ktesting.TContext) {
		// Unassigned pod deleted between Reserve and Permit: Unreserve will
		// drop the pending allocation silently, so RemoveNomination must emit
		// an event.
		testCtx := setup(tCtx)
		signalPending(tCtx, testCtx, preemptionAllocation("gpu-0"))
		requireEvents(tCtx, 1, testCtx.p.RemoveNomination(tCtx, preemptor, nodeName))
	})

	tCtx.Run("reserved-other-devices", func(tCtx ktesting.TContext) {
		testCtx := setup(tCtx)
		signalPending(tCtx, testCtx, preemptionAllocation("gpu-1"))
		requireEvents(tCtx, 1, testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName))
	})

	tCtx.Run("reserved-through-filter", func(tCtx ktesting.TContext) {
		// End-to-end: once victim-claim is deallocated, Filter reuses held
		// gpu-0 and Reserve records it as pending; if Unreserve runs before
		// RemoveNomination, RemoveNomination emits a release event.
		for _, unreserve := range []bool{false, true} {
			tCtx.Run(fmt.Sprintf("unreserve-%v", unreserve), func(tCtx ktesting.TContext) {
				testCtx := setup(tCtx)
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
					claim.Status.Allocation = nil
				})
				state := framework.NewCycleState()
				_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
				require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
				status = filterWithNominated(tCtx, testCtx, state, preemptor, preemptor)
				require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
				require.Equal(tCtx, []string{"gpu-0"}, pickedDevices(tCtx, state))
				status = testCtx.p.Reserve(tCtx, state, assumedPreemptor, nodeName)
				require.True(tCtx, status.IsSuccess(), "Reserve: %v", status)
				if unreserve {
					// Failure handler passes the unassigned pod from the informer to RemoveNomination.
					testCtx.p.Unreserve(tCtx, state, assumedPreemptor, nodeName)
					requireEvents(tCtx, 1, testCtx.p.RemoveNomination(tCtx, preemptor, nodeName))
					return
				}
				requireEvents(tCtx, 0, testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName))
			})
		}
	})

	tCtx.Run("bound", func(tCtx ktesting.TContext) {
		// Claim was already committed with the held device (e.g. by a
		// co-consumer sharing the claim), so ending the nomination frees nothing.
		testCtx := setup(tCtx)
		updateClaim(tCtx, testCtx, "preemptor-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.Allocation = preemptionAllocation("gpu-0")
			claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(preemptor)}
		})
		requireEvents(tCtx, 0, endNomination(tCtx, testCtx, preemptor))
	})

	tCtx.Run("bound-claim-recreated", func(tCtx ktesting.TContext) {
		// Claim was recreated under the same name and allocated the held
		// device, so the device remains in use.
		testCtx := setup(tCtx)
		old, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
		tCtx.ExpectNoError(err)
		tCtx.ExpectNoError(testCtx.client.ResourceV1().ResourceClaims(namespace).Delete(tCtx, "preemptor-claim", metav1.DeleteOptions{}))
		_, err = testCtx.client.ResourceV1().ResourceClaims(namespace).Create(tCtx, preemptionClaim("preemptor-claim").Obj(), metav1.CreateOptions{})
		tCtx.ExpectNoError(err)
		eventually(tCtx, "claim created again", func() bool {
			claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
			return err == nil && claim.UID != old.UID
		})
		updateClaim(tCtx, testCtx, "preemptor-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.Allocation = preemptionAllocation("gpu-0")
		})
		requireEvents(tCtx, 0, endNomination(tCtx, testCtx, preemptor))
	})

	tCtx.Run("renominated-nothing-to-allocate", func(tCtx ktesting.TContext) {
		// Another pod already allocated the shared claim with gpu-0, so
		// re-nomination prepares nil and drops the hold without emitting an event.
		testCtx := setup(tCtx)
		updateClaim(tCtx, testCtx, "preemptor-claim", func(claim *resourceapi.ResourceClaim) {
			claim.Status.Allocation = preemptionAllocation("gpu-0")
		})
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		requireEvents(tCtx, 0, addNomination(tCtx, testCtx, state, preemptor, nodeName))
		require.Nil(tCtx, testCtx.p.store.nominatedAllocation(preemptor.UID), "hold")
	})
}

func requireEvents(tCtx ktesting.TContext, want int, result framework.NominationChangeResult) {
	tCtx.Helper()
	require.Len(tCtx, result.Events, want, "events")
	for _, event := range result.Events {
		require.Equal(tCtx, framework.EventResourceClaimHoldReleased, event.Event)
		oldClaim, newClaim := event.OldObj.(*resourceapi.ResourceClaim), event.NewObj.(*resourceapi.ResourceClaim)
		require.Equal(tCtx, "preemptor-claim", oldClaim.Name)
		require.NotNil(tCtx, oldClaim.Status.Allocation, "old allocation")
		require.Nil(tCtx, newClaim.Status.Allocation, "new allocation")
	}
}

func TestNominationExtensionsGateDisabled(t *testing.T) {
	testNominationExtensionsGateDisabled(ktesting.Init(t))
}

func testNominationExtensionsGateDisabled(tCtx ktesting.TContext) {
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, nil, nil, feature.Features{})
	pod := preemptionPod("pod", "claim").Obj()
	state := framework.NewCycleState()
	prepared := testCtx.p.PrepareNomination(tCtx, state, pod, nodeName)
	require.Nil(tCtx, prepared, "PrepareNomination")
	require.Empty(tCtx, testCtx.p.AddNomination(tCtx, pod, nodeName, &allocation{nodeName: nodeName}).Events, "AddNomination")
	require.Empty(tCtx, testCtx.p.RemoveNomination(tCtx, pod, nodeName).Events, "RemoveNomination")
	testCtx.p.AssumeVictim(tCtx, fromPod(pod).Node(nodeName).Obj())
	eligible, _ := testCtx.p.PodEligibleToPreempt(tCtx, pod, nodeName)
	require.True(tCtx, eligible, "PodEligibleToPreempt")
}
