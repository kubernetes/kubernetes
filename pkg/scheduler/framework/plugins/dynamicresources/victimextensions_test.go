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
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestVictimClaimTracker(t *testing.T) {
	victim := func(name string) map[types.UID]types.NamespacedName {
		return map[types.UID]types.NamespacedName{types.UID(name): {Namespace: namespace, Name: name}}
	}
	newClaim := func(uid string, victims map[types.UID]types.NamespacedName) *victimClaim {
		return &victimClaim{claim: types.NamespacedName{Namespace: namespace, Name: uid}, uid: types.UID(uid), victims: victims}
	}

	t.Run("merge-victims", func(t *testing.T) {
		tracker := newVictimClaimTracker()
		tracker.add("a", newClaim("c", victim("v1")))
		tracker.add("a", newClaim("c", victim("v2")))
		claims := tracker.onNode("a")
		require.Len(t, claims, 1)
		require.Equal(t, map[types.UID]types.NamespacedName{
			"v1": {Namespace: namespace, Name: "v1"},
			"v2": {Namespace: namespace, Name: "v2"},
		}, claims[0].victims)
		require.Empty(t, tracker.onNode("b"))
	})

	t.Run("forget", func(t *testing.T) {
		tracker := newVictimClaimTracker()
		before := newClaim("c", victim("v1"))
		tracker.add("a", before)
		tracker.add("a", newClaim("c", victim("v2")))
		// The instance got replaced, so forgetting the old one does nothing.
		tracker.forget("c", before)
		require.True(t, tracker.isTracked("c"))
		tracker.forget("c", nil)
		require.False(t, tracker.isTracked("c"))
		require.Empty(t, tracker.onNode("a"))
	})
}

func TestAssumeVictim(t *testing.T) {
	testAssumeVictim(ktesting.Init(t))
}

// testAssumeVictim checks which claims of victims get tracked and that
// those which other pods keep using get forgotten.
func testAssumeVictim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Device("instance-3").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim", "unallocated-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim-2").Node(nodeName).Obj()
	running := preemptionPod("running", "victim-claim-2").Node(nodeName).Obj()
	victim3 := preemptionPod("victim3", "victim-claim-3").Node(nodeName).Obj()
	victim4 := preemptionPod("victim4", "victim-claim-3").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		preemptionClaim("unallocated-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		// Shared with a pod which keeps running, so not released.
		st.FromResourceClaim(preemptionClaim("victim-claim-2").Allocation(preemptionAllocation("instance-2")).Obj()).
			ReservedFor(consumerRef(victim2), consumerRef(running)).Obj(),
		// Shared by two victims, released once both are preempted.
		st.FromResourceClaim(preemptionClaim("victim-claim-3").Allocation(preemptionAllocation("instance-3")).Obj()).
			ReservedFor(consumerRef(victim3), consumerRef(victim4)).Obj(),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, victim2, running, victim3, victim4})
	tracked := func() []string {
		var names []string
		for _, claim := range testCtx.p.victimClaims.onNode(nodeName) {
			names = append(names, claim.claim.String())
		}
		slices.Sort(names)
		return names
	}
	reason := func() string {
		_, reason := testCtx.p.PodEligibleToPreempt(tCtx, preemptor, nodeName)
		return reason
	}

	// Victims which are not bound to a node have nothing to release there.
	testCtx.p.AssumeVictim(tCtx, preemptionPod("unassigned", "victim-claim").Obj())
	require.Empty(tCtx, tracked(), "claims of unassigned victim")

	testCtx.p.AssumeVictim(tCtx, victim)
	testCtx.p.AssumeVictim(tCtx, victim2)
	testCtx.p.AssumeVictim(tCtx, victim3)
	require.Equal(tCtx, []string{"default/victim-claim", "default/victim-claim-2", "default/victim-claim-3"}, tracked(), "claims after AssumeVictim")
	require.Equal(tCtx, "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/victim-claim.", reason(),
		"claims which other pods keep using don't block")
	require.Equal(tCtx, []string{"default/victim-claim"}, tracked(), "claims after PodEligibleToPreempt")

	testCtx.p.AssumeVictim(tCtx, victim3)
	testCtx.p.AssumeVictim(tCtx, victim4)
	require.Equal(tCtx, "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/victim-claim, default/victim-claim-3.", reason(),
		"claim of two victims")
}

func TestPodEligibleToPreempt(t *testing.T) {
	testPodEligibleToPreempt(ktesting.Init(t))
}

// testPodEligibleToPreempt checks when a preemptor has to wait for claims
// of preempted pods on its nominated node.
func testPodEligibleToPreempt(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	other := preemptionPod("other", "victim-claim").Node(nodeName).Obj()
	victimClaim := reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim)
	notEligible := "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/victim-claim."

	for name, tc := range map[string]struct {
		victimClaim *resourceapi.ResourceClaim
		pending     bool
		// preemptionFailed skips AssumeVictim.
		preemptionFailed bool
		// change runs after AddNomination and AssumeVictim.
		change     func(tCtx ktesting.TContext, testCtx *testContext)
		nodeName   string
		pod        *v1.Pod
		wantReason string
		wantForget bool
	}{
		"settling": {
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"settling-recorded-node": {
			wantReason: notEligible,
		},
		"settling-other-node": {
			nodeName:   node2Name,
			wantReason: notEligible,
		},
		"deallocated": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
					claim.Status.Allocation = nil
				})
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"victim-terminating": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updatePod(tCtx, testCtx, victim.Name, func(pod *v1.Pod) {
					pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
				})
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"joined-by-other-pod": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor, consumerRef(other))
				})
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"preemption-failed": {
			preemptionFailed: true,
			nodeName:         nodeName,
			wantForget:       true,
		},
		"nomination-ended": {
			// The claim still blocks the node.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.RemoveNomination(tCtx, preemptor, nodeName)
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"nothing-to-allocate": {
			pod:      preemptionPod("no-claims").Obj(),
			nodeName: nodeName,
		},
		"pending-allocation": {
			victimClaim: preemptionClaim("victim-claim").Obj(),
			pending:     true,
			nodeName:    nodeName,
			wantReason:  notEligible,
		},
		"pending-allocation-dropped": {
			victimClaim: preemptionClaim("victim-claim").Obj(),
			pending:     true,
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
				tCtx.ExpectNoError(err)
				testCtx.draManager.ResourceClaims().MaybeRemoveClaimPendingAllocation(claim.UID, true)
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"unreserved-victim-exists": {
			// The PreBind of the victim got cancelled.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"unreserved-victim-gone": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
				deletePod(tCtx, testCtx, victim)
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"unreserved-owned-by-deleted-victim": {
			victimClaim: st.FromResourceClaim(victimClaim).OwnerReference(victim.Name, string(victim.UID), podKind).Obj(),
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
				deletePod(tCtx, testCtx, victim)
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claim := tc.victimClaim
			if claim == nil {
				claim = victimClaim
			}
			claims := []*resourceapi.ResourceClaim{preemptionClaim("preemptor-claim").Obj(), claim}
			testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, other})
			if tc.pending {
				pendingClaim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
				tCtx.ExpectNoError(err)
				pendingClaim = pendingClaim.DeepCopy()
				pendingClaim.Status.Allocation = preemptionAllocation("instance-1")
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(pendingClaim.UID, pendingClaim))
			}
			if tc.preemptionFailed {
				nominateWithoutAssume(tCtx, testCtx, preemptor, nodeName, victim)
			} else {
				nominate(tCtx, testCtx, preemptor, nodeName, victim)
			}
			require.NotNil(tCtx, testCtx.p.nominations.nomination(preemptor.UID), "nomination")
			claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
			tCtx.ExpectNoError(err)
			require.Equal(tCtx, !tc.preemptionFailed, testCtx.p.victimClaims.isTracked(claim.UID), "victim claim tracked after preemption")
			if tc.change != nil {
				tc.change(tCtx, testCtx)
			}

			pod := tc.pod
			if pod == nil {
				pod = preemptor
			}
			eligible, reason := testCtx.p.PodEligibleToPreempt(tCtx, pod, tc.nodeName)
			require.Equal(tCtx, tc.wantReason == "", eligible, "eligible")
			require.Equal(tCtx, tc.wantReason, reason, "reason")
			require.Equal(tCtx, !tc.wantForget, testCtx.p.victimClaims.isTracked(claim.UID), "victim claim still tracked")
		})
	}
}

func TestVictimClaimPruning(t *testing.T) {
	testVictimClaimPruning(ktesting.Init(t))
}

// testVictimClaimPruning checks that a deallocation event ends the tracking.
func testVictimClaimPruning(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
	nominate(tCtx, testCtx, preemptor, nodeName, victim)
	require.Len(tCtx, testCtx.p.victimClaims.onNode(nodeName), 1, "victim claims before deallocation")

	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
	})
	require.Empty(tCtx, testCtx.p.victimClaims.onNode(nodeName), "victim claims after deallocation")
}

func TestPreemptionStatusCodeWithVictimClaims(t *testing.T) {
	testPreemptionStatusCodeWithVictimClaims(ktesting.Init(t))
}

// testPreemptionStatusCodeWithVictimClaims checks that the node stays a
// candidate after the victims are gone and before their claims are
// deallocated, so that DefaultPreemption asks PodEligibleToPreempt.
func testPreemptionStatusCodeWithVictimClaims(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
	// The victim is not in the NodeInfo anymore.
	state := nominate(tCtx, testCtx, preemptor, nodeName, victim)
	status := testCtx.p.Filter(tCtx, state, preemptor, testCtx.nodeInfos[0])
	require.Equal(tCtx, fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"), status)

	testCtx.p.victimClaims.forget(testCtx.p.victimClaims.onNode(nodeName)[0].uid, nil)
	status = testCtx.p.Filter(tCtx, state, preemptor, testCtx.nodeInfos[0])
	require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"), status)
}

func TestPostFilterDeallocatesVictimClaim(t *testing.T) {
	testPostFilterDeallocatesVictimClaim(ktesting.Init(t))
}

// testPostFilterDeallocatesVictimClaim covers a victim whose PreBind was
// cancelled: its claim stays allocated, without consumer or still reserved
// for the victim, and its own PostFilter has to deallocate it.
func testPostFilterDeallocatesVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	for name, unreserved := range map[string]bool{"unreserved": true, "reserved-for-victim": false} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claims := []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
			}
			testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
			nominate(tCtx, testCtx, preemptor, nodeName, fromPod(victim).Node(nodeName).Obj())
			if unreserved {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
			}

			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, victim, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			_, status = testCtx.p.PostFilter(tCtx, state, victim, nil)
			require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "deallocation of ResourceClaim completed"), status)
			claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim-claim", metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			require.Nil(tCtx, claim.Status.Allocation, "allocation")
		})
	}
}

func TestPodGroupPostFilterDeallocatesVictimClaim(t *testing.T) {
	testPodGroupPostFilterDeallocatesVictimClaim(ktesting.Init(t))
}

// testPodGroupPostFilterDeallocatesVictimClaim is the same for a victim in a
// PodGroup, whose pods don't run PostFilter. Filter may not even reach the
// DynamicResources plugin for it, so nothing else would deallocate the claim.
func testPodGroupPostFilterDeallocatesVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	podGroup := &schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "group"}}
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
	nominate(tCtx, testCtx, preemptor, nodeName, fromPod(victim).Node(nodeName).Obj())
	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
	})

	userClaims, err := testCtx.p.podResourceClaims(victim)
	tCtx.ExpectNoError(err)
	podGroupCycleState := framework.NewCycleState()
	podGroupCycleState.Write(stateKey, &podGroupStateData{
		podsStateData: map[types.NamespacedName]*stateData{
			{Namespace: namespace, Name: victim.Name}: {filterResults: &filterResults{}, claims: newClaimStore(userClaims, nil, nil)},
		},
	})
	pgInfo := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(podGroup), UnscheduledPods: []*v1.Pod{victim}}
	_, status := testCtx.p.PodGroupPostFilter(tCtx, podGroupCycleState, pgInfo, nil)
	require.Equal(tCtx, fwk.Unschedulable, status.Code(), "PodGroupPostFilter: %v", status)
	claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim-claim", metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	require.Nil(tCtx, claim.Status.Allocation, "allocation")
}
