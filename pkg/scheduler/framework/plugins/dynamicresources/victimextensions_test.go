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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPreemptionStoreVictimClaims(t *testing.T) {
	pods := func(names ...string) map[types.UID]types.NamespacedName {
		pods := make(map[types.UID]types.NamespacedName)
		for _, name := range names {
			pods[types.UID(name)] = types.NamespacedName{Namespace: namespace, Name: name}
		}
		return pods
	}
	newClaim := func(nodeName string, allocation *resourceapi.AllocationResult, pods map[types.UID]types.NamespacedName) *releasedClaim {
		return &releasedClaim{claim: types.NamespacedName{Namespace: namespace, Name: "c"}, uid: "c", nodeName: nodeName, allocation: allocation, pods: pods}
	}
	podsOf := func(s *preemptionStore, nodeName string) []types.UID {
		var uids []types.UID
		for _, claim := range s.victimClaimsOnNode(nodeName) {
			for uid := range claim.pods {
				uids = append(uids, uid)
			}
		}
		return uids
	}

	t.Run("merge-victims", func(t *testing.T) {
		s := newPreemptionStore()
		s.assumeVictimClaim(newClaim("a", preemptionAllocation("instance-1"), pods("v1")))
		s.assumeVictimClaim(newClaim("a", preemptionAllocation("instance-1"), pods("v2")))
		require.ElementsMatch(t, []types.UID{"v1", "v2"}, podsOf(s, "a"))
	})

	t.Run("reallocated", func(t *testing.T) {
		s := newPreemptionStore()
		s.assumeVictimClaim(newClaim("a", preemptionAllocation("instance-1"), pods("v1")))
		s.assumeVictimClaim(newClaim("b", poolAllocation("b", "instance-2"), pods("v2")))
		require.Empty(t, s.victimClaimsOnNode("a"), "claims on the old node")
		require.Equal(t, []types.UID{"v2"}, podsOf(s, "b"))
	})

	t.Run("forget", func(t *testing.T) {
		s := newPreemptionStore()
		before := newClaim("a", preemptionAllocation("instance-1"), pods("v1"))
		s.assumeVictimClaim(before)
		s.assumeVictimClaim(newClaim("a", preemptionAllocation("instance-1"), pods("v2")))
		// assumeVictimClaim replaced before with a merged entry, so pointer-matched forget is a no-op.
		s.forgetVictimClaim("c", before)
		require.True(t, s.isVictimClaim("c"))
		s.forgetVictimClaim("c", nil)
		require.False(t, s.isVictimClaim("c"))
		require.Empty(t, s.victimClaimsOnNode("a"))
	})
}

// setupVictims enables the DRAPreemption feature gate so NewDRAManager
// initializes preemptionStore.
func setupVictims(tCtx ktesting.TContext, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object) *testContext {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, claims, objs, feature.Features{EnableDRAPreemption: true})
	require.NotNil(tCtx, testCtx.p.store, "preemption store")
	return testCtx
}

func assumeVictims(tCtx ktesting.TContext, testCtx *testContext, victims ...*v1.Pod) {
	for _, victim := range victims {
		testCtx.p.AssumeVictim(tCtx, victim)
	}
}

func eventually(tCtx ktesting.TContext, what string, condition func() bool) {
	tCtx.Helper()
	require.Eventually(tCtx, condition, 10*time.Second, 10*time.Millisecond, what)
}

// updateClaim updates a claim's status and waits until both the claim tracker
// and preemptionStore event handlers have processed the update.
func updateClaim(tCtx ktesting.TContext, testCtx *testContext, name string, update func(claim *resourceapi.ResourceClaim)) {
	tCtx.Helper()
	// AssumeCache invokes event handlers in registration order after updating
	// its store, so a handler registered here runs after existing handlers.
	var mutex sync.Mutex
	seen := sets.New[string]()
	testCtx.draManager.resourceClaimTracker.cache.AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(_, obj any) {
			if claim, ok := obj.(*resourceapi.ResourceClaim); ok && claim.Name == name {
				mutex.Lock()
				defer mutex.Unlock()
				seen.Insert(claim.ResourceVersion)
			}
		},
	})
	claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, name, metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	update(claim)
	claim, err = testCtx.client.ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err)
	eventually(tCtx, "claim update processed by all handlers", func() bool {
		mutex.Lock()
		defer mutex.Unlock()
		return seen.Has(claim.ResourceVersion)
	})
}

// updatePod writes directly to the fake client's object tracker because setup's
// ResourceVersion reactor rejects pods created without a ResourceVersion.
func updatePod(tCtx ktesting.TContext, testCtx *testContext, name string, update func(pod *v1.Pod)) {
	tCtx.Helper()
	pod, err := testCtx.client.CoreV1().Pods(namespace).Get(tCtx, name, metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	update(pod)
	tCtx.ExpectNoError(testCtx.client.Tracker().Update(v1.SchemeGroupVersion.WithResource("pods"), pod, namespace))
	eventually(tCtx, "pod update in informer", func() bool {
		cached, err := testCtx.p.podLister.Pods(namespace).Get(name)
		return err == nil && apiequality.Semantic.DeepEqual(cached, pod)
	})
}

func deletePod(tCtx ktesting.TContext, testCtx *testContext, pod *v1.Pod) {
	tCtx.Helper()
	tCtx.ExpectNoError(testCtx.client.CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{}))
	eventually(tCtx, "pod deletion in informer", func() bool {
		_, err := testCtx.p.podLister.Pods(pod.Namespace).Get(pod.Name)
		return err != nil
	})
}

func TestAssumeVictim(t *testing.T) {
	testAssumeVictim(ktesting.Init(t))
}

func testAssumeVictim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Device("instance-3").Obj()
	victim := preemptionPod("victim", "victim-claim", "group-claim", "unallocated-claim", "shared-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "shared-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		st.FromResourceClaim(preemptionClaim("group-claim").Allocation(preemptionAllocation("instance-2")).Obj()).
			ReservedFor(resourceapi.ResourceClaimConsumerReference{APIGroup: "scheduling.k8s.io", Resource: "podgroups", Name: "group", UID: "group"}).Obj(),
		preemptionClaim("unallocated-claim").Obj(),
		st.FromResourceClaim(preemptionClaim("shared-claim").Allocation(preemptionAllocation("instance-3")).Obj()).
			ReservedFor(consumerRef(victim), consumerRef(victim2)).Obj(),
	}

	for name, tc := range map[string]struct {
		victims    []*v1.Pod
		gate       bool
		wantClaims map[string][]types.UID
	}{
		"victim": {
			victims: []*v1.Pod{victim},
			gate:    true,
			wantClaims: map[string][]types.UID{
				"default/victim-claim": {victim.UID},
				"default/shared-claim": {victim.UID},
			},
		},
		"shared-by-two-victims": {
			victims: []*v1.Pod{victim, victim2},
			gate:    true,
			wantClaims: map[string][]types.UID{
				"default/victim-claim": {victim.UID},
				"default/shared-claim": {victim.UID, victim2.UID},
			},
		},
		"unassigned": {
			victims: []*v1.Pod{fromPod(victim).Node("").Obj()},
			gate:    true,
		},
		"gate-disabled": {
			victims: []*v1.Pod{victim},
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, tc.gate)
			testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, []apiruntime.Object{slice, victim, victim2}, feature.Features{EnableDRAPreemption: tc.gate})
			assumeVictims(tCtx, testCtx, tc.victims...)
			if !tc.gate {
				require.Nil(tCtx, testCtx.p.store, "preemption store")
				return
			}
			got := make(map[string][]types.UID)
			for _, claim := range testCtx.p.store.victimClaimsOnNode(nodeName) {
				for uid := range claim.pods {
					got[claim.claim.String()] = append(got[claim.claim.String()], uid)
				}
			}
			for _, uids := range got {
				slices.Sort(uids)
			}
			if len(tc.wantClaims) == 0 {
				tc.wantClaims = map[string][]types.UID{}
			}
			require.Equal(tCtx, tc.wantClaims, got, "victim claims and their pods")
		})
	}
}

func TestPodEligibleToPreempt(t *testing.T) {
	testPodEligibleToPreempt(ktesting.Init(t))
}

// testPodEligibleToPreempt verifies when settling victim claims block further
// preemption and when stale entries are pruned.
func testPodEligibleToPreempt(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Device("instance-3").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim").Node(nodeName).Obj()
	other := preemptionPod("other", "victim-claim").Node(nodeName).Obj()
	victimClaim := reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim)
	notEligible := "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/victim-claim."

	for name, tc := range map[string]struct {
		victimClaim *resourceapi.ResourceClaim
		pending     bool
		// victims passed to AssumeVictim; nil defaults to {victim}.
		victims    []*v1.Pod
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
		"other-node": {
			nodeName: node2Name,
		},
		"not-nominated": {},
		"no-victim": {
			// Preemption failed before AssumeVictim was called.
			victims:  []*v1.Pod{},
			nodeName: nodeName,
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
			// Another pod now reserves the claim so it is not settling, but the
			// entry stays tracked in case that pod is also preempted.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor, consumerRef(other))
				})
			},
			nodeName: nodeName,
		},
		"shared-before-second-victim": {
			victimClaim: st.FromResourceClaim(victimClaim).ReservedFor(consumerRef(victim), consumerRef(victim2)).Obj(),
			nodeName:    nodeName,
		},
		"shared-after-second-victim": {
			victimClaim: st.FromResourceClaim(victimClaim).ReservedFor(consumerRef(victim), consumerRef(victim2)).Obj(),
			victims:     []*v1.Pod{victim, victim2},
			nodeName:    nodeName,
			wantReason:  notEligible,
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
			// PreBind was canceled so ReservedFor is empty, but the victim pod
			// still exists to deallocate the claim in PostFilter.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"unreserved-victim-gone": {
			// ReservedFor is empty and the non-owning victim is gone, so
			// nothing will deallocate the claim.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
				deletePod(tCtx, testCtx, victim)
			},
			nodeName: nodeName,
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
		"claims-allocated-already": {
			pod:      preemptionPod("allocated", "allocated-claim").Priority(100).Obj(),
			nodeName: nodeName,
		},
		"shared-with-terminating-pod": {
			// A terminating co-consumer does not keep the claim in use.
			victimClaim: st.FromResourceClaim(victimClaim).ReservedFor(consumerRef(victim), consumerRef(other)).Obj(),
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updatePod(tCtx, testCtx, other.Name, func(pod *v1.Pod) {
					pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
				})
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"pending-allocation-replaced": {
			// Pending allocation was dropped and replaced with different
			// devices without a Status.Allocation -> nil transition for
			// claimHandlers to observe.
			victimClaim: preemptionClaim("victim-claim").Obj(),
			pending:     true,
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
				tCtx.ExpectNoError(err)
				testCtx.draManager.ResourceClaims().MaybeRemoveClaimPendingAllocation(claim.UID, true)
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("instance-2")
					claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(other)}
				})
			},
			nodeName:   nodeName,
			wantForget: true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claim := tc.victimClaim
			if claim == nil {
				claim = victimClaim
			}
			claims := []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				preemptionClaim("allocated-claim").Allocation(preemptionAllocation("instance-3")).Obj(),
				claim,
			}
			testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, victim2, other})
			if tc.pending {
				pendingClaim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
				tCtx.ExpectNoError(err)
				pendingClaim = pendingClaim.DeepCopy()
				pendingClaim.Status.Allocation = preemptionAllocation("instance-1")
				tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(pendingClaim.UID, pendingClaim))
			}
			victims := tc.victims
			if victims == nil {
				victims = []*v1.Pod{victim}
			}
			assumeVictims(tCtx, testCtx, victims...)
			var claimUID types.UID
			if len(victims) > 0 {
				recorded := testCtx.p.store.victimClaimsOnNode(nodeName)
				require.Len(tCtx, recorded, 1, "victim claims")
				claimUID = recorded[0].uid
			}
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
			if claimUID != "" {
				// Trigger lazy pruning explicitly in case PodEligibleToPreempt
				// returned early before inspecting nodeName.
				_ = testCtx.p.settlingVictimClaims(nodeName, false)
				require.Equal(tCtx, !tc.wantForget, testCtx.p.store.isVictimClaim(claimUID), "victim claim still tracked")
			}
		})
	}
}

func TestVictimClaimPruning(t *testing.T) {
	testVictimClaimPruning(ktesting.Init(t))
}

func testVictimClaimPruning(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim})
	assumeVictims(tCtx, testCtx, victim)
	require.Len(tCtx, testCtx.p.store.victimClaimsOnNode(nodeName), 1, "victim claims before deallocation")

	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
	})
	require.Empty(tCtx, testCtx.p.store.victimClaimsOnNode(nodeName), "victim claims after deallocation")
}

func TestVictimClaimRecreated(t *testing.T) {
	testVictimClaimRecreated(ktesting.Init(t))
}

// testVictimClaimRecreated verifies lazy pruning when AssumeVictim records a
// stale claim UID after the delete event already fired.
func testVictimClaimRecreated(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim})
	testCtx.p.store.assumeVictimClaim(&releasedClaim{
		claim:      types.NamespacedName{Namespace: namespace, Name: "victim-claim"},
		uid:        "old-uid",
		nodeName:   nodeName,
		allocation: preemptionAllocation("instance-1"),
		pods:       map[types.UID]types.NamespacedName{victim.UID: {Namespace: namespace, Name: victim.Name}},
	})
	require.Empty(tCtx, testCtx.p.settlingVictimClaims(nodeName, false), "settling victim claims")
	require.False(tCtx, testCtx.p.store.isVictimClaim("old-uid"), "record of the old claim still tracked")
}

func TestPreemptionStatusCodeWithVictimClaims(t *testing.T) {
	testPreemptionStatusCodeWithVictimClaims(ktesting.Init(t))
}

// testPreemptionStatusCodeWithVictimClaims verifies that Filter keeps returning
// Unschedulable after victims are deleted while their claims are still settling.
func testPreemptionStatusCodeWithVictimClaims(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
	assumeVictims(tCtx, testCtx, victim)
	// testCtx.nodeInfos[0] has no pods, simulating that victim was already removed from the node.
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	status = testCtx.p.Filter(tCtx, state, preemptor, testCtx.nodeInfos[0])
	require.Equal(tCtx, fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"), status)

	testCtx.p.store.forgetVictimClaim(testCtx.p.store.victimClaimsOnNode(nodeName)[0].uid, nil)
	status = testCtx.p.Filter(tCtx, state, preemptor, testCtx.nodeInfos[0])
	require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims"), status)
}

func TestPostFilterDeallocatesVictimClaim(t *testing.T) {
	testPostFilterDeallocatesVictimClaim(ktesting.Init(t))
}

// testPostFilterDeallocatesVictimClaim verifies that a preempted pod's own
// PostFilter deallocates its tracked victim claim unless another pod reserves it.
func testPostFilterDeallocatesVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	other := preemptionPod("other", "victim-claim").Node(nodeName).Obj()
	for name, tc := range map[string]struct {
		assume          bool
		reservedFor     []*v1.Pod
		wantDeallocated bool
	}{
		"unreserved":             {assume: true, wantDeallocated: true},
		"reserved-for-victim":    {assume: true, reservedFor: []*v1.Pod{victim}, wantDeallocated: true},
		"reserved-for-other-pod": {assume: true, reservedFor: []*v1.Pod{other}},
		"not-a-victim-claim":     {},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claims := []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
			}
			testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, other})
			if tc.assume {
				assumeVictims(tCtx, testCtx, fromPod(victim).Node(nodeName).Obj())
			}
			updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
				claim.Status.ReservedFor = nil
				for _, pod := range tc.reservedFor {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor, consumerRef(pod))
				}
			})

			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, victim, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			_, status = testCtx.p.PostFilter(tCtx, state, victim, nil)
			claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim-claim", metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			if tc.wantDeallocated {
				require.Equal(tCtx, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "deallocation of ResourceClaim completed"), status)
				require.Nil(tCtx, claim.Status.Allocation, "allocation")
			} else {
				require.Equal(tCtx, fwk.NewStatus(fwk.Unschedulable), status)
				require.NotNil(tCtx, claim.Status.Allocation, "allocation")
			}
		})
	}
}

func TestPodGroupPostFilterDeallocatesVictimClaim(t *testing.T) {
	testPodGroupPostFilterDeallocatesVictimClaim(ktesting.Init(t))
}

// testPodGroupPostFilterDeallocatesVictimClaim verifies the same cleanup in
// PodGroupPostFilter, since PodGroup members skip per-pod PostFilter.
func testPodGroupPostFilterDeallocatesVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	podGroup := &schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "group"}}
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
	}
	testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
	assumeVictims(tCtx, testCtx, fromPod(victim).Node(nodeName).Obj())
	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
	})

	userClaims, err := testCtx.p.podResourceClaims(victim)
	tCtx.ExpectNoError(err)
	podGroupCycleState := framework.NewCycleState()
	podGroupCycleState.Write(stateKey, &podGroupStateData{
		podsStateData: map[types.NamespacedName]*stateData{
			{Namespace: namespace, Name: victim.Name}: {claims: newClaimStore(userClaims, nil, nil)},
		},
	})
	pgInfo := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(podGroup), UnscheduledPods: []*v1.Pod{victim}}
	_, status := testCtx.p.PodGroupPostFilter(tCtx, podGroupCycleState, pgInfo, nil)
	require.Equal(tCtx, fwk.Unschedulable, status.Code(), "PodGroupPostFilter: %v", status)
	claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim-claim", metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	require.Nil(tCtx, claim.Status.Allocation, "allocation")
}

func TestIsSchedulableAfterAssignedPodDeleted(t *testing.T) {
	testIsSchedulableAfterAssignedPodDeleted(ktesting.Init(t))
}

func testIsSchedulableAfterAssignedPodDeleted(tCtx ktesting.TContext) {
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	preemptor.Status.NominatedNodeName = nodeName
	waiting := preemptionPod("waiting", "waiting-claim").Obj()
	testCtx := setupVictims(tCtx, nil, nil)
	logger := klog.FromContext(tCtx)
	hint := func(pod, deleted *v1.Pod) fwk.QueueingHint {
		hint, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, pod, deleted, nil)
		tCtx.ExpectNoError(err)
		return hint
	}

	require.Equal(tCtx, fwk.Queue, hint(preemptor, preemptionPod("victim", "victim-claim").Node(nodeName).Obj()), "pod with claims deleted from the nominated node")
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, preemptionPod("elsewhere", "victim-claim").Node(node2Name).Obj()), "pod with claims deleted from another node")
	require.Equal(tCtx, fwk.QueueSkip, hint(waiting, preemptionPod("victim", "victim-claim").Node(nodeName).Obj()), "pod which is not nominated")
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, preemptionPod("unrelated").Node(nodeName).Obj()), "pod without claims")
	failed := preemptionPod("failed", "victim-claim").Node(nodeName).Obj()
	failed.Status.Phase = v1.PodFailed
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, failed), "failed pod whose claim got deallocated")
	completed := preemptionPod("completed", "victim-claim").Node(nodeName).Obj()
	completed.Status.Phase = v1.PodSucceeded
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, completed), "completed pod whose claim got deallocated")
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, preemptionPod("unassigned", "victim-claim").Obj()), "unassigned pod which lost its nomination")
}

func TestEventsToRegisterWithPreemption(t *testing.T) {
	testEventsToRegisterWithPreemption(ktesting.Init(t))
}

func testEventsToRegisterWithPreemption(tCtx ktesting.TContext) {
	assignedPodDelete := fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Delete}
	for _, enabled := range []bool{false, true} {
		testCtx := setupPreemption(tCtx, nil, nil, nil, feature.Features{EnableDRAPreemption: enabled})
		events, err := testCtx.p.EventsToRegister(tCtx)
		tCtx.ExpectNoError(err)
		var found bool
		for _, event := range events {
			if cmp.Equal(event.Event, assignedPodDelete) {
				found = true
			}
		}
		require.Equal(tCtx, enabled, found, "AssignedPod/Delete registered with DRAPreemption=%v", enabled)
	}
}
