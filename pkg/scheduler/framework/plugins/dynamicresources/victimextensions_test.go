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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
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

// setupVictims enables the DRAPreemption feature gate so NewDRAManager
// initializes preemptionTracker.
func setupVictims(tCtx ktesting.TContext, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object) *testContext {
	tCtx.Helper()
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, claims, objs, feature.Features{EnableDRAPreemption: true})
	require.NotNil(tCtx, testCtx.p.preemptionTracker, "preemption tracker")
	return testCtx
}

func assumeVictims(tCtx ktesting.TContext, testCtx *testContext, victims ...*v1.Pod) {
	tCtx.Helper()
	for _, victim := range victims {
		testCtx.p.AssumeVictim(tCtx, victim)
	}
}

func eventually(tCtx ktesting.TContext, what string, condition func() bool) {
	tCtx.Helper()
	require.Eventually(tCtx, condition, 10*time.Second, 10*time.Millisecond, what)
}

// updateClaim updates a claim's status and waits until both the claim tracker
// and preemptionTracker event handlers have processed the update.
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

func TestAssumeVictim(t *testing.T) {
	testAssumeVictim(ktesting.Init(t))
}

func testAssumeVictim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	victim := preemptionPod("victim", "victim-claim", "unallocated-claim", "standalone").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		preemptionClaim("unallocated-claim").Obj(),
		reserve(preemptionClaim("standalone").Allocation(preemptionAllocation("instance-2")).Obj(), victim),
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
			testCtx := setupPreemption(tCtx, []*v1.Node{workerNode}, claims, []apiruntime.Object{slice, victim}, feature.Features{EnableDRAPreemption: tc.gate})
			assumeVictims(tCtx, testCtx, tc.victims...)
			if !tc.gate {
				require.Nil(tCtx, testCtx.p.preemptionTracker, "preemption tracker")
				return
			}
			got := make(map[string][]types.UID)
			for _, claim := range testCtx.p.preemptionTracker.victimClaimsOnNode(nodeName) {
				got[claim.claim.String()] = append(got[claim.claim.String()], claim.podUID)
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
	slice := st.MakeResourceSlice(nodeName, driver).Devices("instance-1", "instance-2", "instance-3", "instance-4").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	other := preemptionPod("other", "victim-claim").Node(nodeName).Obj()
	victimClaim := reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim)
	notEligible := "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/victim-claim."
	// Four victims with unsorted claim names to verify alphabetical ordering
	// and truncation beyond maxClaimsInReason (3).
	var manyVictims []*v1.Pod
	var manyClaims []*resourceapi.ResourceClaim
	for i, name := range []string{"d", "b", "a", "c"} {
		claimName := name + "-claim"
		pod := preemptionPod(name, claimName).Node(nodeName).Obj()
		manyVictims = append(manyVictims, pod)
		manyClaims = append(manyClaims, reserve(preemptionClaim(claimName).Allocation(preemptionAllocation("instance-"+string(rune('1'+i)))).Obj(), pod))
	}
	// stale's template claim status points to an allocated claim owned by
	// another pod.
	stale := st.MakePod().Name("stale").Namespace(namespace).UID("stale").Priority(100).
		PodResourceClaims(v1.PodResourceClaim{Name: "claim", ResourceClaimTemplateName: new("template")}).Obj()
	stale.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{{Name: "claim", ResourceClaimName: new("allocated-claim")}}

	for name, tc := range map[string]struct {
		gateDisabled bool
		victimClaim  *resourceapi.ResourceClaim
		extraClaims  []*resourceapi.ResourceClaim
		pending      bool
		// victims passed to AssumeVictim; nil defaults to {victim}.
		victims    []*v1.Pod
		change     func(tCtx ktesting.TContext, testCtx *testContext)
		staleUID   types.UID
		nodeName   string
		pod        *v1.Pod
		wantReason string
		wantForget bool
	}{
		"settling": {
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"gate-disabled": {
			gateDisabled: true,
			nodeName:     nodeName,
		},
		"more-than-max-claims": {
			extraClaims: manyClaims,
			victims:     manyVictims,
			nodeName:    nodeName,
			wantReason:  "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: default/a-claim, default/b-claim, default/c-claim and 1 more.",
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
		"claim-recreated": {
			// AssumeVictim recorded a stale claim UID after the delete event already fired.
			victims:  []*v1.Pod{},
			staleUID: "old-uid",
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.preemptionTracker.assumeVictimClaim(&releasedClaim{
					claim:            types.NamespacedName{Namespace: namespace, Name: "victim-claim"},
					uid:              "old-uid",
					nodeName:         nodeName,
					allocatedDevices: &preemptionAllocation("instance-1").Devices,
					podUID:           victim.UID,
				})
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"joined-by-other-pod": {
			// Another pod shares the claim by name and now reserves it, so it
			// is not settling, but the entry stays tracked.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor, consumerRef(other))
				})
			},
			nodeName: nodeName,
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
				testCtx.draManager.ResourceClaims().MaybeRemoveClaimPendingAllocation(claim.UID, true /* forceRemove */)
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"unreserved": {
			// Unreserve after a canceled PreBind empties ReservedFor; the
			// victim owns the claim, so it will be deleted with the victim or
			// deallocated in PostFilter.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = nil
				})
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"claims-allocated-already": {
			pod:      preemptionPod("allocated", "allocated-claim").Priority(100).Obj(),
			nodeName: nodeName,
		},
		"claim-of-other-pod": {
			pod:        stale,
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
				testCtx.draManager.ResourceClaims().MaybeRemoveClaimPendingAllocation(claim.UID, true /* forceRemove */)
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
			claims = append(claims, tc.extraClaims...)
			objs := []apiruntime.Object{slice, preemptor, victim, other}
			for _, v := range tc.victims {
				objs = append(objs, v)
			}
			var testCtx *testContext
			if tc.gateDisabled {
				testCtx = setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, claims, objs, feature.Features{})
			} else {
				testCtx = setupVictims(tCtx, claims, objs)
			}
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
			var claimUIDs []types.UID
			if !tc.gateDisabled && len(victims) > 0 {
				recorded := testCtx.p.preemptionTracker.victimClaimsOnNode(nodeName)
				require.Len(tCtx, recorded, len(victims), "victim claims")
				for _, r := range recorded {
					claimUIDs = append(claimUIDs, r.uid)
				}
			}
			if tc.staleUID != "" {
				claimUIDs = append(claimUIDs, tc.staleUID)
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
			if !tc.gateDisabled && len(claimUIDs) > 0 {
				// Trigger lazy pruning explicitly in case PodEligibleToPreempt
				// returned early before inspecting nodeName.
				_ = testCtx.p.settlingVictimClaims(nodeName, false /* firstOnly */)
				for _, claimUID := range claimUIDs {
					require.Equal(tCtx, !tc.wantForget, testCtx.p.preemptionTracker.victimClaim(claimUID) != nil, "victim claim %s still tracked", claimUID)
				}
			}
		})
	}
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
	for name, tc := range map[string]struct {
		forget bool
		want   *fwk.Status
	}{
		"settling":  {want: fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims")},
		"forgotten": {forget: true, want: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "cannot allocate all claims")},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claims := []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
			}
			testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, preemptor, victim})
			assumeVictims(tCtx, testCtx, victim)
			if tc.forget {
				testCtx.p.preemptionTracker.forgetVictimClaim(testCtx.p.preemptionTracker.victimClaimsOnNode(nodeName)[0].uid, nil)
			}
			// testCtx.nodeInfos[0] has no pods, simulating that victim was already removed from the node.
			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			require.Equal(tCtx, tc.want, testCtx.p.Filter(tCtx, state, preemptor, testCtx.nodeInfos[0]))
		})
	}
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

func TestRequeuedVictimClaim(t *testing.T) {
	testRequeuedVictimClaim(ktesting.Init(t))
}

// testRequeuedVictimClaim verifies that a PreBind-preempted victim requeued
// with its claim still allocated (e.g. after Unreserve failed to deallocate
// it) is rejected in Filter and deallocates the claim in PostFilter, as long as
// the claim is awaiting deallocation.
func testRequeuedVictimClaim(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	sharer := preemptionPod("sharer", "victim-claim").Obj()
	beingDeallocated := fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated")
	for name, tc := range map[string]struct {
		// pod to filter; defaults to victim.
		pod         *v1.Pod
		assume      bool
		reservedFor *v1.Pod
		// reallocate replaces the tracked allocation after AssumeVictim.
		reallocate bool
		// postFilterOnly skips Filter, as when the pod fails on all nodes for
		// other reasons.
		postFilterOnly  bool
		wantFilter      *fwk.Status
		wantPostFilter  *fwk.Status
		wantDeallocated bool
	}{
		"victim-claim": {
			assume:          true,
			wantFilter:      beingDeallocated,
			wantPostFilter:  fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "deallocation of ResourceClaim completed"),
			wantDeallocated: true,
		},
		"not-a-victim-claim": {},
		"victim-claim-reserved-for-other-pod": {
			// The other pod keeps the claim allocated anyway, so the victim
			// may share it again.
			assume:      true,
			reservedFor: sharer,
		},
		"victim-claim-allocated-again": {
			// The tracked allocation is gone, so the entry is obsolete.
			assume:     true,
			reallocate: true,
		},
		"victim-claim-allocated-again-postfilter": {
			// PostFilter must not deallocate the new allocation either.
			assume:         true,
			reallocate:     true,
			postFilterOnly: true,
			wantPostFilter: fwk.NewStatus(fwk.Unschedulable),
		},
		"other-pod-while-victim-reserves-claim": {
			// Joining would keep the claim allocated once the victim is gone,
			// so the other pod waits for the deallocation.
			pod:            sharer,
			assume:         true,
			reservedFor:    victim,
			wantFilter:     beingDeallocated,
			wantPostFilter: fwk.NewStatus(fwk.Unschedulable),
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			pod := tc.pod
			if pod == nil {
				pod = victim
			}
			claim := preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj()
			if tc.reservedFor != nil {
				claim = reserve(claim, tc.reservedFor)
			}
			testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{claim}, []apiruntime.Object{slice, victim, sharer})
			if tc.assume {
				assumeVictims(tCtx, testCtx, fromPod(victim).Node(nodeName).Obj())
			}
			if tc.reallocate {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("instance-2")
				})
			}

			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			if !tc.postFilterOnly {
				status = testCtx.p.Filter(tCtx, state, pod, testCtx.nodeInfos[0])
			}
			if tc.wantFilter == nil && !tc.postFilterOnly {
				require.True(tCtx, status.IsSuccess(), "Filter: %v", status)
				if tc.reallocate {
					claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
					tCtx.ExpectNoError(err)
					require.Nil(tCtx, testCtx.p.preemptionTracker.victimClaim(claim.UID), "obsolete entry still tracked")
				}
				return
			}
			if !tc.postFilterOnly {
				require.Equal(tCtx, tc.wantFilter, status, "Filter")
			}
			_, status = testCtx.p.PostFilter(tCtx, state, pod, nil)
			require.Equal(tCtx, tc.wantPostFilter, status, "PostFilter")
			claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "victim-claim", metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			require.Equal(tCtx, tc.wantDeallocated, claim.Status.Allocation == nil, "deallocated")
		})
	}
}

func TestVictimClaimAfterUnreserve(t *testing.T) {
	testVictimClaimAfterUnreserve(ktesting.Init(t))
}

// testVictimClaimAfterUnreserve verifies that a victim preempted before
// PreBind persists its allocation stops blocking preemption once Unreserve
// drops the pending allocation, even if the victim later gets the same device
// again. If PreBind persisted the allocation, the claim stays tracked until
// deallocated even when Unreserve did not observe that allocation yet.
func testVictimClaimAfterUnreserve(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	setup := func(tCtx ktesting.TContext) (*testContext, types.UID) {
		tCtx.Helper()
		testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{preemptionClaim("victim-claim").Obj(), preemptionClaim("preemptor-claim").Obj()}, []apiruntime.Object{slice, victim, preemptor})
		claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "victim-claim")
		tCtx.ExpectNoError(err)
		return testCtx, claim.UID
	}
	schedule := func(tCtx ktesting.TContext, testCtx *testContext) fwk.CycleState {
		tCtx.Helper()
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, victim, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		require.True(tCtx, testCtx.p.Filter(tCtx, state, victim, testCtx.nodeInfos[0]).IsSuccess(), "Filter")
		require.True(tCtx, testCtx.p.Reserve(tCtx, state, victim, nodeName).IsSuccess(), "Reserve")
		return state
	}
	preempt := func(tCtx ktesting.TContext, testCtx *testContext, uid types.UID) fwk.CycleState {
		tCtx.Helper()
		state := schedule(tCtx, testCtx)
		assumeVictims(tCtx, testCtx, fromPod(victim).Node(nodeName).Obj())
		require.NotNil(tCtx, testCtx.p.preemptionTracker.victimClaim(uid), "pending allocation tracked")
		return state
	}

	for name, tc := range map[string]struct {
		storeBeforeUnreserve      bool
		storeAfterUnreserve       bool
		assumeStaleAfterUnreserve bool
		checkEligibility          bool
		wantEligible              bool
		wantTracked               bool
		wantVictimFilter          *fwk.Status
		rescheduleAndPreBind      bool
	}{
		"pending-allocation-dropped": {
			checkEligibility:     true,
			wantEligible:         true,
			wantTracked:          false,
			rescheduleAndPreBind: true,
		},
		"allocation-stored": {
			// PreBind stored the allocation although the pod's cycle failed, so the
			// claim keeps waiting for its deallocation.
			storeBeforeUnreserve: true,
			wantTracked:          true,
		},
		"allocation-stored-response-lost": {
			// PreBind's update went through, but its response got lost, so
			// Unreserve doesn't see the allocation yet. Once it arrives, the
			// preemptor waits, and the victim can't keep the claim.
			storeAfterUnreserve: true,
			checkEligibility:    true,
			wantEligible:        false,
			wantTracked:         true,
			wantVictimFilter:    fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "resourceclaim of preempted pods is being deallocated"),
		},
		"assumed-after-unreserve": {
			// AssumeVictim read the pending allocation before Unreserve dropped
			// it, but stored its entry only afterwards. Reserve forgets the entry
			// when the victim gets the same device again.
			assumeStaleAfterUnreserve: true,
			wantTracked:               true,
			rescheduleAndPreBind:      true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx, uid := setup(tCtx)
			var state fwk.CycleState
			var pendingCopy *resourceapi.AllocationResult
			if tc.assumeStaleAfterUnreserve {
				state = schedule(tCtx, testCtx)
				pending := testCtx.draManager.ResourceClaims().GetPendingAllocation(uid)
				require.NotNil(tCtx, pending, "pending allocation")
				pendingCopy = pending.DeepCopy()
			} else {
				state = preempt(tCtx, testCtx, uid)
				if tc.storeAfterUnreserve {
					pending := testCtx.draManager.ResourceClaims().GetPendingAllocation(uid)
					require.NotNil(tCtx, pending, "pending allocation")
					pendingCopy = pending.DeepCopy()
				}
			}
			if tc.storeBeforeUnreserve {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("instance-1")
					claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(victim)}
				})
			}
			testCtx.p.Unreserve(tCtx, state, victim, nodeName)
			if tc.storeAfterUnreserve {
				updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = pendingCopy
					claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(victim)}
				})
			}
			if tc.assumeStaleAfterUnreserve {
				testCtx.p.preemptionTracker.assumeVictimClaim(&releasedClaim{
					claim:            types.NamespacedName{Namespace: namespace, Name: "victim-claim"},
					uid:              uid,
					nodeName:         nodeName,
					allocatedDevices: &pendingCopy.Devices,
					podUID:           victim.UID,
				})
			}

			if tc.checkEligibility {
				eligible, reason := testCtx.p.PodEligibleToPreempt(tCtx, preemptor, nodeName)
				require.Equal(tCtx, tc.wantEligible, eligible, "eligible after Unreserve: %s", reason)
			}
			require.Equal(tCtx, tc.wantTracked, testCtx.p.preemptionTracker.victimClaim(uid) != nil, "tracked after Unreserve")
			if tc.wantVictimFilter != nil {
				state = framework.NewCycleState()
				_, status := testCtx.p.PreFilter(tCtx, state, victim, testCtx.nodeInfos)
				require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
				require.Equal(tCtx, tc.wantVictimFilter, testCtx.p.Filter(tCtx, state, victim, testCtx.nodeInfos[0]), "Filter of the victim")
			}
			if tc.rescheduleAndPreBind {
				state = schedule(tCtx, testCtx)
				require.Nil(tCtx, testCtx.p.preemptionTracker.victimClaim(uid), "tracked after Reserve")
				require.True(tCtx, testCtx.p.PreBind(tCtx, state, victim, nodeName).IsSuccess(), "PreBind")
				eligible, reason := testCtx.p.PodEligibleToPreempt(tCtx, preemptor, nodeName)
				require.True(tCtx, eligible, "victim scheduled again with the same device: %s", reason)
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

func TestUnreserveDeallocatesClaim(t *testing.T) {
	testUnreserveDeallocatesClaim(ktesting.Init(t))
}

// testUnreserveDeallocatesClaim verifies that Unreserve deallocates a claim
// allocated in this cycle only if PreBind didn't succeed and the pod is still
// its sole consumer.
func testUnreserveDeallocatesClaim(tCtx ktesting.TContext) {
	pod := preemptionPod("pod", "claim").Obj()
	other := preemptionPod("other", "claim").Node(nodeName).Obj()
	for name, tc := range map[string]struct {
		// canceled makes PreBind fail after storing the allocation by
		// canceling the binding-conditions wait.
		canceled        bool
		alsoReservedFor *v1.Pod
		wantReservedFor []resourceapi.ResourceClaimConsumerReference
		wantDeallocated bool
	}{
		"reserved-for-pod":            {canceled: true, wantDeallocated: true},
		"also-reserved-for-other-pod": {canceled: true, alsoReservedFor: other, wantReservedFor: []resourceapi.ResourceClaimConsumerReference{consumerRef(other)}},
		"prebind-succeeded": {
			// A failed Bind call may still have bound the pod on the API server.
			wantReservedFor: []resourceapi.ResourceClaimConsumerReference{},
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
			if tc.canceled {
				slice = fabricSlice
			}
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
			testCtx := setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, []*resourceapi.ResourceClaim{preemptionClaim("claim").Obj()}, []apiruntime.Object{slice, pod, other},
				feature.Features{EnableDRAPreemption: true, EnableDRADeviceBindingConditions: true, EnableDRAResourceClaimDeviceStatus: true})
			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
			require.True(tCtx, testCtx.p.Filter(tCtx, state, pod, testCtx.nodeInfos[0]).IsSuccess(), "Filter")
			require.True(tCtx, testCtx.p.Reserve(tCtx, state, pod, nodeName).IsSuccess(), "Reserve")
			preBindCtx := context.Context(tCtx)
			if tc.canceled {
				canceledCtx, cancel := context.WithCancel(tCtx)
				cancel()
				preBindCtx = canceledCtx
			}
			status = testCtx.p.PreBind(preBindCtx, state, pod, nodeName)
			require.Equal(tCtx, !tc.canceled, status.IsSuccess(), "PreBind: %v", status)
			if tc.alsoReservedFor != nil {
				updateClaim(tCtx, testCtx, "claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor, consumerRef(tc.alsoReservedFor))
				})
			}

			testCtx.p.Unreserve(tCtx, state, pod, nodeName)
			claim, err := testCtx.client.ResourceV1().ResourceClaims(namespace).Get(tCtx, "claim", metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			require.Equal(tCtx, tc.wantReservedFor, claim.Status.ReservedFor, "ReservedFor")
			assumed, err := testCtx.draManager.ResourceClaims().Get(namespace, "claim")
			tCtx.ExpectNoError(err)
			if tc.wantDeallocated {
				require.Nil(tCtx, claim.Status.Allocation, "allocation")
				require.Nil(tCtx, assumed.Status.Allocation, "allocation in the assume cache")
			} else {
				require.NotNil(tCtx, claim.Status.Allocation, "allocation")
			}
		})
	}
}

func TestIsSchedulableAfterAssignedPodDeleted(t *testing.T) {
	testIsSchedulableAfterAssignedPodDeleted(ktesting.Init(t))
}

func testIsSchedulableAfterAssignedPodDeleted(tCtx ktesting.TContext) {
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	preemptor.Status.NominatedNodeName = nodeName
	waiting := preemptionPod("waiting", "waiting-claim").Obj()
	failed := preemptionPod("failed", "victim-claim").Node(nodeName).Obj()
	failed.Status.Phase = v1.PodFailed
	completed := preemptionPod("completed", "victim-claim").Node(nodeName).Obj()
	completed.Status.Phase = v1.PodSucceeded

	// The deleted pods share an allocated claim; a missing or unallocated one
	// would wake pods for another reason (see hasUnallocatedClaim).
	testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
		preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(),
	}, nil)
	logger := klog.FromContext(tCtx)

	for name, tc := range map[string]struct {
		pod      *v1.Pod
		holdNode string
		oldObj   any
		wantHint fwk.QueueingHint
		wantErr  bool
	}{
		"pod-with-claims-deleted-from-nominated-node": {
			pod:      preemptor,
			oldObj:   preemptionPod("victim", "victim-claim").Node(nodeName).Obj(),
			wantHint: fwk.Queue,
		},
		"pod-with-claims-deleted-from-other-node": {
			pod:      preemptor,
			oldObj:   preemptionPod("elsewhere", "victim-claim").Node(node2Name).Obj(),
			wantHint: fwk.QueueSkip,
		},
		"pod-not-nominated": {
			pod:      waiting,
			oldObj:   preemptionPod("victim", "victim-claim").Node(nodeName).Obj(),
			wantHint: fwk.QueueSkip,
		},
		"pod-without-claims": {
			pod:      preemptor,
			oldObj:   preemptionPod("unrelated").Node(nodeName).Obj(),
			wantHint: fwk.QueueSkip,
		},
		"failed-pod": {
			pod:      preemptor,
			oldObj:   failed,
			wantHint: fwk.QueueSkip,
		},
		"completed-pod": {
			pod:      preemptor,
			oldObj:   completed,
			wantHint: fwk.QueueSkip,
		},
		"unassigned-pod": {
			pod:      preemptor,
			oldObj:   preemptionPod("unassigned", "victim-claim").Obj(),
			wantHint: fwk.QueueSkip,
		},
		"unassigned-pod-for-waiting": {
			pod:      waiting,
			oldObj:   preemptionPod("unassigned", "victim-claim").Obj(),
			wantHint: fwk.QueueSkip,
		},
		// Pod status updates can lag behind in-memory nominations, so nominatedTo
		// also checks preemptionTracker holds.
		"hold-on-other-node": {
			pod:      waiting,
			holdNode: node2Name,
			oldObj:   preemptionPod("victim", "victim-claim").Node(nodeName).Obj(),
			wantHint: fwk.QueueSkip,
		},
		"hold-on-nominated-node": {
			pod:      waiting,
			holdNode: nodeName,
			oldObj:   preemptionPod("victim", "victim-claim").Node(nodeName).Obj(),
			wantHint: fwk.Queue,
		},
		"unexpected-object": {
			pod:      preemptor,
			oldObj:   &v1.Node{},
			wantHint: fwk.Queue,
			wantErr:  true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			if tc.holdNode != "" {
				testCtx.p.preemptionTracker.nominate(tc.pod.UID, &podHold{nodeName: tc.holdNode})
				tCtx.Cleanup(func() { testCtx.p.preemptionTracker.release(tc.pod.UID) })
			}
			hint, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, tc.pod, tc.oldObj, nil)
			if tc.wantErr {
				require.Error(tCtx, err)
			} else {
				tCtx.ExpectNoError(err)
			}
			require.Equal(tCtx, tc.wantHint, hint)
		})
	}
}

func TestIsSchedulableAfterAssumedPodForgotten(t *testing.T) {
	testIsSchedulableAfterAssumedPodForgotten(ktesting.Init(t))
}

// testIsSchedulableAfterAssumedPodForgotten verifies that AssignedPod/Delete
// wakes waiting pods when an assumed pod is forgotten after Unreserve dropped
// its pending claim allocation.
func testIsSchedulableAfterAssumedPodForgotten(tCtx ktesting.TContext) {
	waiting := preemptionPod("waiting", "waiting-claim").Obj()
	running := preemptionPod("running", "allocated-claim").Node(node2Name).Obj()
	completed := preemptionPod("completed", "forgotten-claim").Node(node2Name).Obj()
	completed.Status.Phase = v1.PodSucceeded

	testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
		preemptionClaim("forgotten-claim").Obj(),
		preemptionClaim("pending-claim").Obj(),
		reserve(preemptionClaim("allocated-claim").Allocation(poolAllocation(node2Name, "gpu-0")).Obj(), running),
	}, nil)
	logger := klog.FromContext(tCtx)
	pending, err := testCtx.draManager.ResourceClaims().Get(namespace, "pending-claim")
	tCtx.ExpectNoError(err)
	pending = pending.DeepCopy()
	pending.Status.Allocation = poolAllocation(node2Name, "gpu-1")
	tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(pending.UID, pending))

	for name, tc := range map[string]struct {
		oldObj   any
		wantHint fwk.QueueingHint
		wantErr  bool
	}{
		"pending-allocation-dropped": {
			oldObj:   preemptionPod("forgotten", "forgotten-claim").Node(node2Name).Obj(),
			wantHint: fwk.Queue,
		},
		"allocated-and-dropped-claim": {
			oldObj:   preemptionPod("forgotten-sharer", "allocated-claim", "forgotten-claim").Node(node2Name).Obj(),
			wantHint: fwk.Queue,
		},
		"claim-already-deleted": {
			// The garbage collector deletes an unallocated claim together with
			// its pod, so it may be gone already.
			oldObj:   preemptionPod("forgotten-gone", "gone-claim").Node(node2Name).Obj(),
			wantHint: fwk.Queue,
		},
		"claim-still-pending": {
			oldObj:   preemptionPod("binding", "pending-claim").Node(node2Name).Obj(),
			wantHint: fwk.QueueSkip,
		},
		"claim-still-allocated": {
			oldObj:   running,
			wantHint: fwk.QueueSkip,
		},
		"unassigned-pod": {
			// Clearing an unassigned pod's nomination also fires AssignedPod/Delete
			// with Spec.NodeName == "" and unallocated claims; it must be skipped.
			oldObj:   preemptionPod("unassigned", "forgotten-claim").Obj(),
			wantHint: fwk.QueueSkip,
		},
		"completed-pod": {
			oldObj:   completed,
			wantHint: fwk.QueueSkip,
		},
		"unexpected-object": {
			oldObj:   &v1.Node{},
			wantHint: fwk.Queue,
			wantErr:  true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			hint, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, waiting, tc.oldObj, nil)
			if tc.wantErr {
				require.Error(tCtx, err)
			} else {
				tCtx.ExpectNoError(err)
			}
			require.Equal(tCtx, tc.wantHint, hint)
		})
	}
}

func TestEventsToRegisterWithPreemption(t *testing.T) {
	testEventsToRegisterWithPreemption(ktesting.Init(t))
}

func testEventsToRegisterWithPreemption(tCtx ktesting.TContext) {
	assignedPodDelete := fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Delete}
	for name, enabled := range map[string]bool{"disabled": false, "enabled": true} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupPreemption(tCtx, nil, nil, nil, feature.Features{EnableDRAPreemption: enabled})
			events, err := testCtx.p.EventsToRegister(tCtx)
			tCtx.ExpectNoError(err)
			var found bool
			for _, event := range events {
				if cmp.Equal(event.Event, assignedPodDelete) {
					found = true
				}
			}
			require.Equal(tCtx, enabled, found, "AssignedPod/Delete registered")
		})
	}
}
