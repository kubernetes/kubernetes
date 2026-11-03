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

func TestClaimNominations(t *testing.T) {
	victim := func(name string) map[types.UID]types.NamespacedName {
		return map[types.UID]types.NamespacedName{types.UID(name): {Namespace: namespace, Name: name}}
	}
	newClaim := func(uid string, victims map[types.UID]types.NamespacedName) *victimClaim {
		return &victimClaim{claim: types.NamespacedName{Namespace: namespace, Name: uid}, uid: types.UID(uid), victims: victims}
	}
	victimsOf := func(n *claimNominations, uid string) []types.UID {
		var uids []types.UID
		for victimUID := range n.victimClaims[n.victimClaimNodes[types.UID(uid)]][types.UID(uid)].victims {
			uids = append(uids, victimUID)
		}
		return uids
	}

	t.Run("replace-and-remove", func(t *testing.T) {
		n := newClaimNominations()
		n.add("p", &podNomination{nodeName: "a", victims: victim("v1")}, map[string][]*victimClaim{"a": {newClaim("c1", victim("v1"))}})
		n.add("p", &podNomination{nodeName: "b", victims: victim("v2")}, map[string][]*victimClaim{"b": {newClaim("c2", victim("v2"))}})
		require.Equal(t, "b", n.nomination("p").nodeName)
		// An event about the older nomination must not remove the newer one.
		n.remove("p", "a", false)
		require.NotNil(t, n.nomination("p"))
		n.remove("p", "b", false)
		require.Nil(t, n.nomination("p"))
		// Claims of the victims stay.
		require.True(t, n.isVictimClaim("c1"))
		require.True(t, n.isVictimClaim("c2"))
		require.Len(t, n.victimClaimsOnNode("a"), 1)
	})

	t.Run("merge-victims", func(t *testing.T) {
		n := newClaimNominations()
		n.add("p1", &podNomination{nodeName: "a", victims: victim("v1")}, map[string][]*victimClaim{"a": {newClaim("c", victim("v1"))}})
		n.add("p2", &podNomination{nodeName: "a", victims: victim("v2")}, map[string][]*victimClaim{"a": {newClaim("c", victim("v2"))}})
		require.ElementsMatch(t, []types.UID{"v1", "v2"}, victimsOf(n, "c"))
	})

	t.Run("failed-preemption", func(t *testing.T) {
		n := newClaimNominations()
		before := newClaim("c", map[types.UID]types.NamespacedName{"v1": {Name: "v1"}, "v2": {Name: "v2"}})
		n.add("p", &podNomination{nodeName: "a", victims: victim("v1")}, map[string][]*victimClaim{"a": {before}})
		n.remove("p", "a", true)
		require.Nil(t, n.nomination("p"))
		require.Equal(t, []types.UID{"v2"}, victimsOf(n, "c"))
		// The instance got replaced, so forgetting the old one does nothing.
		n.forget("c", before)
		require.True(t, n.isVictimClaim("c"))
		n.forget("c", nil)
		require.False(t, n.isVictimClaim("c"))
		require.Empty(t, n.victimClaimsOnNode("a"))
	})
}

// setupNominations sets up the plugin with the claimNominations store,
// which needs the feature gate.
func setupNominations(tCtx ktesting.TContext, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object) *testContext {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, claims, objs, feature.Features{EnableDRAPreemption: true})
	require.NotNil(tCtx, testCtx.p.nominations, "claimNominations")
	return testCtx
}

// nominate runs PreFilter for the preemptor and then AddNominatedPod.
func nominate(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nodeName string, victims ...*v1.Pod) fwk.CycleState {
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	testCtx.p.AddNominatedPod(tCtx, state, preemptor, nodeName, victims)
	return state
}

func eventually(tCtx ktesting.TContext, what string, condition func() bool) {
	tCtx.Helper()
	require.Eventually(tCtx, condition, 10*time.Second, 10*time.Millisecond, what)
}

// updateClaim updates the claim status and waits until the handlers of the
// claim tracker and the plugin have processed the update.
func updateClaim(tCtx ktesting.TContext, testCtx *testContext, name string, update func(claim *resourceapi.ResourceClaim)) {
	tCtx.Helper()
	// The assume cache calls handlers in the order in which they were added
	// and only after storing an object, so a handler which gets added now
	// sees the update last.
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

// updatePod bypasses the reactors of the fake client because those need a
// ResourceVersion, which pods created by the fake client don't have.
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

func TestPodEligibleToPreempt(t *testing.T) {
	testPodEligibleToPreempt(ktesting.Init(t))
}

// testPodEligibleToPreempt checks when a preemptor has to wait for claims
// of its victims.
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
		// change runs after AddNominatedPod.
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
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.RemoveNominatedPod(tCtx, preemptor, nodeName, true)
			},
			nodeName:   nodeName,
			wantForget: true,
		},
		"preemption-failed-victim-terminating": {
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				updatePod(tCtx, testCtx, victim.Name, func(pod *v1.Pod) {
					pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
				})
				testCtx.p.RemoveNominatedPod(tCtx, preemptor, nodeName, true)
			},
			nodeName:   nodeName,
			wantReason: notEligible,
		},
		"nomination-ended": {
			// The claim still blocks the node.
			change: func(tCtx ktesting.TContext, testCtx *testContext) {
				testCtx.p.RemoveNominatedPod(tCtx, preemptor, "", false)
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
			nominate(tCtx, testCtx, preemptor, nodeName, victim)
			require.NotNil(tCtx, testCtx.p.nominations.nomination(preemptor.UID), "nomination")
			claimUID := testCtx.p.nominations.victimClaimsOnNode(nodeName)[0].uid
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
			require.Equal(tCtx, !tc.wantForget, testCtx.p.nominations.isVictimClaim(claimUID), "victim claim still tracked")
		})
	}
}

func TestAddNominatedPod(t *testing.T) {
	testAddNominatedPod(ktesting.Init(t))
}

func testAddNominatedPod(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim-2").Node(nodeName).Obj()
	running := preemptionPod("running", "shared-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		// Shared with a pod which keeps running, so not released.
		st.FromResourceClaim(preemptionClaim("victim-claim-2").Allocation(preemptionAllocation("instance-2")).Obj()).
			ReservedFor(consumerRef(victim2), consumerRef(running)).Obj(),
	}

	for name, tc := range map[string]struct {
		preemptor       *v1.Pod
		deleted         bool
		wantNomination  bool
		wantVictimClaim []string
	}{
		"released-claims": {
			preemptor:       preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj(),
			wantNomination:  true,
			wantVictimClaim: []string{"default/victim-claim"},
		},
		"cannot-release": {
			preemptor: preemptionPod("preemptor", "preemptor-claim").Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
		},
		"preemptor-gone": {
			preemptor:       preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj(),
			deleted:         true,
			wantVictimClaim: []string{"default/victim-claim"},
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			objs := []apiruntime.Object{slice, victim, victim2, running}
			if !tc.deleted {
				objs = append(objs, tc.preemptor)
			}
			testCtx := setupNominations(tCtx, claims, objs)
			nominate(tCtx, testCtx, tc.preemptor, nodeName, victim, victim2)

			nomination := testCtx.p.nominations.nomination(tc.preemptor.UID)
			require.Equal(tCtx, tc.wantNomination, nomination != nil, "nomination")
			if nomination != nil {
				require.Equal(tCtx, nodeName, nomination.nodeName)
				require.Equal(tCtx, map[types.UID]types.NamespacedName{
					victim.UID:  {Namespace: namespace, Name: victim.Name},
					victim2.UID: {Namespace: namespace, Name: victim2.Name},
				}, nomination.victims)
			}
			var victimClaims []string
			for _, claim := range testCtx.p.nominations.victimClaimsOnNode(nodeName) {
				victimClaims = append(victimClaims, claim.claim.String())
				require.Equal(tCtx, map[types.UID]types.NamespacedName{victim.UID: {Namespace: namespace, Name: victim.Name}}, claim.victims)
			}
			require.Equal(tCtx, tc.wantVictimClaim, victimClaims, "victim claims")
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
	require.Len(tCtx, testCtx.p.nominations.victimClaimsOnNode(nodeName), 1, "victim claims before deallocation")

	updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
	})
	require.Empty(tCtx, testCtx.p.nominations.victimClaimsOnNode(nodeName), "victim claims after deallocation")
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

	testCtx.p.nominations.forget(testCtx.p.nominations.victimClaimsOnNode(nodeName)[0].uid, nil)
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

func TestIsSchedulableAfterAssignedPodDeleted(t *testing.T) {
	testIsSchedulableAfterAssignedPodDeleted(ktesting.Init(t))
}

func testIsSchedulableAfterAssignedPodDeleted(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim-2").Node(nodeName).Obj()
	waiting := preemptionPod("waiting", "waiting-claim").Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		preemptionClaim("waiting-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		reserve(preemptionClaim("victim-claim-2").Allocation(preemptionAllocation("instance-2")).Obj(), victim2),
		preemptionClaim("unallocated-claim").Obj(),
	}
	testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, victim2})
	nominate(tCtx, testCtx, preemptor, nodeName, victim, victim2)
	logger := klog.FromContext(tCtx)
	hint := func(pod, deleted *v1.Pod) fwk.QueueingHint {
		hint, err := testCtx.p.isSchedulableAfterAssignedPodDeleted(logger, pod, deleted, nil)
		tCtx.ExpectNoError(err)
		return hint
	}

	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, victim), "first victim deleted, second one still bound")
	require.Equal(tCtx, fwk.QueueSkip, hint(waiting, victim), "victim of another pod with allocated claim")
	require.Equal(tCtx, fwk.QueueSkip, hint(preemptor, preemptionPod("unrelated").Node(nodeName).Obj()), "unrelated pod without claims")
	require.Equal(tCtx, fwk.Queue, hint(waiting, preemptionPod("forgotten", "unallocated-claim").Node(nodeName).Obj()), "pod without its pending allocation")
	completed := preemptionPod("completed", "unallocated-claim").Node(nodeName).Obj()
	completed.Status.Phase = v1.PodSucceeded
	require.Equal(tCtx, fwk.QueueSkip, hint(waiting, completed), "completed pod whose claim got deallocated")
	require.Equal(tCtx, fwk.QueueSkip, hint(waiting, preemptionPod("unassigned", "unallocated-claim").Obj()), "unassigned pod which lost its nomination")
	deletePod(tCtx, testCtx, victim2)
	require.Equal(tCtx, fwk.Queue, hint(preemptor, victim), "last victim deleted")
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
