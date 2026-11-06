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
	n := newClaimNominations()
	n.add("p", &podNomination{nodeName: "a"})
	n.add("p", &podNomination{nodeName: "b"})
	require.Equal(t, "b", n.nomination("p").nodeName)
	// An event about the older nomination must not remove the newer one.
	n.remove("p", "a")
	require.NotNil(t, n.nomination("p"))
	n.remove("p", "b")
	require.Nil(t, n.nomination("p"))
}

// setupNominations sets up the plugin with the claimNominations store,
// which needs the feature gate.
func setupNominations(tCtx ktesting.TContext, claims []*resourceapi.ResourceClaim, objs []apiruntime.Object) *testContext {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	testCtx := setupPreemption(tCtx, []*v1.Node{workerNode, workerNode2}, claims, objs, feature.Features{EnableDRAPreemption: true})
	require.NotNil(tCtx, testCtx.p.nominations, "claimNominations")
	return testCtx
}

// nominate runs PreFilter for the preemptor, simulates the removal of the
// victims, calls AddNominatedPod and then AssumeVictim for each victim, like
// a successful preemption.
func nominate(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nodeName string, victims ...*v1.Pod) fwk.CycleState {
	state := nominateWithoutAssume(tCtx, testCtx, preemptor, nodeName, victims...)
	for _, victim := range victims {
		testCtx.p.AssumeVictim(tCtx, victim)
	}
	return state
}

// nominateWithoutAssume is nominate for a preemption which failed to
// preempt the victims.
func nominateWithoutAssume(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, nodeName string, victims ...*v1.Pod) fwk.CycleState {
	tCtx.Helper()
	state := framework.NewCycleState()
	_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	var nodeInfo fwk.NodeInfo
	for _, info := range testCtx.nodeInfos {
		if info.Node().Name == nodeName {
			nodeInfo = info
		}
	}
	require.NotNil(tCtx, nodeInfo, "node %s", nodeName)
	// Like the dry run of preemption, which ends with Filter on a copy of
	// the state without the victims. Whether the pod fits does not matter.
	sim := state.Clone()
	for _, victim := range victims {
		podInfo, err := framework.NewPodInfo(victim)
		tCtx.ExpectNoError(err)
		require.True(tCtx, testCtx.p.RemovePod(tCtx, sim, preemptor, podInfo, nodeInfo).IsSuccess(), "RemovePod")
	}
	_ = testCtx.p.Filter(tCtx, sim, preemptor, nodeInfo)
	testCtx.p.AddNominatedPod(tCtx, state, preemptor, nodeName)
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

func TestAddNominatedPod(t *testing.T) {
	testAddNominatedPod(ktesting.Init(t))
}

func TestAddNominatedPodAfterReprieve(t *testing.T) {
	testAddNominatedPodAfterReprieve(ktesting.Init(t))
}

// testAddNominatedPodAfterReprieve checks that AddNominatedPod gets the
// victims which SelectVictimsOnNode ends up with, also if the last Filter ran
// on a copy made by RunFilterPluginsWithNominatedPods for a victim which did
// not get reprieved.
func testAddNominatedPodAfterReprieve(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	victim2 := preemptionPod("victim2", "victim-claim-2").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
		reserve(preemptionClaim("victim-claim-2").Allocation(preemptionAllocation("instance-2")).Obj(), victim2),
	}
	podInfo := func(pod *v1.Pod) fwk.PodInfo {
		podInfo, err := framework.NewPodInfo(pod)
		tCtx.ExpectNoError(err)
		return podInfo
	}
	victimsOf := func(testCtx *testContext) []string {
		var names []string
		for _, name := range testCtx.p.nominations.nomination(preemptor.UID).victims {
			names = append(names, name.Name)
		}
		slices.Sort(names)
		return names
	}

	for name, tc := range map[string]struct {
		reprieved   bool
		wantVictims []string
	}{
		"not-reprieved": {wantVictims: []string{victim.Name, victim2.Name}},
		"reprieved":     {reprieved: true, wantVictims: []string{victim.Name}},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, preemptor, victim, victim2})
			nodeInfo := testCtx.nodeInfos[0]
			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
			require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)

			sim := state.Clone()
			for _, pod := range []*v1.Pod{victim, victim2} {
				require.True(tCtx, testCtx.p.RemovePod(tCtx, sim, preemptor, podInfo(pod), nodeInfo).IsSuccess(), "RemovePod")
			}
			require.True(tCtx, testCtx.p.Filter(tCtx, sim, preemptor, nodeInfo).IsSuccess(), "Filter without victims")
			// Reprieve victim2: the first pass of
			// RunFilterPluginsWithNominatedPods runs on a copy.
			require.True(tCtx, testCtx.p.AddPod(tCtx, sim, preemptor, podInfo(victim2), nodeInfo).IsSuccess(), "AddPod")
			_ = testCtx.p.Filter(tCtx, sim.Clone(), preemptor, nodeInfo)
			if tc.reprieved {
				_ = testCtx.p.Filter(tCtx, sim, preemptor, nodeInfo)
			} else {
				require.True(tCtx, testCtx.p.RemovePod(tCtx, sim, preemptor, podInfo(victim2), nodeInfo).IsSuccess(), "RemovePod")
			}

			testCtx.p.AddNominatedPod(tCtx, state, preemptor, nodeName)
			require.Equal(tCtx, tc.wantVictims, victimsOf(testCtx), "victims of the nomination")
		})
	}
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
		preemptor      *v1.Pod
		deleted        bool
		wantNomination bool
	}{
		"released-claims": {
			preemptor:      preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj(),
			wantNomination: true,
		},
		"cannot-release": {
			preemptor: preemptionPod("preemptor", "preemptor-claim").Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
		},
		"preemptor-gone": {
			preemptor: preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj(),
			deleted:   true,
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			objs := []apiruntime.Object{slice, victim, victim2, running}
			if !tc.deleted {
				objs = append(objs, tc.preemptor)
			}
			testCtx := setupNominations(tCtx, claims, objs)
			nominateWithoutAssume(tCtx, testCtx, tc.preemptor, nodeName, victim, victim2)

			nomination := testCtx.p.nominations.nomination(tc.preemptor.UID)
			require.Equal(tCtx, tc.wantNomination, nomination != nil, "nomination")
			if nomination != nil {
				require.Equal(tCtx, nodeName, nomination.nodeName)
				require.Equal(tCtx, map[types.UID]types.NamespacedName{
					victim.UID:  {Namespace: namespace, Name: victim.Name},
					victim2.UID: {Namespace: namespace, Name: victim2.Name},
				}, nomination.victims)
			}
		})
	}
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
