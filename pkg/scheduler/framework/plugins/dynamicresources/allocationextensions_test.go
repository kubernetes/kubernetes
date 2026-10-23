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
	"maps"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPodAllocations(t *testing.T) {
	held := func(device string) map[types.UID]heldClaim {
		return map[types.UID]heldClaim{"c": {claim: types.NamespacedName{Namespace: namespace, Name: "c"}, allocation: preemptionAllocation(device)}}
	}
	candidate := func(nodeName string, assumed bool, device string) *allocation {
		return &allocation{nodeName: nodeName, assumed: assumed, claims: held(device)}
	}

	t.Run("confirm-candidate", func(t *testing.T) {
		n := newClaimNominations()
		n.setCandidate("p", candidate("a", false, "gpu-0"))
		// A nomination to another node does not get the candidate.
		n.addAllocation("p", "b", false, 0, true)
		require.Empty(t, n.nominatedAllocation("p").claims)
		n.removeAllocation("p", "b", false)
		n.addAllocation("p", "a", false, 0, true)
		require.Equal(t, held("gpu-0"), n.nominatedAllocation("p").claims)
		require.Nil(t, n.pods["p"].candidate, "confirmed candidate")
		require.Equal(t, []types.UID{"p"}, slices.Collect(maps.Keys(n.nominationsOnNode("a"))), "nominations on the node")
		require.Empty(t, n.nominationsOnNode("b"), "nominations on the old node")
		// The end of an older nomination must not remove the newer one.
		removed, _ := n.removeAllocation("p", "b", false)
		require.Nil(t, removed)
		require.Equal(t, "a", n.nominatedNode("p"))
		removed, keeper := n.removeAllocation("p", "a", false)
		require.Equal(t, held("gpu-0"), removed.claims)
		require.Nil(t, keeper)
		require.Empty(t, n.pods, "pods after the nomination ended")
		require.Empty(t, n.nominations, "nominations after the nomination ended")
	})

	t.Run("without-claims", func(t *testing.T) {
		n := newClaimNominations()
		n.addAllocation("p", "a", false, 0, false)
		require.Empty(t, n.pods, "nominated pod without claims")
		n.addAllocation("p", "a", true, 0, false)
		require.Empty(t, n.pods, "assumed pod without candidate")
	})

	t.Run("keepers", func(t *testing.T) {
		n := newClaimNominations()
		n.setCandidate("p", candidate("a", false, "gpu-0"))
		n.addAllocation("p", "a", false, 0, true)
		// A new nomination to the same node.
		n.setCandidate("p", candidate("a", false, "gpu-1"))
		_, keeper := n.removeAllocation("p", "a", false)
		require.Equal(t, held("gpu-1"), keeper.claims, "candidate for the same node")
		n.addAllocation("p", "a", false, 0, true)

		// The assumed pod comes before the end of the nomination.
		n.setCandidate("p", candidate("a", true, "gpu-2"))
		n.addAllocation("p", "a", true, 0, false)
		_, keeper = n.removeAllocation("p", "a", false)
		require.Equal(t, held("gpu-2"), keeper.claims, "assumed pod on the same node")
		require.Empty(t, n.nominatedNode("p"))
		removed, keeper := n.removeAllocation("p", "a", true)
		require.Equal(t, held("gpu-2"), removed.claims, "allocation of the assumed pod")
		require.Nil(t, keeper)
		require.Empty(t, n.pods)
	})

	t.Run("drop-candidate", func(t *testing.T) {
		n := newClaimNominations()
		n.setCandidate("p", candidate("a", false, "gpu-0"))
		n.dropCandidate("p")
		require.Empty(t, n.pods)
	})
}

func TestNominatedNodes(t *testing.T) {
	testNominatedNodes(ktesting.Init(t))
}

// testNominatedNodes checks that the plugin follows the nominations of the
// PodNominator for pods with claims, also without a prepared allocation.
func testNominatedNodes(tCtx ktesting.TContext) {
	pod := preemptionPod("pod", "claim").Priority(100).Obj()
	noClaims := preemptionPod("no-claims").Priority(100).Obj()
	testCtx := setupNominations(tCtx, []*resourceapi.ResourceClaim{preemptionClaim("claim").Obj()}, []apiruntime.Object{pod, noClaims})
	nominations := testCtx.p.nominations

	testCtx.p.AddAllocation(tCtx, pod, nodeName)
	require.Equal(tCtx, nodeName, nominations.nominatedNode(pod.UID), "nominated")
	testCtx.p.AddAllocation(tCtx, noClaims, nodeName)
	require.Empty(tCtx, nominations.nominatedNode(noClaims.UID), "nominated pod without claims")

	// A change of the nominated node is a removal followed by an addition.
	testCtx.p.RemoveAllocation(tCtx, pod, nodeName)
	testCtx.p.AddAllocation(tCtx, pod, node2Name)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "nominated to another node")

	// Assumed pods are not nominated.
	assumed := fromPod(pod).Node(node2Name).Obj()
	testCtx.p.AddAllocation(tCtx, assumed, node2Name)
	testCtx.p.RemoveAllocation(tCtx, assumed, node2Name)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "after the calls for the assumed pod")

	testCtx.p.RemoveAllocation(tCtx, pod, nodeName)
	require.Equal(tCtx, node2Name, nominations.nominatedNode(pod.UID), "after removing a nomination for another node")
	testCtx.p.RemoveAllocation(tCtx, pod, node2Name)
	require.Empty(tCtx, nominations.nominatedNode(pod.UID), "after the nomination ended")
}

func TestPrepareAllocation(t *testing.T) {
	testPrepareAllocation(ktesting.Init(t))
}

// testPrepareAllocation checks which allocation PrepareAllocation takes
// from the scheduling cycle.
func testPrepareAllocation(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Device("gpu-1").Device("gpu-2").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	claims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		preemptionClaim("never-claim").Obj(),
		reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
		reserve(preemptionClaim("allocated-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), victim),
	}
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	never := preemptionPod("never", "never-claim").Priority(100).PreemptionPolicy(v1.PreemptNever).Obj()
	allocated := preemptionPod("allocated", "allocated-claim").Priority(100).Obj()
	deleted := preemptionPod("deleted", "preemptor-claim").Priority(100).Obj()
	candidate := func(testCtx *testContext, pod *v1.Pod) *allocation {
		testCtx.p.nominations.mutex.RLock()
		defer testCtx.p.nominations.mutex.RUnlock()
		if p := testCtx.p.nominations.pods[pod.UID]; p != nil {
			return p.candidate
		}
		return nil
	}
	devices := func(a *allocation) []string {
		var devices []string
		for _, held := range a.claims {
			for _, result := range held.allocation.Devices.Results {
				devices = append(devices, result.Device)
			}
		}
		return devices
	}
	// prepare runs PrepareAllocation for prepared with the CycleState of pod,
	// in which Filter ran on the node with the victims removed.
	prepare := func(tCtx ktesting.TContext, testCtx *testContext, pod, prepared *v1.Pod, victims ...*v1.Pod) {
		state := framework.NewCycleState()
		_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
		require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
		_ = dryRun(tCtx, testCtx, state, pod, nodeName, victims...)
		testCtx.p.PrepareAllocation(tCtx, state, prepared, nodeName)
	}

	tCtx.Run("nominated", func(tCtx ktesting.TContext) {
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, preemptor})
		prepare(tCtx, testCtx, preemptor, preemptor, victim)
		a := candidate(testCtx, preemptor)
		require.NotNil(tCtx, a, "candidate")
		require.False(tCtx, a.assumed, "assumed")
		require.Equal(tCtx, []string{"gpu-0"}, devices(a), "devices from the dry run")
		require.Equal(tCtx, []types.UID{lookupClaimUID(tCtx, testCtx, "victim-claim")}, a.released.UnsortedList(), "released claims")
	})

	tCtx.Run("nominated-cannot-preempt", func(tCtx ktesting.TContext) {
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, never})
		prepare(tCtx, testCtx, never, never, victim)
		require.Nil(tCtx, candidate(testCtx, never), "candidate")
	})

	tCtx.Run("assumed", func(tCtx ktesting.TContext) {
		// Also for pods which cannot preempt.
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, never})
		prepare(tCtx, testCtx, never, fromPod(never).Node(nodeName).Obj())
		a := candidate(testCtx, never)
		require.NotNil(tCtx, a, "candidate")
		require.True(tCtx, a.assumed, "assumed")
		require.Equal(tCtx, []string{"gpu-2"}, devices(a), "devices")
		require.Empty(tCtx, a.released, "released claims")
	})

	tCtx.Run("assumed-nothing-to-allocate", func(tCtx ktesting.TContext) {
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, allocated})
		prepare(tCtx, testCtx, allocated, fromPod(allocated).Node(nodeName).Obj())
		require.Nil(tCtx, candidate(testCtx, allocated), "candidate")
	})

	tCtx.Run("deleted-before", func(tCtx ktesting.TContext) {
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim})
		prepare(tCtx, testCtx, deleted, deleted, victim)
		require.Nil(tCtx, candidate(testCtx, deleted), "candidate")
	})

	tCtx.Run("deleted-after", func(tCtx ktesting.TContext) {
		testCtx := setupNominations(tCtx, claims, []apiruntime.Object{slice, victim, deleted})
		prepare(tCtx, testCtx, deleted, deleted, victim)
		require.NotNil(tCtx, candidate(testCtx, deleted), "candidate")
		deletePod(tCtx, testCtx, deleted)
		eventually(tCtx, "candidate dropped", func() bool {
			return candidate(testCtx, deleted) == nil
		})
	})
}
