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

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPreemptionTrackerHolds(t *testing.T) {
	holdA1 := &podHold{nodeName: nodeName, heldClaims: map[types.UID]heldClaim{"claim-a": {allocatedDevices: &preemptionAllocation("gpu-0").Devices}}}
	holdA2 := &podHold{nodeName: nodeName, heldClaims: map[types.UID]heldClaim{"claim-a": {allocatedDevices: &preemptionAllocation("gpu-1").Devices}}}
	holdB := &podHold{nodeName: node2Name, heldClaims: map[types.UID]heldClaim{"claim-b": {allocatedDevices: &poolAllocation(node2Name, "gpu-0").Devices}}}

	type op struct {
		podUID       types.UID
		nominate     *podHold // nil means release
		wantReturned *podHold
	}
	testcases := map[string]struct {
		ops       []op
		wantHolds map[types.UID]*podHold
	}{
		"nominate-and-release": {
			ops: []op{
				{podUID: "p1", nominate: holdA1, wantReturned: nil},
				{podUID: "p1", wantReturned: holdA1},
				{podUID: "p1", wantReturned: nil},
			},
			wantHolds: map[types.UID]*podHold{"p1": nil},
		},
		"replace-existing-hold": {
			ops: []op{
				{podUID: "p1", nominate: holdA1, wantReturned: nil},
				{podUID: "p1", nominate: holdA2, wantReturned: holdA1},
			},
			wantHolds: map[types.UID]*podHold{"p1": holdA2},
		},
		"independent-pods": {
			ops: []op{
				{podUID: "p1", nominate: holdA1, wantReturned: nil},
				{podUID: "p2", nominate: holdB, wantReturned: nil},
				{podUID: "p1", wantReturned: holdA1},
			},
			wantHolds: map[types.UID]*podHold{"p1": nil, "p2": holdB},
		},
		"release-missing": {
			ops: []op{
				{podUID: "missing", wantReturned: nil},
			},
			wantHolds: map[types.UID]*podHold{"missing": nil},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			s := newPreemptionTracker()
			for i, step := range tc.ops {
				var got *podHold
				if step.nominate != nil {
					got = s.nominate(step.podUID, step.nominate)
				} else {
					got = s.release(step.podUID)
				}
				require.Same(t, step.wantReturned, got, "op %d returned hold", i)
			}
			for podUID, want := range tc.wantHolds {
				require.Same(t, want, s.hold(podUID), "hold of %s", podUID)
			}
		})
	}
}

func TestPreemptionTrackerVictimClaims(t *testing.T) {
	newClaim := func(nodeName, device string, podUID types.UID) *releasedClaim {
		return &releasedClaim{
			claim:            types.NamespacedName{Namespace: namespace, Name: "c"},
			uid:              "c",
			nodeName:         nodeName,
			allocatedDevices: &poolAllocation(nodeName, device).Devices,
			podUID:           podUID,
		}
	}
	podsOf := func(s *preemptionTracker, nodeName string) []types.UID {
		var uids []types.UID
		for _, claim := range s.victimClaimsOnNode(nodeName) {
			uids = append(uids, claim.podUID)
		}
		slices.Sort(uids)
		return uids
	}

	type assumeStep struct {
		nodeName string
		device   string
		podUID   types.UID
	}
	type forgetMode int
	const (
		forgetNone forgetMode = iota
		forgetFirstPointer
		forgetLastPointer
		forgetUnconditional
		forgetUntracked
	)

	testcases := map[string]struct {
		assumes        []assumeStep
		forget         forgetMode
		wantPodsByNode map[string][]types.UID
		wantTracked    bool
		wantDevice     string
	}{
		"reallocated-other-node": {
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
				{nodeName: "b", device: "instance-2", podUID: "v2"},
			},
			wantPodsByNode: map[string][]types.UID{"a": nil, "b": {"v2"}},
			wantTracked:    true,
			wantDevice:     "instance-2",
		},
		"reallocated-same-node": {
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
				{nodeName: "a", device: "instance-2", podUID: "v2"},
			},
			wantPodsByNode: map[string][]types.UID{"a": {"v2"}},
			wantTracked:    true,
			wantDevice:     "instance-2",
		},
		"forget-stale-pointer": {
			// assumeVictimClaim replaced the first entry, so pointer-matched forget is a no-op.
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
				{nodeName: "a", device: "instance-1", podUID: "v1"},
			},
			forget:         forgetFirstPointer,
			wantPodsByNode: map[string][]types.UID{"a": {"v1"}},
			wantTracked:    true,
			wantDevice:     "instance-1",
		},
		"forget-matching-pointer": {
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
			},
			forget:         forgetLastPointer,
			wantPodsByNode: map[string][]types.UID{"a": nil},
			wantTracked:    false,
		},
		"forget-unconditional": {
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
				{nodeName: "a", device: "instance-1", podUID: "v1"},
			},
			forget:         forgetUnconditional,
			wantPodsByNode: map[string][]types.UID{"a": nil},
			wantTracked:    false,
		},
		"forget-untracked": {
			assumes: []assumeStep{
				{nodeName: "a", device: "instance-1", podUID: "v1"},
			},
			forget:         forgetUntracked,
			wantPodsByNode: map[string][]types.UID{"a": {"v1"}},
			wantTracked:    true,
			wantDevice:     "instance-1",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			s := newPreemptionTracker()
			var assumed []*releasedClaim
			for _, step := range tc.assumes {
				c := newClaim(step.nodeName, step.device, step.podUID)
				s.assumeVictimClaim(c)
				assumed = append(assumed, c)
			}
			switch tc.forget {
			case forgetFirstPointer:
				s.forgetVictimClaim("c", assumed[0])
			case forgetLastPointer:
				s.forgetVictimClaim("c", assumed[len(assumed)-1])
			case forgetUnconditional:
				s.forgetVictimClaim("c", nil)
			case forgetUntracked:
				s.forgetVictimClaim("other", nil)
			}
			for node, wantPods := range tc.wantPodsByNode {
				require.Equal(t, wantPods, podsOf(s, node), "victim claims on node %s", node)
			}
			tracked := s.victimClaim("c")
			require.Equal(t, tc.wantTracked, tracked != nil, "claim tracked")
			if tc.wantTracked && tc.wantDevice != "" {
				require.Equal(t, tc.wantDevice, tracked.allocatedDevices.Results[0].Device, "tracked device")
			}
		})
	}
}

func TestPreemptionTrackerClaimHandlers(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)

	allocatedClaim := preemptionClaim("victim-claim").UID("c").Allocation(preemptionAllocation("instance-1")).Obj()
	unallocatedClaim := preemptionClaim("victim-claim").UID("c").Obj()
	reallocatedClaim := preemptionClaim("victim-claim").UID("c").Allocation(preemptionAllocation("instance-2")).Obj()
	otherAllocated := preemptionClaim("other-claim").UID("other").Allocation(preemptionAllocation("instance-2")).Obj()
	otherUnallocated := preemptionClaim("other-claim").UID("other").Obj()

	testcases := map[string]struct {
		invoke      func(h cache.ResourceEventHandler)
		wantTracked bool
	}{
		"update-deallocated": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(allocatedClaim, unallocatedClaim) },
			wantTracked: false,
		},
		"update-still-allocated": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(allocatedClaim, reallocatedClaim) },
			wantTracked: true,
		},
		"update-newly-allocated": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(unallocatedClaim, allocatedClaim) },
			wantTracked: true,
		},
		"update-unrelated-claim-deallocated": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(otherAllocated, otherUnallocated) },
			wantTracked: true,
		},
		"update-unexpected-old-object": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(&v1.Node{}, unallocatedClaim) },
			wantTracked: true,
		},
		"update-unexpected-new-object": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnUpdate(allocatedClaim, &v1.Node{}) },
			wantTracked: true,
		},
		"delete-claim": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnDelete(allocatedClaim) },
			wantTracked: false,
		},
		"delete-tombstone": {
			invoke: func(h cache.ResourceEventHandler) {
				h.OnDelete(cache.DeletedFinalStateUnknown{Key: "default/victim-claim", Obj: allocatedClaim})
			},
			wantTracked: false,
		},
		"delete-unexpected-object": {
			invoke:      func(h cache.ResourceEventHandler) { h.OnDelete(&v1.Node{}) },
			wantTracked: true,
		},
		"delete-unexpected-tombstone": {
			invoke: func(h cache.ResourceEventHandler) {
				h.OnDelete(cache.DeletedFinalStateUnknown{Key: "default/victim-claim", Obj: &v1.Node{}})
			},
			wantTracked: true,
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			s := newPreemptionTracker()
			s.assumeVictimClaim(&releasedClaim{
				claim:            types.NamespacedName{Namespace: namespace, Name: "victim-claim"},
				uid:              "c",
				nodeName:         nodeName,
				allocatedDevices: &allocatedClaim.Status.Allocation.Devices,
				podUID:           "victim",
			})
			tc.invoke(s.claimHandlers(logger))
			require.Equal(tCtx, tc.wantTracked, s.victimClaim("c") != nil, "claim tracked")
		})
	}
}

func TestVictimClaimPruning(t *testing.T) {
	testVictimClaimPruning(ktesting.Init(t))
}

func testVictimClaimPruning(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()

	for name, action := range map[string]func(tCtx ktesting.TContext, testCtx *testContext){
		"deallocated": func(tCtx ktesting.TContext, testCtx *testContext) {
			updateClaim(tCtx, testCtx, "victim-claim", func(claim *resourceapi.ResourceClaim) {
				claim.Status.ReservedFor = nil
				claim.Status.Allocation = nil
			})
		},
		"deleted": func(tCtx ktesting.TContext, testCtx *testContext) {
			tCtx.ExpectNoError(testCtx.client.ResourceV1().ResourceClaims(namespace).Delete(tCtx, "victim-claim", metav1.DeleteOptions{}))
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			claims := []*resourceapi.ResourceClaim{
				reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("instance-1")).Obj(), victim),
			}
			testCtx := setupVictims(tCtx, claims, []apiruntime.Object{slice, victim})
			assumeVictims(tCtx, testCtx, victim)
			require.Len(tCtx, testCtx.p.preemptionTracker.victimClaimsOnNode(nodeName), 1, "victim claims before action")

			action(tCtx, testCtx)
			eventually(tCtx, "victim claim forgotten", func() bool {
				return len(testCtx.p.preemptionTracker.victimClaimsOnNode(nodeName)) == 0
			})
		})
	}
}
