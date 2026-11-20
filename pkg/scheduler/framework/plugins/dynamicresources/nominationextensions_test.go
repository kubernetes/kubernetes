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
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func TestNominationExtensions(t *testing.T) {
	testNominationExtensions(ktesting.Init(t))
}

// testNominationExtensions verifies when AddNomination and RemoveNomination
// emit EventResourceClaimHoldReleased because held devices became free without
// a ResourceClaim event.
func testNominationExtensions(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Devices("gpu-0", "gpu-1").Obj()
	victim := preemptionPod("victim", "victim-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	assumedPreemptor := fromPod(preemptor).Node(nodeName).Obj()
	setup := func(tCtx ktesting.TContext) *testContext {
		tCtx.Helper()
		testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			reserve(preemptionClaim("victim-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), victim),
		}, []apiruntime.Object{slice, victim, preemptor})
		requireEvents(tCtx, 0, nominate(tCtx, testCtx, preemptor, nil, victim))
		require.Equal(tCtx, []string{"gpu-0"}, heldDevices(testCtx, preemptor), "held devices")
		return testCtx
	}
	signalPending := func(tCtx ktesting.TContext, testCtx *testContext, allocation *resourceapi.AllocationResult) {
		tCtx.Helper()
		claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
		tCtx.ExpectNoError(err)
		claim = claim.DeepCopy()
		claim.Status.Allocation = allocation
		tCtx.ExpectNoError(testCtx.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim))
	}

	singleClaimCases := map[string]struct {
		// unnominated skips the initial nomination holding gpu-0.
		unnominated bool
		// action performs the nomination change(s) under test and returns the
		// result whose event count is checked.
		action     func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult
		wantEvents int
		wantHeld   []string
		wantNode   string
	}{
		"nomination-ended": {
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				return endNomination(tCtx, testCtx, preemptor)
			},
			wantEvents: 1,
		},
		"renominated-same-devices": {
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				return nominate(tCtx, testCtx, preemptor, nil, victim)
			},
			wantHeld: []string{"gpu-0"},
		},
		"renominated-other-devices": {
			// Re-nominating without removing victim selects free gpu-1, releasing held gpu-0.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				return nominate(tCtx, testCtx, preemptor, nil)
			},
			wantEvents: 1,
			wantHeld:   []string{"gpu-1"},
		},
		"renominated-other-node": {
			// PodNominator calls RemoveNomination on the old node before AddNomination on the new node.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				hold := &podHold{nodeName: node2Name, heldClaims: testCtx.p.preemptionTracker.hold(preemptor.UID).heldClaims}
				requireEvents(tCtx, 1, endNomination(tCtx, testCtx, preemptor))
				return testCtx.p.AddNomination(tCtx, preemptor, node2Name, hold)
			},
			wantHeld: []string{"gpu-0"},
			wantNode: node2Name,
		},
		"renominated-claim-recreated": {
			// Claim was recreated under the same name with a new UID; keeping the same device emits no release event.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				recreated := make(map[types.UID]heldClaim)
				for uid, held := range testCtx.p.preemptionTracker.hold(preemptor.UID).heldClaims {
					recreated[uid+"-recreated"] = held
				}
				return testCtx.p.AddNomination(tCtx, preemptor, nodeName, &podHold{nodeName: nodeName, heldClaims: recreated})
			},
			wantHeld: []string{"gpu-0"},
		},
		"nothing-prepared": {
			// Re-nominating with nil prepared state drops the existing hold and
			// emits one event; repeating with no active hold is a no-op.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				first := testCtx.p.AddNomination(tCtx, preemptor, nodeName, nil)
				require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices after first nil")
				requireEvents(tCtx, 0, testCtx.p.AddNomination(tCtx, preemptor, nodeName, nil))
				return first
			},
			wantEvents: 1,
		},
		"not-nominated": {
			unnominated: true,
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				return endNomination(tCtx, testCtx, preemptor)
			},
		},
		// Assumed pod's nomination ends after Reserve and Permit while its allocation is still pending.
		"reserved-held-devices": {
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				signalPending(tCtx, testCtx, preemptionAllocation("gpu-0"))
				return testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName)
			},
		},
		"reserved-held-devices-new-share": {
			// Freshly generated ShareIDs do not prevent matching held devices.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				allocation := preemptionAllocation("gpu-0")
				allocation.Devices.Results[0].ShareID = new(types.UID("share"))
				signalPending(tCtx, testCtx, allocation)
				return testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName)
			},
		},
		"ended-during-own-cycle": {
			// Unassigned pod deleted between Reserve and Permit. Unreserve will
			// drop the pending allocation silently, so RemoveNomination must emit
			// an event.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				signalPending(tCtx, testCtx, preemptionAllocation("gpu-0"))
				return testCtx.p.RemoveNomination(tCtx, preemptor, nodeName)
			},
			wantEvents: 1,
		},
		"reserved-other-devices": {
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				signalPending(tCtx, testCtx, preemptionAllocation("gpu-1"))
				return testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName)
			},
			wantEvents: 1,
		},
		"reserved-different-device-count": {
			// Holding two devices while reserving only one frees a held device.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				claim, err := testCtx.draManager.ResourceClaims().Get(namespace, "preemptor-claim")
				tCtx.ExpectNoError(err)
				testCtx.p.AddNomination(tCtx, preemptor, nodeName, &podHold{
					nodeName: nodeName,
					heldClaims: map[types.UID]heldClaim{
						claim.UID: {
							claim:            types.NamespacedName{Namespace: namespace, Name: "preemptor-claim"},
							allocatedDevices: &preemptionAllocation("gpu-0", "gpu-1").Devices,
						},
					},
				})
				signalPending(tCtx, testCtx, preemptionAllocation("gpu-0"))
				return testCtx.p.RemoveNomination(tCtx, assumedPreemptor, nodeName)
			},
			wantEvents: 1,
		},
		"bound": {
			// Claim was already committed with the held device, so ending the
			// nomination frees nothing.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				updateClaim(tCtx, testCtx, "preemptor-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("gpu-0")
					claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{consumerRef(preemptor)}
				})
				return endNomination(tCtx, testCtx, preemptor)
			},
		},
		"bound-claim-recreated": {
			// Claim was recreated under the same name and allocated the held
			// device, so the device remains in use.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
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
				return endNomination(tCtx, testCtx, preemptor)
			},
		},
		"renominated-nothing-to-allocate": {
			// preemptor-claim is allocated with gpu-0 already, so re-nomination
			// prepares nil and drops the hold without emitting an event.
			action: func(tCtx ktesting.TContext, testCtx *testContext) framework.NominationChangeResult {
				updateClaim(tCtx, testCtx, "preemptor-claim", func(claim *resourceapi.ResourceClaim) {
					claim.Status.Allocation = preemptionAllocation("gpu-0")
				})
				state := framework.NewCycleState()
				_, status := testCtx.p.PreFilter(tCtx, state, preemptor, testCtx.nodeInfos)
				require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
				return addNomination(tCtx, testCtx, state, preemptor, nodeName)
			},
		},
	}

	for name, tc := range singleClaimCases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			var testCtx *testContext
			if tc.unnominated {
				testCtx = setupVictims(tCtx, nil, nil)
			} else {
				testCtx = setup(tCtx)
			}
			requireEvents(tCtx, tc.wantEvents, tc.action(tCtx, testCtx))
			if tc.wantHeld == nil {
				require.Empty(tCtx, heldDevices(testCtx, preemptor), "held devices")
			} else {
				require.Equal(tCtx, tc.wantHeld, heldDevices(testCtx, preemptor), "held devices")
			}
			if tc.wantNode != "" {
				hold := testCtx.p.preemptionTracker.hold(preemptor.UID)
				require.NotNil(tCtx, hold, "hold")
				require.Equal(tCtx, tc.wantNode, hold.nodeName, "hold node")
			}
		})
	}

	// twoClaims builds a two-claim hold so partial-release tests exercise both
	// map iteration orders.
	twoClaims := func(deviceA, deviceB string) map[types.UID]heldClaim {
		claims := map[types.UID]heldClaim{
			"claim-a": {claim: types.NamespacedName{Namespace: namespace, Name: "claim-a"}, allocatedDevices: &preemptionAllocation(deviceA).Devices},
		}
		if deviceB != "" {
			claims["claim-b"] = heldClaim{claim: types.NamespacedName{Namespace: namespace, Name: "claim-b"}, allocatedDevices: &preemptionAllocation(deviceB).Devices}
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
				testCtx.p.AddNomination(tCtx, preemptor, nodeName, &podHold{nodeName: nodeName, heldClaims: twoClaims("gpu-0", "gpu-1")})
				result := testCtx.p.AddNomination(tCtx, preemptor, nodeName, &podHold{nodeName: nodeName, heldClaims: newer})
				require.Len(tCtx, result.Events, 1, "events")
				require.Equal(tCtx, "claim-b", result.Events[0].OldObj.(*resourceapi.ResourceClaim).Name, "released claim")
			}
		})
	}

	tCtx.Run("reserved-through-filter", func(tCtx ktesting.TContext) {
		// Once victim-claim is deallocated, Filter picks held gpu-0 and Reserve
		// records it as pending; if Unreserve runs before RemoveNomination,
		// RemoveNomination emits a release event.
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

func TestPrepareNomination(t *testing.T) {
	testPrepareNomination(ktesting.Init(t))
}

func testPrepareNomination(tCtx ktesting.TContext) {
	slice := st.MakeResourceSlice(nodeName, driver).Devices("gpu-0", "gpu-1").Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Obj()
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()

	type step struct {
		remove     *v1.Pod
		add        *v1.Pod
		filter     bool
		wantPicked []string
	}
	testcases := map[string]struct {
		pod           *v1.Pod
		deallocateV2  bool
		wantPreFilter fwk.Code
		steps         []step
		wantDevices   []string
	}{
		"final-state": {
			// Both victims are removed, then each reprieve attempt passes
			// DynamicResources (updating pickedDevices to gpu-1) before a later
			// plugin rejects and re-removes it; PrepareNomination must return the
			// allocation for both victims removed.
			steps: []step{
				{remove: v1Pod},
				{remove: v2Pod, filter: true},
				{add: v2Pod, filter: true},
				{remove: v2Pod},
				{add: v1Pod, filter: true, wantPicked: []string{"gpu-1"}},
				{remove: v1Pod},
			},
			wantDevices: []string{"gpu-0"},
		},
		"last-step-reprieved": {
			// v1's reprieve succeeds, so the final state has only v2 removed.
			steps: []step{
				{remove: v1Pod},
				{remove: v2Pod, filter: true},
				{add: v1Pod, filter: true},
			},
			wantDevices: []string{"gpu-1"},
		},
		"not-filtered": {
			// RemovePod ran without a subsequent Filter call for that victim set.
			steps: []step{
				{remove: v1Pod},
			},
		},
		"no-claims": {
			// Pod without claims skips PreFilter; PrepareNomination is a no-op.
			pod:           preemptionPod("no-claims").Priority(100).Obj(),
			wantPreFilter: fwk.Skip,
		},
		"cannot-preempt": {
			// PreemptNever pods never acquire holds even if Filter allocated a free device.
			pod:          fromPod(preemptor).PreemptionPolicy(v1.PreemptNever).Obj(),
			deallocateV2: true,
			steps:        []step{{filter: true}},
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			pod := tc.pod
			if pod == nil {
				pod = preemptor
			}
			v2Claim := reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod)
			if tc.deallocateV2 {
				v2Claim = preemptionClaim("v2-claim").Obj()
			}
			testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
				preemptionClaim("preemptor-claim").Obj(),
				reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
				v2Claim,
			}, []apiruntime.Object{slice, v1Pod, v2Pod, pod})

			state := framework.NewCycleState()
			_, status := testCtx.p.PreFilter(tCtx, state, pod, testCtx.nodeInfos)
			wantPreFilter := tc.wantPreFilter
			if wantPreFilter == 0 {
				wantPreFilter = fwk.Success
			}
			require.Equal(tCtx, wantPreFilter, status.Code(), "PreFilter: %v", status)

			cs := state.Clone()
			nodeInfo := testCtx.nodeInfos[0]
			for i, s := range tc.steps {
				if s.remove != nil {
					removePod(tCtx, testCtx, cs, pod, s.remove, nodeInfo)
				}
				if s.add != nil {
					addPod(tCtx, testCtx, cs, pod, s.add, nodeInfo)
				}
				if s.filter {
					status := testCtx.p.Filter(tCtx, cs, pod, nodeInfo)
					require.True(tCtx, status.IsSuccess(), "step %d: Filter: %v", i, status)
					if s.wantPicked != nil {
						require.Equal(tCtx, s.wantPicked, pickedDevices(tCtx, cs), "step %d: picked devices", i)
					}
				}
			}

			prepared := testCtx.p.PrepareNomination(tCtx, state, pod, nodeName)
			if tc.wantDevices == nil {
				require.Nil(tCtx, prepared, "hold")
				return
			}
			hold, ok := prepared.(*podHold)
			require.True(tCtx, ok, "PrepareNomination returned %T", prepared)
			require.NotNil(tCtx, hold, "PrepareNomination returned a typed nil")
			require.Equal(tCtx, nodeName, hold.nodeName, "node of the hold")
			var devices []string
			for _, held := range hold.heldClaims {
				for _, result := range held.allocatedDevices.Results {
					devices = append(devices, result.Device)
				}
			}
			require.Equal(tCtx, tc.wantDevices, devices, "devices of the final state")
		})
	}
}

type podsFilter struct {
	name   string
	reject func(pods podNames) bool
}

type podNames map[string]bool

func (pl *podsFilter) Name() string {
	return pl.name
}

func (pl *podsFilter) Filter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	pods := podNames{}
	for _, podInfo := range nodeInfo.GetPods() {
		pods[podInfo.GetPod().Name] = true
	}
	if pl.reject != nil && pl.reject(pods) {
		return fwk.NewStatus(fwk.Unschedulable, "rejected by "+pl.name)
	}
	return nil
}

// setupDefaultPreemption wires a single DynamicResources instance between two
// mock Filter plugins ("Before" and "After", standing in for plugins like
// NodeResourcesFit without creating an import cycle) inside one Framework and
// DefaultPreemption, then runs PreFilter for preemptor.
func setupDefaultPreemption(
	tCtx ktesting.TContext,
	claims []*resourceapi.ResourceClaim,
	objs []apiruntime.Object,
	preemptor *v1.Pod,
	pods, nominated []*v1.Pod,
	podGroups []*schedulingv1beta1.PodGroup,
	before, after func(pods podNames) bool,
) (*defaultpreemption.DefaultPreemption, *DynamicResources, *internalcache.Snapshot, fwk.CycleState) {
	tCtx.Helper()
	genericWorkload := len(podGroups) > 0
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.GenericWorkload, genericWorkload)

	draFeatures := feature.Features{EnableDRAPreemption: true, EnableGenericWorkload: genericWorkload}
	env := setupDRAManager(tCtx, claims, []*resourceapi.DeviceClass{deviceClass}, podGroups, objs, draFeatures, false /* failPatch */, nil)

	var draPlugin *DynamicResources
	newPlugin := func(ctx context.Context, _ apiruntime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		pl, err := New(ctx, &config.DynamicResourcesArgs{}, fh, draFeatures)
		if err != nil {
			return nil, err
		}
		draPlugin = pl.(*DynamicResources)
		return pl, nil
	}
	filterPlugin := func(pl *podsFilter) tf.RegisterPluginFunc {
		return tf.RegisterFilterPlugin(pl.name, func(context.Context, apiruntime.Object, fwk.Handle) (fwk.Plugin, error) {
			return pl, nil
		})
	}
	nodes := []*v1.Node{workerNode, workerNode2}
	podGroupManager := internalcache.New(tCtx, nil, genericWorkload, false /* compositePodGroupEnabled */)
	for _, podGroup := range podGroups {
		podGroupManager.AddGenericPodGroup(fwk.NewGenericPodGroup(podGroup))
	}
	snapshot := internalcache.NewTestSnapshotWithPodGroups(pods, nodes, podGroups, nil)
	preemptionFeatures := feature.Features{EnableGenericWorkload: genericWorkload}
	// PodNominator ignores nominated pods that are not also present in the scheduling queue.
	nominator := internalqueue.NewTestQueueWithInformerFactory(tCtx, (&queuesort.PrioritySort{}).Less, env.informerFactory)
	for _, pod := range nominated {
		nominator.Add(tCtx, pod)
		podInfo, err := framework.NewPodInfo(pod)
		tCtx.ExpectNoError(err)
		nominator.AddNominatedPod(tCtx.Logger(), podInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName})
	}
	fh, err := tf.NewFramework(tCtx, []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		filterPlugin(&podsFilter{name: "Before", reject: before}),
		tf.RegisterPluginAsExtensions(Name, newPlugin, "PreFilter", "Filter"),
		filterPlugin(&podsFilter{name: "After", reject: after}),
	}, "",
		runtime.WithClientSet(env.client),
		runtime.WithInformerFactory(env.informerFactory),
		runtime.WithEventRecorder(&events.FakeRecorder{}),
		runtime.WithSharedDRAManager(env.draManager),
		runtime.WithPodGroupManager(podGroupManager),
		runtime.WithPodNominator(nominator),
		runtime.WithSnapshotSharedLister(snapshot),
		runtime.WithMutableSnapshotLister(snapshot),
		runtime.WithPreemptionManager(func(fh fwk.Handle) fwk.PreemptionManager {
			return preemption.NewPreemptionManager(fh, preemptionFeatures)
		}),
	)
	tCtx.ExpectNoError(err, "create framework")
	tCtx.Cleanup(func() { runtime.WaitForShutdown(fh) })
	require.NotNil(tCtx, draPlugin, "DynamicResources plugin initialized")

	env.startAndSync(tCtx)

	pl, err := defaultpreemption.New(tCtx, &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 10, MinCandidateNodesAbsolute: 100}, fh, preemptionFeatures)
	tCtx.ExpectNoError(err, "create DefaultPreemption")

	state := framework.NewCycleState()
	_, status, _ := fh.RunPreFilterPlugins(tCtx, state, preemptor)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	return pl, draPlugin, snapshot, state
}

func TestPrepareNominationWithDefaultPreemption(t *testing.T) {
	testPrepareNominationWithDefaultPreemption(ktesting.Init(t))
}

// testPrepareNominationWithDefaultPreemption verifies that PrepareNomination
// captures the allocation from the final victim set chosen by
// DefaultPreemption's simulation, even when intermediate reprieve attempts ran
// Filter later or when another Filter plugin rejected a candidate state.
func testPrepareNominationWithDefaultPreemption(tCtx ktesting.TContext) {
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Priority(10).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Priority(20).Obj()
	// v3 has no claims (simulating a pod blocking CPU/memory rather than DRA).
	v3Pod := preemptionPod("v3").Node(nodeName).Priority(30).Obj()
	holder := preemptionPod("holder").Priority(200).Obj()
	defaultPods := []*v1.Pod{v1Pod, v2Pod, v3Pod}
	defaultClaims := func() []*resourceapi.ResourceClaim {
		return []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
			reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod),
		}
	}
	sliceWithDevices := func(numDevices int) *resourceapi.ResourceSlice {
		slice := st.MakeResourceSlice(nodeName, driver)
		for i := range numDevices {
			slice = slice.Device(fmt.Sprintf("gpu-%d", i))
		}
		return slice.Obj()
	}

	podGroup := st.MakePodGroup().Name("group").Namespace(namespace).UID("group").Priority(10).DisruptionModeAll().Obj()
	podA := preemptionPod("a", "a-claim").Node(node2Name).Priority(10).PodGroupName("group").Obj()
	podG := preemptionPod("g", "g-claim").Node(nodeName).Priority(10).PodGroupName("group").Obj()
	podB := preemptionPod("b", "b-claim").Node(node2Name).Priority(20).Obj()
	podGroupClaims := []*resourceapi.ResourceClaim{
		preemptionClaim("preemptor-claim").Obj(),
		reserve(preemptionClaim("a-claim").Allocation(poolAllocation(node2Name, "gpu-0")).Obj(), podA),
		reserve(preemptionClaim("b-claim").Allocation(poolAllocation(node2Name, "gpu-1")).Obj(), podB),
		reserve(preemptionClaim("g-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), podG),
	}
	podGroupSlices := []*resourceapi.ResourceSlice{
		st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Obj(),
		st.MakeResourceSlice(node2Name, driver).Devices("gpu-0", "gpu-1").Obj(),
	}

	testcases := map[string]struct {
		slices       []*resourceapi.ResourceSlice
		claims       []*resourceapi.ResourceClaim
		pods         []*v1.Pod
		nominated    []*v1.Pod
		podGroups    []*schedulingv1beta1.PodGroup
		rejectBefore func(pods podNames) bool
		rejectAfter  func(pods podNames) bool
		evalOrder    []string
		// wantVictims maps each evaluated node name to its expected sorted
		// victim names; nil entry means SelectVictimsOnNode must fail on that node.
		wantVictims map[string][]string
		prepareNode string
		wantDevices []string
	}{
		"plugin-before-rejects": {
			// "Before" rejects whenever v3 coexists with v1 or v2, so v3 is
			// reprieved while v1 and v2 remain victims.
			slices: []*resourceapi.ResourceSlice{sliceWithDevices(2)},
			claims: defaultClaims(),
			pods:   defaultPods,
			rejectBefore: func(pods podNames) bool {
				return pods["v3"] && (pods["v1"] || pods["v2"])
			},
			evalOrder:   []string{nodeName},
			wantVictims: map[string][]string{nodeName: {"v1", "v2"}},
			prepareNode: nodeName,
			wantDevices: []string{nodeName + "/gpu-0"},
		},
		"plugin-after-rejects": {
			// DynamicResources succeeds when v2 or v1 is added back (leaving gpu-1
			// as the most recent Filter result), but "After" rejects both reprieves,
			// so the final state still removes both v1 and v2. Nominating holder
			// also exercises pass-1-only rejections.
			slices:    []*resourceapi.ResourceSlice{sliceWithDevices(2)},
			claims:    defaultClaims(),
			pods:      defaultPods,
			nominated: []*v1.Pod{holder},
			rejectAfter: func(pods podNames) bool {
				return pods["v1"] || pods["v2"]
			},
			evalOrder:   []string{nodeName},
			wantVictims: map[string][]string{nodeName: {"v1", "v2"}},
			prepareNode: nodeName,
			wantDevices: []string{nodeName + "/gpu-0"},
		},
		"last-add-back-stays": {
			// gpu-2 is already free and only v3 blocks scheduling, so both v2 and
			// v1 are reprieved and PrepareNomination must use the state after
			// adding v1 back.
			slices: []*resourceapi.ResourceSlice{sliceWithDevices(3)},
			claims: defaultClaims(),
			pods:   defaultPods,
			rejectBefore: func(pods podNames) bool {
				return pods["v3"]
			},
			evalOrder:   []string{nodeName},
			wantVictims: map[string][]string{nodeName: {"v3"}},
			prepareNode: nodeName,
			wantDevices: []string{nodeName + "/gpu-2"},
		},
		"podgroup-victim-on-two-nodes/node-first": {
			// PodGroup {a (on node2), g (on node)} is evicted as a unit, but
			// victims in a PodGroup release nothing (alpha). node's simulation,
			// which also removes a on node2, therefore frees nothing, and node2's
			// own simulation settles on b, regardless of execution order.
			slices:    podGroupSlices,
			claims:    podGroupClaims,
			pods:      []*v1.Pod{podA, podB, podG},
			podGroups: []*schedulingv1beta1.PodGroup{podGroup},
			rejectAfter: func(pods podNames) bool {
				return pods["b"]
			},
			evalOrder:   []string{nodeName, node2Name},
			wantVictims: map[string][]string{nodeName: nil, node2Name: {"b"}},
			prepareNode: node2Name,
			wantDevices: []string{node2Name + "/gpu-1"},
		},
		"podgroup-victim-on-two-nodes/node2-first": {
			slices:    podGroupSlices,
			claims:    podGroupClaims,
			pods:      []*v1.Pod{podA, podB, podG},
			podGroups: []*schedulingv1beta1.PodGroup{podGroup},
			rejectAfter: func(pods podNames) bool {
				return pods["b"]
			},
			evalOrder:   []string{node2Name, nodeName},
			wantVictims: map[string][]string{nodeName: nil, node2Name: {"b"}},
			prepareNode: node2Name,
			wantDevices: []string{node2Name + "/gpu-1"},
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			objs := []apiruntime.Object{preemptor}
			for _, s := range tc.slices {
				objs = append(objs, s)
			}
			for _, pod := range tc.pods {
				objs = append(objs, pod)
			}
			for _, pod := range tc.nominated {
				objs = append(objs, pod)
			}
			pl, draPlugin, snapshot, state := setupDefaultPreemption(tCtx, tc.claims, objs, preemptor, tc.pods, tc.nominated, tc.podGroups, tc.rejectBefore, tc.rejectAfter)

			for _, node := range tc.evalOrder {
				nodeInfo, err := snapshot.NodeInfos().Get(node)
				tCtx.ExpectNoError(err, "get node")
				possibleVictims, err := pl.Evaluator.GetVictimsOnNode(tCtx, nodeInfo)
				tCtx.ExpectNoError(err, "get possible victims")
				victims, _, status := pl.SelectVictimsOnNode(tCtx, state.Clone(), preemptor, nodeInfo.Snapshot(), possibleVictims, nil)
				want := tc.wantVictims[node]
				if want == nil {
					require.False(tCtx, status.IsSuccess(), "select victims on %s: %v", node, status)
					continue
				}
				require.True(tCtx, status.IsSuccess(), "select victims on %s: %v", node, status)
				var names []string
				for _, pod := range victims {
					names = append(names, pod.Name)
				}
				slices.Sort(names)
				require.Equal(tCtx, want, names, "victims on %s", node)
			}

			hold, _ := draPlugin.PrepareNomination(tCtx, state, preemptor, tc.prepareNode).(*podHold)
			require.NotNil(tCtx, hold, "hold")
			require.Equal(tCtx, tc.prepareNode, hold.nodeName, "node of the hold")
			var devices []string
			for _, held := range hold.heldClaims {
				for _, result := range held.allocatedDevices.Results {
					devices = append(devices, result.Pool+"/"+result.Device)
				}
			}
			require.Equal(tCtx, tc.wantDevices, devices, "devices of the final state")
		})
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
	require.Empty(tCtx, testCtx.p.AddNomination(tCtx, pod, nodeName, &podHold{nodeName: nodeName}).Events, "AddNomination")
	require.Empty(tCtx, testCtx.p.RemoveNomination(tCtx, pod, nodeName).Events, "RemoveNomination")
}
