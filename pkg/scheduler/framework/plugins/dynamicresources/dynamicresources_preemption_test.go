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
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestRemovePod(t *testing.T) {
	tCtx := ktesting.Init(t)

	testCases := []struct {
		name   string
		setup  func(t *testing.T) (podInfo fwk.PodInfo, claims map[string]*resourceapi.ResourceClaim, initialState *stateData)
		verify func(t *testing.T, state *stateData, status *fwk.Status)
	}{
		{
			name: "allocated-unused-claim-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-a").Namespace("default").UID("pod-a-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if !state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be released")
				}
			},
		},
		{
			name: "single-user-claim-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-populated"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{Resource: "pods", UID: "pod-populated-uid"},
				}

				pod := st.MakePod().Name("pod-populated").Namespace("default").UID("pod-populated-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if !state.simulatedReleasedClaims.Has("claim-populated") {
					t.Errorf("expected claim-populated to be released")
				}
			},
		},
		{
			name: "shared-claim-single-user-out-not-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "shared-claim"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{Resource: "pods", UID: "pod-a-uid"},
					{Resource: "pods", UID: "pod-b-uid"},
				}

				pod := st.MakePod().Name("pod-shared-a").Namespace("default").UID("pod-a-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("shared-claim") {
					t.Errorf("expected shared-claim NOT to be released")
				}
				if users, ok := state.simulatedRemovedPodsForClaim["shared-claim"]; !ok || !users.Has("pod-a-uid") {
					t.Errorf("expected pod-a-uid to be recorded as removed")
				}
			},
		},
		{
			name: "shared-claim-all-users-out-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "shared-claim"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{Resource: "pods", UID: "pod-a-uid"},
					{Resource: "pods", UID: "pod-b-uid"},
				}

				pod := st.MakePod().Name("pod-shared-b").Namespace("default").UID("pod-b-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: map[types.UID]sets.Set[types.UID]{"shared-claim": sets.New[types.UID]("pod-a-uid")},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if !state.simulatedReleasedClaims.Has("shared-claim") {
					t.Errorf("expected shared-claim to be released")
				}
			},
		},
		{
			name: "pod-group-claim-not-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "pod-group-claim"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{APIGroup: "scheduling.k8s.io", Resource: "podgroups", Name: "my-group", UID: "group-uid"},
				}

				pod := st.MakePod().Name("pod-group-a").Namespace("default").UID("pod-group-a-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("pod-group-claim") {
					t.Errorf("expected pod-group-claim NOT to be released because it is held by a PodGroup")
				}
				if users, ok := state.simulatedRemovedPodsForClaim["pod-group-claim"]; !ok || !users.Has("pod-group-a-uid") {
					t.Errorf("expected pod-group-a-uid to be recorded as removed")
				}
			},
		},
		{
			name: "shared-claim-count-matches-but-uids-differ-not-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "spurious-claim"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{Resource: "pods", UID: "never-simulated-pod-uid"},
				}

				pod := st.MakePod().Name("pod-spurious").Namespace("default").UID("pod-spurious-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("spurious-claim") {
					t.Errorf("expected spurious-claim NOT to be released because the simulated UIDs do not match ReservedFor")
				}
				if users, ok := state.simulatedRemovedPodsForClaim["spurious-claim"]; !ok || !users.Has("pod-spurious-uid") {
					t.Errorf("expected pod-spurious-uid to be recorded as removed")
				}
			},
		},
		{
			name: "duplicate-claim-name-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-dup-claim").Namespace("default").UID("pod-dup-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{
					{Name: "my-claim-1", ResourceClaimName: &claimName},
					{Name: "my-claim-2", ResourceClaimName: &claimName},
				}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
					allocatedState: &structured.AllocatedState{
						AllocatedDevices:         sets.New[structured.DeviceID](structured.MakeDeviceID("test-driver", "pool-1", "device-1")),
						AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
						AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
					},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if users, ok := state.simulatedRemovedPodsForClaim["claim-1"]; !ok || users.Len() != 1 {
					t.Errorf("expected pod-dup-uid to be recorded exactly once, got %v users", users.Len())
				}
				if !state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be released")
				}
				if state.allocatedState.AllocatedDevices.Len() != 0 {
					t.Errorf("expected devices to be removed strictly once")
				}
			},
		},
		{
			name: "missing-claim-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "missing-claim"
				pod := st.MakePod().Name("pod-missing-claim").Namespace("default").UID("pod-missing-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if len(state.simulatedRemovedPodsForClaim) != 0 {
					t.Errorf("expected no claims to be recorded as removed for missing claims, got %v", state.simulatedRemovedPodsForClaim)
				}
			},
		},
		{
			name: "nil-resource-claim-name-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				templateName := "my-template"
				pod := st.MakePod().Name("pod-nil-claim").Namespace("default").UID("pod-nil-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimTemplateName: &templateName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if len(state.simulatedRemovedPodsForClaim) != 0 {
					t.Errorf("expected no removal tracking, got %v", state.simulatedRemovedPodsForClaim)
				}
			},
		},
		{
			name: "direct-claim-released",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-direct").Namespace("default").UID("pod-direct-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if !state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be released")
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podInfo, claims, initialState := tc.setup(t)
			mockManager := &mockPreemptionDRAManager{claims: claims}
			pl := &DynamicResources{
				draManager: mockManager,
				fts: feature.Features{
					EnableDRAPreemption: true,
				},
			}

			cycleState := framework.NewCycleState()
			cycleState.Write(stateKey, initialState)

			nodeInfo := framework.NewNodeInfo()

			status := pl.RemovePod(tCtx, cycleState, nil, podInfo, nodeInfo)

			tc.verify(t, initialState, status)
		})
	}
}

func TestAddPod(t *testing.T) {
	tCtx := ktesting.Init(t)

	testCases := []struct {
		name   string
		setup  func(t *testing.T) (podInfo fwk.PodInfo, claims map[string]*resourceapi.ResourceClaim, initialState *stateData)
		verify func(t *testing.T, state *stateData, status *fwk.Status)
	}{
		{
			name: "released-claim-readded",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-a").Namespace("default").UID("pod-a-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](claim.UID),
					allocatedState: &structured.AllocatedState{
						AllocatedDevices:         sets.New[structured.DeviceID](),
						AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
						AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
					},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be removed from released claims")
				}
				if state.allocatedState.AllocatedDevices.Len() == 0 {
					t.Errorf("expected devices to be restored to allocatedState")
				}
			},
		},
		{
			name: "shared-claim-clear-counter",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-a").Namespace("default").UID("pod-a-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: map[types.UID]sets.Set[types.UID]{claim.UID: sets.New[types.UID]("pod-a-uid")},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if users, ok := state.simulatedRemovedPodsForClaim["claim-1"]; ok && users.Has("pod-a-uid") {
					t.Errorf("expected pod-a-uid to be cleared from simulatedRemovedPodsForClaim")
				}
			},
		},
		{
			name: "duplicate-claim-name-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-dup-claim").Namespace("default").UID("pod-dup-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{
					{Name: "my-claim-1", ResourceClaimName: &claimName},
					{Name: "my-claim-2", ResourceClaimName: &claimName},
				}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](claim.UID),
					simulatedRemovedPodsForClaim: map[types.UID]sets.Set[types.UID]{claim.UID: sets.New[types.UID]("pod-dup-uid")},
					allocatedState: &structured.AllocatedState{
						AllocatedDevices:         sets.New[structured.DeviceID](),
						AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
						AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
					},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be removed from released claims strictly once")
				}
				if users, ok := state.simulatedRemovedPodsForClaim["claim-1"]; ok && users.Has("pod-dup-uid") {
					t.Errorf("expected pod-dup-uid to be completely cleared exactly once")
				}
				if state.allocatedState.AllocatedDevices.Len() != 1 {
					t.Errorf("expected devices to be strictly restored only once")
				}
			},
		},
		{
			name: "missing-claim-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "missing-claim"
				pod := st.MakePod().Name("pod-missing-claim").Namespace("default").UID("pod-missing-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims:      sets.New[types.UID](),
					simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
					allocatedState: &structured.AllocatedState{
						AllocatedDevices:         sets.New[structured.DeviceID](),
						AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
						AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
					},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if len(state.simulatedReleasedClaims) != 0 {
					t.Errorf("expected no released claims")
				}
				if len(state.simulatedRemovedPodsForClaim) != 0 {
					t.Errorf("expected no removed pods tracking")
				}
			},
		},
		{
			name: "nil-resource-claim-name-ignored",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				templateName := "my-template"
				pod := st.MakePod().Name("pod-nil-claim").Namespace("default").UID("pod-nil-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimTemplateName: &templateName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](),
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
			},
		},
		{
			name: "direct-claim-readded",
			setup: func(t *testing.T) (fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData) {
				claimName := "claim-1"
				claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
				claim.UID = types.UID(claimName)
				claim.Status.Allocation = &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Driver: "test-driver", Pool: "pool-1", Device: "device-1"},
						},
					},
				}

				pod := st.MakePod().Name("pod-direct").Namespace("default").UID("pod-direct-uid").Obj()
				pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
				podInfo, _ := framework.NewPodInfo(pod)

				initialState := &stateData{
					simulatedReleasedClaims: sets.New[types.UID](claim.UID),
					allocatedState: &structured.AllocatedState{
						AllocatedDevices:         sets.New[structured.DeviceID](),
						AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
						AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
					},
				}
				return podInfo, map[string]*resourceapi.ResourceClaim{claimName: claim}, initialState
			},
			verify: func(t *testing.T, state *stateData, status *fwk.Status) {
				if status != nil {
					t.Errorf("expected no error, got %v", status)
				}
				if state.simulatedReleasedClaims.Has("claim-1") {
					t.Errorf("expected claim-1 to be removed from released claims")
				}
				if state.allocatedState.AllocatedDevices.Len() == 0 {
					t.Errorf("expected devices to be restored to allocatedState")
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podInfo, claims, initialState := tc.setup(t)
			mockManager := &mockPreemptionDRAManager{claims: claims}
			pl := &DynamicResources{
				draManager: mockManager,
				fts: feature.Features{
					EnableDRAPreemption: true,
				},
			}

			cycleState := framework.NewCycleState()
			cycleState.Write(stateKey, initialState)

			nodeInfo := framework.NewNodeInfo()

			status := pl.AddPod(tCtx, cycleState, nil, podInfo, nodeInfo)

			tc.verify(t, initialState, status)
		})
	}
}

func TestSharedClaimsPreemption(t *testing.T) {
	// Create a mock SharedDRAManager that returns a shared claim.
	claimName := "claim-1"
	claim := st.MakeResourceClaim().Namespace("default").Name(claimName).Obj()
	claim.UID = types.UID(claimName)
	claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
		{Resource: "pods", UID: "victim-1-uid"},
		{Resource: "pods", UID: "victim-2-uid"},
	}

	mockManager := &mockPreemptionDRAManager{
		claims: map[string]*resourceapi.ResourceClaim{
			claimName: claim,
		},
	}

	pl := &DynamicResources{
		draManager: mockManager,
		fts: feature.Features{
			EnableDRAPreemption: true,
		},
	}

	state := &stateData{
		simulatedReleasedClaims:      sets.New[types.UID](),
		simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
	}
	cycleState := framework.NewCycleState()
	cycleState.Write(stateKey, state)

	// Victim 1
	pod1 := st.MakePod().Name("victim-1").Namespace("default").UID("victim-1-uid").Obj()
	pod1.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              "my-claim",
			ResourceClaimName: &claimName,
		},
	}
	podInfo1, err := framework.NewPodInfo(pod1)
	if err != nil {
		t.Fatalf("NewPodInfo failed: %v", err)
	}

	// Victim 2
	pod2 := st.MakePod().Name("victim-2").Namespace("default").UID("victim-2-uid").Obj()
	pod2.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              "my-claim",
			ResourceClaimName: &claimName,
		},
	}
	podInfo2, err := framework.NewPodInfo(pod2)
	if err != nil {
		t.Fatalf("NewPodInfo failed: %v", err)
	}

	nodeInfo := framework.NewNodeInfo()

	// 1. Remove Victim 1. Claim should NOT be released.
	status := pl.RemovePod(context.Background(), cycleState, nil, podInfo1, nodeInfo)
	if status != nil {
		t.Errorf("unexpected status %v", status)
	}
	if state.simulatedReleasedClaims.Has(claim.UID) {
		t.Errorf("expected claim-1 NOT to be marked as released after removing only victim-1")
	}

	// 2. Remove Victim 2. Claim SHOULD be released!
	status = pl.RemovePod(context.Background(), cycleState, nil, podInfo2, nodeInfo)
	if status != nil {
		t.Errorf("unexpected status %v", status)
	}
	if !state.simulatedReleasedClaims.Has(claim.UID) {
		t.Errorf("expected claim-1 to be marked as released after removing both victims")
	}

	// 3. Add back Victim 1 (Backtrack step).
	// The claim MUST transition back to UN-released, but Victim 2 should remain registered!
	status = pl.AddPod(context.Background(), cycleState, nil, podInfo1, nodeInfo)
	if status != nil {
		t.Errorf("unexpected status %v", status)
	}
	if state.simulatedReleasedClaims.Has(claim.UID) {
		t.Errorf("expected claim-1 to be removed from released claims after restoring victim-1")
	}
	if users, ok := state.simulatedRemovedPodsForClaim[claim.UID]; !ok || !users.Has("victim-2-uid") {
		t.Errorf("expected victim-2-uid to remain registered in simulatedRemovedPodsForClaim")
	}
	if users := state.simulatedRemovedPodsForClaim[claim.UID]; users.Has("victim-1-uid") {
		t.Errorf("expected victim-1-uid to be cleared from simulatedRemovedPodsForClaim")
	}

	// 4. Add back Victim 2 (Complete restoration step).
	// State should return back to fully clean.
	status = pl.AddPod(context.Background(), cycleState, nil, podInfo2, nodeInfo)
	if status != nil {
		t.Errorf("unexpected status %v", status)
	}
	if users, ok := state.simulatedRemovedPodsForClaim[claim.UID]; ok && users.Len() != 0 {
		t.Errorf("expected simulatedRemovedPodsForClaim to be empty after restoring all victims, got %v", users)
	}
}

type mockPreemptionDRAManager struct {
	fwk.SharedDRAManager
	claims map[string]*resourceapi.ResourceClaim
}

func (m *mockPreemptionDRAManager) ResourceClaims() fwk.ResourceClaimTracker {
	return &mockPreemptionClaimTracker{claims: m.claims}
}

type mockPreemptionClaimTracker struct {
	fwk.ResourceClaimTracker
	claims map[string]*resourceapi.ResourceClaim
}

func (t *mockPreemptionClaimTracker) Get(namespace, name string) (*resourceapi.ResourceClaim, error) {
	if claim, ok := t.claims[name]; ok {
		return claim, nil
	}
	return nil, apierrors.NewNotFound(resourceapi.Resource("resourceclaims"), name)
}

func TestPreemptionEndToEnd(t *testing.T) {
	tCtx := ktesting.Init(t)

	// Create test-specific class
	testClassName := "test-class"
	testClass := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: testClassName},
	}

	// Create a slice with 3 devices on "test-driver"
	testSlice := st.MakeResourceSlice("node-1", "test-driver").
		Device("dev-1").
		Device("dev-2").
		Device("dev-3").
		Obj()

	// Preemptor claims (unallocated)
	preemptorClaim1 := st.MakeResourceClaim().Namespace("default").Name("preemptor-claim-1").Request(testClassName).Obj()
	preemptorClaim1.UID = "preemptor-claim-1-uid"
	preemptorClaim2 := st.MakeResourceClaim().Namespace("default").Name("preemptor-claim-2").Request(testClassName).Obj()
	preemptorClaim2.UID = "preemptor-claim-2-uid"

	// Victim claims (allocated)
	makeAlloc := func(devName string) *resourceapi.AllocationResult {
		return &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{
					{
						Driver:  "test-driver",
						Pool:    "node-1",
						Device:  devName,
						Request: "req",
					},
				},
			},
		}
	}

	victimClaim1 := preemptorClaim1.DeepCopy()
	victimClaim1.Name = "victim-claim-1"
	victimClaim1.UID = "victim-claim-1-uid"
	victimClaim1.Status.Allocation = makeAlloc("dev-1")

	victimSharedClaim := preemptorClaim1.DeepCopy()
	victimSharedClaim.Name = "victim-shared-claim"
	victimSharedClaim.UID = "victim-shared-claim-uid"
	victimSharedClaim.Status.Allocation = makeAlloc("dev-2")
	victimSharedClaim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
		{Resource: "pods", UID: "victim-shared-a-uid"},
		{Resource: "pods", UID: "victim-shared-b-uid"},
	}

	victimClaim2 := preemptorClaim1.DeepCopy()
	victimClaim2.Name = "victim-claim-2"
	victimClaim2.UID = "victim-claim-2-uid"
	victimClaim2.Status.Allocation = makeAlloc("dev-3")

	mockManager := &mockPreemptionDRAManagerWithSlices{
		claims: map[string]*resourceapi.ResourceClaim{
			victimClaim1.Name:      victimClaim1,
			victimSharedClaim.Name: victimSharedClaim,
			victimClaim2.Name:      victimClaim2,
		},
	}

	makeVictim := func(name, uid string, claimNames ...string) fwk.PodInfo {
		pod := st.MakePod().Name(name).Namespace("default").UID(uid).Obj()
		for _, cn := range claimNames {
			claimNameCopy := cn
			pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims, v1.PodResourceClaim{
				Name:              "claim-" + cn,
				ResourceClaimName: &claimNameCopy,
			})
		}
		pi, _ := framework.NewPodInfo(pod)
		return pi
	}

	victim1 := makeVictim("victim-1", "victim-1-uid", victimClaim1.Name)
	victimSharedA := makeVictim("victim-a", "victim-shared-a-uid", victimSharedClaim.Name)
	victimSharedB := makeVictim("victim-b", "victim-shared-b-uid", victimSharedClaim.Name)
	victim2 := makeVictim("victim-2", "victim-2-uid", victimClaim2.Name)
	victimMulti := makeVictim("victim-multi", "victim-multi-uid", victimClaim1.Name, victimClaim2.Name)

	type op struct {
		action string
		pod    fwk.PodInfo
	}

	testCases := []struct {
		name             string
		preemptorClaims  []*resourceapi.ResourceClaim
		ops              []op
		wantFilterStatus *fwk.Status
	}{
		{
			name:             "baseline-exhaustion-no-preemption",
			preemptorClaims:  []*resourceapi.ResourceClaim{preemptorClaim1},
			ops:              []op{},
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"),
		},
		{
			name:            "successful-preemption-single-victim",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1},
			ops: []op{
				{"remove", victim1}, // frees dev-1
			},
			wantFilterStatus: nil,
		},
		{
			name:            "partial-preemption-shared-claim-fails",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1},
			ops: []op{
				{"remove", victimSharedA}, // dev-2 not freed yet
			},
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"),
		},
		{
			name:            "full-preemption-shared-claim-succeeds",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1},
			ops: []op{
				{"remove", victimSharedA},
				{"remove", victimSharedB}, // frees dev-2
			},
			wantFilterStatus: nil,
		},
		{
			name:            "multi-claim-preemption",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1, preemptorClaim2},
			ops: []op{
				{"remove", victim1}, // frees dev-1
				{"remove", victim2}, // frees dev-3
			},
			wantFilterStatus: nil,
		},
		{
			name:            "preemption-rollback",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1},
			ops: []op{
				{"remove", victim1}, // frees dev-1
				{"add", victim1},    // takes back dev-1
			},
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, "cannot allocate all claims"),
		},
		{
			name:            "single-victim-multiple-claims",
			preemptorClaims: []*resourceapi.ResourceClaim{preemptorClaim1, preemptorClaim2},
			ops: []op{
				{"remove", victimMulti}, // frees dev-1 and dev-3 simultaneously
			},
			wantFilterStatus: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			feats := feature.Features{
				EnableDRAPreemption: true,
			}

			testCtx := setup(tCtx, nil, nil, nil, []*resourceapi.DeviceClass{testClass}, nil, []apiruntime.Object{testSlice}, feats, false, nil)
			pl := testCtx.p
			pl.fts.EnableDRAPreemption = true

			mockManager.SharedDRAManager = testCtx.draManager
			pl.draManager = mockManager
			pl.enabled = true

			state := &stateData{
				claims:                       newClaimStore(tc.preemptorClaims, nil, nil),
				informationsForClaim:         make([]informationForClaim, len(tc.preemptorClaims)),
				nodeAllocations:              make(map[string]nodeAllocation),
				simulatedReleasedClaims:      sets.New[types.UID](),
				simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
				allocatedState: &structured.AllocatedState{
					AllocatedDevices:         sets.New[structured.DeviceID](),
					AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
					AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
				},
			}

			// Pre-allocate the victim claims so they exhaust the devices (dev-1, dev-2, dev-3)
			pl.addClaimToAllocatedState(state.allocatedState, victimClaim1)
			pl.addClaimToAllocatedState(state.allocatedState, victimSharedClaim)
			pl.addClaimToAllocatedState(state.allocatedState, victimClaim2)

			testSlices, _ := testCtx.draManager.ResourceSlices().ListWithDeviceTaintRules()
			allocator, _ := structured.NewAllocator(tCtx, AllocatorFeatures(feats), *state.allocatedState, testCtx.draManager.DeviceClasses(), testSlices, nil)
			state.allocator = allocator
			state.resourceSlices = testSlices

			cycleState := framework.NewCycleState()
			cycleState.Write(stateKey, state)

			node := st.MakeNode().Name("node-1").Obj()
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(node)

			preemptorPod := st.MakePod().Name("preemptor").Namespace("default").Obj()

			for _, op := range tc.ops {
				if op.action == "remove" {
					pl.RemovePod(tCtx, cycleState, preemptorPod, op.pod, nodeInfo)
				} else if op.action == "add" {
					pl.AddPod(tCtx, cycleState, preemptorPod, op.pod, nodeInfo)
				}
			}

			status := pl.Filter(tCtx, cycleState, preemptorPod, nodeInfo)

			if status.Code() != tc.wantFilterStatus.Code() {
				t.Errorf("expected status code %v, got %v", tc.wantFilterStatus.Code(), status.Code())
			}
			if tc.wantFilterStatus.Code() == fwk.Error || tc.wantFilterStatus.Code() == fwk.UnschedulableAndUnresolvable {
				if status.Message() != tc.wantFilterStatus.Message() {
					t.Errorf("expected status message %q, got %q", tc.wantFilterStatus.Message(), status.Message())
				}
			}
		})
	}
}

type mockPreemptionDRAManagerWithSlices struct {
	fwk.SharedDRAManager
	claims map[string]*resourceapi.ResourceClaim
}

func (m *mockPreemptionDRAManagerWithSlices) ResourceClaims() fwk.ResourceClaimTracker {
	return &mockPreemptionClaimTracker{claims: m.claims}
}

func TestPreemptionSimulationSymmetry(t *testing.T) {
	tCtx := ktesting.Init(t)

	type op struct {
		action string // "remove", "add"
		podKey string // e.g. "pod-a", "pod-b"
	}

	testCases := []struct {
		name   string
		setup  func(t *testing.T) (pods map[string]fwk.PodInfo, claims map[string]*resourceapi.ResourceClaim, initialState *stateData, feats feature.Features)
		ops    []op
		verify func(t *testing.T, state *stateData)
	}{
		{
			name: "single-pod-remove-add",
			ops: []op{
				{"remove", "pod-a"},
				{"add", "pod-a"},
			},
			setup: func(t *testing.T) (map[string]fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData, feature.Features) {
				claim := makeTestClaim("claim-1", "test-driver", "pool-1", "device-1", nil, nil, "pod-a-uid", "pod-b-uid")
				pod := makeTestPodInfo("pod-a", "pod-a-uid", "claim-1")
				return map[string]fwk.PodInfo{"pod-a": pod}, map[string]*resourceapi.ResourceClaim{"claim-1": claim}, newEmptyStateData(), feature.Features{EnableDRAPreemption: true}
			},
			verify: verifyCleanSymmetryState("claim-1", structured.MakeDeviceID("test-driver", "pool-1", "device-1")),
		},
		{
			name: "shared-claims-remove-A-remove-B-add-A-add-B",
			ops: []op{
				{"remove", "pod-a"},
				{"remove", "pod-b"},
				{"add", "pod-a"},
				{"add", "pod-b"},
			},
			setup: func(t *testing.T) (map[string]fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData, feature.Features) {
				claim := makeTestClaim("claim-1", "test-driver", "pool-1", "device-1", nil, nil, "pod-a-uid", "pod-b-uid")
				podA := makeTestPodInfo("pod-a", "pod-a-uid", "claim-1")
				podB := makeTestPodInfo("pod-b", "pod-b-uid", "claim-1")
				return map[string]fwk.PodInfo{"pod-a": podA, "pod-b": podB}, map[string]*resourceapi.ResourceClaim{"claim-1": claim}, newEmptyStateData(), feature.Features{EnableDRAPreemption: true}
			},
			verify: verifyCleanSymmetryState("claim-1", structured.MakeDeviceID("test-driver", "pool-1", "device-1")),
		},
		{
			name: "shared-claims-remove-A-remove-B-add-B-add-A",
			ops: []op{
				{"remove", "pod-a"},
				{"remove", "pod-b"},
				{"add", "pod-b"},
				{"add", "pod-a"},
			},
			setup: func(t *testing.T) (map[string]fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData, feature.Features) {
				claim := makeTestClaim("claim-1", "test-driver", "pool-1", "device-1", nil, nil, "pod-a-uid", "pod-b-uid")
				podA := makeTestPodInfo("pod-a", "pod-a-uid", "claim-1")
				podB := makeTestPodInfo("pod-b", "pod-b-uid", "claim-1")
				return map[string]fwk.PodInfo{"pod-a": podA, "pod-b": podB}, map[string]*resourceapi.ResourceClaim{"claim-1": claim}, newEmptyStateData(), feature.Features{EnableDRAPreemption: true}
			},
			verify: verifyCleanSymmetryState("claim-1", structured.MakeDeviceID("test-driver", "pool-1", "device-1")),
		},
		{
			name: "consumable-capacity-remove-add",
			ops: []op{
				{"remove", "pod-a"},
				{"add", "pod-a"},
			},
			setup: func(t *testing.T) (map[string]fwk.PodInfo, map[string]*resourceapi.ResourceClaim, *stateData, feature.Features) {
				claim := makeTestClaim("claim-consumable", "test-driver", "pool-1", "device-1", ptr.To(types.UID("share-1")), map[resourceapi.QualifiedName]resource.Quantity{
					resourceapi.QualifiedName("cpu"): resource.MustParse("5"),
				}, "pod-a-uid")
				pod := makeTestPodInfo("pod-a", "pod-a-uid", "claim-consumable")
				initialState := newEmptyStateData()
				feats := feature.Features{
					EnableDRAPreemption:         true,
					EnableDRAConsumableCapacity: true,
				}
				return map[string]fwk.PodInfo{"pod-a": pod}, map[string]*resourceapi.ResourceClaim{"claim-consumable": claim}, initialState, feats
			},
			verify: verifyConsumableSymmetryState("claim-consumable", structured.MakeSharedDeviceID(structured.MakeDeviceID("test-driver", "pool-1", "device-1"), ptr.To(types.UID("share-1")))),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			pods, claims, state, feats := tc.setup(t)
			mockManager := &mockPreemptionDRAManager{
				claims: claims,
			}
			pl := &DynamicResources{
				draManager: mockManager,
				fts:        feats,
			}

			// Pre-populate initial allocatedState
			for _, claim := range claims {
				pl.addClaimToAllocatedState(state.allocatedState, claim)
			}

			cycleState := framework.NewCycleState()
			cycleState.Write(stateKey, state)

			nodeInfo := framework.NewNodeInfo()

			// Run Operations
			for _, operation := range tc.ops {
				pod := pods[operation.podKey]
				if operation.action == "remove" {
					pl.RemovePod(tCtx, cycleState, nil, pod, nodeInfo)
				} else if operation.action == "add" {
					pl.AddPod(tCtx, cycleState, nil, pod, nodeInfo)
				}
			}

			// Verify Invariants after simulation completes
			tc.verify(t, state)
		})
	}
}

// Declarative Test Factory Helpers

func makeTestClaim(name string, driver, pool, device string, shareID *types.UID, capacity map[resourceapi.QualifiedName]resource.Quantity, reservingUIDs ...string) *resourceapi.ResourceClaim {
	claim := st.MakeResourceClaim().Namespace("default").Name(name).Obj()
	claim.UID = types.UID(name)
	for _, rUID := range reservingUIDs {
		claim.Status.ReservedFor = append(claim.Status.ReservedFor, resourceapi.ResourceClaimConsumerReference{
			Resource: "pods",
			UID:      types.UID(rUID),
		})
	}
	if driver != "" {
		result := resourceapi.DeviceRequestAllocationResult{
			Driver: driver,
			Pool:   pool,
			Device: device,
		}
		if shareID != nil {
			result.ShareID = shareID
			if len(capacity) > 0 {
				result.ConsumedCapacity = capacity
			}
		}
		claim.Status.Allocation = &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{result},
			},
		}
	}
	return claim
}

func makeTestPodInfo(name, uid, claimName string) fwk.PodInfo {
	pod := st.MakePod().Name(name).Namespace("default").UID(uid).Obj()
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "my-claim", ResourceClaimName: &claimName}}
	podInfo, _ := framework.NewPodInfo(pod)
	return podInfo
}

func newEmptyStateData() *stateData {
	return &stateData{
		simulatedReleasedClaims:      sets.New[types.UID](),
		simulatedRemovedPodsForClaim: make(map[types.UID]sets.Set[types.UID]),
		allocatedState: &structured.AllocatedState{
			AllocatedDevices:         sets.New[structured.DeviceID](),
			AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
			AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
		},
	}
}

// Reusable Assertion Closures

func verifyCleanSymmetryState(claimUID string, expectedDeviceID structured.DeviceID) func(t *testing.T, state *stateData) {
	return func(t *testing.T, state *stateData) {
		if state.simulatedReleasedClaims.Len() != 0 {
			t.Errorf("expected no released claims, got %d", state.simulatedReleasedClaims.Len())
		}
		if users, ok := state.simulatedRemovedPodsForClaim[types.UID(claimUID)]; ok && users.Len() != 0 {
			t.Errorf("expected no removed users, got %v", users.Len())
		}
		if state.allocatedState.AllocatedDevices.Len() != 1 || !state.allocatedState.AllocatedDevices.Has(expectedDeviceID) {
			t.Errorf("expected devices to be restored to initial state")
		}
	}
}

func verifyConsumableSymmetryState(claimUID string, expectedSharedDeviceID structured.SharedDeviceID) func(t *testing.T, state *stateData) {
	return func(t *testing.T, state *stateData) {
		if state.simulatedReleasedClaims.Len() != 0 {
			t.Errorf("expected no released claims, got %d", state.simulatedReleasedClaims.Len())
		}
		if users, ok := state.simulatedRemovedPodsForClaim[types.UID(claimUID)]; ok && users.Len() != 0 {
			t.Errorf("expected no removed users, got %v", users.Len())
		}
		if state.allocatedState.AllocatedDevices.Len() != 0 {
			t.Errorf("expected no dedicated devices, got %d", state.allocatedState.AllocatedDevices.Len())
		}
		if state.allocatedState.AllocatedSharedDeviceIDs.Len() != 1 || !state.allocatedState.AllocatedSharedDeviceIDs.Has(expectedSharedDeviceID) {
			t.Errorf("expected shared devices to be restored to initial state")
		}
		if len(state.allocatedState.AggregatedCapacity) != 1 {
			t.Errorf("expected consumable capacity to be restored to initial state")
		}
	}
}
