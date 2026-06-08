/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/utils/ptr"

	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	hugePages2Mi = v1.ResourceName(v1.ResourceHugePagesPrefix + "2Mi")
	hugePages1Gi = v1.ResourceName(v1.ResourceHugePagesPrefix + "1Gi")
)

// mockDRAManager is a mock implementation of fwk.SharedDRAManager.
type mockDRAManager struct {
	fwk.SharedDRAManager
	claims         []*resourceapi.ResourceClaim
	resourceSlices []*resourceapi.ResourceSlice
}

func (m *mockDRAManager) ResourceClaims() fwk.ResourceClaimTracker {
	return m
}

func (m *mockDRAManager) Get(namespace, name string) (*resourceapi.ResourceClaim, error) {
	for _, claim := range m.claims {
		if claim.Namespace == namespace && claim.Name == name {
			return claim, nil
		}
	}
	return nil, fmt.Errorf("claim %s/%s not found", namespace, name)
}

func (m *mockDRAManager) AssumeClaimAfterAPICall(claim *resourceapi.ResourceClaim) error {
	return nil
}

func (m *mockDRAManager) AssumedClaimRestore(namespace, name string) {}

func (m *mockDRAManager) GetPendingAllocation(uid types.UID) *resourceapi.AllocationResult {
	return nil
}

func (m *mockDRAManager) SignalClaimPendingAllocation(uid types.UID, claim *resourceapi.ResourceClaim) error {
	return nil
}

func (m *mockDRAManager) MaybeRemoveClaimPendingAllocation(_ types.UID, _ bool) (deleted bool) {
	return false
}

func (m *mockDRAManager) List() ([]*resourceapi.ResourceClaim, error) {
	return nil, nil
}

func (m *mockDRAManager) GatherAllocatedState() (*structured.AllocatedState, error) {
	return nil, nil
}

func (m *mockDRAManager) ListAllAllocatedDevices() (sets.Set[structured.DeviceID], error) {
	return nil, nil
}

func (m *mockDRAManager) ResourceSlices() fwk.ResourceSliceLister {
	return m
}

func (m *mockDRAManager) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return m.resourceSlices, nil
}

func TestValidateNodeAllocatableDRAClaimSharing(t *testing.T) {
	claimName := "node-allocatable-claim"
	claimNameSpace := "test-ns"
	claim1Key := types.NamespacedName{Namespace: claimNameSpace, Name: claimName}

	tests := []struct {
		name       string
		pod        *v1.Pod
		claim      *resourceapi.ResourceClaim
		nodeInfo   *framework.NodeInfo
		wantStatus *fwk.Status
	}{
		{
			name:  "empty claim",
			pod:   st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").Obj(),
			claim: nil,
			nodeInfo: &framework.NodeInfo{
				NodeAllocatableDRAClaimStates: map[types.NamespacedName]*fwk.NodeAllocatableDRAClaimState{},
			},
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "claim not in node info",
			pod:  st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("pod-uid").Obj(),
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      claimName,
					Namespace: claimNameSpace,
					UID:       "claim-uid",
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NodeAllocatableDRAClaimStates = map[types.NamespacedName]*fwk.NodeAllocatableDRAClaimState{}
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "claim shared, current pod not in consumers",
			pod:  st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("pod-uid").Obj(),
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      claimName,
					Namespace: claimNameSpace,
					UID:       "claim-uid",
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NodeAllocatableDRAClaimStates = map[types.NamespacedName]*fwk.NodeAllocatableDRAClaimState{
					claim1Key: {ConsumerPods: sets.New[types.UID]("other-pod-uid", "another-pod-uid")},
				}
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node allocatable resource claim node-allocatable-claim shared by multiple pods"),
		},
		{
			name: "claim shared, current pod in consumers",
			pod:  st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("pod-uid").Obj(),
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      claimName,
					Namespace: claimNameSpace,
					UID:       "claim-uid",
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NodeAllocatableDRAClaimStates = map[types.NamespacedName]*fwk.NodeAllocatableDRAClaimState{
					claim1Key: {ConsumerPods: sets.New[types.UID]("pod-uid", "another-pod-uid")},
				}
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node allocatable resource claim node-allocatable-claim shared by multiple pods"),
		},
		{
			name: "claim only used by other pod",
			pod:  st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("pod-uid").Obj(),
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      claimName,
					Namespace: claimNameSpace,
					UID:       "claim-uid",
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NodeAllocatableDRAClaimStates = map[types.NamespacedName]*fwk.NodeAllocatableDRAClaimState{
					claim1Key: {ConsumerPods: sets.New[types.UID]("other-pod-uid")},
				}
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node allocatable resource claim node-allocatable-claim is already used by another pod"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DynamicResources{}
			gotStatus := pl.validateNodeAllocatableDRAClaimSharing(tt.pod, tt.nodeInfo, tt.claim)
			if diff := cmp.Diff(tt.wantStatus, gotStatus); diff != "" {
				t.Errorf("validateDRAClaimShareState() returned diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBuildNodeAllocatableDRAInfo(t *testing.T) {
	cpuDevicePerInstance := resourceapi.Device{
		Name: "cpu0",
		NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
			v1.ResourceCPU: {
				AllocationMultiplier: ptr.To(resource.MustParse("1")),
			},
		},
	}

	cpuDeviceCapacity := resourceapi.Device{
		Name: "cpu0",
		NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
			v1.ResourceCPU: {
				CapacityKey: ptr.To(resourceapi.QualifiedName("dra.example.com/cpu")),
			},
		},
	}

	cpuMemDeviceCapacity := resourceapi.Device{
		Name: "device1",
		NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
			v1.ResourceCPU: {
				CapacityKey: ptr.To(resourceapi.QualifiedName("dra.example.com/cpu")),
			},
			v1.ResourceMemory: {
				CapacityKey: ptr.To(resourceapi.QualifiedName("dra.example.com/memory")),
			},
		},
	}

	gpuDeviceAux := resourceapi.Device{
		Name: "gpu0",
		NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
			v1.ResourceCPU: {
				AllocationMultiplier: ptr.To(resource.MustParse("2")),
			},
			v1.ResourceMemory: {
				AllocationMultiplier: ptr.To(resource.MustParse("4Gi")),
			},
		},
	}

	makeSlice := func(name string, devices ...resourceapi.Device) *resourceapi.ResourceSlice {
		return &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test-ns"},
			Spec: resourceapi.ResourceSliceSpec{
				Pool:     resourceapi.ResourcePool{Name: "pool1"},
				NodeName: ptr.To("test-node"),
				Driver:   "dra.example.com",
				Devices:  devices,
			},
		}
	}

	makeClaim := func(name, uid string) *resourceapi.ResourceClaim {
		return &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test-ns", UID: types.UID(uid)},
		}
	}

	allocResult := func(pool, device string, consumed ...map[resourceapi.QualifiedName]resource.Quantity) *resourceapi.AllocationResult {
		res := resourceapi.DeviceRequestAllocationResult{
			Pool:   pool,
			Device: device,
		}
		if len(consumed) > 0 {
			res.ConsumedCapacity = consumed[0]
		}
		return &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{res},
			},
		}
	}

	claimNameSpace := "test-ns"

	tests := []struct {
		name                            string
		pod                             *v1.Pod
		claims                          []*resourceapi.ResourceClaim
		resourceSlices                  []*resourceapi.ResourceSlice
		nodeAllocatableClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult
		want                            []v1.NodeAllocatableResourceClaimStatus
		wantErr                         bool
	}{
		{
			name: "empty",
			pod:  &v1.Pod{},
			want: []v1.NodeAllocatableResourceClaimStatus{},
		},
		{
			name: "one container, one claim, per instance quantity",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "node-allocatable-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("node-allocatable-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDevicePerInstance)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "node-allocatable-claim", UID: "claim-uid"}: allocResult("pool1", "cpu0"),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "node-allocatable-claim",

				Containers: []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("1"),
				},
			}},
		},
		{
			name: "Pod with Standard and DRA CPU and Memory Request",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Claims: []v1.ResourceClaim{{Name: "node-allocatable-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("node-allocatable-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuMemDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "node-allocatable-claim", UID: "claim-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu":    resource.MustParse("4"),
					"dra.example.com/memory": resource.MustParse("8Gi"),
				}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "node-allocatable-claim",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("8Gi"),
				},
			}},
		},
		{
			name: "Fungible GPU/CPU claim - GPU selected",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Claims:   []v1.ResourceClaim{{Name: "fungible-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("fungible-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("gpu-slice", resourceapi.Device{Name: "gpu0"})},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "fungible-claim", UID: "claim-uid"}: allocResult("pool1", "gpu0"),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{},
		},
		{
			name: "Fungible GPU/CPU claim - CPU selected",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Claims:   []v1.ResourceClaim{{Name: "fungible-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("fungible-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("cpu-slice", cpuDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "fungible-claim", UID: "claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("30"),
				}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "fungible-claim",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("30"),
				},
			}},
		},
		{
			name: "Combined CPU request and Auxiliary Request",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
						Claims:   []v1.ResourceClaim{{Name: "gpu-claim"}, {Name: "cpu-claim"}},
					},
				}}).
				Obj(),
			claims: []*resourceapi.ResourceClaim{
				makeClaim("cpu-claim", "cpu-claim-uid"),
				makeClaim("gpu-claim", "gpu-claim-uid"),
			},
			resourceSlices: []*resourceapi.ResourceSlice{
				makeSlice("cpu-slice", cpuDeviceCapacity),
				makeSlice("gpu-slice", gpuDeviceAux),
			},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "cpu-claim", UID: "cpu-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("10"),
				}),
				{Name: "gpu-claim", UID: "gpu-claim-uid"}: allocResult("pool1", "gpu0"),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "cpu-claim",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("10"),
				},
			}, {
				ResourceClaimName: "gpu-claim",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("2"),
					v1.ResourceMemory: resource.MustParse("4Gi"),
				},
			}},
		},
		{
			name: "Pod Level Resources with shared CPU claim and sidecar containers",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Resources(v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("11"),
						v1.ResourceMemory: resource.MustParse("10Gi"),
					},
				}).
				Containers([]v1.Container{
					{Name: "c1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "cpu-claim"}}}},
					{Name: "c2", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "cpu-claim"}}}},
					{Name: "sidecar1"},
					{Name: "sidecar2"},
				}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("cpu-claim", "cpu-claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("cpu-slice", cpuDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "cpu-claim", UID: "cpu-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("10"),
				}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "cpu-claim",
				Containers:        []string{"c1", "c2"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("10"),
				},
			}},
		},
		{
			name: "Multiple Claims per Container",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
						Claims:   []v1.ResourceClaim{{Name: "claim1"}, {Name: "claim2"}},
					},
				}}).
				Obj(),
			claims: []*resourceapi.ResourceClaim{
				makeClaim("claim1", "claim1-uid"),
				makeClaim("claim2", "claim2-uid"),
			},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDeviceCapacity), makeSlice("slice2", cpuMemDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("4"),
				}),
				{Name: "claim2", UID: "claim2-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/memory": resource.MustParse("8Gi"),
				}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "claim1",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("4"),
				},
			}, {
				ResourceClaimName: "claim2",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceMemory: resource.MustParse("8Gi"),
				},
			}},
		},
		{
			name: "Unreferenced Claims",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{Name: "c1"}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("unref-claim", "unref-claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "unref-claim", UID: "unref-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("4"),
				}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "unref-claim",
				Containers:        []string{},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("4"),
				},
			}},
		},
		{
			name: "Combined Capacity and PerAllocatedUnitQuantity",
			pod: st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").
				Containers([]v1.Container{{
					Name:      "c1",
					Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}},
				}}).
				Obj(),
			claims: []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{
				makeSlice("slice1", resourceapi.Device{
					Name: "device1",
					NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
						v1.ResourceCPU: {
							CapacityKey:          ptr.To(resourceapi.QualifiedName("dra.example.com/cores")),
							AllocationMultiplier: ptr.To(resource.MustParse("2")),
						},
					},
				}),
			},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{"dra.example.com/cores": resource.MustParse("4")}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "claim1",
				Containers:        []string{"c1"},
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: resource.MustParse("8"),
				},
			}},
		},
		{
			name:           "Capacity Key Missing in Allocation - Should be Ignored",
			pod:            st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").Containers([]v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}}}}).Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDeviceCapacity)},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{"dra.example.com/wrong": resource.MustParse("4")}),
			},
			want: []v1.NodeAllocatableResourceClaimStatus{},
		},
		{
			name:           "Invalid -  Device Not Found",
			pod:            st.MakePod().Name("test-pod").Namespace(claimNameSpace).UID("test-uid").Containers([]v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}}}}).Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", resourceapi.Device{Name: "device1"})},
			nodeAllocatableClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "nonexistent-device"),
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			draManager := &mockDRAManager{
				claims:         tt.claims,
				resourceSlices: tt.resourceSlices,
			}

			pl := &DynamicResources{
				draManager: draManager,
			}

			claimNametoUID := make(map[string]types.UID)
			for _, claim := range tt.claims {
				claimNametoUID[claim.Name] = claim.UID
			}

			got, err := pl.buildNodeAllocatableDRAInfo(tt.pod, tt.nodeAllocatableClaimAllocations, claimNametoUID)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildNodeAllocatableDRAInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf(`buildNodeAllocatableDRAInfo() diff (-want +got):
%s`, diff)
			}
		})
	}
}

func TestPatchNodeAllocatableResourceClaimStatus(t *testing.T) {
	pod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("pod-uid").Obj()

	tests := []struct {
		name                               string
		assumedPodStatus                   v1.PodStatus
		finalPodNodeAllocatableClaimStatus []v1.NodeAllocatableResourceClaimStatus
		wantPatch                          bool
		setPatchError                      error
		wantStatus                         *fwk.Status
	}{
		{
			name:       "no node allocatable resource claims for this pod",
			wantPatch:  false,
			wantStatus: nil,
		},
		{
			name: "assumed pod status same as new status",
			assumedPodStatus: v1.PodStatus{
				NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim1",
						Containers:        []string{"c1"},
						Resources: map[v1.ResourceName]resource.Quantity{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			finalPodNodeAllocatableClaimStatus: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "claim1",
					Containers:        []string{"c1"},
					Resources: map[v1.ResourceName]resource.Quantity{
						v1.ResourceCPU: resource.MustParse("1"),
					},
				},
			},
			wantPatch:  true,
			wantStatus: nil,
		},
		{
			name: "assumed pod status different from new status",
			assumedPodStatus: v1.PodStatus{
				NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim1",
						Containers:        []string{"c1"},
						Resources: map[v1.ResourceName]resource.Quantity{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			finalPodNodeAllocatableClaimStatus: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "claim1",
					Containers:        []string{"c1"},
					Resources: map[v1.ResourceName]resource.Quantity{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
			},
			wantPatch:  false,
			wantStatus: statusError(klog.TODO(), errors.New("assumed pod status does not match calculated status to be patched")),
		},
		{
			name: "pod status patch error",
			assumedPodStatus: v1.PodStatus{
				NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim1",
						Containers:        []string{"c1"},
						Resources: map[v1.ResourceName]resource.Quantity{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			finalPodNodeAllocatableClaimStatus: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "claim1",
					Containers:        []string{"c1"},
					Resources: map[v1.ResourceName]resource.Quantity{
						v1.ResourceCPU: resource.MustParse("1"),
					},
				},
			},
			wantPatch:     true,
			setPatchError: errors.New("inject patch error"),
			wantStatus:    statusError(klog.TODO(), fmt.Errorf("updating pod test-ns/test-pod NodeAllocatableResourceClaimStatuses: %w", errors.New("inject patch error"))),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			podToUpdate := pod.DeepCopy()
			podToUpdate.Status = *tt.assumedPodStatus.DeepCopy()

			fakeClient := fake.NewSimpleClientset(podToUpdate)
			pl := &DynamicResources{
				clientset: fakeClient,
				fts:       feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate),
			}
			if tt.setPatchError != nil {
				fakeClient.PrependReactor("patch", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, tt.setPatchError
				})
			}
			status := pl.patchNodeAllocatableResourceClaimStatus(ctx, podToUpdate, tt.finalPodNodeAllocatableClaimStatus)

			if tt.wantStatus != nil && status != nil {
				if tt.wantStatus.Code() != status.Code() {
					t.Errorf("patchNodeAllocatableResourceClaimStatus() status code = %v, want %v", status.Code(), tt.wantStatus.Code())
				}
				if tt.wantStatus.AsError().Error() != status.AsError().Error() {
					t.Errorf("patchNodeAllocatableResourceClaimStatus() status error = %v, want %v", status.AsError().Error(), tt.wantStatus.AsError().Error())
				}
			} else if tt.wantStatus != status {
				t.Errorf("patchNodeAllocatableResourceClaimStatus() status = %v, want %v", status, tt.wantStatus)
			}

			actions := fakeClient.Actions()
			gotPatch := false
			for _, action := range actions {
				if action.Matches("patch", "pods") && action.GetSubresource() == "status" {
					gotPatch = true
					break
				}
			}

			if gotPatch != tt.wantPatch {
				t.Errorf("patchNodeAllocatableResourceClaimStatus() gotPatch = %v, want %v", gotPatch, tt.wantPatch)
			}
		})
	}
}

func TestClearNodeAllocatableResourceClaimStatus(t *testing.T) {
	pod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("pod-uid").Obj()

	tests := []struct {
		name             string
		initialPodStatus v1.PodStatus
		wantPatch        bool
	}{
		{
			name:             "no status to clear",
			initialPodStatus: v1.PodStatus{},
			wantPatch:        false,
		},
		{
			name: "status cleared",
			initialPodStatus: v1.PodStatus{
				NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim1",
						Containers:        []string{"c1"},
						Resources: map[v1.ResourceName]resource.Quantity{
							v1.ResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			wantPatch: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			podToUpdate := pod.DeepCopy()
			podToUpdate.Status = *tt.initialPodStatus.DeepCopy()

			fakeClient := fake.NewSimpleClientset(podToUpdate)
			pl := &DynamicResources{
				clientset: fakeClient,
				fts:       feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate),
			}

			pl.clearNodeAllocatableResourceClaimStatus(ctx, podToUpdate)

			actions := fakeClient.Actions()
			gotPatch := false
			for _, action := range actions {
				if action.Matches("patch", "pods") && action.GetSubresource() == "status" {
					gotPatch = true
					break
				}
			}

			if gotPatch != tt.wantPatch {
				t.Errorf("patchNodeAllocatableResourceClaimStatus() gotPatch = %v, want %v", gotPatch, tt.wantPatch)
			}
		})
	}
}

func TestNodeFitsNativeResources(t *testing.T) {
	tests := []struct {
		name       string
		podRequest *framework.Resource
		nodeInfo   *framework.NodeInfo
		wantStatus *fwk.Status
	}{
		{
			name:       "empty pod request",
			podRequest: &framework.Resource{},
			nodeInfo:   framework.NewNodeInfo(),
			wantStatus: nil,
		},
		{
			name: "sufficient cpu and memory",
			podRequest: &framework.Resource{
				MilliCPU: 1000,
				Memory:   1024 * 1024 * 1024,
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					v1.ResourceCPU:    "2000m",
					v1.ResourceMemory: "2Gi",
				}).Obj())
				return ni
			}(),
			wantStatus: nil,
		},
		{
			name: "insufficient cpu",
			podRequest: &framework.Resource{
				MilliCPU: 3000,
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					v1.ResourceCPU: "2000m",
				}).Obj())
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `Insufficient cpu`),
		},
		{
			name: "insufficient memory",
			podRequest: &framework.Resource{
				Memory: 3 * 1024 * 1024 * 1024,
			},
			nodeInfo: func() *framework.NodeInfo {
				existigPod := v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceMemory: resource.MustParse("4Gi"),
									},
								},
							},
						},
					},
				}
				ni := framework.NewNodeInfo(&existigPod)
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					v1.ResourceMemory: "5Gi",
				}).Obj())
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, `Insufficient memory`),
		},
		{
			name: "sufficient hugepages",
			podRequest: &framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{
					hugePages2Mi: 1,
					hugePages1Gi: 2,
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					hugePages2Mi: "1",
					hugePages1Gi: "2",
				}).Obj())
				return ni
			}(),
			wantStatus: nil,
		},
		{
			name: "insufficient hugepages 2Mi",
			podRequest: &framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{
					hugePages2Mi: 2,
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					hugePages2Mi: "1",
				}).Obj())
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `Insufficient hugepages-2Mi`),
		},
		{
			name: "insufficient hugepages 1Gi",
			podRequest: &framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{
					hugePages1Gi: 3,
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					hugePages1Gi: "2",
				}).Obj())
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `Insufficient hugepages-1Gi`),
		},
		{
			name: "all resources sufficient",
			podRequest: &framework.Resource{
				MilliCPU: 1000,
				Memory:   1024 * 1024 * 1024,
				ScalarResources: map[v1.ResourceName]int64{
					hugePages2Mi: 1,
				},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					v1.ResourceCPU:    "2000m",
					v1.ResourceMemory: "2Gi",
					hugePages2Mi:      "1",
				}).Obj())
				return ni
			}(),
			wantStatus: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DynamicResources{}
			status := pl.nodeFitsResources(tt.nodeInfo, tt.podRequest)
			if diff := cmp.Diff(tt.wantStatus, status, cmp.Comparer(func(a, b *fwk.Status) bool {
				if a == nil && b == nil {
					return true
				}
				if a == nil || b == nil {
					return false
				}
				if a.Code() != b.Code() {
					return false
				}
				// Only compare reasons if the code is Unschedulable
				if a.Code() == fwk.Unschedulable {
					return cmp.Diff(a.Reasons(), b.Reasons()) == ""
				}
				return true
			})); diff != "" {
				t.Errorf(`nodeFitsNativeResources() diff (-want +got):
%s`, diff)
			}
		})
	}
}

func TestValidatePodLevelRequestsCoverDRA(t *testing.T) {
	tests := []struct {
		name                  string
		pod                   *v1.Pod
		nodeAllocatableStatus []v1.NodeAllocatableResourceClaimStatus
		requestWithPodLevel   v1.ResourceList
		wantStatusCode        fwk.Code
		wantErrorMessage      string
	}{
		{
			name: "PodLevelResources enabled, no pod resources set",
			pod: st.MakePod().Name("test-pod").
				Containers([]v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
						Claims: []v1.ResourceClaim{{Name: "dra-claim"}},
					},
				}}).
				Obj(),
			nodeAllocatableStatus: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "dra-claim",
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("512Mi"),
				},
			}},
			requestWithPodLevel: v1.ResourceList{},
			wantStatusCode:      fwk.Success,
		},
		{
			name: "PodLevel resources sufficient",
			pod: st.MakePod().Name("test-pod").
				Resources(v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2"),
						v1.ResourceMemory: resource.MustParse("2Gi"),
					},
				}).
				Containers([]v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
						Claims: []v1.ResourceClaim{{Name: "dra-claim"}},
					},
				}}).
				Obj(),
			nodeAllocatableStatus: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "dra-claim",
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("512Mi"),
				},
			}},
			requestWithPodLevel: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("2"),
				v1.ResourceMemory: resource.MustParse("2Gi"),
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "PodLevel insufficient for one of the resource",
			pod: st.MakePod().Name("test-pod").
				Resources(v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("2Gi"),
					},
				}).
				Containers([]v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("800m"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
						Claims: []v1.ResourceClaim{{Name: "dra-claim"}},
					},
				}}).
				Obj(),
			nodeAllocatableStatus: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "dra-claim",
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("300m"), // 800 + 300 = 1100 > 1000
					v1.ResourceMemory: resource.MustParse("512Mi"),
				},
			}},
			requestWithPodLevel: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("2Gi"),
			},
			wantStatusCode:   fwk.UnschedulableAndUnresolvable,
			wantErrorMessage: "pod level request for cpu is insufficient to cover the aggregated container and node-allocatable DRA requests",
		},
		{
			name: "PodLevel insufficient for multiple resources",
			pod: st.MakePod().Name("test-pod").
				Resources(v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				}).
				Containers([]v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1100m"),
							v1.ResourceMemory: resource.MustParse("1100Mi"),
						},
						Claims: []v1.ResourceClaim{{Name: "dra-claim"}},
					},
				}}).
				Obj(),
			nodeAllocatableStatus: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "dra-claim",
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			}},
			requestWithPodLevel: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			wantStatusCode:   fwk.UnschedulableAndUnresolvable,
			wantErrorMessage: "insufficient to cover the aggregated container and node-allocatable DRA requests",
		},
		{
			name: "PodLevel resources sufficient with overheads",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("4"),
							v1.ResourceMemory: resource.MustParse("3Gi"),
						},
					},
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Claims: []v1.ResourceClaim{{Name: "dra-claim"}},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("512Mi"),
					},
				},
			},
			nodeAllocatableStatus: []v1.NodeAllocatableResourceClaimStatus{{
				ResourceClaimName: "dra-claim",
				Resources: map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU:    resource.MustParse("1"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			}},
			requestWithPodLevel: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("3Gi"),
			},
			wantStatusCode: fwk.Success,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DynamicResources{
				fts: feature.Features{EnablePodLevelResources: true},
			}
			tt.pod.Status.NodeAllocatableResourceClaimStatuses = tt.nodeAllocatableStatus
			gotStatus := pl.validatePodLevelRequestsCoverDRA(klog.TODO(), tt.pod, tt.requestWithPodLevel)
			if diff := cmp.Diff(tt.wantStatusCode, gotStatus.Code()); diff != "" {
				t.Errorf("validatePodLevelRequestsCoverDRA() returned diff (-want +got):\n%s", diff)
			}
			if tt.wantStatusCode != fwk.Success && tt.wantErrorMessage != "" {
				if !strings.Contains(gotStatus.Message(), tt.wantErrorMessage) {
					t.Errorf("validatePodLevelRequestsCoverDRA() returned error %q, want it to contain %q", gotStatus.Message(), tt.wantErrorMessage)
				}
			}
		})
	}
}
