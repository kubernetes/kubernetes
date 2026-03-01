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
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
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

func (m *mockDRAManager) ClaimHasPendingAllocation(uid types.UID) bool {
	return false
}

func (m *mockDRAManager) SignalClaimPendingAllocation(uid types.UID, claim *resourceapi.ResourceClaim) error {
	return nil
}

func (m *mockDRAManager) RemoveClaimPendingAllocation(uid types.UID) bool {
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

func TestDeviceClassManagesNativeResource(t *testing.T) {
	tests := []struct {
		name                     string
		deviceClassName          string
		deviceClasses            []*resourceapi.DeviceClass
		enableDRANativeResources bool
		wantManages              bool
		wantStatus               *fwk.Status
	}{
		{
			name:                     "feature disabled",
			enableDRANativeResources: false,
			wantManages:              false,
			wantStatus:               nil,
		},
		{
			name:                     "empty device class name",
			deviceClassName:          "",
			enableDRANativeResources: true,
			wantManages:              false,
			wantStatus:               nil,
		},
		{
			name:                     "device class not found",
			deviceClassName:          "test-class",
			enableDRANativeResources: true,
			wantManages:              false,
			wantStatus:               fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `device class test-class does not exist`),
		},
		{
			name:                     "device class does not manage native resources",
			deviceClassName:          "test-class",
			enableDRANativeResources: true,
			deviceClasses: []*resourceapi.DeviceClass{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test-class"},
					Spec: resourceapi.DeviceClassSpec{
						ManagesNativeResources: ptr.To(false),
					},
				},
			},
			wantManages: false,
			wantStatus:  nil,
		},
		{
			name:                     "device class manages native resources",
			deviceClassName:          "test-class",
			enableDRANativeResources: true,
			deviceClasses: []*resourceapi.DeviceClass{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test-class"},
					Spec: resourceapi.DeviceClassSpec{
						ManagesNativeResources: ptr.To(true),
					},
				},
			},
			wantManages: true,
			wantStatus:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			fakeClient := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

			for _, dc := range tt.deviceClasses {
				err := informerFactory.Resource().V1().DeviceClasses().Informer().GetIndexer().Add(dc)
				if err != nil {
					t.Fatalf("failed to add device class to informer: %v", err)
				}
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANativeResources, tt.enableDRANativeResources)

			pl := &DynamicResources{

				fts:        feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate),
				draManager: NewDRAManager(ctx, assumecache.NewAssumeCache(klog.FromContext(ctx), informerFactory.Core().V1().Pods().Informer(), "", "", nil), nil, informerFactory),
			}

			gotManages, gotStatus := pl.deviceClassManagesNativeResource(klog.FromContext(ctx), tt.deviceClassName)

			if gotManages != tt.wantManages {
				t.Errorf("deviceClassManagesNativeResource() gotManages = %v, want %v", gotManages, tt.wantManages)
			}
			if diff := cmp.Diff(tt.wantStatus, gotStatus); diff != "" {
				t.Errorf(`deviceClassManagesNativeResource() gotStatus diff (-want +got):
%s`, diff)
			}
		})
	}
}

func TestBuildNativeDRAInfo(t *testing.T) {
	cpuDevicePerInstance := resourceapi.Device{
		Name: "cpu0",
		NativeResourceMappings: map[v1.ResourceName]resourceapi.NativeResourceMapping{
			v1.ResourceCPU: {
				PerAllocatedUnitQuantity: ptr.To(resource.MustParse("1")),
			},
		},
	}

	cpuDeviceCapacity := resourceapi.Device{
		Name: "cpu0",
		NativeResourceMappings: map[v1.ResourceName]resourceapi.NativeResourceMapping{
			v1.ResourceCPU: {
				CapacityKey: ptr.To(resourceapi.QualifiedName("dra.example.com/cpu")),
			},
		},
	}

	cpuMemDeviceCapacity := resourceapi.Device{
		Name: "device1",
		NativeResourceMappings: map[v1.ResourceName]resourceapi.NativeResourceMapping{
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
		NativeResourceMappings: map[v1.ResourceName]resourceapi.NativeResourceMapping{
			v1.ResourceCPU: {
				PerAllocatedUnitQuantity: ptr.To(resource.MustParse("2")),
			},
			v1.ResourceMemory: {
				PerAllocatedUnitQuantity: ptr.To(resource.MustParse("4Gi")),
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

	tests := []struct {
		name                   string
		pod                    *v1.Pod
		claims                 []*resourceapi.ResourceClaim
		resourceSlices         []*resourceapi.ResourceSlice
		nativeClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult
		want                   []v1.PodNativeResourceClaimStatus
		wantErr                bool
	}{
		{
			name: "empty",
			pod:  &v1.Pod{},
			want: []v1.PodNativeResourceClaimStatus{},
		},
		{
			name: "one container, one claim, per instance quantity",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "native-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("native-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDevicePerInstance)},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "native-claim", UID: "claim-uid"}: allocResult("pool1", "cpu0"),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "native-claim", UID: "claim-uid"},
				Containers: []string{"c1"},
				Resources: []v1.NativeResourceAllocation{{
					ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1"),
				}},
			}},
		},
		{
			name: "Pod with Standard and DRA CPU and Memory Request",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
				Containers([]v1.Container{{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Claims: []v1.ResourceClaim{{Name: "native-claim"}},
					},
				}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("native-claim", "claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuMemDeviceCapacity)},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "native-claim", UID: "claim-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu":    resource.MustParse("4"),
					"dra.example.com/memory": resource.MustParse("8Gi"),
				}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "native-claim", UID: "claim-uid"},
				Containers: []string{"c1"},
				Resources: []v1.NativeResourceAllocation{
					{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("4")},
					{ResourceName: v1.ResourceMemory, Quantity: resource.MustParse("8Gi")},
				},
			}},
		},
		{
			name: "Fungible GPU/CPU claim - GPU selected",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
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
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "fungible-claim", UID: "claim-uid"}: allocResult("pool1", "gpu0"),
			},
			want: []v1.PodNativeResourceClaimStatus{},
		},
		{
			name: "Fungible GPU/CPU claim - CPU selected",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
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
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "fungible-claim", UID: "claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("30"),
				}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "fungible-claim", UID: "claim-uid"},
				Containers: []string{"c1"},
				Resources: []v1.NativeResourceAllocation{{
					ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("30"),
				}},
			}},
		},
		{
			name: "Combined Native CPU request and Auxiliary Request",
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
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "cpu-claim", UID: "cpu-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("10"),
				}),
				{Name: "gpu-claim", UID: "gpu-claim-uid"}: allocResult("pool1", "gpu0"),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "cpu-claim", UID: "cpu-claim-uid"},
				Containers: []string{"c1"},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("10")}},
			}, {
				ClaimInfo:  v1.ObjectReference{Name: "gpu-claim", UID: "gpu-claim-uid"},
				Containers: []string{"c1"},
				Resources: []v1.NativeResourceAllocation{{
					ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("2"),
				}, {
					ResourceName: v1.ResourceMemory, Quantity: resource.MustParse("4Gi"),
				}},
			}},
		},
		{
			name: "Pod Level Resources with shared CPU claim and sidecar containers",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
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
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "cpu-claim", UID: "cpu-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("10"),
				}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "cpu-claim", UID: "cpu-claim-uid"},
				Containers: []string{"c1", "c2"},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("10")}},
			}},
		},
		{
			name: "Multiple Claims per Container",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
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
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("4"),
				}),
				{Name: "claim2", UID: "claim2-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/memory": resource.MustParse("8Gi"),
				}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
				Containers: []string{"c1"},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("4")}},
			}, {
				ClaimInfo:  v1.ObjectReference{Name: "claim2", UID: "claim2-uid"},
				Containers: []string{"c1"},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceMemory, Quantity: resource.MustParse("8Gi")}},
			}},
		},
		{
			name: "Unreferenced Claims",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
				Containers([]v1.Container{{Name: "c1"}}).
				Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("unref-claim", "unref-claim-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDeviceCapacity)},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "unref-claim", UID: "unref-claim-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{
					"dra.example.com/cpu": resource.MustParse("4"),
				}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "unref-claim", UID: "unref-claim-uid"},
				Containers: []string{},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("4")}},
			}},
		},
		{
			name: "Combined Capacity and PerAllocatedUnitQuantity",
			pod: st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").
				Containers([]v1.Container{{
					Name:      "c1",
					Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}},
				}}).
				Obj(),
			claims: []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{
				makeSlice("slice1", resourceapi.Device{
					Name: "device1",
					NativeResourceMappings: map[v1.ResourceName]resourceapi.NativeResourceMapping{
						v1.ResourceCPU: {
							CapacityKey:              ptr.To(resourceapi.QualifiedName("dra.example.com/cores")),
							PerAllocatedUnitQuantity: ptr.To(resource.MustParse("2")),
						},
					},
				}),
			},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "device1", map[resourceapi.QualifiedName]resource.Quantity{"dra.example.com/cores": resource.MustParse("4")}),
			},
			want: []v1.PodNativeResourceClaimStatus{{
				ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
				Containers: []string{"c1"},
				Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("8")}},
			}},
		},
		{
			name:           "Capacity Key Missing in Allocation - Should be Ignored",
			pod:            st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").Containers([]v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}}}}).Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", cpuDeviceCapacity)},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
				{Name: "claim1", UID: "claim1-uid"}: allocResult("pool1", "cpu0", map[resourceapi.QualifiedName]resource.Quantity{"dra.example.com/wrong": resource.MustParse("4")}),
			},
			want: []v1.PodNativeResourceClaimStatus{},
		},
		{
			name:           "Invalid -  Device Not Found",
			pod:            st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").Containers([]v1.Container{{Name: "c1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}}}}).Obj(),
			claims:         []*resourceapi.ResourceClaim{makeClaim("claim1", "claim1-uid")},
			resourceSlices: []*resourceapi.ResourceSlice{makeSlice("slice1", resourceapi.Device{Name: "device1"})},
			nativeClaimAllocations: map[v1.ObjectReference]*resourceapi.AllocationResult{
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

			got, err := pl.buildNativeDRAInfo(tt.pod, tt.nativeClaimAllocations, claimNametoUID)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildNativeDRAInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf(`buildNativeDRAInfo() diff (-want +got):
%s`, diff)
			}
		})
	}
}

func TestValidateNativeDRAClaims(t *testing.T) {
	tests := []struct {
		name                      string
		pod                       *v1.Pod
		nativeResourceClaimStatus []v1.PodNativeResourceClaimStatus
		nodeInfo                  *framework.NodeInfo
		wantErr                   bool
	}{
		{
			name:    "no native resource claims",
			pod:     st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			wantErr: false,
		},
		{
			name: "claim used by this pod only",
			pod:  st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			nativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
				{ClaimInfo: v1.ObjectReference{UID: "claim-uid"}},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NativeDRAClaimStates = map[types.UID]*fwk.NativeDRAClaimAllocationState{
					"claim-uid": {ConsumerPods: sets.New[types.UID]("pod-uid")},
				}
				return ni
			}(),
			wantErr: false,
		},
		{
			name: "claim not in node info",
			pod:  st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			nativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
				{ClaimInfo: v1.ObjectReference{UID: "claim-uid"}},
			},
			nodeInfo: framework.NewNodeInfo(),
			wantErr:  false,
		},
		{
			name: "claim shared, current pod not in consumers",
			pod:  st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			nativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
				{ClaimInfo: v1.ObjectReference{UID: "claim-uid"}},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NativeDRAClaimStates = map[types.UID]*fwk.NativeDRAClaimAllocationState{
					"claim-uid": {ConsumerPods: sets.New[types.UID]("other-pod-uid", "another-pod-uid")},
				}
				return ni
			}(),
			wantErr: true,
		},
		{
			name: "claim shared, current pod in consumers",
			pod:  st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			nativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
				{ClaimInfo: v1.ObjectReference{UID: "claim-uid"}},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NativeDRAClaimStates = map[types.UID]*fwk.NativeDRAClaimAllocationState{
					"claim-uid": {ConsumerPods: sets.New[types.UID]("pod-uid", "another-pod-uid")},
				}
				return ni
			}(),
			wantErr: true,
		},
		{
			name: "claim only used by other pod",
			pod:  st.MakePod().Name("test-pod").UID("pod-uid").Obj(),
			nativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
				{ClaimInfo: v1.ObjectReference{UID: "claim-uid"}},
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.NativeDRAClaimStates = map[types.UID]*fwk.NativeDRAClaimAllocationState{
					"claim-uid": {ConsumerPods: sets.New[types.UID]("other-pod-uid")},
				}
				return ni
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DynamicResources{}
			err := pl.validateNativeDRAClaims(tt.pod, tt.nodeInfo, tt.nativeResourceClaimStatus)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateNativeDRAClaims() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPatchNativeResourceClaimStatus(t *testing.T) {
	nodeName := "test-node"
	pod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("pod-uid").Obj()

	tests := []struct {
		name                     string
		enableDRANativeResources bool
		hasNativeResourceClaims  bool
		initialPodStatus         v1.PodStatus
		finalPodStatus           []v1.PodNativeResourceClaimStatus
		wantPatch                bool
		setPatchError            error
		wantStatus               *fwk.Status
	}{
		{
			name:                     "feature disabled",
			enableDRANativeResources: false,
			hasNativeResourceClaims:  true,
			wantPatch:                false,
			wantStatus:               nil,
		},
		{
			name:                     "no native resource claims for this pod",
			enableDRANativeResources: true,
			hasNativeResourceClaims:  false,
			wantPatch:                false,
			wantStatus:               nil,
		},
		{
			name:                     "initial status empty, new status added",
			enableDRANativeResources: true,
			hasNativeResourceClaims:  true,
			initialPodStatus:         v1.PodStatus{},
			finalPodStatus: []v1.PodNativeResourceClaimStatus{
				{
					ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
					Containers: []string{"c1"},
					Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1")}},
				},
			},
			wantPatch:  true,
			wantStatus: nil,
		},
		{
			name:                     "initial status same as new status",
			enableDRANativeResources: true,
			hasNativeResourceClaims:  true,
			initialPodStatus: v1.PodStatus{
				NativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
					{
						ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
						Containers: []string{"c1"},
						Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1")}},
					},
				},
			},
			finalPodStatus: []v1.PodNativeResourceClaimStatus{
				{
					ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
					Containers: []string{"c1"},
					Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1")}},
				},
			},
			wantPatch:  false,
			wantStatus: nil,
		},
		{
			name:                     "initial status different from new status",
			enableDRANativeResources: true,
			hasNativeResourceClaims:  true,
			initialPodStatus: v1.PodStatus{
				NativeResourceClaimStatus: []v1.PodNativeResourceClaimStatus{
					{
						ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
						Containers: []string{"c1"},
						Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1")}},
					},
				},
			},
			finalPodStatus: []v1.PodNativeResourceClaimStatus{
				{
					ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
					Containers: []string{"c1"},
					Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("2")}},
				},
			},
			wantPatch:  true,
			wantStatus: nil,
		},
		{
			name:                     "pod status patch error",
			enableDRANativeResources: true,
			hasNativeResourceClaims:  true,
			initialPodStatus:         v1.PodStatus{},
			finalPodStatus: []v1.PodNativeResourceClaimStatus{
				{
					ClaimInfo:  v1.ObjectReference{Name: "claim1", UID: "claim1-uid"},
					Containers: []string{"c1"},
					Resources:  []v1.NativeResourceAllocation{{ResourceName: v1.ResourceCPU, Quantity: resource.MustParse("1")}},
				},
			},
			wantPatch:     true,
			setPatchError: errors.New("inject patch error"),
			wantStatus:    statusError(klog.TODO(), fmt.Errorf("updating pod test-ns/test-pod NativeResourceClaimStatus: %w", errors.New("inject patch error"))),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANativeResources, tt.enableDRANativeResources)

			podToUpdate := pod.DeepCopy()
			podToUpdate.Status = *tt.initialPodStatus.DeepCopy()

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
			state := &stateData{
				hasNativeResourceClaims: tt.hasNativeResourceClaims,
				nodeAllocations: map[string]nodeAllocation{
					nodeName: {
						nativeResourceClaimStatus: tt.finalPodStatus,
					},
				},
			}
			status := pl.patchNativeResourceClaimStatus(ctx, state, podToUpdate, nodeName)

			if tt.wantStatus != nil && status != nil {
				if tt.wantStatus.Code() != status.Code() {
					t.Errorf("patchNativeResourceClaimStatus() status code = %v, want %v", status.Code(), tt.wantStatus.Code())
				}
				if tt.wantStatus.AsError().Error() != status.AsError().Error() {
					t.Errorf("patchNativeResourceClaimStatus() status error = %v, want %v", status.AsError().Error(), tt.wantStatus.AsError().Error())
				}
			} else if tt.wantStatus != status {
				t.Errorf("patchNativeResourceClaimStatus() status = %v, want %v", status, tt.wantStatus)
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
				t.Errorf("patchNativeResourceClaimStatus() gotPatch = %v, want %v", gotPatch, tt.wantPatch)
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
			name:       "empty_pod_request_fits",
			podRequest: &framework.Resource{},
			nodeInfo:   framework.NewNodeInfo(),
			wantStatus: nil,
		},
		{
			name: "sufficient_cpu_and_memory",
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
			name: "insufficient_cpu_fails",
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
			wantStatus: fwk.NewStatus(fwk.Unschedulable, `Insufficient resources: cpu (requested: 3000, used: 0, capacity: 2000)`),
		},
		{
			name: "insufficient_memory_fails",
			podRequest: &framework.Resource{
				Memory: 3 * 1024 * 1024 * 1024,
			},
			nodeInfo: func() *framework.NodeInfo {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name("test-node").Capacity(map[v1.ResourceName]string{
					v1.ResourceMemory: "2Gi",
				}).Obj())
				return ni
			}(),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, `Insufficient resources: memory (requested: 3221225472, used: 0, capacity: 2147483648)`),
		},
		{
			name: "sufficient_hugepages",
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
			name: "insufficient_hugepages_2Mi",
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
			wantStatus: fwk.NewStatus(fwk.Unschedulable, `Insufficient resources: hugepages-2Mi (requested: 2, used: 0, capacity: 1)`),
		},
		{
			name: "insufficient_hugepages_1Gi",
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
			wantStatus: fwk.NewStatus(fwk.Unschedulable, `Insufficient resources: hugepages-1Gi (requested: 3, used: 0, capacity: 2)`),
		},
		{
			name: "all_resources_sufficient",
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
			status := pl.nodeFitsNativeResources(tt.nodeInfo, tt.podRequest)
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
