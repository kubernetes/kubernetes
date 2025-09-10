/*
Copyright 2023 The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

func TestResourceAllocationScorerCalculateRequests(t *testing.T) {
	const oneMi = 1048576
	tests := []struct {
		name     string
		pod      v1.Pod
		expected map[v1.ResourceName]int64
	}{
		{
			name: "overhead only",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    1000,
				v1.ResourceMemory: oneMi,
			},
		},
		{
			name: "1x requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    util.DefaultMilliCPURequest,
				v1.ResourceMemory: util.DefaultMemoryRequest,
			},
		},
		{
			name: "2x requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{}, {},
					},
				},
			},
			// should accumulate once per container without a request
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    2 * util.DefaultMilliCPURequest,
				v1.ResourceMemory: 2 * util.DefaultMemoryRequest,
			},
		},
		{
			name: "request container + requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    1000 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "container + requestless container + overhead",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    2000 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: 2*oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "init container + container + requestless container + overhead",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("3"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    4000,
				v1.ResourceMemory: 2*oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "requestless init container + small init container + small container ",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1m"),
									v1.ResourceMemory: resource.MustParse("1"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("3m"),
									v1.ResourceMemory: resource.MustParse("3"),
								},
							},
						},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    util.DefaultMilliCPURequest,
				v1.ResourceMemory: util.DefaultMemoryRequest,
			},
		},
		{
			name: "requestless init container + small init container + small container + requestless container ",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1m"),
									v1.ResourceMemory: resource.MustParse("1"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("3m"),
									v1.ResourceMemory: resource.MustParse("3"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    3 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: 3 + util.DefaultMemoryRequest,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var scorer resourceAllocationScorer
			for n, exp := range tc.expected {
				got := scorer.calculatePodResourceRequest(&tc.pod, n)
				if got != exp {
					t.Errorf("expected %s = %d, got %d", n, exp, got)
				}
			}
		})
	}
}

func TestCalculateResourceAllocatableRequest(t *testing.T) {
	// Initialize test variables
	nodeName := "resource-node"
	driverName := "test-driver"
	testClaim := "test-claim"
	explicitExtendedResource := v1.ResourceName(extendedResourceName)
	implicitExtendedResource := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + deviceClassName)
	celCache := cel.NewCache(10, cel.Features{EnableConsumableCapacity: true})
	draFeatures := structured.Features{
		AdminAccess:            true,
		PrioritizedList:        true,
		PartitionableDevices:   true,
		DeviceTaints:           true,
		DeviceBindingAndStatus: true,
		ConsumableCapacity:     true,
	}

	// Define test cases
	tests := map[string]struct {
		enableDRAExtendedResource bool
		node                      *v1.Node
		extendedResource          v1.ResourceName
		objects                   []apiruntime.Object
		podRequest                int64
		expectedAllocatable       int64
		expectedRequested         int64
	}{
		"device-plugin-resource-feature-disabled": {
			enableDRAExtendedResource: false,
			node:                      st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{explicitExtendedResource: "4"}).Obj(),
			extendedResource:          explicitExtendedResource,
			podRequest:                1,
			expectedAllocatable:       4,
			expectedRequested:         1,
		},
		"device-plugin-resource-feature-enabled": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{explicitExtendedResource: "4"}).Obj(),
			extendedResource:          explicitExtendedResource,
			podRequest:                1,
			expectedAllocatable:       4,
			expectedRequested:         1,
		},
		"DRA-backed-resource-explicit": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice(nodeName, driverName).Device("device-1").Obj(),
			},
			podRequest:          1,
			expectedAllocatable: 1,
			expectedRequested:   1,
		},
		"DRA-backed-resource-implicit": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          implicitExtendedResource,
			objects: []apiruntime.Object{
				&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: deviceClassName,
					},
				},
				st.MakeResourceSlice(nodeName, driverName).Device("device-1").Obj(),
			},
			podRequest:          1,
			expectedAllocatable: 1,
			expectedRequested:   1,
		},
		"DRA-backed-resource-no-slices": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects:                   []apiruntime.Object{deviceClassWithExtendResourceName},
			podRequest:                1,
			expectedAllocatable:       0,
			expectedRequested:         0,
		},
		"DRA-backed-resource-with-allocated-device": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice(nodeName, driverName).Devices("device-1", "device-2").Obj(),
				// Create a resource claim that fully allocates device-1
				st.MakeResourceClaim().
					Name(testClaim).
					Request(deviceClassName).
					Allocation(&resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1",
									Driver:  driverName,
									Pool:    nodeName,
									Device:  "device-1",
								},
							},
						},
					}).
					Obj(),
			},
			podRequest:          1,
			expectedAllocatable: 2,
			expectedRequested:   2, // 1 allocated + 1 requested
		},
		"DRA-backed-resource-with-shared-device-allocation": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice(nodeName, driverName).Devices("device-1", "device-2").Obj(),
				// Create a resource claim with shared device allocation (consumable capacity)
				st.MakeResourceClaim().
					Name(testClaim).
					Request(deviceClassName).
					Allocation(&resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1",
									Driver:  driverName,
									Pool:    nodeName,
									Device:  "device-1",
									ShareID: ptr.To(types.UID("share-123")), // Shared device allocation
								},
							},
						},
					}).
					Obj(),
			},
			podRequest:          1,
			expectedAllocatable: 2,
			expectedRequested:   2, // 1 allocated (shared) + 1 requested
		},
		"DRA-backed-resource-multiple-devices-mixed-allocation": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice(nodeName, driverName).Devices("device-1", "device-2", "device-3").Obj(),
				// Mix of fully allocated and shared device allocations
				st.MakeResourceClaim().
					Name("test-claim-1").
					Request(deviceClassName).
					Allocation(&resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1",
									Driver:  driverName,
									Pool:    nodeName,
									Device:  "device-1",
									// No ShareID = fully allocated device
								},
							},
						},
					}).
					Obj(),
				st.MakeResourceClaim().
					Name("test-claim-2").
					Request(deviceClassName).
					Allocation(&resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1",
									Driver:  driverName,
									Pool:    nodeName,
									Device:  "device-2",
									ShareID: ptr.To(types.UID("share-456")), // Shared device allocation
								},
							},
						},
					}).
					Obj(),
				// device-3 remains unallocated
			},
			podRequest:          1,
			expectedAllocatable: 3,
			expectedRequested:   3, // 2 allocated (1 full + 1 shared) + 1 requested
		},
		"DRA-backed-resource-with-per-device-node-selection": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			objects: []apiruntime.Object{
				&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: deviceClassName,
					},
					Spec: resourceapi.DeviceClassSpec{
						ExtendedResourceName: &extendedResourceName,
						Selectors: []resourceapi.DeviceSelector{
							{
								CEL: &resourceapi.CELDeviceSelector{
									// Realistic GPU selection: match test driver with at least 8GB memory and SOME- model
									Expression: `device.driver == "test-driver" && device.capacity["test-driver"].memory.compareTo(quantity("8Gi")) >= 0 && device.attributes["test-driver"].model.startsWith("SOME-")`,
								},
							},
						},
					},
				},
				// Create a custom resource slice with per-device node selection
				&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{Name: "per-device-slice"},
					Spec: resourceapi.ResourceSliceSpec{
						Driver:                 driverName, // Match the CEL expression driver
						Pool:                   resourceapi.ResourcePool{Name: "per-device-pool", ResourceSliceCount: 1},
						PerDeviceNodeSelection: ptr.To(true), // Enable per-device node selection
						// Note: No NodeName, AllNodes, or NodeSelector at slice level when PerDeviceNodeSelection is true
						Devices: []resourceapi.Device{
							{
								Name:     "device-1",
								NodeName: ptr.To(nodeName), // This device matches the test node
								Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
									"model": {StringValue: ptr.To("SOME-XYZ")},
								},
								Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
									"memory": {Value: resource.MustParse("16Gi")}, // 16GB GPU - matches CEL (>= 8GB)
								},
							},
							{
								Name:     "device-2",
								NodeName: ptr.To("other-node"), // This device matches a different node
								Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
									"model": {StringValue: ptr.To("SOME-ZYX")},
								},
								Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
									"memory": {Value: resource.MustParse("12Gi")}, // 12GB GPU - matches CEL (>= 8GB)
								},
							},
							{
								Name:     "device-3",
								AllNodes: ptr.To(true), // This device matches all nodes
								Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
									"model": {StringValue: ptr.To("SOME-XZY")},
								},
								Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
									"memory": {Value: resource.MustParse("24Gi")}, // 24GB GPU - matches CEL (>= 8GB)
								},
							},
						},
					},
				},
				// Create a resource claim that allocates gpu-1
				st.MakeResourceClaim().
					Name("test-claim").
					Request(deviceClassName).
					Allocation(&resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1",
									Driver:  driverName,
									Pool:    "per-device-pool",
									Device:  "device-1",
								},
							},
						},
					}).
					Obj(),
			},
			podRequest:          1,
			expectedAllocatable: 2, // Only device-1 (matches test-node) and device-3 (matches all nodes)
			expectedRequested:   2, // 1 allocated (device-1) + 1 requested
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			// Setup environment, create required objects
			logger, tCtx := ktesting.NewTestContext(t)
			tCtx, cancel := context.WithCancel(tCtx)
			defer cancel()

			client := fake.NewClientset(tc.objects...)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			resourceSliceTrackerOpts := tracker.Options{
				SliceInformer: informerFactory.Resource().V1().ResourceSlices(),
				TaintInformer: informerFactory.Resource().V1alpha3().DeviceTaintRules(),
				ClassInformer: informerFactory.Resource().V1().DeviceClasses(),
				KubeClient:    client,
			}
			resourceSliceTracker, err := tracker.StartTracker(tCtx, resourceSliceTrackerOpts)
			require.NoError(t, err, "couldn't start resource slice tracker")
			draManager := dynamicresources.NewDRAManager(
				tCtx,
				assumecache.NewAssumeCache(
					logger,
					informerFactory.Resource().V1().ResourceClaims().Informer(),
					"resource claim",
					"",
					nil),
				resourceSliceTracker,
				informerFactory)

			informerFactory.Start(tCtx.Done())
			t.Cleanup(func() {
				// Now we can wait for all goroutines to stop.
				informerFactory.Shutdown()
			})
			informerFactory.WaitForCacheSync(tCtx.Done())

			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(tc.node)

			scorer := &resourceAllocationScorer{
				enableDRAExtendedResource: tc.enableDRAExtendedResource,
				draManager:                draManager,
				draFeatures:               draFeatures,
				celCache:                  celCache,
			}

			// Test calculateResourceAllocatableRequest API
			allocatable, requested := scorer.calculateResourceAllocatableRequest(tCtx, nodeInfo, tc.extendedResource, tc.podRequest)
			assert.Equal(t, tc.expectedAllocatable, allocatable)
			assert.Equal(t, tc.expectedRequested, requested)
		})
	}
}
