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
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	"k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
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
		name                    string
		pod                     v1.Pod
		useExtraNodeAllocatable bool
		expected                map[v1.ResourceName]int64
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
		{
			name:                    "extra node allocatable resources included when useExtraNodeAllocatable is true",
			useExtraNodeAllocatable: true,
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					ExtraNodeAllocatableResources: makeExtraNodeAllocatableResources("50m"),
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    150,
				v1.ResourceMemory: oneMi,
			},
		},
		{
			name:                    "extra node allocatable resources ignored when useExtraNodeAllocatable is false",
			useExtraNodeAllocatable: false,
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					ExtraNodeAllocatableResources: makeExtraNodeAllocatableResources("50m"),
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    100,
				v1.ResourceMemory: oneMi,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var scorer resourceAllocationScorer
			for n, exp := range tc.expected {
				got := scorer.calculatePodResourceRequest(&tc.pod, n, tc.useExtraNodeAllocatable)
				if got != exp {
					t.Errorf("expected %s = %d, got %d", n, exp, got)
				}
			}
		})
	}
}

func TestCalculateResourceAllocatableRequest(t *testing.T) {
	testCalculateResourceAllocatableRequest(ktesting.Init(t))
}
func testCalculateResourceAllocatableRequest(tCtx ktesting.TContext) {
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
		draObjects                []apiruntime.Object
		expectedAllocatable       int64
		expectedAllocated         int64
	}{
		"device-plugin-resource-feature-disabled": {
			enableDRAExtendedResource: false,
			node:                      st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{explicitExtendedResource: "4"}).Obj(),
			extendedResource:          explicitExtendedResource,
			expectedAllocatable:       4,
			expectedAllocated:         0,
		},
		"device-plugin-resource-feature-enabled": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{explicitExtendedResource: "4"}).Obj(),
			extendedResource:          explicitExtendedResource,
			expectedAllocatable:       4,
			expectedAllocated:         0,
		},
		"DRA-backed-resource-explicit": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice(nodeName, driverName).Device("device-1").Obj(),
			},
			expectedAllocatable: 1, // 1 device `device-1`
			expectedAllocated:   0,
		},
		"DRA-backed-resource-implicit": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          implicitExtendedResource,
			draObjects: []apiruntime.Object{
				&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: deviceClassName,
					},
				},
				st.MakeResourceSlice(nodeName, driverName).Device("device-1").Obj(),
			},
			expectedAllocatable: 1, // 1 device `device-1`
			expectedAllocated:   0,
		},
		"DRA-backed-resource-no-slices": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
			},
			expectedAllocatable: 0,
			expectedAllocated:   0,
		},
		"DRA-backed-resource-with-allocated-device": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
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
			expectedAllocatable: 2, // 2 allocatable devices `device-1`, `device-2`
			expectedAllocated:   1, // 1 allocated by `testClaim`
		},
		"DRA-backed-resource-with-shared-device-allocation": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
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
			expectedAllocatable: 2, // 2 allocatable devices `device-1`, `device-2`
			expectedAllocated:   1, // 1 allocated (shared) by `testClaim`
		},
		"DRA-backed-resource-multiple-devices-mixed-allocation": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
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
			expectedAllocatable: 3, // 3 allocatable devices `device-1`, `device-2`, `device-3`
			expectedAllocated:   2, // 2 allocated (1 full `test-claim-1` + 1 shared `test-claim-2`)
		},
		"DRA-backed-resource-with-per-device-node-selection": {
			enableDRAExtendedResource: true,
			node:                      st.MakeNode().Name(nodeName).Label("zone", "us-east-1a").Obj(),
			extendedResource:          explicitExtendedResource,
			draObjects: []apiruntime.Object{
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
								Name: "device-3",
								// Use a node selector to ensure nodeMatches is exercised for this device
								NodeSelector: &v1.NodeSelector{
									NodeSelectorTerms: []v1.NodeSelectorTerm{
										{
											MatchExpressions: []v1.NodeSelectorRequirement{
												{
													Key:      "zone",
													Operator: v1.NodeSelectorOpIn,
													Values:   []string{"us-east-1a"},
												},
											},
										},
									},
								},
								Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
									"model": {StringValue: ptr.To("SOME-XZY")},
								},
								Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
									"memory": {Value: resource.MustParse("24Gi")}, // 24GB GPU - matches CEL (>= 8GiB)
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
			expectedAllocatable: 2, // device-1 matches the test node and device-3 matches via its selector
			expectedAllocated:   1, // 1 allocated (device-1)
		},
	}

	for name, tc := range tests {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			// Setup environment, create required objects
			if !tc.enableDRAExtendedResource {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(tCtx, utilfeature.DefaultFeatureGate, version.MustParse("1.36"))
			}
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, tc.enableDRAExtendedResource)

			draManager := newTestDRAManager(tCtx, tc.draObjects...)

			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(tc.node)

			scorer := &resourceAllocationScorer{
				enableDRAExtendedResource: tc.enableDRAExtendedResource,
				draManager:                draManager,
				draFeatures:               draFeatures,
				DRACaches: DRACaches{
					celCache: celCache,
				},
			}

			var draPreScoreState *draPreScoreState
			if tc.enableDRAExtendedResource {
				var status *fwk.Status
				draPreScoreState, status = getDRAPreScoredParams(draManager, []config.ResourceSpec{{Name: string(tc.extendedResource)}})
				if status != nil {
					tCtx.Fatalf("getting DRA pre-scored params failed: %v", status)
				}
			}

			// Test calculateResourceAllocatableRequest API
			allocatable, allocated := scorer.calculateResourceAllocatableRequest(tCtx, nodeInfo, tc.extendedResource, draPreScoreState)
			if !cmp.Equal(allocatable, tc.expectedAllocatable) {
				tCtx.Errorf("Expected allocatable=%v, but got %v", tc.expectedAllocatable, allocatable)
			}
			if !cmp.Equal(allocated, tc.expectedAllocated) {
				tCtx.Errorf("Expected allocated=%v, but got %v", tc.expectedAllocated, allocated)
			}
		})
	}
}

// newTestDRAManager creates a DefaultDRAManager for testing purposes.
// Only usable in a syntest bubble.
func newTestDRAManager(tCtx ktesting.TContext, objects ...apiruntime.Object) *dynamicresources.DefaultDRAManager {
	tCtx = tCtx.WithCancel()
	client := fake.NewClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sliceInformer := draapi.NewInformerForResourceSlice(informerFactory)
	resourceSliceTrackerOpts := tracker.Options{
		SliceLister:   draapi.NewResourceSliceLister(sliceInformer.GetIndexer()),
		SliceInformer: sliceInformer,
		KubeClient:    client,
	}
	resourceSliceTracker, err := tracker.StartTracker(tCtx, resourceSliceTrackerOpts)
	tCtx.ExpectNoError(err, "couldn't start resource slice tracker")
	logger := tCtx.Logger()
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

	handlerOptions := cache.HandlerOptions{Logger: &logger}
	cache := draManager.DeviceClassResolver().(*extendedresourcecache.ExtendedResourceCache)
	handle, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandlerWithOptions(cache, handlerOptions)
	tCtx.ExpectNoError(err, "add device class informer event handler")
	tCtx.Cleanup(func() {
		_ = informerFactory.Resource().V1().DeviceClasses().Informer().RemoveEventHandler(handle)
	})

	informerFactory.StartWithContext(tCtx)
	tCtx.Cleanup(func() {
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSyncWithContext(tCtx)

	// Wait for full initialization of manager, including
	// processing of all informer events.
	tCtx.Wait()

	return draManager
}

func TestCalculateNodeAllocatableRequestSaturatesRequested(t *testing.T) {
	// Adding the scored pod to a node total already at MaxInt64 must not wrap it
	// negative, which would read as free capacity.
	tCtx := ktesting.Init(t)

	node := st.MakeNode().Name("saturated-node").Capacity(map[v1.ResourceName]string{
		v1.ResourceCPU:    "8",
		v1.ResourceMemory: "16Gi",
	}).Obj()

	// Both requests project to MaxInt64, so the node totals saturate.
	hugePod := st.MakePod().Name("huge").Namespace("ns").UID("huge").
		Req(map[v1.ResourceName]string{
			v1.ResourceCPU:    "9223372036854775807",
			v1.ResourceMemory: "9223372036854775807",
		}).Obj()

	nodeInfo := framework.NewNodeInfo(hugePod)
	nodeInfo.SetNode(node)

	if got := nodeInfo.GetNonZeroRequested().GetMilliCPU(); got != math.MaxInt64 {
		t.Fatalf("precondition: node requested milliCPU = %d, want MaxInt64 (NodeInfo did not saturate)", got)
	}
	if got := nodeInfo.GetNonZeroRequested().GetMemory(); got != math.MaxInt64 {
		t.Fatalf("precondition: node requested memory = %d, want MaxInt64 (NodeInfo did not saturate)", got)
	}

	scorer := &resourceAllocationScorer{
		resources: []config.ResourceSpec{
			{Name: string(v1.ResourceCPU)},
			{Name: string(v1.ResourceMemory)},
		},
	}

	requested, allocated, _ := scorer.calculateNodeAllocatableRequest(tCtx, nodeInfo, []int64{1, 1}, nil)

	for i, name := range []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory} {
		if allocated[i] != math.MaxInt64 {
			t.Errorf("%s: allocated = %d, want MaxInt64", name, allocated[i])
		}
		if requested[i] != math.MaxInt64 {
			t.Errorf("%s: requested = %d, want MaxInt64; a wrap to a negative reads as spare capacity", name, requested[i])
		}
	}
}

// TestDRAMatchCachesAcrossCycles verifies that changing selector inputs does not
// reuse stale results, while unchanged inputs remain cached across cycles.
func TestDRAMatchCachesAcrossCycles(t *testing.T) {
	testDRAMatchCachesAcrossCycles(ktesting.Init(t))
}

func testDRAMatchCachesAcrossCycles(tCtx ktesting.TContext) {
	const (
		nodeName   = "match-cache-node"
		driverName = "test-driver"
	)
	resource := v1.ResourceName(extendedResourceName)
	celClass := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: "cel-class"},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &extendedResourceName,
			Selectors: []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{
					Expression: `device.driver == "test-driver" && device.attributes["test-driver"].model == "SOME-GOOD"`,
				},
			}},
		},
	}
	draFeatures := structured.Features{
		AdminAccess:          true,
		PrioritizedList:      true,
		PartitionableDevices: true,
		DeviceTaints:         true,
		ConsumableCapacity:   true,
	}
	var generation int64
	sliceWithModel := func(model string) *resourceapi.ResourceSlice {
		slice := st.MakeResourceSlice(nodeName, driverName).Obj()
		generation++
		slice.UID = "match-cache-slice"
		slice.Generation = generation
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
			Name: "device-1",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"model": {StringValue: new(model)},
			},
		})
		return slice
	}
	pod := st.MakePod().Name("pod").Obj()
	podWithRequests := st.MakePod().Name("pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "100m"}).Obj()
	node := st.MakeNode().Name(nodeName).Obj()
	enableFeature := func(tCtx ktesting.TContext) {
		featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
	}
	capacityViaFit := func(tCtx ktesting.TContext, f *Fit, pod *v1.Pod, node *v1.Node) int64 {
		cycleState := framework.NewCycleState()
		if status := f.PreScore(tCtx, cycleState, pod, nil); !status.IsSuccess() {
			tCtx.Fatalf("PreScore failed: %v", status)
		}
		state, err := getPreScoreState(cycleState)
		if err != nil {
			tCtx.Fatal(err)
		}
		nodeInfo := framework.NewNodeInfo()
		nodeInfo.SetNode(node)
		allocatable, _ := f.calculateResourceAllocatableRequest(tCtx, nodeInfo, resource, state.draPreScoreState)
		return allocatable
	}
	capacityViaBalancedAllocation := func(tCtx ktesting.TContext, ba *BalancedAllocation, pod *v1.Pod, node *v1.Node) int64 {
		cycleState := framework.NewCycleState()
		if status := ba.PreScore(tCtx, cycleState, pod, nil); !status.IsSuccess() {
			tCtx.Fatalf("PreScore failed: %v", status)
		}
		state, err := getBalancedAllocationPreScoreState(cycleState)
		if err != nil {
			tCtx.Fatal(err)
		}
		nodeInfo := framework.NewNodeInfo()
		nodeInfo.SetNode(node)
		allocatable, _ := ba.calculateResourceAllocatableRequest(tCtx, nodeInfo, resource, state.draPreScoreState)
		return allocatable
	}

	tCtx.SyncTest("device attributes changed between cycles", func(tCtx ktesting.TContext) {
		enableFeature(tCtx)
		f := &Fit{
			enableDRAExtendedResource: true,
			resourceAllocationScorer: &resourceAllocationScorer{
				enableDRAExtendedResource: true,
				draManager:                newTestDRAManager(tCtx, celClass, sliceWithModel("SOME-GOOD")),
				draFeatures:               draFeatures,
				resources:                 []config.ResourceSpec{{Name: string(resource)}},
				DRACaches:                 DRACaches{celCache: cel.NewCache(10, cel.Features{})},
			},
		}
		if got := capacityViaFit(tCtx, f, pod, node); got != 1 {
			tCtx.Errorf("first cycle: expected capacity 1, got %d", got)
		}
		before, ok := f.deviceMatchCache.Load(types.UID("match-cache-slice"))
		if !ok {
			tCtx.Fatal("first cycle did not cache the slice")
		}
		if got := capacityViaFit(tCtx, f, pod, node); got != 1 {
			tCtx.Errorf("unchanged cycle: capacity %d, want 1", got)
		}
		after, _ := f.deviceMatchCache.Load(types.UID("match-cache-slice"))
		if before != after {
			tCtx.Error("PreScore discarded unchanged slice results")
		}

		// The driver re-publishes the slice for the same pool with a changed
		// device attribute, so the device no longer matches the class.
		f.draManager = newTestDRAManager(tCtx, celClass, sliceWithModel("SOME-BAD"))
		if got := capacityViaFit(tCtx, f, pod, node); got != 0 {
			tCtx.Errorf("after attribute change: expected capacity 0, got %d (stale match result reused)", got)
		}
	})

	tCtx.SyncTest("node labels changed between cycles", func(tCtx ktesting.TContext) {
		enableFeature(tCtx)
		slice := st.MakeResourceSliceWithPerDeviceNodeSelection("match-cache-pool", driverName).Obj()
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
			Name: "device-1",
			NodeSelector: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{{
					MatchExpressions: []v1.NodeSelectorRequirement{{
						Key:      "zone",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"us-east-1a"},
					}},
				}},
			},
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"model": {StringValue: new("SOME-GOOD")},
			},
		})
		draManager := newTestDRAManager(tCtx, celClass, slice)
		f := &Fit{
			enableDRAExtendedResource: true,
			resourceAllocationScorer: &resourceAllocationScorer{
				enableDRAExtendedResource: true,
				draManager:                draManager,
				draFeatures:               draFeatures,
				resources:                 []config.ResourceSpec{{Name: string(resource)}},
				DRACaches:                 DRACaches{celCache: cel.NewCache(10, cel.Features{})},
			},
		}
		ba := &BalancedAllocation{resourceAllocationScorer: resourceAllocationScorer{
			enableDRAExtendedResource: true,
			draManager:                draManager,
			draFeatures:               draFeatures,
			resources:                 []config.ResourceSpec{{Name: string(v1.ResourceCPU)}, {Name: string(resource)}},
			DRACaches:                 DRACaches{celCache: cel.NewCache(10, cel.Features{})},
		}}
		for _, step := range []struct {
			label string
			want  int64
		}{
			{"", 0}, {"us-east-1a", 1}, {"us-east-1b", 0}, {"us-east-1a", 1}, {"", 0},
		} {
			node := st.MakeNode().Name(nodeName).Obj()
			if step.label != "" {
				node.Labels = map[string]string{"zone": step.label}
			}
			if got := capacityViaFit(tCtx, f, pod, node); got != step.want {
				tCtx.Errorf("Fit with zone %q: capacity %d, want %d", step.label, got, step.want)
			}
			if got := capacityViaBalancedAllocation(tCtx, ba, podWithRequests, node); got != step.want {
				tCtx.Errorf("BalancedAllocation with zone %q: capacity %d, want %d", step.label, got, step.want)
			}
		}
	})

	tCtx.SyncTest("device attributes changed between cycles (balanced allocation)", func(tCtx ktesting.TContext) {
		enableFeature(tCtx)
		// The pod has a CPU request, so BalancedAllocation must not skip it
		// as best-effort (it checks the requests of its own resource list).
		ba := &BalancedAllocation{
			resourceAllocationScorer: resourceAllocationScorer{
				enableDRAExtendedResource: true,
				draManager:                newTestDRAManager(tCtx, celClass, sliceWithModel("SOME-GOOD")),
				draFeatures:               draFeatures,
				resources:                 []config.ResourceSpec{{Name: string(v1.ResourceCPU)}, {Name: string(resource)}},
				DRACaches:                 DRACaches{celCache: cel.NewCache(10, cel.Features{})},
			},
		}
		if got := capacityViaBalancedAllocation(tCtx, ba, podWithRequests, node); got != 1 {
			tCtx.Errorf("first cycle: expected capacity 1, got %d", got)
		}
		before, ok := ba.deviceMatchCache.Load(types.UID("match-cache-slice"))
		if !ok {
			tCtx.Fatal("first cycle did not cache the slice")
		}
		if got := capacityViaBalancedAllocation(tCtx, ba, podWithRequests, node); got != 1 {
			tCtx.Errorf("unchanged cycle: capacity %d, want 1", got)
		}
		after, _ := ba.deviceMatchCache.Load(types.UID("match-cache-slice"))
		if before != after {
			tCtx.Error("PreScore discarded unchanged slice results")
		}

		ba.draManager = newTestDRAManager(tCtx, celClass, sliceWithModel("SOME-BAD"))
		if got := capacityViaBalancedAllocation(tCtx, ba, podWithRequests, node); got != 0 {
			tCtx.Errorf("after attribute change: expected capacity 0, got %d (stale match result reused)", got)
		}
	})
}

func TestDRAMatchCachesWithoutPreScore(t *testing.T) {
	tCtx := ktesting.Init(t)
	for _, balanced := range []bool{false, true} {
		name := "fit"
		if balanced {
			name = "balanced allocation"
		}
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			class := &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{Name: "fallback-class"},
				Spec: resourceapi.DeviceClassSpec{
					ExtendedResourceName: &extendedResourceName,
					Selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{
						Expression: `device.attributes["test-driver"].model == "GOOD"`,
					}}},
				},
			}
			scorer := &resourceAllocationScorer{
				enableDRAExtendedResource: true,
				resources:                 []config.ResourceSpec{{Name: string(v1.ResourceCPU)}, {Name: extendedResourceName}},
				// Expose DRA capacity through Score so the assertion does not depend on a scoring strategy.
				scorer:    func(_, _, allocatable []int64) int64 { return allocatable[1] },
				DRACaches: DRACaches{celCache: cel.NewCache(10, cel.Features{})},
			}
			pod := st.MakePod().Name("pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "100m", v1.ResourceName(extendedResourceName): "1"}).Obj()
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(st.MakeNode().Name("fallback-node").Obj())
			fit := &Fit{enableDRAExtendedResource: true, resourceAllocationScorer: scorer}
			ba := &BalancedAllocation{resourceAllocationScorer: resourceAllocationScorer{
				enableDRAExtendedResource: true,
				resources:                 scorer.resources,
				scorer:                    scorer.scorer,
				DRACaches:                 DRACaches{celCache: scorer.celCache},
			}}
			for i, tc := range []struct {
				model string
				want  int64
			}{{"GOOD", 1}, {"BAD", 0}, {"GOOD", 1}} {
				slice := st.MakeResourceSlice("fallback-node", "test-driver").Obj()
				slice.Spec.Devices = []resourceapi.Device{{
					Name: "device-1",
					Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"model": {StringValue: new(tc.model)},
					},
				}}
				slice.UID = "fallback-slice"
				slice.Generation = int64(i + 1)
				manager := newTestDRAManager(tCtx, class, slice)
				fit.draManager = manager
				ba.draManager = manager
				var workers sync.WaitGroup
				for range 8 {
					workers.Go(func() {
						state := framework.NewCycleState()
						var got int64
						var status *fwk.Status
						if balanced {
							got, status = ba.Score(tCtx, state, pod, nodeInfo)
						} else {
							got, status = fit.Score(tCtx, state, pod, nodeInfo)
						}
						if !status.IsSuccess() {
							tCtx.Errorf("Score failed: %v", status)
						}
						if got != tc.want {
							tCtx.Errorf("model %s: expected capacity %d, got %d", tc.model, tc.want, got)
						}
					})
				}
				workers.Wait()
			}
		})
	}
}

func TestDeviceMatchCacheVersions(t *testing.T) {
	for _, capacitySelector := range []bool{false, true} {
		t.Run(fmt.Sprintf("capacity-selector=%t", capacitySelector), func(t *testing.T) {
			scorer := &resourceAllocationScorer{DRACaches: DRACaches{celCache: cel.NewCache(10, cel.Features{})}}
			expression := `device.attributes["test-driver"].model == "good"`
			if capacitySelector {
				expression = `device.capacity["test-driver"].memory.compareTo(quantity("8Gi")) >= 0`
			}
			class := &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{
					Selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{Expression: expression}}},
				},
			}
			makeSlice := func(uid types.UID, generation int64, matches bool) *draapi.ResourceSlice {
				external := st.MakeResourceSlice("node", "test-driver").Obj()
				external.UID, external.Generation = uid, generation
				model, capacity := "bad", "4Gi"
				if matches {
					model, capacity = "good", "16Gi"
				}
				external.Spec.Devices = []resourceapi.Device{{
					Name: "gpu",
					Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"model": {StringValue: &model},
					},
					Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
						"memory": {Value: resource.MustParse(capacity)},
					},
				}}
				internal := &draapi.ResourceSlice{}
				if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(external, internal, nil); err != nil {
					t.Fatal(err)
				}
				return internal
			}
			check := func(slice *draapi.ResourceSlice, want bool) {
				t.Helper()
				got, err := scorer.deviceMatchesClass(context.Background(), slice, slice.Spec.Devices[0], class)
				if err != nil || got != want {
					t.Fatalf("UID %s generation %d: match = %v, err = %v, want %v", slice.UID, slice.Generation, got, err, want)
				}
			}
			// Slices without UIDs must not share results, even with identical keys.
			for _, matches := range []bool{true, false, true} {
				check(makeSlice("", 1, matches), matches)
			}
			if _, ok := scorer.deviceMatchCache.Load(types.UID("")); ok {
				t.Fatal("slice without UID retained cached results")
			}
			old := makeSlice("slice-a", 1, true)
			check(old, true)
			oldResults := scorer.matchResultsForSlice(old)
			scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: old.UID, Generation: old.Generation - 1}})
			check(old, true)
			if scorer.matchResultsForSlice(old) != oldResults {
				t.Fatal("unchanged slice lost cached results")
			}
			latest := makeSlice(old.UID, 2, false)
			check(latest, false)
			latestResults := scorer.matchResultsForSlice(latest)
			check(old, true)
			scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: old.UID, Generation: old.Generation - 1}})
			if scorer.matchResultsForSlice(latest) != latestResults {
				t.Fatal("older snapshot evicted newer generation")
			}
			scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: latest.UID, Generation: latest.Generation - 1}})
			if v, _ := scorer.deviceMatchCache.Load(old.UID); v != latestResults {
				t.Fatal("current generation was not retained")
			}
			key := deviceMatchKey{expression: expression, device: latest.Spec.Devices[0].Name}
			if v, ok := latestResults.results.Load(key); !ok || v != false {
				t.Fatal("negative match was not cached")
			}
			scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: old.UID, Generation: latest.Generation}})
			if _, ok := scorer.deviceMatchCache.Load(old.UID); ok {
				t.Fatal("deleted slice retained")
			}
			recreated := makeSlice("slice-b", 1, true)
			check(recreated, true)
			for gen := int64(2); gen <= 100; gen++ {
				slice := makeSlice(recreated.UID, gen, gen%2 == 0)
				check(slice, gen%2 == 0)
				scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: slice.UID, Generation: slice.Generation - 1}})
			}
			count := 0
			scorer.deviceMatchCache.Range(func(_, _ any) bool { count++; return true })
			if count != 1 {
				t.Fatalf("after repeated updates: got %d slices, want 1", count)
			}
			scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: recreated.UID, Generation: 100}})
			count = 0
			scorer.deviceMatchCache.Range(func(_, _ any) bool { count++; return true })
			if count != 0 {
				t.Fatalf("after deletion: got %d slices, want 0", count)
			}
		})
	}
}

func TestDRAMatchCachesConcurrentSnapshots(t *testing.T) {
	scorer := &resourceAllocationScorer{DRACaches: DRACaches{celCache: cel.NewCache(10, cel.Features{})}}
	class := &resourceapi.DeviceClass{
		Spec: resourceapi.DeviceClassSpec{
			Selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{
				Expression: `device.attributes["test-driver"].enabled`,
			}}},
		},
	}
	snapshots := make([]*draapi.ResourceSlice, 2)
	for i := range snapshots {
		external := st.MakeResourceSlice("node", "test-driver").Obj()
		external.UID, external.Generation = "shared-slice", int64(i+1)
		external.Spec.Devices = []resourceapi.Device{{
			Name: "gpu",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"enabled": {BoolValue: new(i == 0)},
			},
		}}
		snapshots[i] = &draapi.ResourceSlice{}
		if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(external, snapshots[i], nil); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	for worker := range 16 {
		wg.Go(func() {
			for j := range 100 {
				i := (worker + j) % 2
				slice := snapshots[i]
				got, err := scorer.deviceMatchesClass(context.Background(), slice, slice.Spec.Devices[0], class)
				if err != nil || got != (i == 0) {
					t.Errorf("generation %d: match = %v, err = %v", slice.Generation, got, err)
					return
				}
				scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: snapshots[1].UID, Generation: snapshots[1].Generation - 1}})
			}
		})
	}
	wg.Wait()
	scorer.matchResultsForSlice(snapshots[1])
	scorer.invalidateDeviceMatchCache(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{UID: snapshots[0].UID, Generation: snapshots[0].Generation - 1}})
	if value, _ := scorer.deviceMatchCache.Load(snapshots[1].UID); value.(*sliceMatchResults).generation != 2 {
		t.Fatal("older snapshot replaced or pruned newer generation")
	}
}

func TestDeviceMatchCacheEvents(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	slice := &resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "slice", UID: "uid", Generation: 10}}
	client := fake.NewClientset(slice)
	factory := informers.NewSharedInformerFactory(client, 0)
	informer := factory.Resource().V1().ResourceSlices().TypedInformer()
	caches := []*DRACaches{{}, {}}
	for _, c := range caches {
		if err := c.watchResourceSlices(ctx, informer); err != nil {
			t.Fatal(err)
		}
	}
	factory.Start(ctx.Done())
	defer func() { cancel(); factory.Shutdown() }()
	if !cache.WaitForCacheSync(ctx.Done(), informer.HasSynced) {
		t.Fatal("ResourceSlice informer did not sync")
	}
	snapshot := &draapi.ResourceSlice{ObjectMeta: slice.ObjectMeta}
	oldResults := make([]*sliceMatchResults, len(caches))
	for i, c := range caches {
		oldResults[i] = c.matchResultsForSlice(snapshot)
		if c.matchResultsForSlice(snapshot) != oldResults[i] {
			t.Fatal("stable slice lost cached results")
		}
	}
	latest := slice.DeepCopy()
	latest.Generation++
	if _, err := client.ResourceV1().ResourceSlices().Update(ctx, latest, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	awaitEmpty := func() {
		t.Helper()
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			for _, c := range caches {
				if _, ok := c.deviceMatchCache.Load(slice.UID); ok {
					return false, nil
				}
			}
			return true, nil
		}); err != nil {
			t.Fatalf("cache invalidation was not delivered: %v", err)
		}
	}
	awaitEmpty()
	snapshot.Generation++
	for _, c := range caches {
		current := c.matchResultsForSlice(snapshot)
		c.invalidateDeviceMatchCache(slice)
		if c.matchResultsForSlice(snapshot) != current {
			t.Fatal("delayed old event removed newer results")
		}
	}
	if err := client.ResourceV1().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}
	awaitEmpty()
	var workers sync.WaitGroup
	for _, c := range caches {
		workers.Go(func() {
			for range 100 {
				c.matchResultsForSlice(snapshot)
				c.invalidateDeviceMatchCache(latest)
			}
		})
	}
	workers.Wait()
	awaitEmpty()
	// A worker publishing after the final delete notification must clean up itself.
	for _, c := range caches {
		c.matchResultsForSlice(snapshot)
		if _, ok := c.deviceMatchCache.Load(slice.UID); ok {
			t.Fatal("old snapshot resurrected deleted UID")
		}
	}
	recreated := latest.DeepCopy()
	recreated.UID = "new-uid"
	recreated.Generation = 1
	if _, err := client.ResourceV1().ResourceSlices().Create(ctx, recreated, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		obj, exists, err := informer.GetStore().GetByKey(slice.Name)
		return exists && obj.(*resourceapi.ResourceSlice).UID == recreated.UID, err
	}); err != nil {
		t.Fatal(err)
	}
	for i, c := range caches {
		result := c.matchResultsForSlice(&draapi.ResourceSlice{ObjectMeta: recreated.ObjectMeta})
		if result == oldResults[i] {
			t.Fatal("recreated slice reused deleted results")
		}
		if _, ok := c.deviceMatchCache.Load(recreated.UID); !ok {
			t.Fatal("recreated slice was not cached")
		}
	}
	cancel()
}
