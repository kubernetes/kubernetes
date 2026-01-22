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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	"k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/utils/ktesting"
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
			node:                      st.MakeNode().Name(nodeName).Label("zone", "us-east-1a").Obj(),
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
			podRequest:          1,
			expectedAllocatable: 2, // device-1 matches the test node and device-3 matches via its selector
			expectedRequested:   2, // 1 allocated (device-1) + 1 requested
		},
	}

	for name, tc := range tests {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			// Setup environment, create required objects
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, tc.enableDRAExtendedResource)

			draManager := newTestDRAManager(tCtx, tc.objects...)

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
			allocatable, requested := scorer.calculateResourceAllocatableRequest(tCtx, nodeInfo, tc.extendedResource, tc.podRequest, draPreScoreState)
			if !cmp.Equal(allocatable, tc.expectedAllocatable) {
				tCtx.Errorf("Expected allocatable=%v, but got allocatable=%v", tc.expectedAllocatable, allocatable)
			}
			if !cmp.Equal(requested, tc.expectedRequested) {
				tCtx.Errorf("Expected requested=%v, but got requested=%v", tc.expectedRequested, requested)
			}
		})
	}
}

// newTestDRAManager creates a DefaultDRAManager for testing purposes.
// Only usable in a syntest bubble.
func newTestDRAManager(tCtx ktesting.TContext, objects ...apiruntime.Object) *dynamicresources.DefaultDRAManager {
	tCtx = ktesting.WithCancel(tCtx)
	client := fake.NewClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	resourceSliceTrackerOpts := tracker.Options{
		SliceInformer: informerFactory.Resource().V1().ResourceSlices(),
		TaintInformer: informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		ClassInformer: informerFactory.Resource().V1().DeviceClasses(),
		KubeClient:    client,
	}
	resourceSliceTracker, err := tracker.StartTracker(tCtx, resourceSliceTrackerOpts)
	tCtx.ExpectNoError(err, "couldn't start resource slice tracker")
	draManager := dynamicresources.NewDRAManager(
		tCtx,
		assumecache.NewAssumeCache(
			tCtx.Logger(),
			informerFactory.Resource().V1().ResourceClaims().Informer(),
			"resource claim",
			"",
			nil),
		resourceSliceTracker,
		informerFactory)

	cache := draManager.DeviceClassResolver().(*extendedresourcecache.ExtendedResourceCache)
	handle, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(cache)
	tCtx.ExpectNoError(err, "add device class informer event handler")
	tCtx.Cleanup(func() {
		_ = informerFactory.Resource().V1().DeviceClasses().Informer().RemoveEventHandler(handle)
	})

	informerFactory.Start(tCtx.Done())
	tCtx.Cleanup(func() {
		tCtx.Cancel("test has completed")
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())

	// Wait for full initialization of manager, including
	// processing of all informer events.
	tCtx.Wait()

	return draManager
}

// getCachedDeviceMatch checks the cache for a DeviceMatches result
// returns (matches, found)
func (r *resourceAllocationScorer) getCachedDeviceMatch(expression string, driver string, poolName string, deviceName string) (bool, bool) {
	key := buildDeviceMatchCacheKey(expression, driver, poolName, deviceName)

	if value, ok := r.deviceMatchCache.Load(key); ok {
		return value.(bool), true
	}

	return false, false
}

// setCachedDeviceMatch stores a DeviceMatches result in the cache
func (r *resourceAllocationScorer) setCachedDeviceMatch(expression string, driver string, poolName string, deviceName string, matches bool) {
	key := buildDeviceMatchCacheKey(expression, driver, poolName, deviceName)
	r.deviceMatchCache.Store(key, matches)
}

func TestDeviceMatchCaching(t *testing.T) {
	// Create a scorer with caching enabled
	scorer := &resourceAllocationScorer{
		DRACaches: DRACaches{
			celCache: cel.NewCache(10, cel.Features{}),
		},
	}

	expression := `device.attributes["example.com"].test_attr == "test-value"`
	driverName := "example.com/test-driver"
	poolName := "example-pool"
	deviceName := "device-1"

	// Test cache operations
	// Initially, cache should be empty
	matches, found := scorer.getCachedDeviceMatch(expression, driverName, poolName, deviceName)
	if found {
		t.Errorf("Cache should be empty initially, but found an entry")
	}
	if matches {
		t.Errorf("Matches should be false initially, but got true")
	}

	// Store a result in cache
	scorer.setCachedDeviceMatch(expression, driverName, poolName, deviceName, true)

	// Retrieve from cache
	matches, found = scorer.getCachedDeviceMatch(expression, driverName, poolName, deviceName)
	if !found {
		t.Errorf("Result should be found in cache, but was not")
	}
	if !matches {
		t.Errorf("Cached result should match what we stored, expected true but got false")
	}

	// Test caching with error
	scorer.setCachedDeviceMatch(expression, driverName, poolName, deviceName, false)

	matches, found = scorer.getCachedDeviceMatch(expression, driverName, poolName, deviceName)
	if !found {
		t.Errorf("Result should be found in cache")
	}
	if matches {
		t.Errorf("Cached result should match what we stored, expected false but got true")
	}

	// Test that different devices have different cache keys
	matches, found = scorer.getCachedDeviceMatch(expression, driverName, poolName, "device-2")
	if found {
		t.Errorf("Different device should not hit cache")
	}
	if matches {
		t.Errorf("Matches should be false for uncached entry")
	}

	// Test that different pools have different cache keys
	matches, found = scorer.getCachedDeviceMatch(expression, driverName, "other-pool", deviceName)
	if found {
		t.Errorf("Different pool should not hit cache")
	}
	if matches {
		t.Errorf("Matches should be false for uncached entry")
	}
}

func BenchmarkDeviceMatchCaching(b *testing.B) {
	expression := `device.attributes["example.com"].test_attr == "test-value"`
	driverName := "example.com/test-driver"
	poolName := "example-pool"
	deviceName := "device-1"

	// Create a scorer with caching enabled
	scorer := &resourceAllocationScorer{
		DRACaches: DRACaches{
			celCache: cel.NewCache(10, cel.Features{}),
		},
	}

	b.Run("WithoutCache", func(b *testing.B) {
		// Disable caching by creating a new scorer each time
		for i := 0; i < b.N; i++ {
			freshScorer := &resourceAllocationScorer{
				DRACaches: DRACaches{
					celCache: cel.NewCache(10, cel.Features{}),
				},
			}

			// This will always be a cache miss
			_, _ = freshScorer.getCachedDeviceMatch(expression, driverName, poolName, deviceName)
		}
	})

	b.Run("WithCache", func(b *testing.B) {
		// Pre-warm the cache
		scorer.setCachedDeviceMatch(expression, driverName, poolName, deviceName, true)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// This should always be a cache hit
			_, _ = scorer.getCachedDeviceMatch(expression, driverName, poolName, deviceName)
		}
	})
}

// getCachedNodeMatch checks the cache for a NodeMatches result
func (r *resourceAllocationScorer) getCachedNodeMatch(nodeName string, nodeNameToMatch string, allNodesMatch bool, nodeSelectorHash string) (bool, bool) {
	key := buildNodeMatchCacheKey(nodeName, nodeNameToMatch, allNodesMatch, nodeSelectorHash)

	if value, ok := r.nodeMatchCache.Load(key); ok {
		return value.(bool), true
	}

	return false, false
}

// setCachedNodeMatch stores a NodeMatches result in the cache
func (r *resourceAllocationScorer) setCachedNodeMatch(nodeName string, nodeNameToMatch string, allNodesMatch bool, nodeSelectorHash string, matches bool) {
	key := buildNodeMatchCacheKey(nodeName, nodeNameToMatch, allNodesMatch, nodeSelectorHash)
	r.nodeMatchCache.Store(key, matches)
}

func TestNodeMatchCaching(t *testing.T) {
	// Create a scorer with caching enabled
	scorer := &resourceAllocationScorer{
		DRACaches: DRACaches{
			celCache: cel.NewCache(10, cel.Features{}),
		},
	}

	// Create test node
	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node-1",
			Labels: map[string]string{
				"zone":          "us-east-1a",
				"instance.type": "gpu-xlarge",
			},
		},
	}

	// Test cases for different NodeMatches scenarios
	testCases := []struct {
		name            string
		nodeNameToMatch string
		allNodesMatch   bool
		nodeSelector    *v1.NodeSelector
		expectedMatch   bool
	}{
		{
			name:            "exact node name match",
			nodeNameToMatch: "test-node-1",
			allNodesMatch:   false,
			nodeSelector:    nil,
			expectedMatch:   true,
		},
		{
			name:            "node name mismatch",
			nodeNameToMatch: "different-node",
			allNodesMatch:   false,
			nodeSelector:    nil,
			expectedMatch:   false,
		},
		{
			name:            "all nodes match",
			nodeNameToMatch: "",
			allNodesMatch:   true,
			nodeSelector:    nil,
			expectedMatch:   true,
		},
		{
			name:            "node selector match",
			nodeNameToMatch: "",
			allNodesMatch:   false,
			nodeSelector: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "zone",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{"us-east-1a", "us-east-1b"},
							},
						},
					},
				},
			},
			expectedMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nodeSelectorStr := tc.nodeSelector.String()

			// First call should be a cache miss
			_, found1 := scorer.getCachedNodeMatch(testNode.Name, tc.nodeNameToMatch, tc.allNodesMatch, nodeSelectorStr)
			assert.False(t, found1, "Cache should be empty initially")

			// Simulate setting a cached result
			scorer.setCachedNodeMatch(testNode.Name, tc.nodeNameToMatch, tc.allNodesMatch, nodeSelectorStr, tc.expectedMatch)

			// Second call should be a cache hit
			matches2, found2 := scorer.getCachedNodeMatch(testNode.Name, tc.nodeNameToMatch, tc.allNodesMatch, nodeSelectorStr)
			assert.True(t, found2, "Result should be found in cache")
			assert.Equal(t, tc.expectedMatch, matches2, "Cached result should match expected value")
		})
	}
}

func TestNodeMatchesCacheHit(t *testing.T) {
	scorer := &resourceAllocationScorer{
		DRACaches: DRACaches{
			celCache: cel.NewCache(1, cel.Features{}),
		},
	}

	node := st.MakeNode().Name("cache-node").Label("zone", "us-east-1a").Obj()
	selector := &v1.NodeSelector{
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
	}

	firstMatch, err := scorer.nodeMatches(node, "", false, selector)
	if err != nil {
		t.Fatalf("unexpected error while evaluating selector: %v", err)
	}
	if !firstMatch {
		t.Fatalf("expected nodeMatches to return true for the selector")
	}

	secondMatch, err := scorer.nodeMatches(node, "", false, selector)
	if err != nil {
		t.Fatalf("unexpected error while hitting cache: %v", err)
	}
	if !secondMatch {
		t.Fatalf("expected cached nodeMatches to be true")
	}
}

func BenchmarkNodeMatchCaching(b *testing.B) {
	// Create test node
	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node-1",
			Labels: map[string]string{
				"zone":          "us-east-1a",
				"instance.type": "gpu-xlarge",
			},
		},
	}

	// Create a complex NodeSelector to make the test meaningful
	nodeSelector := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      "zone",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"us-east-1a", "us-east-1b", "us-west-2a"},
					},
					{
						Key:      "instance.type",
						Operator: v1.NodeSelectorOpExists,
					},
				},
			},
		},
	}

	b.Run("WithoutCache", func(b *testing.B) {
		nodeSelectorStr := nodeSelector.String()

		// Create a new scorer for each iteration to avoid caching
		for i := 0; i < b.N; i++ {
			freshScorer := &resourceAllocationScorer{
				DRACaches: DRACaches{
					celCache: cel.NewCache(10, cel.Features{}),
				},
			}

			// This will always be a cache miss
			_, _ = freshScorer.getCachedNodeMatch(testNode.Name, "", false, nodeSelectorStr)
		}
	})

	b.Run("WithCache", func(b *testing.B) {
		nodeSelectorStr := nodeSelector.String()

		// Create a scorer and pre-warm the cache
		scorer := &resourceAllocationScorer{
			DRACaches: DRACaches{
				celCache: cel.NewCache(10, cel.Features{}),
			},
		}

		// Pre-warm the cache
		scorer.setCachedNodeMatch(testNode.Name, "", false, nodeSelectorStr, true)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// This should always be a cache hit
			_, _ = scorer.getCachedNodeMatch(testNode.Name, "", false, nodeSelectorStr)
		}
	})
}
