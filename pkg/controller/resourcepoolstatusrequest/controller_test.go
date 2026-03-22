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

package resourcepoolstatusrequest

import (
	"fmt"
	"testing"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
)

func TestCalculatePoolStatus(t *testing.T) {
	testCases := []struct {
		name               string
		request            *resourcev1alpha3.ResourcePoolStatusRequest
		slices             []*resourcev1.ResourceSlice
		claims             []*resourcev1.ResourceClaim
		expectedPools      int
		expectedTotal      int32
		expectedAlloc      int32
		expectedPoolCount  int32   // expected PoolCount value (total matching pools before limit)
		expectedTruncated  bool    // true when len(Pools) < PoolCount
		checkNodeName      bool    // when true, verify first pool's NodeName matches expectedNodeName
		expectedNodeName   *string // expected NodeName for the first pool (nil means no NodeName)
		checkIncomplete    bool    // when true, verify incomplete pool behavior
		expectedIncomplete int     // number of pools expected to have ValidationError set
	}{
		{
			name: "single-pool-no-allocations",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTotal: 4,
			expectedAlloc: 0,
		},
		{
			name: "single-pool-with-allocations",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
			},
			claims: []*resourcev1.ResourceClaim{
				makeAllocatedClaim("claim-1", "default", "test.example.com", "pool-1", "device-0"),
				makeAllocatedClaim("claim-2", "default", "test.example.com", "pool-1", "device-1"),
			},
			expectedPools: 1,
			expectedTotal: 4,
			expectedAlloc: 2,
		},
		{
			name: "multiple-pools",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-2", "node-2", 8),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 2,
			expectedTotal: 12,
			expectedAlloc: 0,
		},
		{
			name: "filter-by-driver",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "other.example.com", "pool-2", "node-2", 8),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTotal: 4,
			expectedAlloc: 0,
		},
		{
			name: "filter-by-pool-name",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver:   "test.example.com",
					PoolName: new("pool-1"),
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-2", "node-2", 8),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTotal: 4,
			expectedAlloc: 0,
		},
		{
			name: "multiple-slices-same-pool",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-1", "node-1", 4),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTotal: 8,
			expectedAlloc: 0,
		},
		{
			name: "limit-pools",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
					Limit:  new(int32(1)),
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-2", "node-2", 8),
			},
			claims:            []*resourcev1.ResourceClaim{},
			expectedPools:     1,
			expectedPoolCount: 2,
			expectedTruncated: true,
		},
		{
			name: "no-matching-pools",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "nonexistent.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 0,
			expectedTotal: 0,
			expectedAlloc: 0,
		},
		{
			name: "older-generation-slices-ignored",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				// Old generation slice (should be ignored)
				makeSliceWithGeneration("slice-old", "test.example.com", "pool-1", "node-1", 8, 1),
				// New generation slices (should be counted)
				makeSliceWithGeneration("slice-new-1", "test.example.com", "pool-1", "node-1", 4, 2),
				makeSliceWithGeneration("slice-new-2", "test.example.com", "pool-1", "node-1", 4, 2),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTotal: 8, // 4+4 from gen 2 only, not 8 from gen 1
			expectedAlloc: 0,
		},
		{
			name: "incomplete-pool-has-validation-error",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				// Pool expects 5 slices but only 3 are published
				makeSliceWithExpectedCount("slice-1", "test.example.com", "pool-1", "node-1", 4, 1, 5),
				makeSliceWithExpectedCount("slice-2", "test.example.com", "pool-1", "node-1", 4, 1, 5),
				makeSliceWithExpectedCount("slice-3", "test.example.com", "pool-1", "node-1", 4, 1, 5),
			},
			claims:             []*resourcev1.ResourceClaim{},
			expectedPools:      1,
			checkIncomplete:    true,
			expectedIncomplete: 1,
		},
		{
			name: "mix-of-complete-and-incomplete-pools",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				// pool-1: complete (1/1 slices)
				makeSliceWithExpectedCount("slice-1", "test.example.com", "pool-1", "node-1", 4, 1, 1),
				// pool-2: incomplete (2/3 slices)
				makeSliceWithExpectedCount("slice-2a", "test.example.com", "pool-2", "node-2", 4, 1, 3),
				makeSliceWithExpectedCount("slice-2b", "test.example.com", "pool-2", "node-2", 4, 1, 3),
			},
			claims:             []*resourcev1.ResourceClaim{},
			expectedPools:      2,
			expectedTotal:      4, // only pool-1 (complete) contributes to total
			expectedAlloc:      0,
			checkIncomplete:    true,
			expectedIncomplete: 1,
		},
		{
			name: "same-node-pool-sets-nodename",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-1", "node-1", 4),
			},
			claims:           []*resourcev1.ResourceClaim{},
			expectedPools:    1,
			expectedTotal:    8,
			expectedAlloc:    0,
			checkNodeName:    true,
			expectedNodeName: new("node-1"),
		},
		{
			name: "mixed-node-pool-clears-nodename",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-1", "node-2", 4),
			},
			claims:           []*resourcev1.ResourceClaim{},
			expectedPools:    1,
			expectedTotal:    8,
			expectedAlloc:    0,
			checkNodeName:    true,
			expectedNodeName: nil,
		},
		{
			name: "nil-nodename-slices-no-nodename",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSliceWithoutNode("slice-1", "test.example.com", "pool-1", 4),
				makeSliceWithoutNode("slice-2", "test.example.com", "pool-1", 4),
			},
			claims:           []*resourcev1.ResourceClaim{},
			expectedPools:    1,
			expectedTotal:    8,
			expectedAlloc:    0,
			checkNodeName:    true,
			expectedNodeName: nil,
		},
		{
			name: "all-pools-complete-no-validation-errors",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
			},
			slices: []*resourcev1.ResourceSlice{
				// pool-1: complete (2/2 slices)
				makeSliceWithExpectedCount("slice-1a", "test.example.com", "pool-1", "node-1", 4, 1, 2),
				makeSliceWithExpectedCount("slice-1b", "test.example.com", "pool-1", "node-1", 4, 1, 2),
			},
			claims:             []*resourcev1.ResourceClaim{},
			expectedPools:      1,
			expectedTotal:      8,
			expectedAlloc:      0,
			checkIncomplete:    true,
			expectedIncomplete: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			// Create fake client and informers
			fakeClient := fake.NewClientset()
			informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

			// Create controller
			controller, err := NewController(ctx, fakeClient,
				informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
				informerFactory.Resource().V1().ResourceSlices(),
				informerFactory.Resource().V1().ResourceClaims(),
			)
			if err != nil {
				t.Fatalf("Failed to create controller: %v", err)
			}

			// Add slices to the informer store
			for _, slice := range tc.slices {
				err := informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
				if err != nil {
					t.Fatalf("Failed to add slice to informer: %v", err)
				}
			}

			// Add claims to the informer store
			for _, claim := range tc.claims {
				err := informerFactory.Resource().V1().ResourceClaims().Informer().GetStore().Add(claim)
				if err != nil {
					t.Fatalf("Failed to add claim to informer: %v", err)
				}
			}

			// Calculate pool status (reads directly from listers)
			status := controller.calculatePoolStatus(ctx, tc.request)

			// Verify pool count
			if len(status.Pools) != tc.expectedPools {
				t.Errorf("Expected %d pools, got %d", tc.expectedPools, len(status.Pools))
			}

			// Verify PoolCount field
			if tc.expectedPoolCount > 0 {
				if status.PoolCount == nil {
					t.Errorf("Expected PoolCount to be %d, got nil", tc.expectedPoolCount)
				} else if *status.PoolCount != tc.expectedPoolCount {
					t.Errorf("Expected PoolCount %d, got %d", tc.expectedPoolCount, *status.PoolCount)
				}
			}

			// Verify truncation: len(Pools) < PoolCount means truncated
			if tc.expectedTruncated {
				if status.PoolCount == nil || int32(len(status.Pools)) >= *status.PoolCount {
					t.Errorf("Expected truncation (len(Pools)=%d < PoolCount), but PoolCount=%v", len(status.Pools), status.PoolCount)
				}
			}

			// Count device totals from complete pools only (those without ValidationError)
			if tc.expectedPools > 0 && !tc.expectedTruncated {
				var totalDevices, allocatedDevices int32
				for _, pool := range status.Pools {
					if pool.TotalDevices != nil {
						totalDevices += *pool.TotalDevices
					}
					if pool.AllocatedDevices != nil {
						allocatedDevices += *pool.AllocatedDevices
					}
				}
				if totalDevices != tc.expectedTotal {
					t.Errorf("Expected total devices %d, got %d", tc.expectedTotal, totalDevices)
				}
				if allocatedDevices != tc.expectedAlloc {
					t.Errorf("Expected allocated devices %d, got %d", tc.expectedAlloc, allocatedDevices)
				}
			}

			// Verify NodeName when test specifies expected value
			if tc.checkNodeName && len(status.Pools) == 1 {
				pool := status.Pools[0]
				if tc.expectedNodeName != nil {
					if pool.NodeName == nil {
						t.Errorf("Expected NodeName %q, got nil", *tc.expectedNodeName)
					} else if *pool.NodeName != *tc.expectedNodeName {
						t.Errorf("Expected NodeName %q, got %q", *tc.expectedNodeName, *pool.NodeName)
					}
				} else if pool.NodeName != nil {
					t.Errorf("Expected NodeName to be nil, got %q", *pool.NodeName)
				}
			}

			// Verify incomplete pool counts (pools with ValidationError set)
			if tc.checkIncomplete {
				incompleteCount := 0
				for _, pool := range status.Pools {
					if pool.ValidationError != nil {
						incompleteCount++
						// Incomplete pools should have nil device counts and nil ResourceSliceCount
						if pool.TotalDevices != nil {
							t.Errorf("Incomplete pool %s/%s should have nil TotalDevices, got %d", pool.Driver, pool.PoolName, *pool.TotalDevices)
						}
						if pool.AllocatedDevices != nil {
							t.Errorf("Incomplete pool %s/%s should have nil AllocatedDevices, got %d", pool.Driver, pool.PoolName, *pool.AllocatedDevices)
						}
						if pool.ResourceSliceCount != nil {
							t.Errorf("Incomplete pool %s/%s should have nil ResourceSliceCount, got %d", pool.Driver, pool.PoolName, *pool.ResourceSliceCount)
						}
					}
				}
				if incompleteCount != tc.expectedIncomplete {
					t.Errorf("Expected %d incomplete pools (with ValidationError), got %d", tc.expectedIncomplete, incompleteCount)
				}
			}

			// Verify conditions are set
			if len(status.Conditions) == 0 {
				t.Error("Expected Conditions to be set")
			}
		})
	}
}

func TestSyncRequest(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create fake client and informers
	request := &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
	}

	fakeClient := fake.NewClientset(request)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	// Create controller
	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add request to informer
	err = informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(request)
	if err != nil {
		t.Fatalf("Failed to add request to informer: %v", err)
	}

	// Add a slice
	slice := makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4)
	err = informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
	if err != nil {
		t.Fatalf("Failed to add slice to informer: %v", err)
	}

	// Sync the request
	err = controller.syncRequest(ctx, "test-request")
	if err != nil {
		t.Fatalf("syncRequest failed: %v", err)
	}

	// Verify the status was updated (fake client should have received UpdateStatus call)
	actions := fakeClient.Actions()
	var foundUpdate bool
	for _, action := range actions {
		if action.GetVerb() == "update" && action.GetSubresource() == "status" {
			foundUpdate = true
			break
		}
	}
	if !foundUpdate {
		t.Error("Expected status update action, but none found")
	}
}

func TestSyncRequestRequeuesIncompletePool(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	request := &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
	}

	fakeClient := fake.NewClientset(request)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add request to informer
	err = informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(request)
	if err != nil {
		t.Fatalf("Failed to add request to informer: %v", err)
	}

	// Add an incomplete pool (expects 3 slices, only 1 published)
	slice := makeSliceWithExpectedCount("slice-1", "test.example.com", "pool-1", "node-1", 4, 1, 3)
	err = informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
	if err != nil {
		t.Fatalf("Failed to add slice to informer: %v", err)
	}

	// syncRequest should always return an error for incomplete pools,
	// letting processNextWorkItem handle retry counting and drop logic.
	err = controller.syncRequest(ctx, "test-request")
	if err == nil {
		t.Fatal("Expected syncRequest to return error for incomplete pool requeue, got nil")
	}

	// Verify no status update was made
	for _, action := range fakeClient.Actions() {
		if action.GetVerb() == "update" && action.GetSubresource() == "status" {
			t.Error("Should not update status when requeueing for incomplete pools")
		}
	}

	// Even after retries are exhausted, syncRequest still returns an error;
	// it is processNextWorkItem that decides to drop the key.
	for range maxRetries {
		controller.workqueue.AddRateLimited("test-request")
		key, _ := controller.workqueue.Get()
		controller.workqueue.Done(key)
	}

	fakeClient.ClearActions()
	err = controller.syncRequest(ctx, "test-request")
	if err == nil {
		t.Fatal("Expected syncRequest to still return error for incomplete pools after retries exhausted")
	}
}

func TestSkipProcessedRequest(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	now := metav1.Now()
	// Create a request that's already been processed (Status is non-nil with conditions)
	request := &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
		Status: &resourcev1alpha3.ResourcePoolStatusRequestStatus{
			PoolCount: new(int32(0)),
			Conditions: []metav1.Condition{
				{
					Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
					Status:             metav1.ConditionTrue,
					LastTransitionTime: now,
					Reason:             "CalculationComplete",
					Message:            "Calculated status for 0 pools",
				},
			},
		},
	}

	fakeClient := fake.NewClientset(request)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	err = informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(request)
	if err != nil {
		t.Fatalf("Failed to add request to informer: %v", err)
	}

	// Sync should skip the already-processed request
	err = controller.syncRequest(ctx, "test-request")
	if err != nil {
		t.Fatalf("syncRequest failed: %v", err)
	}

	// Verify no update was made
	actions := fakeClient.Actions()
	for _, action := range actions {
		if action.GetVerb() == "update" && action.GetSubresource() == "status" {
			t.Error("Should not update status for already-processed request")
		}
	}
}

// Helper functions

func makeSlice(name, driver, pool, node string, deviceCount int) *resourcev1.ResourceSlice {
	return makeSliceWithGenerationAndCount(name, driver, pool, node, deviceCount, 1, 1)
}

func makeSliceWithExpectedCount(name, driver, pool, node string, deviceCount int, generation int64, resourceSliceCount int64) *resourcev1.ResourceSlice {
	return makeSliceWithGenerationAndCount(name, driver, pool, node, deviceCount, generation, resourceSliceCount)
}

func makeSliceWithGeneration(name, driver, pool, node string, deviceCount int, generation int64) *resourcev1.ResourceSlice {
	return makeSliceWithGenerationAndCount(name, driver, pool, node, deviceCount, generation, 1)
}

func makeSliceWithGenerationAndCount(name, driver, pool, node string, deviceCount int, generation int64, resourceSliceCount int64) *resourcev1.ResourceSlice {
	devices := make([]resourcev1.Device, deviceCount)
	for i := range deviceCount {
		devices[i] = resourcev1.Device{
			Name: fmt.Sprintf("device-%d", i),
		}
	}

	return &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   driver,
			NodeName: &node,
			Pool: resourcev1.ResourcePool{
				Name:               pool,
				Generation:         generation,
				ResourceSliceCount: resourceSliceCount,
			},
			Devices: devices,
		},
	}
}

func makeAllocatedClaim(name, namespace, driver, pool, device string) *resourcev1.ResourceClaim {
	return &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: resourcev1.ResourceClaimSpec{},
		Status: resourcev1.ResourceClaimStatus{
			Allocation: &resourcev1.AllocationResult{
				Devices: resourcev1.DeviceAllocationResult{
					Results: []resourcev1.DeviceRequestAllocationResult{
						{
							Driver: driver,
							Pool:   pool,
							Device: device,
						},
					},
				},
			},
		},
	}
}

func TestIsOlderThan(t *testing.T) {
	testCases := []struct {
		name     string
		time     time.Time
		duration time.Duration
		expected bool
	}{
		{
			name:     "zero-time-is-not-older",
			time:     time.Time{},
			duration: time.Hour,
			expected: false,
		},
		{
			name:     "recent-time-is-not-older",
			time:     time.Now().Add(-30 * time.Minute),
			duration: time.Hour,
			expected: false,
		},
		{
			name:     "old-time-is-older",
			time:     time.Now().Add(-2 * time.Hour),
			duration: time.Hour,
			expected: true,
		},
		{
			name:     "just-under-boundary-is-not-older",
			time:     time.Now().Add(-time.Hour + time.Minute),
			duration: time.Hour,
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := isOlderThan(tc.time, tc.duration)
			if result != tc.expected {
				t.Errorf("isOlderThan(%v, %v) = %v, expected %v", tc.time, tc.duration, result, tc.expected)
			}
		})
	}
}

func TestShouldDeleteRequest(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	fakeClient := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	testCases := []struct {
		name           string
		request        *resourcev1alpha3.ResourcePoolStatusRequest
		expectedDelete bool
	}{
		{
			name: "completed-request-not-expired",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test-request",
					CreationTimestamp: metav1.Now(),
				},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
				Status: &resourcev1alpha3.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(0)),
					Conditions: []metav1.Condition{
						{
							Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
							Status:             metav1.ConditionTrue,
							LastTransitionTime: metav1.Now(),
							Reason:             "CalculationComplete",
							Message:            "Calculated status for 0 pools",
						},
					},
				},
			},
			expectedDelete: false,
		},
		{
			name: "completed-request-expired",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test-request",
					CreationTimestamp: metav1.NewTime(time.Now().Add(-2 * time.Hour)),
				},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
				Status: &resourcev1alpha3.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(0)),
					Conditions: []metav1.Condition{
						{
							Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
							Status:             metav1.ConditionTrue,
							LastTransitionTime: metav1.NewTime(time.Now().Add(-2 * time.Hour)),
							Reason:             "CalculationComplete",
							Message:            "Calculated status for 0 pools",
						},
					},
				},
			},
			expectedDelete: true,
		},
		{
			name: "pending-request-not-expired",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test-request",
					CreationTimestamp: metav1.Now(),
				},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
				// No status - request is pending
			},
			expectedDelete: false,
		},
		{
			name: "pending-request-expired",
			request: &resourcev1alpha3.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test-request",
					CreationTimestamp: metav1.NewTime(time.Now().Add(-25 * time.Hour)),
				},
				Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
				},
				// No status - pending request expired
			},
			expectedDelete: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := controller.shouldDeleteRequest(tc.request)
			if result != tc.expectedDelete {
				t.Errorf("shouldDeleteRequest() = %v, expected %v", result, tc.expectedDelete)
			}
		})
	}
}

func TestCleanupExpiredRequests(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create requests: one expired, one not expired
	expiredRequest := &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "expired-request",
			CreationTimestamp: metav1.NewTime(time.Now().Add(-2 * time.Hour)),
		},
		Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
		Status: &resourcev1alpha3.ResourcePoolStatusRequestStatus{
			PoolCount: new(int32(0)),
			Conditions: []metav1.Condition{
				{
					Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
					Status:             metav1.ConditionTrue,
					LastTransitionTime: metav1.NewTime(time.Now().Add(-2 * time.Hour)),
					Reason:             "CalculationComplete",
					Message:            "Calculated status for 0 pools",
				},
			},
		},
	}

	activeRequest := &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "active-request",
			CreationTimestamp: metav1.Now(),
		},
		Spec: resourcev1alpha3.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
		Status: &resourcev1alpha3.ResourcePoolStatusRequestStatus{
			PoolCount: new(int32(0)),
			Conditions: []metav1.Condition{
				{
					Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
					Status:             metav1.ConditionTrue,
					LastTransitionTime: metav1.Now(),
					Reason:             "CalculationComplete",
					Message:            "Calculated status for 0 pools",
				},
			},
		},
	}

	fakeClient := fake.NewClientset(expiredRequest, activeRequest)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add requests to informer store
	err = informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(expiredRequest)
	if err != nil {
		t.Fatalf("Failed to add expired request to informer: %v", err)
	}
	err = informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests().Informer().GetStore().Add(activeRequest)
	if err != nil {
		t.Fatalf("Failed to add active request to informer: %v", err)
	}

	// Run cleanup
	controller.cleanupExpiredRequests(ctx)

	// Verify that delete was called for expired request
	actions := fakeClient.Actions()
	var deletedExpired, deletedActive bool
	for _, action := range actions {
		if action.GetVerb() == "delete" {
			deleteAction, ok := action.(interface{ GetName() string })
			if ok {
				if deleteAction.GetName() == "expired-request" {
					deletedExpired = true
				}
				if deleteAction.GetName() == "active-request" {
					deletedActive = true
				}
			}
		}
	}

	if !deletedExpired {
		t.Error("Expected expired-request to be deleted")
	}
	if deletedActive {
		t.Error("Did not expect active-request to be deleted")
	}
}

func makeSliceWithoutNode(name, driver, pool string, deviceCount int) *resourcev1.ResourceSlice {
	devices := make([]resourcev1.Device, deviceCount)
	for i := range deviceCount {
		devices[i] = resourcev1.Device{
			Name: fmt.Sprintf("device-%d", i),
		}
	}

	return &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourcev1.ResourceSliceSpec{
			Driver: driver,
			Pool: resourcev1.ResourcePool{
				Name:               pool,
				Generation:         1,
				ResourceSliceCount: 1,
			},
			Devices: devices,
		},
	}
}

// new is a generic helper to create a pointer to a value.
func new[T any](v T) *T {
	return &v
}
