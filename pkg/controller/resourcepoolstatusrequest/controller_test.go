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

	resourcev1 "k8s.io/api/resource/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestCalculatePoolStatus(t *testing.T) {
	testCases := []struct {
		name          string
		request       *resourcev1alpha1.ResourcePoolStatusRequest
		slices        []*resourcev1.ResourceSlice
		claims        []*resourcev1.ResourceClaim
		expectedPools int
		expectedTotal int32
		expectedAlloc int32
		expectedTrunc bool
	}{
		{
			name: "single-pool-no-allocations",
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
					Driver:   "test.example.com",
					PoolName: "pool-1",
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
					Driver: "test.example.com",
					Limit:  ptr.To(int32(1)),
				},
			},
			slices: []*resourcev1.ResourceSlice{
				makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4),
				makeSlice("slice-2", "test.example.com", "pool-2", "node-2", 8),
			},
			claims:        []*resourcev1.ResourceClaim{},
			expectedPools: 1,
			expectedTrunc: true,
		},
		{
			name: "no-matching-pools",
			request: &resourcev1alpha1.ResourcePoolStatusRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
				Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
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
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			// Create fake client and informers
			fakeClient := fake.NewClientset()
			informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

			// Create controller
			controller, err := NewController(ctx, fakeClient, informerFactory)
			if err != nil {
				t.Fatalf("Failed to create controller: %v", err)
			}

			// Add slices to the cache
			for _, slice := range tc.slices {
				err := informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
				if err != nil {
					t.Fatalf("Failed to add slice to informer: %v", err)
				}
			}

			// Add claims to the cache
			for _, claim := range tc.claims {
				err := informerFactory.Resource().V1().ResourceClaims().Informer().GetStore().Add(claim)
				if err != nil {
					t.Fatalf("Failed to add claim to informer: %v", err)
				}
			}

			// Rebuild caches
			controller.rebuildPoolCache(ktesting.NewLogger(t, ktesting.NewConfig()))
			controller.rebuildAllocationCache(ktesting.NewLogger(t, ktesting.NewConfig()))

			// Calculate pool status
			status := controller.calculatePoolStatus(ctx, tc.request)

			// Verify results
			if len(status.Pools) != tc.expectedPools {
				t.Errorf("Expected %d pools, got %d", tc.expectedPools, len(status.Pools))
			}

			if tc.expectedPools > 0 && !tc.expectedTrunc {
				var totalDevices, allocatedDevices int32
				for _, pool := range status.Pools {
					totalDevices += pool.TotalDevices
					allocatedDevices += pool.AllocatedDevices
				}
				if totalDevices != tc.expectedTotal {
					t.Errorf("Expected total devices %d, got %d", tc.expectedTotal, totalDevices)
				}
				if allocatedDevices != tc.expectedAlloc {
					t.Errorf("Expected allocated devices %d, got %d", tc.expectedAlloc, allocatedDevices)
				}
			}

			if status.Truncated != tc.expectedTrunc {
				t.Errorf("Expected truncated=%v, got %v", tc.expectedTrunc, status.Truncated)
			}

			// Verify observation time is set
			if status.ObservationTime == nil {
				t.Error("Expected ObservationTime to be set")
			}
		})
	}
}

func TestSyncRequest(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create fake client and informers
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
	}

	fakeClient := fake.NewClientset(request)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	// Create controller
	controller, err := NewController(ctx, fakeClient, informerFactory)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add request to informer
	err = informerFactory.Resource().V1alpha1().ResourcePoolStatusRequests().Informer().GetStore().Add(request)
	if err != nil {
		t.Fatalf("Failed to add request to informer: %v", err)
	}

	// Add a slice
	slice := makeSlice("slice-1", "test.example.com", "pool-1", "node-1", 4)
	err = informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
	if err != nil {
		t.Fatalf("Failed to add slice to informer: %v", err)
	}

	// Rebuild caches
	controller.rebuildPoolCache(ktesting.NewLogger(t, ktesting.NewConfig()))
	controller.rebuildAllocationCache(ktesting.NewLogger(t, ktesting.NewConfig()))

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

func TestSkipProcessedRequest(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create a request that's already been processed
	now := metav1.Now()
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
		},
		Status: resourcev1alpha1.ResourcePoolStatusRequestStatus{
			ObservationTime: &now, // Already processed
		},
	}

	fakeClient := fake.NewClientset(request)
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient, informerFactory)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	err = informerFactory.Resource().V1alpha1().ResourcePoolStatusRequests().Informer().GetStore().Add(request)
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

func TestRebuildPoolCache(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	fakeClient := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient, informerFactory)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add slices to informer
	slices := []*resourcev1.ResourceSlice{
		makeSlice("slice-1", "driver-a", "pool-1", "node-1", 4),
		makeSlice("slice-2", "driver-a", "pool-1", "node-1", 4),
		makeSlice("slice-3", "driver-a", "pool-2", "node-2", 8),
		makeSlice("slice-4", "driver-b", "pool-3", "node-3", 2),
	}

	for _, slice := range slices {
		err := informerFactory.Resource().V1().ResourceSlices().Informer().GetStore().Add(slice)
		if err != nil {
			t.Fatalf("Failed to add slice: %v", err)
		}
	}

	// Rebuild cache
	controller.rebuildPoolCache(ktesting.NewLogger(t, ktesting.NewConfig()))

	// Verify pool data
	controller.poolDataMu.RLock()
	defer controller.poolDataMu.RUnlock()

	if len(controller.poolData) != 3 {
		t.Errorf("Expected 3 pools, got %d", len(controller.poolData))
	}

	pool1 := controller.poolData["driver-a/pool-1"]
	if pool1 == nil {
		t.Fatal("pool-1 not found")
	}
	if pool1.totalDevices != 8 {
		t.Errorf("Expected pool-1 totalDevices=8, got %d", pool1.totalDevices)
	}
	if pool1.sliceCount != 2 {
		t.Errorf("Expected pool-1 sliceCount=2, got %d", pool1.sliceCount)
	}
}

func TestRebuildAllocationCache(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	fakeClient := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)

	controller, err := NewController(ctx, fakeClient, informerFactory)
	if err != nil {
		t.Fatalf("Failed to create controller: %v", err)
	}

	// Add claims to informer
	claims := []*resourcev1.ResourceClaim{
		makeAllocatedClaim("claim-1", "ns1", "driver-a", "pool-1", "device-0"),
		makeAllocatedClaim("claim-2", "ns1", "driver-a", "pool-1", "device-1"),
		makeAllocatedClaim("claim-3", "ns2", "driver-a", "pool-2", "device-0"),
		makeUnallocatedClaim("claim-4", "ns1"),
	}

	for _, claim := range claims {
		err := informerFactory.Resource().V1().ResourceClaims().Informer().GetStore().Add(claim)
		if err != nil {
			t.Fatalf("Failed to add claim: %v", err)
		}
	}

	// Rebuild cache
	controller.rebuildAllocationCache(ktesting.NewLogger(t, ktesting.NewConfig()))

	// Verify allocation data
	controller.poolDataMu.RLock()
	defer controller.poolDataMu.RUnlock()

	if len(controller.allocationData) != 2 {
		t.Errorf("Expected 2 allocation entries, got %d", len(controller.allocationData))
	}

	if controller.allocationData["driver-a/pool-1"] != 2 {
		t.Errorf("Expected pool-1 allocations=2, got %d", controller.allocationData["driver-a/pool-1"])
	}

	if controller.allocationData["driver-a/pool-2"] != 1 {
		t.Errorf("Expected pool-2 allocations=1, got %d", controller.allocationData["driver-a/pool-2"])
	}
}

// Helper functions

func makeSlice(name, driver, pool, node string, deviceCount int) *resourcev1.ResourceSlice {
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
			NodeName: ptr.To(node),
			Pool: resourcev1.ResourcePool{
				Name:       pool,
				Generation: 1,
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

func makeUnallocatedClaim(name, namespace string) *resourcev1.ResourceClaim {
	return &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec:   resourcev1.ResourceClaimSpec{},
		Status: resourcev1.ResourceClaimStatus{},
	}
}

// Compile-time check that our test helpers work with the lister
var _ = labels.Everything()
