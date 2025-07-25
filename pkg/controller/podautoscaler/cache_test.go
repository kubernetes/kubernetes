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

package podautoscaler

import (
	"context"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
)

// setupTestEnv creates and returns the basic test environment components
func setupTestEnv() (dynamic.Interface, meta.RESTMapper, *mockMonitor) {
	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{})
	mockMonitor := newMockMonitor()
	return dynamicClient, mapper, mockMonitor
}

func createTestDeployment(ctx context.Context, client dynamic.Interface) (*unstructured.Unstructured, error) {
	deployment := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "test-deployment",
				"namespace": "default",
				"uid":       "test-uid",
			},
		},
	}

	return client.Resource(schema.GroupVersionResource{
		Group: "apps", Version: "v1", Resource: "deployments",
	}).Namespace("default").Create(ctx, deployment, metav1.CreateOptions{})
}

func TestControllerCache_GetResource(t *testing.T) {
	ctx := context.Background()
	dynamicClient, mapper, monitor := setupTestEnv()
	cache := NewControllerCache(dynamicClient, mapper, 5*time.Minute, monitor)

	deployment, err := createTestDeployment(ctx, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	tests := []struct {
		name      string
		ownerRef  metav1.OwnerReference
		namespace string
		wantErr   bool
	}{
		{
			name: "valid owner reference",
			ownerRef: metav1.OwnerReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
				UID:        deployment.GetUID(),
			},
			namespace: "default",
			wantErr:   false,
		},
		{
			name: "invalid api version",
			ownerRef: metav1.OwnerReference{
				APIVersion: "invalid/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			namespace: "default",
			wantErr:   true,
		},
		{
			name: "non-existent resource",
			ownerRef: metav1.OwnerReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "non-existent",
			},
			namespace: "default",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resource, err := cache.GetResource(tt.namespace, tt.ownerRef)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetResource() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && resource == nil {
				t.Error("GetResource() returned nil resource when error not expected")
			}
		})
	}
}

func TestControllerCache_CacheExpiry(t *testing.T) {
	ctx := context.Background()
	dynamicClient, mapper, monitor := setupTestEnv()
	cacheTTL := 100 * time.Millisecond
	cache := NewControllerCache(dynamicClient, mapper, cacheTTL, monitor)

	deployment, err := createTestDeployment(ctx, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	ownerRef := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       deployment.GetName(),
		UID:        deployment.GetUID(),
	}

	// First request - should be a cache miss
	resource1, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("First GetResource() failed: %v", err)
	}

	if monitor.GetCacheMiss("deployments") != 1 {
		t.Error("Expected cache miss on first request")
	}

	// Immediate second request - should be a cache hit
	resource2, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("Second GetResource() failed: %v", err)
	}

	if monitor.GetCacheHit("deployments") != 1 {
		t.Error("Expected cache hit on second request")
	}

	// Wait for cache to expire
	time.Sleep(cacheTTL + 50*time.Millisecond)

	// Third request - should be a cache miss again
	resource3, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("Third GetResource() failed: %v", err)
	}

	if monitor.GetCacheMiss("deployments") != 2 {
		t.Error("Expected second cache miss after expiry")
	}

	// Verify resources
	if resource1 == nil || resource2 == nil || resource3 == nil {
		t.Error("Unexpected nil resource")
	}
}

func TestControllerCache_Cleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dynamicClient, mapper, monitor := setupTestEnv()
	cacheTTL := 100 * time.Millisecond
	cache := NewControllerCache(dynamicClient, mapper, cacheTTL, monitor)

	deployment, err := createTestDeployment(ctx, dynamicClient)
	if err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	cache.Start(ctx, 50*time.Millisecond)

	ownerRef := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       deployment.GetName(),
		UID:        deployment.GetUID(),
	}

	// Add resource to cache
	_, err = cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("GetResource() failed: %v", err)
	}

	// Wait for cleanup
	time.Sleep(cacheTTL + 200*time.Millisecond)

	// Verify cache was cleaned by checking for a cache miss
	_, err = cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("GetResource() after cleanup failed: %v", err)
	}

	if monitor.GetCacheMiss("deployments") != 2 {
		t.Error("Expected cache miss after cleanup")
	}
}
