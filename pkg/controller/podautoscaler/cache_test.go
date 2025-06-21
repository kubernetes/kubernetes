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
func setupTestEnv() (dynamic.Interface, meta.RESTMapper) {
	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{})
	return dynamicClient, mapper
}

func cleanupResource(client dynamic.Interface, t *testing.T) {
	err := client.
		Resource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}).
		Namespace("default").
		Delete(context.TODO(), "test-deployment", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete test deployment: %v", err)
	}
}

func createTestDeployment(client dynamic.Interface, t *testing.T) {
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

	_, err := client.
		Resource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}).
		Namespace("default").
		Create(context.TODO(), deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}
}

func TestControllerCache_GetResource(t *testing.T) {
	dynamicClient, mapper := setupTestEnv()
	cache := NewControllerCache(dynamicClient, mapper, 5*time.Minute)

	// create and clean resource after
	createTestDeployment(dynamicClient, t)
	defer cleanupResource(dynamicClient, t)

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
				UID:        "test-uid",
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

// TODO: need to add metric for cache hit - this test is incomplete
func TestControllerCache_CacheExpiry(t *testing.T) {
	// Setup with short TTL for testing
	dynamicClient, mapper := setupTestEnv()
	cacheTTL := 100 * time.Millisecond
	cache := NewControllerCache(dynamicClient, mapper, cacheTTL)

	// create and clean resource after
	createTestDeployment(dynamicClient, t)
	defer cleanupResource(dynamicClient, t)

	ownerRef := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       "test-deployment",
	}

	// First request should hit the API - check cache miss +1
	resource1, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("First GetResource() failed: %v", err)
	}

	// Immediate second request should hit the cache - check cache hit +1
	resource2, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("Second GetResource() failed: %v", err)
	}

	if resource1 != resource2 {
		t.Error("Cache miss: second request returned different resource")
	}

	// Wait for cache to expire
	time.Sleep(cacheTTL + 50*time.Millisecond)

	// Third request should hit the API again - check cache miss +1
	resource3, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("Third GetResource() failed: %v", err)
	}

	if resource1 == resource3 {
		t.Error("Cache did not expire: third request returned same resource instance")
	}
}

func TestControllerCache_Cleanup(t *testing.T) {
	// Setup
	dynamicClient, mapper := setupTestEnv()
	cacheTTL := 100 * time.Millisecond
	cache := NewControllerCache(dynamicClient, mapper, cacheTTL)

	// create and clean resource after
	createTestDeployment(dynamicClient, t)
	defer cleanupResource(dynamicClient, t)

	// Start cleanup routine
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go cache.Start(ctx, 50*time.Millisecond)

	ownerRef := metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       "test-deployment",
	}

	// Add resource to cache
	_, err := cache.GetResource("default", ownerRef)
	if err != nil {
		t.Fatalf("GetResource() failed: %v", err)
	}

	// Verify cache has entry
	cache.mutex.RLock()
	initialSize := len(cache.resources)
	cache.mutex.RUnlock()

	if initialSize == 0 {
		t.Error("Cache is empty after adding resource")
	}

	// Wait for cleanup
	time.Sleep(cacheTTL + 200*time.Millisecond)

	// Verify cache was cleaned
	cache.mutex.RLock()
	finalSize := len(cache.resources)
	cache.mutex.RUnlock()

	if finalSize != 0 {
		t.Errorf("Cache was not cleaned up, still has %d entries", finalSize)
	}
}
