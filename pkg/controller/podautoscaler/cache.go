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
	"fmt"
	"sync"
	"time"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// ControllerCacheEntry stores a cached controller resource
type ControllerCacheEntry struct {
	Resource    *unstructured.Unstructured
	Error       error
	LastFetched time.Time
}

// ControllerCache provides caching for controller resources
type ControllerCache struct {
	mutex         sync.RWMutex
	resources     map[string]*ControllerCacheEntry
	dynamicClient dynamic.Interface
	restMapper    apimeta.RESTMapper
	cacheTTL      time.Duration
}

// NewControllerCache creates a new controller cache
func NewControllerCache(dynamicClient dynamic.Interface, restMapper apimeta.RESTMapper, cacheTTL time.Duration) *ControllerCache {
	return &ControllerCache{
		resources:     make(map[string]*ControllerCacheEntry),
		dynamicClient: dynamicClient,
		restMapper:    restMapper,
		cacheTTL:      cacheTTL,
	}
}

// Start starts a background goroutine to periodically clean up expired entries
func (c *ControllerCache) Start(ctx context.Context, cleanupInterval time.Duration) {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.cleanup()
		}
	}
}

// cleanup removes expired entries from the cache
func (c *ControllerCache) cleanup() {
	now := time.Now()
	expiredTime := now.Add(-c.cacheTTL)

	c.mutex.Lock()
	defer c.mutex.Unlock()

	for key, entry := range c.resources {
		if entry.LastFetched.Before(expiredTime) {
			delete(c.resources, key)
		}
	}
}

// makeResourceKey creates a cache key from GVR, namespace and name
func (c *ControllerCache) makeResourceKey(gvr schema.GroupVersionResource, namespace, name string) string {
	return fmt.Sprintf("%s/%s/%s/%s", gvr.String(), namespace, name, "")
}

// GetResource gets a resource from cache or API server using owner reference
func (c *ControllerCache) GetResource(namespace string, ownerRef metav1.OwnerReference) (*unstructured.Unstructured, error) {
	gv, err := schema.ParseGroupVersion(ownerRef.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to parse group version %s: %v", ownerRef.APIVersion, err)
	}

	// Get GVR for the owner
	gvk := schema.GroupVersionKind{
		Group:   gv.Group,
		Version: gv.Version,
		Kind:    ownerRef.Kind,
	}

	// Try to get the proper GVR using REST mapper
	var gvr schema.GroupVersionResource
	if c.restMapper != nil {
		mapping, err := c.restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err == nil {
			gvr = mapping.Resource
		} else {
			// Fallback to guessing
			gvr, _ = apimeta.UnsafeGuessKindToResource(gvk)
		}
	} else {
		// Fallback to guessing
		gvr, _ = apimeta.UnsafeGuessKindToResource(gvk)
	}

	return c.getResourceByGVR(gvr, namespace, ownerRef.Name)
}

// getResourceByGVR gets a resource by GVR from cache or API server
func (c *ControllerCache) getResourceByGVR(gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error) {
	key := c.makeResourceKey(gvr, namespace, name)

	// Check cache first
	c.mutex.RLock()
	entry, found := c.resources[key]
	c.mutex.RUnlock()

	now := time.Now()
	if found && now.Sub(entry.LastFetched) < c.cacheTTL {
		if entry.Error != nil {
			return nil, entry.Error
		}
		// in cache we can return it
		return entry.Resource, nil
	}

	// Not in cache or too old, fetch from API
	var resource *unstructured.Unstructured
	var err error
	resource, err = c.dynamicClient.Resource(gvr).Namespace(namespace).Get(context.TODO(), name, metav1.GetOptions{})

	// Update cache
	c.mutex.Lock()
	c.resources[key] = &ControllerCacheEntry{
		Resource:    resource,
		Error:       err,
		LastFetched: now,
	}
	c.mutex.Unlock()

	return resource, err
}
