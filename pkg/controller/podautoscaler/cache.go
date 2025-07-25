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
	"sync/atomic"
	"time"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/monitor"
)

// ControllerCacheEntry stores a cached controller resource
type controllerCacheEntry struct {
	lastFetched int64 // atomic time
	Key         string
	resource    *unstructured.Unstructured
	GVR         schema.GroupVersionResource
	Namespace   string
	Name        string
}

func (e *controllerCacheEntry) LastFetched() time.Time {
	return time.Unix(0, atomic.LoadInt64(&e.lastFetched))
}

func (e *controllerCacheEntry) SetLastFetched(t time.Time) {
	atomic.StoreInt64(&e.lastFetched, t.UnixNano())
}

func (e *controllerCacheEntry) IsExpired(ttl time.Duration) bool {
	return time.Since(e.LastFetched()) > ttl
}

type ControllerCacheInterface interface {
	GetResource(namespace string, ownerRef metav1.OwnerReference) (*unstructured.Unstructured, error)
	SetResource(gvr schema.GroupVersionResource, namespace, name string, resource *unstructured.Unstructured, err error)
	DeleteResource(gvr schema.GroupVersionResource, namespace, name string)
	cleanup()
}

// ControllerCache provides caching for controller resources
type ControllerCache struct {
	store         cache.Store
	dynamicClient dynamic.Interface
	restMapper    apimeta.RESTMapper
	cacheTTL      time.Duration
	monitor       monitor.Monitor
}

// NewControllerCache creates a new controller cache
func NewControllerCache(dynamicClient dynamic.Interface, restMapper apimeta.RESTMapper, cacheTTL time.Duration, monitor monitor.Monitor) *ControllerCache {
	keyFunc := func(obj interface{}) (string, error) {
		entry, ok := obj.(*controllerCacheEntry)
		if !ok {
			return "", fmt.Errorf("invalid cache entry")
		}
		return fmt.Sprintf("%s/%s/%s", entry.GVR.String(), entry.Namespace, entry.Name), nil
	}
	return &ControllerCache{
		store:         cache.NewStore(keyFunc),
		dynamicClient: dynamicClient,
		restMapper:    restMapper,
		monitor:       monitor,
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

// makeResourceKey creates a cache key from GVR, namespace and name
func (c *ControllerCache) makeResourceKey(gvr schema.GroupVersionResource, namespace, name string) string {
	return fmt.Sprintf("%s/%s/%s", gvr.String(), namespace, name)
}

func (c *ControllerCache) DeleteResource(gvr schema.GroupVersionResource, namespace, name string) {
	key := c.makeResourceKey(gvr, namespace, name)
	if item, exists, _ := c.store.GetByKey(key); exists {
		c.store.Delete(item)
	}
}

// GetResource gets a resource from cache or API server using owner reference
func (c *ControllerCache) GetResource(namespace string, ownerRef metav1.OwnerReference) (*unstructured.Unstructured, error) {
	logger := klog.Background() // TODO(omerap12): propagate context
	logger.V(5).Info("Getting resource", "namespace", namespace, "ownerRef", ownerRef)

	gv, err := schema.ParseGroupVersion(ownerRef.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to parse group version %s: %w", ownerRef.APIVersion, err)
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
	obj, err := c.getResourceByGVR(gvr, namespace, ownerRef.Name)
	if err != nil {
		return obj, err
	}
	return obj, nil
}

// SetResource explicitly sets a resource in the cache
func (c *ControllerCache) SetResource(gvr schema.GroupVersionResource, namespace, name string, resource *unstructured.Unstructured) error {
	key := c.makeResourceKey(gvr, namespace, name)
	entry := &controllerCacheEntry{
		Key:       key,
		resource:  resource,
		GVR:       gvr,
		Namespace: namespace,
		Name:      name,
	}
	entry.SetLastFetched(time.Now())

	if err := c.store.Add(entry); err != nil {
		return err
	}
	return nil
}

// getResourceByGVR gets a resource by GVR from cache or API server
func (c *ControllerCache) getResourceByGVR(gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error) {
	logger := klog.Background() // TODO(omerap12): propagate context
	key := c.makeResourceKey(gvr, namespace, name)
	obj, err := c.getResourceByKey(key)
	if err != nil {
		return nil, err
	}
	if obj != nil {
		// cache hit
		logger.V(5).Info("Cache hit", "gvr", gvr, "namespace", namespace, "name", name)
		c.monitor.ObserveCacheHit(gvr.Resource)
		return obj, nil
	}
	// object is not in cache
	logger.V(5).Info("Cache miss", "gvr", gvr, "namespace", namespace, "name", name)
	c.monitor.ObserveCacheMiss(gvr.Resource)
	var resource *unstructured.Unstructured
	resource, err = c.dynamicClient.Resource(gvr).Namespace(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		logger.Error(err, "Failed to get resource", "gvr", gvr, "namespace", namespace, "name", name) // TODO(omerap12): add monitor for errors
		return nil, err
	}
	err = c.SetResource(gvr, namespace, name, resource)
	return resource, err
}

func (c *ControllerCache) getResourceByKey(key string) (*unstructured.Unstructured, error) {
	item, exists, err := c.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, nil
	}

	entry := item.(*controllerCacheEntry)
	if entry.IsExpired(c.cacheTTL) {
		c.store.Delete(entry)
		return nil, nil
	}

	return entry.resource, nil
}

func (c *ControllerCache) cleanup() {
	// Get all items and check expiration
	items := c.store.List()
	for _, item := range items {
		if entry, ok := item.(*controllerCacheEntry); ok {
			if entry.IsExpired(c.cacheTTL) {
				c.store.Delete(entry)
			}
		}
	}
}
