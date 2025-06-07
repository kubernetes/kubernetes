package podautoscaler

import (
	"context"
	"fmt"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
)

// ControllerCacheEntry stores a cached controller resource
type ControllerCacheEntry struct {
	Resource    interface{}
	Error       error
	LastFetched time.Time
}

// ControllerCache provides caching for controller resources
type ControllerCache struct {
	mutex        sync.RWMutex
	deployments  map[string]*ControllerCacheEntry
	replicaSets  map[string]*ControllerCacheEntry
	statefulSets map[string]*ControllerCacheEntry
	daemonSets   map[string]*ControllerCacheEntry
	client       appsv1client.AppsV1Interface
	cacheTTL     time.Duration
}

// NewControllerCache creates a new controller cache
func NewControllerCache(client appsv1client.AppsV1Interface, cacheTTL time.Duration) *ControllerCache {
	return &ControllerCache{
		deployments:  make(map[string]*ControllerCacheEntry),
		replicaSets:  make(map[string]*ControllerCacheEntry),
		statefulSets: make(map[string]*ControllerCacheEntry),
		daemonSets:   make(map[string]*ControllerCacheEntry),
		client:       client,
		cacheTTL:     cacheTTL,
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

	for key, entry := range c.deployments {
		if entry.LastFetched.Before(expiredTime) {
			delete(c.deployments, key)
		}
	}

	for key, entry := range c.replicaSets {
		if entry.LastFetched.Before(expiredTime) {
			delete(c.replicaSets, key)
		}
	}

	for key, entry := range c.statefulSets {
		if entry.LastFetched.Before(expiredTime) {
			delete(c.statefulSets, key)
		}
	}

	for key, entry := range c.daemonSets {
		if entry.LastFetched.Before(expiredTime) {
			delete(c.daemonSets, key)
		}
	}
}

// makeKey creates a cache key from namespace and name
func makeKey(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}

// GetDeployment gets a deployment from cache or API server
func (c *ControllerCache) GetDeployment(namespace, name string) (*appsv1.Deployment, error) {
	key := makeKey(namespace, name)

	// Check cache first
	c.mutex.RLock()
	entry, found := c.deployments[key]
	c.mutex.RUnlock()

	now := time.Now()
	if found && now.Sub(entry.LastFetched) < c.cacheTTL {
		if entry.Error != nil {
			return nil, entry.Error
		}
		return entry.Resource.(*appsv1.Deployment), nil
	}

	// Not in cache or too old, fetch from API
	deployment, err := c.client.Deployments(namespace).Get(context.TODO(), name, metav1.GetOptions{})

	// Update cache
	c.mutex.Lock()
	c.deployments[key] = &ControllerCacheEntry{
		Resource:    deployment,
		Error:       err,
		LastFetched: now,
	}
	c.mutex.Unlock()

	return deployment, err
}

// GetReplicaSet gets a replica set from cache or API server
func (c *ControllerCache) GetReplicaSet(namespace, name string) (*appsv1.ReplicaSet, error) {
	key := makeKey(namespace, name)

	// Check cache first
	c.mutex.RLock()
	entry, found := c.replicaSets[key]
	c.mutex.RUnlock()

	now := time.Now()
	if found && now.Sub(entry.LastFetched) < c.cacheTTL {
		if entry.Error != nil {
			return nil, entry.Error
		}
		return entry.Resource.(*appsv1.ReplicaSet), nil
	}

	// Not in cache or too old, fetch from API
	replicaSet, err := c.client.ReplicaSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})

	// Update cache
	c.mutex.Lock()
	c.replicaSets[key] = &ControllerCacheEntry{
		Resource:    replicaSet,
		Error:       err,
		LastFetched: now,
	}
	c.mutex.Unlock()

	return replicaSet, err
}

// GetStatefulSet gets a stateful set from cache or API server
func (c *ControllerCache) GetStatefulSet(namespace, name string) (*appsv1.StatefulSet, error) {
	key := makeKey(namespace, name)

	// Check cache first
	c.mutex.RLock()
	entry, found := c.statefulSets[key]
	c.mutex.RUnlock()

	now := time.Now()
	if found && now.Sub(entry.LastFetched) < c.cacheTTL {
		if entry.Error != nil {
			return nil, entry.Error
		}
		return entry.Resource.(*appsv1.StatefulSet), nil
	}

	// Not in cache or too old, fetch from API
	statefulSet, err := c.client.StatefulSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})

	// Update cache
	c.mutex.Lock()
	c.statefulSets[key] = &ControllerCacheEntry{
		Resource:    statefulSet,
		Error:       err,
		LastFetched: now,
	}
	c.mutex.Unlock()

	return statefulSet, err
}

// GetDaemonSet gets a daemon set from cache or API server
func (c *ControllerCache) GetDaemonSet(namespace, name string) (*appsv1.DaemonSet, error) {
	key := makeKey(namespace, name)

	// Check cache first
	c.mutex.RLock()
	entry, found := c.daemonSets[key]
	c.mutex.RUnlock()

	now := time.Now()
	if found && now.Sub(entry.LastFetched) < c.cacheTTL {
		if entry.Error != nil {
			return nil, entry.Error
		}
		return entry.Resource.(*appsv1.DaemonSet), nil
	}

	// Not in cache or too old, fetch from API
	daemonSet, err := c.client.DaemonSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})

	// Update cache
	c.mutex.Lock()
	c.daemonSets[key] = &ControllerCacheEntry{
		Resource:    daemonSet,
		Error:       err,
		LastFetched: now,
	}
	c.mutex.Unlock()

	return daemonSet, err
}
