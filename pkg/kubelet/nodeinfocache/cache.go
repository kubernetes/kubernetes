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

package nodeinfocache

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// Cache maintains a cached NodeInfo for the kubelet's node.
// Thread-safe wrapper around framework.NodeInfo with incremental updates.
type Cache struct {
	mu       sync.RWMutex
	nodeInfo *framework.NodeInfo
}

// New creates a new empty NodeInfo cache.
func New() *Cache {
	ni := framework.NewNodeInfo()
	// Set a placeholder node so that NodeInfo.RemovePod error messages
	// can safely access Node().Name before SetNode is called.
	ni.SetNode(&v1.Node{})
	return &Cache{
		nodeInfo: ni,
	}
}

// SetNode updates the cached node metadata.
func (c *Cache) SetNode(node *v1.Node) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nodeInfo.SetNode(node)
}

// AddPod adds a pod to the cache incrementally.
func (c *Cache) AddPod(pod *v1.Pod) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nodeInfo.AddPod(pod)
}

// RemovePod removes a pod from the cache.
// If the pod is not found (e.g. it was rejected during admission), the error is logged and ignored.
func (c *Cache) RemovePod(logger klog.Logger, pod *v1.Pod) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.nodeInfo.RemovePod(logger, pod); err != nil {
		logger.V(4).Info("Pod not found in cache during remove", "pod", klog.KObj(pod), "err", err)
	}
}

// UpdatePod updates a pod in the cache (remove old, add new).
// If the old pod is not found (e.g. it was rejected during admission), the error is logged and ignored.
func (c *Cache) UpdatePod(logger klog.Logger, oldPod, newPod *v1.Pod) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.nodeInfo.RemovePod(logger, oldPod); err != nil {
		logger.V(4).Info("Pod not found in cache during update", "pod", klog.KObj(oldPod), "err", err)
	}
	c.nodeInfo.AddPod(newPod)
}

// Snapshot returns a deep copy of the cached NodeInfo for safe concurrent use.
func (c *Cache) Snapshot() *framework.NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.nodeInfo.SnapshotConcrete()
}

// PodCount returns the number of pods in the cache.
func (c *Cache) PodCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.nodeInfo.Pods)
}
