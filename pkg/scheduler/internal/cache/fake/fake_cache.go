/*
Copyright 2015 The Kubernetes Authors.

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

package fake

import (
	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

// Cache is used for testing
type Cache struct {
	AssumeFunc       func(*v1.Pod)
	ForgetFunc       func(*v1.Pod)
	IsAssumedPodFunc func(*v1.Pod) bool
	GetPodFunc       func(*v1.Pod) *v1.Pod
}

// AssumePod is a fake method for testing.
func (c *Cache) AssumePod(pod *v1.Pod) error {
	c.AssumeFunc(pod)
	return nil
}

// FinishBinding is a fake method for testing.
func (c *Cache) FinishBinding(pod *v1.Pod) error { return nil }

// ForgetPod is a fake method for testing.
func (c *Cache) ForgetPod(pod *v1.Pod) error {
	c.ForgetFunc(pod)
	return nil
}

// AddPod is a fake method for testing.
func (c *Cache) AddPod(pod *v1.Pod) error { return nil }

// UpdatePod is a fake method for testing.
func (c *Cache) UpdatePod(oldPod, newPod *v1.Pod) error { return nil }

// RemovePod is a fake method for testing.
func (c *Cache) RemovePod(pod *v1.Pod) error { return nil }

// IsAssumedPod is a fake method for testing.
func (c *Cache) IsAssumedPod(pod *v1.Pod) (bool, error) {
	return c.IsAssumedPodFunc(pod), nil
}

// GetPod is a fake method for testing.
func (c *Cache) GetPod(pod *v1.Pod) (*v1.Pod, error) {
	return c.GetPodFunc(pod), nil
}

// AddNode is a fake method for testing.
func (c *Cache) AddNode(node *v1.Node) error { return nil }

// UpdateNode is a fake method for testing.
func (c *Cache) UpdateNode(oldNode, newNode *v1.Node) error { return nil }

// RemoveNode is a fake method for testing.
func (c *Cache) RemoveNode(node *v1.Node) error { return nil }

// AddCSINode is a fake method for testing.
func (c *Cache) AddCSINode(csiNode *storagev1beta1.CSINode) error { return nil }

// UpdateCSINode is a fake method for testing.
func (c *Cache) UpdateCSINode(oldCSINode, newCSINode *storagev1beta1.CSINode) error { return nil }

// RemoveCSINode is a fake method for testing.
func (c *Cache) RemoveCSINode(csiNode *storagev1beta1.CSINode) error { return nil }

// UpdateNodeInfoSnapshot is a fake method for testing.
func (c *Cache) UpdateNodeInfoSnapshot(nodeSnapshot *internalcache.NodeInfoSnapshot) error {
	return nil
}

// List is a fake method for testing.
func (c *Cache) List(s labels.Selector) ([]*v1.Pod, error) { return nil, nil }

// FilteredList is a fake method for testing.
func (c *Cache) FilteredList(filter algorithm.PodFilter, selector labels.Selector) ([]*v1.Pod, error) {
	return nil, nil
}

// Snapshot is a fake method for testing
func (c *Cache) Snapshot() *internalcache.Snapshot {
	return &internalcache.Snapshot{}
}

// NodeTree is a fake method for testing.
func (c *Cache) NodeTree() *internalcache.NodeTree { return nil }

// GetNodeInfo is a fake method for testing.
func (c *Cache) GetNodeInfo(nodeName string) (*v1.Node, error) {
	return nil, nil
}

// ListNodes is a fake method for testing.
func (c *Cache) ListNodes() []*v1.Node {
	return nil
}

// GetCSINodeInfo is a fake method for testing.
func (c *Cache) GetCSINodeInfo(nodeName string) (*storagev1beta1.CSINode, error) {
	return nil, nil
}
