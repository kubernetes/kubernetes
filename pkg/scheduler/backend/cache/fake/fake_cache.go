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
	"k8s.io/klog/v2"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
)

// Cache is used for testing
type Cache struct {
	internalcache.Cache
	AssumeFunc         func(*v1.Pod)
	ForgetFunc         func(*v1.Pod)
	IsAssumedPodFunc   func(*v1.Pod) bool
	GetPodFunc         func(*v1.Pod) *v1.Pod
	UpdateSnapshotFunc func(nodeSnapshot *internalcache.Snapshot) error
}

// AssumePod allows to mock this method for testing.
func (c *Cache) AssumePod(logger klog.Logger, pod *v1.Pod) error {
	if c.AssumeFunc != nil {
		c.AssumeFunc(pod)
		return nil
	}
	return c.Cache.AssumePod(logger, pod)
}

// ForgetPod allows to mock this method for testing.
func (c *Cache) ForgetPod(logger klog.Logger, pod *v1.Pod) error {
	if c.ForgetFunc != nil {
		c.ForgetFunc(pod)
		return nil
	}
	return c.Cache.ForgetPod(logger, pod)
}

// IsAssumedPod allows to mock this method for testing.
func (c *Cache) IsAssumedPod(pod *v1.Pod) (bool, error) {
	if c.IsAssumedPodFunc != nil {
		return c.IsAssumedPodFunc(pod), nil
	}
	return c.Cache.IsAssumedPod(pod)
}

// GetPod allows to mock this method for testing.
func (c *Cache) GetPod(pod *v1.Pod) (*v1.Pod, error) {
	if c.GetPodFunc != nil {
		return c.GetPodFunc(pod), nil
	}
	return c.Cache.GetPod(pod)
}

// UpdateSnapshot allows to mock this method for testing.
func (c *Cache) UpdateSnapshot(logger klog.Logger, nodeSnapshot *internalcache.Snapshot) error {
	if c.UpdateSnapshotFunc != nil {
		return c.UpdateSnapshotFunc(nodeSnapshot)
	}
	return c.Cache.UpdateSnapshot(logger, nodeSnapshot)
}
