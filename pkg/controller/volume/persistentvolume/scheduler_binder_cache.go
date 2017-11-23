/*
Copyright 2017 The Kubernetes Authors.

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

package persistentvolume

import (
	"sync"

	"k8s.io/api/core/v1"
)

// podBindingCache stores PV binding decisions per pod per node.
// Pod entries are removed when the Pod is deleted or updated to
// no longer be schedulable.
type PodBindingCache interface {
	// UpdateBindings will update the cache with the given bindings for the
	// pod and node.
	UpdateBindings(pod *v1.Pod, node string, bindings []*bindingInfo)

	// DeleteBindings will remove all cached bindings for the given pod.
	DeleteBindings(pod *v1.Pod)

	// GetBindings will return the cached bindings for the given pod and node.
	GetBindings(pod *v1.Pod, node string) []*bindingInfo
}

type podBindingCache struct {
	mutex sync.Mutex

	// Key = pod name
	// Value = nodeBindings
	bindings map[string]nodeBindings
}

// Key = nodeName
// Value = array of bindingInfo
type nodeBindings map[string][]*bindingInfo

func NewPodBindingCache() PodBindingCache {
	return &podBindingCache{bindings: map[string]nodeBindings{}}
}

func (c *podBindingCache) DeleteBindings(pod *v1.Pod) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	podName := getPodName(pod)
	delete(c.bindings, podName)
}

func (c *podBindingCache) UpdateBindings(pod *v1.Pod, node string, bindings []*bindingInfo) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	podName := getPodName(pod)
	nodeBinding, ok := c.bindings[podName]
	if !ok {
		nodeBinding = nodeBindings{}
		c.bindings[podName] = nodeBinding
	}
	nodeBinding[node] = bindings
}

func (c *podBindingCache) GetBindings(pod *v1.Pod, node string) []*bindingInfo {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	podName := getPodName(pod)
	nodeBindings, ok := c.bindings[podName]
	if !ok {
		return nil
	}
	return nodeBindings[node]
}
