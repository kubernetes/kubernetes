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

package endpoint

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	clientcache "k8s.io/client-go/tools/cache"

	"github.com/golang/glog"
)

type Cache interface {
	// AddPod adds pod to the corresponding node cache
	AddPod(pod *v1.Pod) error

	// UpdatePod removes oldPod's information and adds newPod's information.
	UpdatePod(oldPod, newPod *v1.Pod) error

	// RemovePod removes a pod. The pod's information would be subtracted from assigned node.
	RemovePod(pod *v1.Pod) error

	// AddNode adds overall information about node.
	AddNode(node *v1.Node) error

	// RemoveNode removes overall information about node.
	RemoveNode(node *v1.Node) error

	// ListPodsOnNode fetch all pods on the given node.
	ListPodsOnNode(node string) ([]*v1.Pod, error)
}

type NodePodsCache struct {
	// This mutex guards all fields within this cache struct.
	mu sync.Mutex
	// We need to know what pods in the node when node changes, in order to enqueue pod's service.
	NodePods map[string][]*v1.Pod
}

func NewNodePodsCache() *NodePodsCache {
	return &NodePodsCache{
		NodePods: make(map[string][]*v1.Pod),
	}
}

func (cache *NodePodsCache) AddPod(pod *v1.Pod) error {
	node := pod.Spec.NodeName
	// We only accept scheduled pod - pod.Spec.NodeName should not be empty
	if node == "" {
		return nil
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()
	_, ok := cache.NodePods[node]
	if !ok {
		// We may receive pod create event before node create event
		cache.NodePods[node] = []*v1.Pod{pod}
		return nil
	}
	cache.NodePods[node] = append(cache.NodePods[node], pod)
	return nil
}

func (cache *NodePodsCache) UpdatePod(oldPod, newPod *v1.Pod) error {
	if err := cache.RemovePod(oldPod); err != nil {
		glog.Errorf("Error removing pod: %v", err)
		return err
	}
	return cache.AddPod(newPod)
}

func (cache *NodePodsCache) RemovePod(pod *v1.Pod) error {
	key, err := getPodKey(pod)
	if err != nil {
		return err
	}

	node := pod.Spec.NodeName
	// Ignore unscheduled pod
	if node == "" {
		return nil
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	pods, ok := cache.NodePods[node]
	if !ok {
		return fmt.Errorf("error removing pod from non-exist node %s", node)
	}
	for i := range pods {
		k, err := getPodKey(pods[i])
		if err != nil {
			return err
		}
		if k == key {
			// delete the element
			cache.NodePods[node] = append(cache.NodePods[node][:i], cache.NodePods[node][i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("no corresponding pod %s in pods of node %s", pod.Name, node)
}

func (cache *NodePodsCache) AddNode(node *v1.Node) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	_, ok := cache.NodePods[node.Name]
	if !ok {
		cache.NodePods[node.Name] = []*v1.Pod{}
	}
	// We may have already updated nodePodsCache in some caches where pod create event arrived
	// before node create event, see AddPod().
	return nil
}

func (cache *NodePodsCache) RemoveNode(node *v1.Node) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// We may receive error when call RemovePod() if there are any pods on this node.
	// Since notifications about pods are delivered in a different watch, and thus can potentially
	// be observed later, even though they happened before node removal.
	// However, if we don't delete it here, we may never have chance to do it.
	delete(cache.NodePods, node.Name)

	return nil
}

func (cache *NodePodsCache) ListPodsOnNode(node string) ([]*v1.Pod, error) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	return cache.NodePods[node], nil
}

// getPodKey returns the string key of a pod.
func getPodKey(pod *v1.Pod) (string, error) {
	return clientcache.MetaNamespaceKeyFunc(pod)
}
