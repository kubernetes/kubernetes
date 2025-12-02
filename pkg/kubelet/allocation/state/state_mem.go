/*
Copyright 2021 The Kubernetes Authors.

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

package state

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

type stateMemory struct {
	sync.RWMutex
	podResources PodResourceInfoMap
}

var _ State = &stateMemory{}

// NewStateMemory creates new State to track resources resourcesated to pods
func NewStateMemory(resources PodResourceInfoMap) State {
	if resources == nil {
		resources = PodResourceInfoMap{}
	}
	klog.V(2).InfoS("Initialized new in-memory state store for pod resource information tracking")
	return &stateMemory{
		podResources: resources,
	}
}

func (s *stateMemory) GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	resourceInfo, ok := s.podResources[podUID]
	if !ok {
		return v1.ResourceRequirements{}, ok
	}

	resources, ok := resourceInfo.ContainerResources[containerName]
	if !ok {
		return v1.ResourceRequirements{}, ok
	}
	return *resources.DeepCopy(), ok
}

// GetPodLevelResources returns current resources information at pod-level
func (s *stateMemory) GetPodLevelResources(podUID types.UID) (*v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	pr, ok := s.podResources[podUID]
	if !ok {
		return nil, ok
	}

	return pr.PodLevelResources.DeepCopy(), ok
}

func (s *stateMemory) GetPodResourceInfoMap() PodResourceInfoMap {
	s.RLock()
	defer s.RUnlock()
	return s.podResources.Clone()
}

func (s *stateMemory) GetPodResourceInfo(podUID types.UID) (PodResourceInfo, bool) {
	s.RLock()
	defer s.RUnlock()

	resourceInfo, ok := s.podResources[podUID]
	return resourceInfo, ok
}

func (s *stateMemory) SetContainerResources(podUID types.UID, containerName string, resources v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.podResources[podUID]
	if !ok {
		podInfo = PodResourceInfo{
			ContainerResources: make(map[string]v1.ResourceRequirements),
		}
	}

	if podInfo.ContainerResources == nil {
		podInfo.ContainerResources = make(map[string]v1.ResourceRequirements)
	}

	podInfo.ContainerResources[containerName] = resources
	s.podResources[podUID] = podInfo

	klog.V(3).InfoS("Updated container resource information", "podUID", podUID, "containerName", containerName, "resources", resources)
	return nil
}

func (s *stateMemory) SetPodLevelResources(podUID types.UID, resources *v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.podResources[podUID]
	if !ok {
		podInfo.PodLevelResources = &v1.ResourceRequirements{}
	}

	podInfo.PodLevelResources = resources

	s.podResources[podUID] = podInfo

	klog.V(3).InfoS("Updated pod-level resource info", "podUID", podUID, "resources", resources)
	return nil
}

func (s *stateMemory) SetPodResourceInfo(podUID types.UID, resourceInfo PodResourceInfo) error {
	s.Lock()
	defer s.Unlock()

	s.podResources[podUID] = resourceInfo
	klog.V(3).InfoS("Updated pod resource information", "podUID", podUID, "information", resourceInfo)
	return nil
}

func (s *stateMemory) RemovePod(podUID types.UID) error {
	s.Lock()
	defer s.Unlock()
	delete(s.podResources, podUID)
	klog.V(3).InfoS("Deleted pod resource information", "podUID", podUID)
	return nil
}

func (s *stateMemory) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	s.Lock()
	defer s.Unlock()

	for podUID := range s.podResources {
		if _, ok := remainingPods[types.UID(podUID)]; !ok {
			delete(s.podResources, podUID)
		}
	}
}
