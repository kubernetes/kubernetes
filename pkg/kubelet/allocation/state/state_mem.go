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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

type stateMemory struct {
	sync.RWMutex
	podResources PodResourceInfoMap
}

var _ State = &stateMemory{}

// NewStateMemory creates new State to track resources resourcesated to pods
func NewStateMemory(logger klog.Logger, resources PodResourceInfoMap) State {
	if resources == nil {
		resources = PodResourceInfoMap{}
	}
	logger.V(2).Info("Initialized new in-memory state store for pod resource information tracking")
	return &stateMemory{
		podResources: resources,
	}
}

func (s *stateMemory) GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	resourceInfo, ok := s.podResources[podUID]
	if !ok {
		return v1.ResourceRequirements{}, false
	}

	for c := range podutil.ContainerIter(resourceInfo.PodSpec, podutil.InitContainers|podutil.Containers) {
		if c.Name == containerName {
			return *c.Resources.DeepCopy(), true
		}
	}
	return v1.ResourceRequirements{}, false
}

// GetPodLevelResources returns current resources information at pod-level
func (s *stateMemory) GetPodLevelResources(podUID types.UID) (*v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	pr, ok := s.podResources[podUID]
	if !ok {
		return nil, false
	}

	if pr.PodSpec.Resources == nil {
		return nil, false
	}
	return pr.PodSpec.Resources.DeepCopy(), true
}

// GetEmptyDirVolumeLimit returns current resources information for emptyDir volume
func (s *stateMemory) GetEmptyDirVolumeLimit(podUID types.UID, volumeName string) (*resource.Quantity, bool) {
	s.RLock()
	defer s.RUnlock()

	pr, ok := s.podResources[podUID]
	if !ok {
		return nil, false
	}

	for _, vol := range pr.PodSpec.Volumes {
		if vol.Name == volumeName && vol.EmptyDir != nil {
			if vol.EmptyDir.SizeLimit == nil {
				return nil, false
			}
			sizeLimitCopy := vol.EmptyDir.SizeLimit.DeepCopy()
			return &sizeLimitCopy, true
		}
	}
	return nil, false
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

func (s *stateMemory) SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, resources v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.podResources[podUID]
	if !ok {
		podInfo = PodResourceInfo{
			PodSpec: &v1.PodSpec{},
		}
	}

	found := false
	for c := range podutil.ContainerIter(podInfo.PodSpec, podutil.InitContainers|podutil.Containers) {
		if c.Name == containerName {
			c.Resources = resources
			found = true
			break
		}
	}
	if !found {
		podInfo.PodSpec.Containers = append(podInfo.PodSpec.Containers, v1.Container{
			Name:      containerName,
			Resources: resources,
		})
	}

	s.podResources[podUID] = podInfo
	logger.V(3).Info("Updated container resource information in PodSpec", "podUID", podUID, "containerName", containerName, "resources", resources)
	return nil
}

func (s *stateMemory) SetPodLevelResources(logger klog.Logger, podUID types.UID, resources *v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.podResources[podUID]
	if !ok {
		podInfo = PodResourceInfo{
			PodSpec: &v1.PodSpec{},
		}
	}

	podInfo.PodSpec.Resources = resources
	s.podResources[podUID] = podInfo

	logger.V(3).Info("Updated pod-level resource info in PodSpec", "podUID", podUID, "resources", resources)
	return nil
}

func (s *stateMemory) SetEmptyDirVolumeLimit(podUID types.UID, volumeName string, limit *resource.Quantity) error {
	logger := klog.TODO()
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.podResources[podUID]
	if !ok {
		podInfo = PodResourceInfo{
			PodSpec: &v1.PodSpec{},
		}
	}

	found := false
	for i, vol := range podInfo.PodSpec.Volumes {
		if vol.Name == volumeName && vol.EmptyDir != nil {
			podInfo.PodSpec.Volumes[i].EmptyDir.SizeLimit = limit
			found = true
			break
		}
	}
	if !found {
		podInfo.PodSpec.Volumes = append(podInfo.PodSpec.Volumes, v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{
					SizeLimit: limit,
				},
			},
		})
	}

	s.podResources[podUID] = podInfo
	logger.V(3).Info("Updated emptyDir volume limit in PodSpec", "podUID", podUID, "volumeName", volumeName, "limit", limit)
	return nil
}

func (s *stateMemory) SetPodResourceInfo(logger klog.Logger, podUID types.UID, resourceInfo PodResourceInfo) error {
	s.Lock()
	defer s.Unlock()

	s.podResources[podUID] = resourceInfo
	logger.V(3).Info("Updated pod resource information", "podUID", podUID, "information", resourceInfo)
	return nil
}

func (s *stateMemory) RemovePod(logger klog.Logger, podUID types.UID) error {
	s.Lock()
	defer s.Unlock()
	delete(s.podResources, podUID)
	logger.V(3).Info("Deleted pod resource information", "podUID", podUID)
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
