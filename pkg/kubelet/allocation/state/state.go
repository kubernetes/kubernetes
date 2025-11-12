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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

// PodResourceInfo stores resource requirements for containers within a pod.
type PodResourceInfo struct {
	// ContainerResources maps container names to their respective ResourceRequirements.
	ContainerResources map[string]v1.ResourceRequirements
	PodLevelResources  *v1.ResourceRequirements
}

// PodResourceInfoMap maps pod UIDs to their corresponding PodResourceInfo,
// tracking resource requirements for all containers within each pod.
type PodResourceInfoMap map[types.UID]PodResourceInfo

// Clone returns a copy of PodResourceInfoMap
func (pr PodResourceInfoMap) Clone() PodResourceInfoMap {
	prCopy := make(PodResourceInfoMap)
	for podUID, podInfo := range pr {
		prCopy[podUID] = PodResourceInfo{
			ContainerResources: make(map[string]v1.ResourceRequirements),
			PodLevelResources:  podInfo.PodLevelResources.DeepCopy(),
		}
		for containerName, containerInfo := range podInfo.ContainerResources {
			prCopy[podUID].ContainerResources[containerName] = *containerInfo.DeepCopy()
		}
	}
	return prCopy
}

// Reader interface used to read current pod resource state
type Reader interface {
	GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)
	GetPodResourceInfoMap() PodResourceInfoMap
	GetPodResourceInfo(podUID types.UID) (PodResourceInfo, bool)
	GetPodLevelResources(podUID types.UID) (*v1.ResourceRequirements, bool)
}

type writer interface {
	SetContainerResources(podUID types.UID, containerName string, resources v1.ResourceRequirements) error
	SetPodResourceInfo(podUID types.UID, resourceInfo PodResourceInfo) error
	SetPodLevelResources(podUID types.UID, alloc *v1.ResourceRequirements) error
	RemovePod(podUID types.UID) error
	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])
}

// State interface provides methods for tracking and setting pod resources
type State interface {
	Reader
	writer
}
