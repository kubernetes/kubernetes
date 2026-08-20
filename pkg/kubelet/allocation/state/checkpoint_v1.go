/*
Copyright The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

// PodResourceInfo stores resource requirements for containers within a pod.
type PodResourceInfo struct {
	// ContainerResources maps container names to their respective ResourceRequirements.
	ContainerResources map[string]v1.ResourceRequirements

	// PodLevelResources represents resource requirements that apply to the entire pod, if any.
	PodLevelResources *v1.ResourceRequirements

	// EmptyDirVolumeLimits maps emptyDir volume names to their respective resource limits, if any.
	EmptyDirVolumeLimits map[string]*resource.Quantity
}

// PodResourceInfoMap maps pod UIDs to their corresponding PodResourceInfo,
// tracking resource requirements for all containers within each pod.
type PodResourceInfoMap map[types.UID]PodResourceInfo

type PodResourceCheckpointInfo struct {
	Entries PodResourceInfoMap `json:"entries,omitempty"`
}

// Clone returns a copy of PodResourceInfoMap
func (pr PodResourceInfoMap) Clone() PodResourceInfoMap {
	prCopy := make(PodResourceInfoMap)
	for podUID, podInfo := range pr {
		newPodInfo := PodResourceInfo{
			ContainerResources: make(map[string]v1.ResourceRequirements),
			PodLevelResources:  podInfo.PodLevelResources.DeepCopy(),
		}
		for containerName, containerInfo := range podInfo.ContainerResources {
			newPodInfo.ContainerResources[containerName] = *containerInfo.DeepCopy()
		}
		if podInfo.EmptyDirVolumeLimits != nil {
			newPodInfo.EmptyDirVolumeLimits = make(map[string]*resource.Quantity)
			for volumeName, volumeLimit := range podInfo.EmptyDirVolumeLimits {
				if volumeLimit == nil {
					newPodInfo.EmptyDirVolumeLimits[volumeName] = nil
				} else {
					vl := volumeLimit.DeepCopy()
					newPodInfo.EmptyDirVolumeLimits[volumeName] = &vl
				}
			}
		}
		prCopy[podUID] = newPodInfo
	}
	return prCopy
}
