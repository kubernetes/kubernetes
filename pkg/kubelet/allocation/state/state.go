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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// Reader interface used to read current pod resource state
type Reader interface {
	GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)
	GetPodResourceInfoMap() PodResourceInfoMap
	GetPodResourceInfo(podUID types.UID) (PodResourceInfo, bool)
	GetPodLevelResources(podUID types.UID) (*v1.ResourceRequirements, bool)
	GetEmptyDirVolumeLimit(podUID types.UID, volumeName string) (*resource.Quantity, bool)
}

type writer interface {
	SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, containerType podutil.ContainerType, resources v1.ResourceRequirements) error
	SetPodResourceInfo(logger klog.Logger, podUID types.UID, resourceInfo PodResourceInfo) error
	SetPodLevelResources(logger klog.Logger, podUID types.UID, alloc *v1.ResourceRequirements) error
	SetEmptyDirVolumeLimit(podUID types.UID, volumeName string, limit *resource.Quantity) error
	RemovePod(logger klog.Logger, podUID types.UID) error
	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])
}

// State interface provides methods for tracking and setting pod resources
type State interface {
	Reader
	writer
}
