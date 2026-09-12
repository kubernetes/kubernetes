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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

type stateMemory struct {
	sync.RWMutex
	pods PodMap
}

var _ State = &stateMemory{}

func newStateMemory(logger klog.Logger, pods PodMap) *stateMemory {
	if pods == nil {
		pods = PodMap{}
	}
	logger.V(2).Info("Initialized new in-memory state store for pod resource information tracking")
	return &stateMemory{
		pods: pods,
	}
}

// NewStateMemory creates new State to track resources allocated to pods
func NewStateMemory(logger klog.Logger, pods PodMap) State {
	return newStateMemory(logger, pods.Clone())
}

func (s *stateMemory) GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok || pod == nil {
		return v1.ResourceRequirements{}, false
	}

	for c := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
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

	pod, ok := s.pods[podUID]
	if !ok || pod == nil || pod.Spec.Resources == nil {
		return nil, false
	}

	return pod.Spec.Resources.DeepCopy(), true
}

// GetEmptyDirVolumeLimit returns current resources information for emptyDir volume
func (s *stateMemory) GetEmptyDirVolumeLimit(podUID types.UID, volumeName string) (*resource.Quantity, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok || pod == nil {
		return nil, false
	}

	for _, vol := range pod.Spec.Volumes {
		if vol.Name == volumeName && vol.EmptyDir != nil {
			if vol.EmptyDir.SizeLimit == nil || vol.EmptyDir.SizeLimit.IsZero() {
				return nil, false
			}
			sizeLimitCopy := vol.EmptyDir.SizeLimit.DeepCopy()
			return &sizeLimitCopy, true
		}
	}
	return nil, false
}

func (s *stateMemory) GetPodMap() PodMap {
	s.RLock()
	defer s.RUnlock()
	return s.pods.Clone()
}

func (s *stateMemory) GetPodUIDs() []types.UID {
	s.RLock()
	defer s.RUnlock()
	uids := make([]types.UID, 0, len(s.pods))
	for uid := range s.pods {
		uids = append(uids, uid)
	}
	return uids
}

func (s *stateMemory) GetPod(podUID types.UID) (*v1.Pod, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok || pod == nil {
		return nil, false
	}
	return pod.DeepCopy(), true
}

func (s *stateMemory) HasPod(podUID types.UID) bool {
	s.RLock()
	defer s.RUnlock()

	_, ok := s.pods[podUID]
	return ok
}

func (s *stateMemory) toPodList() *v1.PodList {
	s.RLock()
	defer s.RUnlock()

	podList := &v1.PodList{Items: make([]v1.Pod, 0, len(s.pods))}
	for _, pod := range s.pods {
		if pod != nil {
			podList.Items = append(podList.Items, *pod.DeepCopy())
		}
	}
	return podList
}

func (s *stateMemory) SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, containerType podutil.ContainerType, resources v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	pod, ok := s.pods[podUID]
	if !ok || pod == nil {
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: podUID,
			},
		}
	} else {
		pod = pod.DeepCopy()
	}

	found := false
	for c := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if c.Name == containerName {
			c.Resources = *resources.DeepCopy()
			found = true
			break
		}
	}
	if !found {
		newContainer := v1.Container{
			Name:      containerName,
			Resources: *resources.DeepCopy(),
		}
		if containerType == podutil.InitContainers {
			pod.Spec.InitContainers = append(pod.Spec.InitContainers, newContainer)
		} else {
			pod.Spec.Containers = append(pod.Spec.Containers, newContainer)
		}
	}

	s.pods[podUID] = pod
	logger.V(3).Info("Updated container resource information in PodSpec", "podUID", podUID, "containerName", containerName, "resources", resources)
	return nil
}

func (s *stateMemory) SetPodLevelResources(logger klog.Logger, podUID types.UID, resources *v1.ResourceRequirements) error {
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.pods[podUID]
	if !ok || podInfo == nil {
		podInfo = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: podUID,
			},
		}
	} else {
		podInfo = podInfo.DeepCopy()
	}

	if resources == nil {
		podInfo.Spec.Resources = nil
	} else {
		podInfo.Spec.Resources = resources.DeepCopy()
	}
	s.pods[podUID] = podInfo

	logger.V(3).Info("Updated pod-level resource info in PodSpec", "podUID", podUID, "resources", resources)
	return nil
}

func (s *stateMemory) SetEmptyDirVolumeLimit(podUID types.UID, volumeName string, limit *resource.Quantity) error {
	logger := klog.TODO()
	s.Lock()
	defer s.Unlock()

	podInfo, ok := s.pods[podUID]
	if !ok || podInfo == nil {
		podInfo = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: podUID,
			},
		}
	} else {
		podInfo = podInfo.DeepCopy()
	}

	found := false
	for i, vol := range podInfo.Spec.Volumes {
		if vol.Name == volumeName && vol.EmptyDir != nil {
			if limit == nil {
				podInfo.Spec.Volumes[i].EmptyDir.SizeLimit = nil
			} else {
				limitCopy := limit.DeepCopy()
				podInfo.Spec.Volumes[i].EmptyDir.SizeLimit = &limitCopy
			}
			podInfo.Spec.Volumes[i].EmptyDir.Medium = v1.StorageMediumMemory
			found = true
			break
		}
	}
	if !found {
		var limitCopy *resource.Quantity
		if limit != nil {
			lc := limit.DeepCopy()
			limitCopy = &lc
		}
		podInfo.Spec.Volumes = append(podInfo.Spec.Volumes, v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{
					Medium:    v1.StorageMediumMemory,
					SizeLimit: limitCopy,
				},
			},
		})
	}

	s.pods[podUID] = podInfo
	logger.V(3).Info("Updated emptyDir volume limit in PodSpec", "podUID", podUID, "volumeName", volumeName, "limit", limit)
	return nil
}

func (s *stateMemory) SetPod(logger klog.Logger, pod *v1.Pod) error {
	s.Lock()
	defer s.Unlock()

	if pod == nil {
		return nil
	}
	s.pods[pod.UID] = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       pod.UID,
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: *pod.Spec.DeepCopy(),
	}
	logger.V(3).Info("Updated allocated pod", "podUID", pod.UID)
	return nil
}

func (s *stateMemory) RemovePod(logger klog.Logger, podUID types.UID) error {
	s.Lock()
	defer s.Unlock()
	delete(s.pods, podUID)
	logger.V(3).Info("Deleted pod resource information", "podUID", podUID)
	return nil
}

func (s *stateMemory) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	s.Lock()
	defer s.Unlock()

	for podUID := range s.pods {
		if _, ok := remainingPods[types.UID(podUID)]; !ok {
			delete(s.pods, podUID)
		}
	}
}
