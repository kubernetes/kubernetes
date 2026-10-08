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
	"fmt"
	"maps"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// stateMemory never modifies a stored pod. It replaces it with an updated copy that shares whatever
// the update leaves alone, so toPodList can hand out stored pods without copying them.
type stateMemory struct {
	sync.RWMutex
	pods PodMap
}

var _ State = &stateMemory{}

// NewStateMemory creates new State to track resources resourcesated to pods
func NewStateMemory(logger klog.Logger, pods PodMap) State {
	return newStateMemory(logger, pods)
}

func newStateMemory(logger klog.Logger, pods PodMap) *stateMemory {
	if pods == nil {
		pods = PodMap{}
	}
	logger.V(2).Info("Initialized new in-memory state store for pod resource information tracking")
	return &stateMemory{
		pods: pods,
	}
}

func (s *stateMemory) GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok {
		return v1.ResourceRequirements{}, ok
	}

	// Look in all kinds of containers: a pod migrated from a V1 checkpoint has all of them in Spec.Containers.
	for c := range podutil.ContainerIter(&pod.Spec, podutil.AllContainers) {
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
	if !ok {
		return nil, ok
	}

	return pod.Spec.Resources.DeepCopy(), ok
}

// GetEmptyDirVolumeLimit returns current resources information for emptyDir volume
func (s *stateMemory) GetEmptyDirVolumeLimit(podUID types.UID, volumeName string) (*resource.Quantity, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok {
		return nil, ok
	}

	for _, vol := range pod.Spec.Volumes {
		if vol.Name == volumeName && vol.EmptyDir != nil && vol.EmptyDir.SizeLimit != nil {
			sizeLimitCopy := vol.EmptyDir.SizeLimit.DeepCopy()
			return &sizeLimitCopy, ok
		}
	}
	return nil, false
}

func (s *stateMemory) GetPodUIDs() []types.UID {
	s.RLock()
	defer s.RUnlock()
	return slices.Collect(maps.Keys(s.pods))
}

func (s *stateMemory) GetPod(podUID types.UID) (*v1.Pod, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	return pod.DeepCopy(), ok
}

func (s *stateMemory) HasPod(podUID types.UID) bool {
	s.RLock()
	defer s.RUnlock()

	_, ok := s.pods[podUID]
	return ok
}

// toPodList returns the stored pods, in no particular order.
func (s *stateMemory) toPodList() *v1.PodList {
	s.RLock()
	defer s.RUnlock()

	podList := &v1.PodList{Items: make([]v1.Pod, 0, len(s.pods))}
	for _, pod := range s.pods {
		podList.Items = append(podList.Items, *pod)
	}
	return podList
}

// updatePod stores the result of applying mutate to a copy of the pod, which is a new pod with only
// the UID set if none is stored. The copy is shallow, so mutate must replace what it changes
// instead of modifying it in place. If mutate fails, the stored pod is left as it was.
func (s *stateMemory) updatePod(podUID types.UID, mutate func(pod *v1.Pod) error) error {
	s.Lock()
	defer s.Unlock()

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}}
	if stored, ok := s.pods[podUID]; ok {
		podCopy := *stored
		pod = &podCopy
	}
	if err := mutate(pod); err != nil {
		return err
	}
	s.pods[podUID] = pod
	return nil
}

func (s *stateMemory) SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, containerType podutil.ContainerType, resources v1.ResourceRequirements) error {
	err := s.updatePod(podUID, func(pod *v1.Pod) error {
		pod.Spec.InitContainers = slices.Clone(pod.Spec.InitContainers)
		pod.Spec.Containers = slices.Clone(pod.Spec.Containers)
		pod.Spec.EphemeralContainers = slices.Clone(pod.Spec.EphemeralContainers)

		for c := range podutil.ContainerIter(&pod.Spec, podutil.AllContainers) {
			if c.Name == containerName {
				c.Resources = *resources.DeepCopy()
				return nil
			}
		}

		container := v1.Container{Name: containerName, Resources: *resources.DeepCopy()}
		switch containerType {
		case podutil.InitContainers:
			pod.Spec.InitContainers = append(pod.Spec.InitContainers, container)
		case podutil.Containers:
			pod.Spec.Containers = append(pod.Spec.Containers, container)
		case podutil.EphemeralContainers:
			pod.Spec.EphemeralContainers = append(pod.Spec.EphemeralContainers,
				v1.EphemeralContainer{EphemeralContainerCommon: v1.EphemeralContainerCommon(container)})
		default:
			return fmt.Errorf("cannot add container %q: unsupported container type %d", containerName, containerType)
		}
		return nil
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated container resource information", "podUID", podUID, "containerName", containerName, "resources", resources)
	return nil
}

func (s *stateMemory) SetPodLevelResources(logger klog.Logger, podUID types.UID, resources *v1.ResourceRequirements) error {
	err := s.updatePod(podUID, func(pod *v1.Pod) error {
		pod.Spec.Resources = resources.DeepCopy()
		return nil
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated pod-level resource info", "podUID", podUID, "resources", resources)
	return nil
}

func (s *stateMemory) SetEmptyDirVolumeLimit(podUID types.UID, volumeName string, limit *resource.Quantity) error {
	logger := klog.TODO()

	var limitCopy *resource.Quantity
	if limit != nil {
		lc := limit.DeepCopy()
		limitCopy = &lc
	}

	err := s.updatePod(podUID, func(pod *v1.Pod) error {
		pod.Spec.Volumes = slices.Clone(pod.Spec.Volumes)
		for i := range pod.Spec.Volumes {
			vol := &pod.Spec.Volumes[i]
			if vol.Name == volumeName && vol.EmptyDir != nil {
				// Replace EmptyDir instead of modifying it, since the stored pod shares it.
				emptyDir := *vol.EmptyDir
				emptyDir.SizeLimit = limitCopy
				vol.EmptyDir = &emptyDir
				return nil
			}
		}

		// Nothing reads the medium from here, so it is not recorded.
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name:         volumeName,
			VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: limitCopy}},
		})
		return nil
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated emptyDir volume limit", "podUID", podUID, "volumeName", volumeName, "limit", limit)
	return nil
}

func (s *stateMemory) SetPod(logger klog.Logger, pod *v1.Pod) error {
	// Only the identity and the spec of the pod are kept. The spec is copied because the caller owns the pod.
	stored := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: pod.UID, Name: pod.Name, Namespace: pod.Namespace},
		Spec:       *pod.Spec.DeepCopy(),
	}

	s.Lock()
	defer s.Unlock()

	s.pods[pod.UID] = stored
	logger.V(3).Info("Updated pod resource information", "podUID", pod.UID)
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
