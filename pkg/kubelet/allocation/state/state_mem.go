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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// stateMemory never modifies a stored pod. It replaces it with an updated copy that shares whatever
// the update leaves alone, so toPodList and GetPodReadOnly can hand out stored pods without copying them.
type stateMemory struct {
	sync.RWMutex
	pods PodMap
	// maskFn mutates a copy of the pod before it is stored in SetPod, allowing callers
	// to drop fields not tracked by the state store (e.g. actuated state stripping
	// non-resizable resources to minimize memory usage).
	maskFn func(*v1.Pod)
}

var _ State = &stateMemory{}

// NewStateMemory creates new State to track resources allocated to pods
func NewStateMemory(logger klog.Logger, pods PodMap, maskFn func(*v1.Pod)) State {
	return newStateMemory(logger, pods, maskFn)
}

func newStateMemory(logger klog.Logger, pods PodMap, maskFn func(*v1.Pod)) *stateMemory {
	if pods == nil {
		pods = PodMap{}
	}
	logger.V(2).Info("Initialized new in-memory state store for pod resource information tracking")
	return &stateMemory{
		pods:   pods,
		maskFn: maskFn,
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
	for c := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if c.Name == containerName {
			return *c.Resources.DeepCopy(), true
		}
	}
	return v1.ResourceRequirements{}, false
}

func (s *stateMemory) GetContainerVolumeMounts(podUID types.UID, containerName string) ([]v1.VolumeMount, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok {
		return nil, false
	}

	for c := range podutil.ContainerIter(&pod.Spec, podutil.AllContainers) {
		if c.Name == containerName {
			if len(c.VolumeMounts) == 0 {
				return nil, true
			}
			mounts := make([]v1.VolumeMount, len(c.VolumeMounts))
			for i := range c.VolumeMounts {
				mounts[i] = *c.VolumeMounts[i].DeepCopy()
			}
			return mounts, true
		}
	}
	return nil, false
}

// GetPodVolumes returns current volumes information for a pod
func (s *stateMemory) GetPodVolumes(podUID types.UID) ([]v1.Volume, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok || len(pod.Spec.Volumes) == 0 {
		return nil, false
	}

	volumes := make([]v1.Volume, len(pod.Spec.Volumes))
	for i := range pod.Spec.Volumes {
		volumes[i] = *pod.Spec.Volumes[i].DeepCopy()
	}
	return volumes, true
}

func (s *stateMemory) GetContainer(podUID types.UID, containerName string, containerType podutil.ContainerType) (*v1.Container, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	if !ok {
		return nil, false
	}
	for c := range podutil.ContainerIter(&pod.Spec, containerType) {
		if c.Name == containerName {
			return c.DeepCopy(), true
		}
	}
	return nil, false
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
		if vol.Name == volumeName && vol.EmptyDir != nil {
			if vol.EmptyDir.SizeLimit == nil {
				return nil, true
			}
			sizeLimitCopy := *vol.EmptyDir.SizeLimit
			return &sizeLimitCopy, true
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

func (s *stateMemory) GetPodReadOnly(podUID types.UID) (*v1.Pod, bool) {
	s.RLock()
	defer s.RUnlock()

	pod, ok := s.pods[podUID]
	return pod, ok
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

// updateContainer applies mutate to the stored container of the given name and type, or else to a
// new container with only the name set, which it appends to the list for containerType. Only a pod
// migrated from a V1 checkpoint can have the name under another type, since V1 didn't record
// container types (see migrateV1ToV2), so the name is first moved from the other lists.
func (s *stateMemory) updateContainer(podUID types.UID, containerName string, containerType podutil.ContainerType, mutate func(*v1.Container)) error {
	return s.updatePod(podUID, func(pod *v1.Pod) error {
		// Copy on write: storeState encodes the stored pods without holding the lock.
		pod.Spec.InitContainers = slices.Clone(pod.Spec.InitContainers)
		pod.Spec.Containers = slices.Clone(pod.Spec.Containers)
		pod.Spec.EphemeralContainers = slices.Clone(pod.Spec.EphemeralContainers)

		container := v1.Container{Name: containerName}
		for c, existingType := range podutil.ContainerIter(&pod.Spec, podutil.AllContainers) {
			if c.Name == containerName {
				if existingType == containerType {
					mutate(c)
					return nil
				}
				container = *c
				break
			}
		}

		// Remove the container from any other list before appending it to the list for containerType.
		// TODO: Remove the move between lists in v1.39, with the V1 checkpoint format.
		switch containerType {
		case podutil.InitContainers, podutil.Containers, podutil.EphemeralContainers:
			pod.Spec.InitContainers = slices.DeleteFunc(pod.Spec.InitContainers, func(c v1.Container) bool { return c.Name == containerName })
			pod.Spec.Containers = slices.DeleteFunc(pod.Spec.Containers, func(c v1.Container) bool { return c.Name == containerName })
			pod.Spec.EphemeralContainers = slices.DeleteFunc(pod.Spec.EphemeralContainers, func(c v1.EphemeralContainer) bool { return c.Name == containerName })
		}

		mutate(&container)
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
}

func (s *stateMemory) SetContainer(logger klog.Logger, podUID types.UID, containerType podutil.ContainerType, container v1.Container) error {
	err := s.updateContainer(podUID, container.Name, containerType, func(c *v1.Container) {
		*c = *container.DeepCopy()
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated container information", "podUID", podUID, "containerName", container.Name, "resources", container.Resources, "volumeMounts", container.VolumeMounts)
	return nil
}

func (s *stateMemory) SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, containerType podutil.ContainerType, resources v1.ResourceRequirements) error {
	err := s.updateContainer(podUID, containerName, containerType, func(c *v1.Container) {
		c.Resources = *resources.DeepCopy()
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated container resource information", "podUID", podUID, "containerName", containerName, "resources", resources)
	return nil
}

func (s *stateMemory) RemoveContainers(logger klog.Logger, podUID types.UID, containerType podutil.ContainerType, containerNames ...string) error {
	_, err := s.removeContainers(logger, podUID, containerType, containerNames...)
	return err
}

// removeContainers removes the named containers of the given type, and returns whether any were
// removed.
func (s *stateMemory) removeContainers(logger klog.Logger, podUID types.UID, containerType podutil.ContainerType, containerNames ...string) (bool, error) {
	s.Lock()
	defer s.Unlock()

	stored, ok := s.pods[podUID]
	if !ok {
		return false, nil
	}
	podCopy := *stored
	remove := func(c v1.Container) bool { return slices.Contains(containerNames, c.Name) }
	var removed bool
	switch containerType {
	case podutil.InitContainers:
		podCopy.Spec.InitContainers = slices.DeleteFunc(slices.Clone(stored.Spec.InitContainers), remove)
		removed = len(podCopy.Spec.InitContainers) != len(stored.Spec.InitContainers)
	case podutil.Containers:
		podCopy.Spec.Containers = slices.DeleteFunc(slices.Clone(stored.Spec.Containers), remove)
		removed = len(podCopy.Spec.Containers) != len(stored.Spec.Containers)
	case podutil.EphemeralContainers:
		podCopy.Spec.EphemeralContainers = slices.DeleteFunc(slices.Clone(stored.Spec.EphemeralContainers), func(c v1.EphemeralContainer) bool {
			return remove(v1.Container(c.EphemeralContainerCommon))
		})
		removed = len(podCopy.Spec.EphemeralContainers) != len(stored.Spec.EphemeralContainers)
	default:
		return false, fmt.Errorf("cannot remove containers %q: unsupported container type %d", containerNames, containerType)
	}
	if !removed {
		return false, nil
	}
	s.pods[podUID] = &podCopy
	logger.V(3).Info("Removed container information", "podUID", podUID, "containerNames", containerNames)
	return true, nil
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

func (s *stateMemory) SetEmptyDirVolumeLimit(logger klog.Logger, podUID types.UID, volumeName string, emptyDir *v1.EmptyDirVolumeSource) error {
	err := s.updatePod(podUID, func(pod *v1.Pod) error {
		pod.Spec.Volumes = slices.Clone(pod.Spec.Volumes)
		for i := range pod.Spec.Volumes {
			vol := &pod.Spec.Volumes[i]
			if vol.Name == volumeName && vol.EmptyDir != nil {
				vol.EmptyDir = emptyDir.DeepCopy()
				return nil
			}
		}

		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name:         volumeName,
			VolumeSource: v1.VolumeSource{EmptyDir: emptyDir.DeepCopy()},
		})
		return nil
	})
	if err != nil {
		return err
	}

	logger.V(3).Info("Updated emptyDir volume limit", "podUID", podUID, "volumeName", volumeName, "emptyDir", emptyDir)
	return nil
}

func (s *stateMemory) SetPodVolume(logger klog.Logger, podUID types.UID, volume v1.Volume) error {
	_, err := s.setPodVolume(logger, podUID, volume)
	return err
}

func (s *stateMemory) setPodVolume(logger klog.Logger, podUID types.UID, volume v1.Volume) (bool, error) {
	s.Lock()
	defer s.Unlock()

	volCopy := *volume.DeepCopy()
	stored, ok := s.pods[podUID]
	if !ok {
		s.pods[podUID] = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: podUID},
			Spec:       v1.PodSpec{Volumes: []v1.Volume{volCopy}},
		}
		logger.V(3).Info("Updated pod volume information", "podUID", podUID, "volumeName", volume.Name)
		return true, nil
	}

	for i := range stored.Spec.Volumes {
		if stored.Spec.Volumes[i].Name == volume.Name {
			// Preserve an already-actuated EmptyDir.SizeLimit, which is updated via SetEmptyDirVolumeLimit.
			if stored.Spec.Volumes[i].EmptyDir != nil && stored.Spec.Volumes[i].EmptyDir.SizeLimit != nil && volCopy.EmptyDir != nil {
				sizeLimitCopy := *stored.Spec.Volumes[i].EmptyDir.SizeLimit
				volCopy.EmptyDir.SizeLimit = &sizeLimitCopy
			}
			if apiequality.Semantic.DeepEqual(stored.Spec.Volumes[i], volCopy) {
				return false, nil
			}
			pod := *stored
			pod.Spec.Volumes = slices.Clone(stored.Spec.Volumes)
			pod.Spec.Volumes[i] = volCopy
			s.pods[podUID] = &pod
			logger.V(3).Info("Updated pod volume information", "podUID", podUID, "volumeName", volume.Name)
			return true, nil
		}
	}

	pod := *stored
	pod.Spec.Volumes = append(slices.Clone(stored.Spec.Volumes), volCopy)
	s.pods[podUID] = &pod
	logger.V(3).Info("Updated pod volume information", "podUID", podUID, "volumeName", volume.Name)
	return true, nil
}

func (s *stateMemory) RemovePodVolume(logger klog.Logger, podUID types.UID, volumeName string) error {
	_, err := s.removePodVolume(logger, podUID, volumeName)
	return err
}

func (s *stateMemory) removePodVolume(logger klog.Logger, podUID types.UID, volumeName string) (bool, error) {
	s.Lock()
	defer s.Unlock()

	stored, ok := s.pods[podUID]
	if !ok {
		return false, nil
	}

	idx := slices.IndexFunc(stored.Spec.Volumes, func(v v1.Volume) bool { return v.Name == volumeName })
	if idx == -1 {
		return false, nil
	}

	pod := *stored
	pod.Spec.Volumes = slices.Delete(slices.Clone(pod.Spec.Volumes), idx, idx+1)
	s.pods[podUID] = &pod
	logger.V(3).Info("Removed pod volume information", "podUID", podUID, "volumeName", volumeName)
	return true, nil
}

func (s *stateMemory) SetPod(logger klog.Logger, pod *v1.Pod) error {
	// Only the identity and the spec of the pod are kept. The spec is copied because the caller owns the pod.
	stored := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: pod.UID, Name: pod.Name, Namespace: pod.Namespace},
		Spec:       *pod.Spec.DeepCopy(),
	}
	if s.maskFn != nil {
		s.maskFn(stored)
	}

	s.Lock()
	defer s.Unlock()

	if s.maskFn != nil && len(stored.Spec.Volumes) == 0 {
		if existing, ok := s.pods[pod.UID]; ok && len(existing.Spec.Volumes) > 0 {
			stored.Spec.Volumes = slices.Clone(existing.Spec.Volumes)
		}
	}

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
