/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"fmt"
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

const (
	podCgroupNamePrefix = "pod#"
)

// podContainerManagerImpl implements podContainerManager interface.
// It is the general implementation which allows pod level container
// management if qos Cgroup is enabled.
type podContainerManagerImpl struct {
	// nodeInfo stores information about the node resource capacity
	nodeInfo *api.Node
	// qosContainersInfo hold absolute paths of the top level qos containers
	qosContainersInfo QOSContainersInfo
	// Stores the mounted cgroup subsystems
	subsystems *CgroupSubsystems
	// cgroupManager is the cgroup Manager Object responsible for managing all
	// pod cgroups.
	cgroupManager CgroupManager
}

// Make sure that podContainerManagerImpl implements the PodContainerManager interface
var _ PodContainerManager = &podContainerManagerImpl{}

// applyLimits sets pod cgroup resource limits
// It also updates the resource limits on top level qos containers.
func (m *podContainerManagerImpl) applyLimits(pod *api.Pod) error {
	// This function will house the logic for setting the resource parameters
	// on the pod container config and updating top level qos container configs
	return nil
}

// Exists checks if the pod's cgroup already exists
func (m *podContainerManagerImpl) Exists(pod *api.Pod) bool {
	podContainerName := m.GetPodContainerName(pod)
	return m.cgroupManager.Exists(podContainerName)
}

// EnsureExists takes a pod as argument and makes sure that
// pod cgroup exists if qos cgroup hierarchy flag is enabled.
// If the pod level container doesen't already exist it is created.
func (m *podContainerManagerImpl) EnsureExists(pod *api.Pod) error {
	podContainerName := m.GetPodContainerName(pod)
	// check if container already exist
	alreadyExists := m.Exists(pod)
	if !alreadyExists {
		// Create the pod container
		containerConfig := &CgroupConfig{
			Name:               podContainerName,
			ResourceParameters: &ResourceConfig{},
		}
		if err := m.cgroupManager.Create(containerConfig); err != nil {
			return fmt.Errorf("failed to create container for %v : %v", podContainerName, err)
		}
	}
	// Apply appropriate resource limits on the pod container
	// Top level qos containers limits are not updated
	// until we figure how to maintain the desired state in the kubelet.
	// Because maintaining the desired state is difficult without checkpointing.
	if err := m.applyLimits(pod); err != nil {
		return fmt.Errorf("failed to apply resource limits on container for %v : %v", podContainerName, err)
	}
	return nil
}

// GetPodContainerName is a util func takes in a pod as an argument
// and returns the pod's cgroup name. We follow a pod cgroup naming format
// which is opaque and deterministic. Given a pod it's cgroup would be named
// "pod-UID" where the UID is the Pod UID
func (m *podContainerManagerImpl) GetPodContainerName(pod *api.Pod) string {
	podQOS := qos.GetPodQOS(pod)
	// Get the parent QOS container name
	var parentContainer string
	switch podQOS {
	case qos.Guaranteed:
		parentContainer = m.qosContainersInfo.Guaranteed
	case qos.Burstable:
		parentContainer = m.qosContainersInfo.Burstable
	case qos.BestEffort:
		parentContainer = m.qosContainersInfo.BestEffort
	}
	podContainer := podCgroupNamePrefix + string(pod.UID)
	// Get the absolute path of the cgroup
	return path.Join(parentContainer, podContainer)
}

// Destroy destroys the pod container cgroup paths
func (m *podContainerManagerImpl) Destroy(podCgroup string) error {
	// This will house the logic for destroying the pod cgroups.
	// Will be handled in the next PR.
	return nil
}

// podContainerManagerNoop implements podContainerManager interface.
// It is a no-op implementation and basically does nothing
// podContainerManagerNoop is used in case the QoS cgroup Hierarchy is not
// enabled, so Exists() returns true always as the cgroupRoot
// is expected to always exist.
type podContainerManagerNoop struct {
	cgroupRoot string
}

// Make sure that podContainerManagerStub implements the PodContainerManager interface
var _ PodContainerManager = &podContainerManagerNoop{}

func (m *podContainerManagerNoop) Exists(_ *api.Pod) bool {
	return true
}

func (m *podContainerManagerNoop) EnsureExists(_ *api.Pod) error {
	return nil
}

func (m *podContainerManagerNoop) GetPodContainerName(_ *api.Pod) string {
	return m.cgroupRoot
}

// Destroy destroys the pod container cgroup paths
func (m *podContainerManagerNoop) Destroy(_ string) error {
	return nil
}
