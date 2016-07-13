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
	"os"
	"path"
	"path/filepath"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	podCgroupNamePrefix = "pod#"
	minimumMemoryValue  = int64(1)
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
// "pod#UID" where the UID is the Pod UID
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

// Scans through all the subsystems pod cgroups directory.
// Will also scan through the container cgroups that still exist under
// the pod cgroup and haven't been Garbage Collected yet
func (m *podContainerManagerImpl) getPIDsToKill(podCgroup string) []int {
	// Get a list of processes that we need to kill
	pidsToKill := sets.NewInt()
	var pids []int
	for _, val := range m.subsystems.MountPoints {
		dir := path.Join(val, podCgroup)
		_, err := os.Stat(dir)
		if os.IsNotExist(err) {
			// The subsystem pod cgroup is already deleted
			// do nothing, continue
			continue
		}
		// Get a list of pids that are still charged to the pod's cgroup
		pids, err = getCgroupProcs(dir)
		if err != nil {
			continue
		}
		pidsToKill.Insert(pids...)

		// WalkFunc which is called for each file and directory in the pod cgroup dir
		visitor := func(path string, info os.FileInfo, err error) error {
			if !info.IsDir() {
				return nil
			}
			pids, err = getCgroupProcs(path)
			if err != nil {
				return err
			}
			pidsToKill.Insert(pids...)
			return nil
		}
		// Walk through the pod cgroup directory to check if
		// container cgroups haven't been GCed yet. Get attached processes to
		// all such unwanted containers under the pod cgroup
		err = filepath.Walk(dir, visitor)
	}
	return pidsToKill.List()
}

// Scan through the whole cgroup directory and kill all processes either
// attached to the pod cgroup or to a container cgroup under the pod cgroup
func (m *podContainerManagerImpl) tryKillingCgroupProcesses(podCgroup string) error {
	pidsToKill := m.getPIDsToKill(podCgroup)
	// No pids charged to the terminated pod cgroup return
	if len(pidsToKill) == 0 {
		return nil
	}

	var errlist []error
	// os.Kill often errors out,
	// We try killing all the pids multiple times
	for i := 0; i < 5; i++ {
		if i != 0 {
			glog.V(3).Infof("Attempt %v failed to kill all unwanted process. Retyring", i)
		}
		errlist = []error{}
		for _, pid := range pidsToKill {
			p, err := os.FindProcess(pid)
			if err != nil {
				// Process not running anymore, do nothing
				continue
			}
			glog.V(3).Infof("Attempt to kill process with pid: %v", pid)
			if err := p.Kill(); err != nil {
				glog.V(3).Infof("failed to kill process with pid: %v", pid)
				errlist = append(errlist, err)
			}
		}
		if len(errlist) == 0 {
			glog.V(3).Infof("successfully killed all unwanted processes.")
			return nil
		}
	}
	return utilerrors.NewAggregate(errlist)
}

// Destroy destroys the pod container cgroup paths
func (m *podContainerManagerImpl) Destroy(podCgroup string) error {
	// Try killing all the processes attached to the pod cgroup
	if err := m.tryKillingCgroupProcesses(podCgroup); err != nil {
		glog.V(3).Infof("failed to kill all the processes attached to the %v cgroups", podCgroup)
		return fmt.Errorf("failed to kill all the processes attached to the %v cgroups : %v", podCgroup, err)
	}

	// Now its safe to remove the pod's cgroup
	containerConfig := &CgroupConfig{
		Name:               podCgroup,
		ResourceParameters: &ResourceConfig{},
	}
	if err := m.cgroupManager.Destroy(containerConfig); err != nil {
		return fmt.Errorf("failed to delete cgroup paths for %v : %v", podCgroup, err)
	}
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
