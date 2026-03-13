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
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	libcontainercgroups "github.com/opencontainers/cgroups"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

const (
	podCgroupNamePrefix = "pod"
)

// podContainerManagerImpl implements podContainerManager interface.
// It is the general implementation which allows pod level container
// management if qos Cgroup is enabled.
//
type podContainerManagerImpl struct {
	// qosContainersInfo hold absolute paths of the top level qos containers
	qosContainersInfo QOSContainersInfo
	// Stores the mounted cgroup subsystems
	subsystems *CgroupSubsystems
	// cgroupManager is the cgroup Manager Object responsible for managing all
	// pod cgroups.
	cgroupManager CgroupManager
	// Maximum number of pids in a pod
	podPidsLimit int64
	// enforceCPULimits controls whether cfs quota is enforced or not
	enforceCPULimits bool
	// cpuCFSQuotaPeriod is the cfs period value, cfs_period_us, setting per
	// node for all containers in usec
	cpuCFSQuotaPeriod uint64
	// podContainerManager is the ContainerManager running on the machine
	podContainerManager ContainerManager
}

// Make sure that podContainerManagerImpl implements the PodContainerManager interface
var _ PodContainerManager = &podContainerManagerImpl{}

// Exists checks if the pod's cgroup already exists
func (m *podContainerManagerImpl) Exists(pod *v1.Pod) bool {
	podContainerName, _ := m.GetPodContainerName(pod)
	return m.cgroupManager.Exists(podContainerName)
}

// EnsureExists takes a pod as argument and makes sure that
// pod cgroup exists if qos cgroup hierarchy flag is enabled.
// If the pod level container doesn't already exist it is created.
func (m *podContainerManagerImpl) EnsureExists(logger klog.Logger, pod *v1.Pod) error {
	// check if container already exist
	alreadyExists := m.Exists(pod)
	if !alreadyExists {
		enforceCPULimits := m.enforceCPULimits
		if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DisableCPUQuotaWithExclusiveCPUs) && m.podContainerManager.PodHasExclusiveCPUs(pod) {
			logger.V(2).Info("Disabled CFS quota", "pod", klog.KObj(pod))
			enforceCPULimits = false
		}
		enforceMemoryQoS := false
		if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.MemoryQoS) &&
			libcontainercgroups.IsCgroup2UnifiedMode() {
			enforceMemoryQoS = true
		}
		// Create the pod container
		podContainerName, _ := m.GetPodContainerName(pod)
		containerConfig := &CgroupConfig{
			Name: podContainerName,
			Resources: &ResourceConfig{
				CPUQuota:        m.cpuCFSQuotaPeriod,
				EnforceCPULimits: enforceCPULimits,
				MemoryQoS:       enforceMemoryQoS,
			},
		}
		return m.cgroupManager.Create(containerConfig)
	}
	return nil
}

// ReconcilePodMemoryMin clears pod-level memory.min cgroup setting when MemoryQoS is disabled or policy is None.
// This prevents stale memory.min values from lingering on pod cgroups when MemoryQoS is toggled off or policy changes.
func (m *podContainerManagerImpl) ReconcilePodMemoryMin(logger klog.Logger, memoryQoSEnabled bool, memoryQoSPolicyNone bool) error {
	if memoryQoSEnabled && !memoryQoSPolicyNone {
		// MemoryQoS is enabled and policy is not None, no need to clear memory.min
		return nil
	}

	var errs []error
	// Iterate over all pod cgroups and clear memory.min
	for _, podCgroup := range m.cgroupManager.GetPodCgroups() {
		resourceConfig := &ResourceConfig{
			Unified: map[string]string{
				"memory.min": "0",
			},
		}
		err := m.cgroupManager.Set(podCgroup, resourceConfig)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to clear memory.min for pod cgroup %q: %w", podCgroup.String(), err))
			logger.Error(err, "Failed to clear memory.min for pod cgroup", "podCgroup", podCgroup.String())
		}
	}
	return utilerrors.NewAggregate(errs)
}

// GetPodContainerName returns the cgroup name for the pod container
func (m *podContainerManagerImpl) GetPodContainerName(pod *v1.Pod) (CgroupName, error) {
	if pod == nil {
		return CgroupName{}, errors.New("pod is nil")
	}
	podUID := string(pod.UID)
	podCgroupName := NewCgroupName(RootCgroupName, string(v1qos.GetPodQOS(pod)), podCgroupNamePrefix+podUID)
	return podCgroupName, nil
}

// GetPodCgroups returns all pod cgroups managed by the cgroupManager
func (m *podContainerManagerImpl) GetPodCgroups() []CgroupName {
	// This is a helper method to get all pod cgroups
	// Implementation depends on cgroupManager interface
	// For demonstration, assume cgroupManager has a method GetPodCgroups
	if mgr, ok := m.cgroupManager.(interface{ GetPodCgroups() []CgroupName }); ok {
		return mgr.GetPodCgroups()
	}
	return nil
}
