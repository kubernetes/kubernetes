/*
Copyright 2017 The Kubernetes Authors.

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
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

type QOSContainerManager interface {
	Start(*v1.Node, ActivePodsFunc) error
	GetQOSContainersInfo() QOSContainersInfo
	UpdateCgroups() error
}

type qosContainerManagerImpl struct {
	sync.Mutex
	NodeConfig
	nodeInfo          *v1.Node
	qosContainersInfo QOSContainersInfo
	subsystems        *CgroupSubsystems
	cgroupManager     CgroupManager
	activePods        ActivePodsFunc
}

func NewQOSContainerManager(subsystems *CgroupSubsystems, nodeConfig NodeConfig) (QOSContainerManager, error) {
	if !nodeConfig.CgroupsPerQOS {
		return &qosContainerManagerNoop{
			cgroupRoot: CgroupName(nodeConfig.CgroupRoot),
		}, nil
	}

	// this does default to / when enabled, but this tests against regressions.
	if nodeConfig.CgroupRoot == "" {
		return nil, fmt.Errorf("invalid configuration: cgroups-per-qos was specified and cgroup-root was not specified. To enable the QoS cgroup hierarchy you need to specify a valid cgroup-root")
	}

	// we need to check that the cgroup root actually exists for each subsystem
	// of note, we always use the cgroupfs driver when performing this check since
	// the input is provided in that format.
	// this is important because we do not want any name conversion to occur.
	cgroupManager := NewCgroupManager(subsystems, "cgroupfs")
	if !cgroupManager.Exists(CgroupName(nodeConfig.CgroupRoot)) {
		return nil, fmt.Errorf("invalid configuration: cgroup-root doesn't exist")
	}
	glog.Infof("[Container Manager] verified cgroup-root exists: %v", nodeConfig.CgroupRoot)

	if len(nodeConfig.QOSReserveRequests) > 0 {
		glog.Infof("[Container Manager] QoS reserve requests %v", nodeConfig.QOSReserveRequests)
	}

	return &qosContainerManagerImpl{
		NodeConfig:    nodeConfig,
		subsystems:    subsystems,
		cgroupManager: NewCgroupManager(subsystems, nodeConfig.CgroupDriver),
	}, nil
}

func (m *qosContainerManagerImpl) GetQOSContainersInfo() QOSContainersInfo {
	return m.qosContainersInfo
}

func (m *qosContainerManagerImpl) Start(nodeInfo *v1.Node, activePods ActivePodsFunc) error {
	// Top level for Qos containers are created only for Burstable
	// and Best Effort classes
	qosClasses := [2]v1.PodQOSClass{v1.PodQOSBurstable, v1.PodQOSBestEffort}

	// Create containers for both qos classes
	for _, qosClass := range qosClasses {
		// get the container's absolute name
		absoluteContainerName := CgroupName(path.Join(m.CgroupRoot, string(qosClass)))
		// containerConfig object stores the cgroup specifications
		containerConfig := &CgroupConfig{
			Name:               absoluteContainerName,
			ResourceParameters: &ResourceConfig{},
		}
		// check if it exists
		if !m.cgroupManager.Exists(absoluteContainerName) {
			if err := m.cgroupManager.Create(containerConfig); err != nil {
				return fmt.Errorf("failed to create top level %v QOS cgroup : %v", qosClass, err)
			}
		}
	}

	// Store the top level qos container names
	m.qosContainersInfo = QOSContainersInfo{
		Guaranteed: m.CgroupRoot,
		Burstable:  path.Join(m.CgroupRoot, string(v1.PodQOSBurstable)),
		BestEffort: path.Join(m.CgroupRoot, string(v1.PodQOSBestEffort)),
	}
	m.nodeInfo = nodeInfo
	m.activePods = activePods

	// TODO: We already have hooks in killPod() and syncPod() that
	// update the QoS reserve on pod entry/exit. This is really only
	// needed in the case that one of the QoS cgroups is over its
	// resoruce limit and we are wanting to put continuous downward
	// pressure on the cgroup such that we can eventually get to the
	// desired limit.
	go wait.Until(func() {
		err := m.UpdateCgroups()
		if err != nil {
			glog.Warningf("[ContainerManager] Failed to reserve QoS requests: %v", err)
		}
	}, 30*time.Second, wait.NeverStop)

	return nil
}

// setMemoryReserve sums the memory limits of all pods in a QOS class,
// calculates QOS class memory limits, and set those limits in the
// CgroupConfig for each QOS class.
func (m *qosContainerManagerImpl) setMemoryReserve(configs map[v1.PodQOSClass]*CgroupConfig, percentReserve int64) {
	qosMemoryRequests := map[v1.PodQOSClass]int64{
		v1.PodQOSGuaranteed: 0,
		v1.PodQOSBurstable:  0,
	}

	// Sum the pod limits for pods in each QOS class
	pods := m.activePods()
	for _, pod := range pods {
		podMemoryRequest := int64(0)
		qosClass := qos.GetPodQOS(pod)
		if qosClass == v1.PodQOSBestEffort {
			// limits are not set for Best Effort pods
			continue
		}
		for _, container := range pod.Spec.Containers {
			podMemoryRequest += container.Resources.Requests.Memory().Value()
		}
		qosMemoryRequests[qosClass] += podMemoryRequest
	}

	allocatable := m.nodeInfo.Status.Allocatable.Memory().Value()
	if allocatable == 0 {
		glog.V(2).Infof("[Container Manager] Memory capacity reported as 0, might be in standalone mode.  Not setting QOS memory limts.")
		return
	}

	for qos, limits := range qosMemoryRequests {
		glog.V(2).Infof("[Container Manager] %s pod requests total %d bytes (reserve %d%%)", qos, limits, percentReserve)
	}

	// Calculate QOS memory limits
	burstableLimit := allocatable - (qosMemoryRequests[v1.PodQOSGuaranteed] * percentReserve / 100)
	bestEffortLimit := burstableLimit - (qosMemoryRequests[v1.PodQOSBurstable] * percentReserve / 100)
	configs[v1.PodQOSBurstable].ResourceParameters.Memory = &burstableLimit
	configs[v1.PodQOSBestEffort].ResourceParameters.Memory = &bestEffortLimit
}

// retrySetMemoryReserve checks for any QoS cgroups over the limit
// that was attempted to be set in the first Update() and adjusts
// their memory limit to the usage to prevent further growth.
func (m *qosContainerManagerImpl) retrySetMemoryReserve(configs map[v1.PodQOSClass]*CgroupConfig, percentReserve int64) {
	// Unreclaimable memory usage may already exceeded the desired limit
	// Attempt to set the limit near the current usage to put pressure
	// on the cgroup and prevent further growth.
	for qos, config := range configs {
		stats, err := m.cgroupManager.GetResourceStats(config.Name)
		if err != nil {
			glog.V(2).Infof("[Container Manager] %v", err)
			return
		}
		usage := stats.MemoryStats.Usage

		// Because there is no good way to determine of the original Update()
		// on the memory resource was successful, we determine failure of the
		// first attempt by checking if the usage is above the limit we attempt
		// to set.  If it is, we assume the first attempt to set the limit failed
		// and try again setting the limit to the usage.  Otherwise we leave
		// the CgroupConfig as is.
		if configs[qos].ResourceParameters.Memory != nil && usage > *configs[qos].ResourceParameters.Memory {
			configs[qos].ResourceParameters.Memory = &usage
		}
	}
}

func (m *qosContainerManagerImpl) UpdateCgroups() error {
	m.Lock()
	defer m.Unlock()

	qosConfigs := map[v1.PodQOSClass]*CgroupConfig{
		v1.PodQOSBurstable: {
			Name:               CgroupName(m.qosContainersInfo.Burstable),
			ResourceParameters: &ResourceConfig{},
		},
		v1.PodQOSBestEffort: {
			Name:               CgroupName(m.qosContainersInfo.BestEffort),
			ResourceParameters: &ResourceConfig{},
		},
	}

	for resource, percentReserve := range m.QOSReserveRequests {
		switch resource {
		case v1.ResourceMemory:
			m.setMemoryReserve(qosConfigs, percentReserve)
		}
	}
	updateSuccess := true
	for _, config := range qosConfigs {
		err := m.cgroupManager.Update(config)
		if err != nil {
			updateSuccess = false
		}
	}
	if updateSuccess {
		glog.V(2).Infof("[Container Manager] QoS reserve requests set successfully")
		return nil
	}

	// If the resource can adjust the ResourceConfig to increase likelihood of
	// success, call the adjustment function here.  Otherwise, the Update() will
	// be called again with the same values.
	for resource, percentReserve := range m.QOSReserveRequests {
		switch resource {
		case v1.ResourceMemory:
			m.retrySetMemoryReserve(qosConfigs, percentReserve)
		}
	}

	for _, config := range qosConfigs {
		err := m.cgroupManager.Update(config)
		if err != nil {
			return err
		}
	}

	glog.V(2).Infof("[Container Manager] QoS reserve requests set successfully on retry")
	return nil
}

type qosContainerManagerNoop struct {
	cgroupRoot CgroupName
}

var _ QOSContainerManager = &qosContainerManagerNoop{}

func (m *qosContainerManagerNoop) GetQOSContainersInfo() QOSContainersInfo {
	return QOSContainersInfo{}
}

func (m *qosContainerManagerNoop) Start(_ *v1.Node, _ ActivePodsFunc) error {
	return nil
}

func (m *qosContainerManagerNoop) UpdateCgroups() error {
	return nil
}
