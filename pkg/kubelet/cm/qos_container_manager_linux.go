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

	"github.com/golang/glog"

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

	return nil
}

func (m *qosContainerManagerImpl) setCPUCgroupConfig(configs map[v1.PodQOSClass]*CgroupConfig) error {
	pods := m.activePods()
	burstablePodCPURequest := int64(0)
	for i := range pods {
		pod := pods[i]
		qosClass := qos.GetPodQOS(pod)
		if qosClass != v1.PodQOSBurstable {
			// we only care about the burstable qos tier
			continue
		}
		req, _, err := v1.PodRequestsAndLimits(pod)
		if err != nil {
			return err
		}
		if request, found := req[v1.ResourceCPU]; found {
			burstablePodCPURequest += request.MilliValue()
		}
	}

	// make sure best effort is always 2 shares
	bestEffortCPUShares := int64(MinShares)
	configs[v1.PodQOSBestEffort].ResourceParameters.CpuShares = &bestEffortCPUShares

	// set burstable shares based on current observe state
	burstableCPUShares := MilliCPUToShares(burstablePodCPURequest)
	if burstableCPUShares < int64(MinShares) {
		burstableCPUShares = int64(MinShares)
	}
	configs[v1.PodQOSBurstable].ResourceParameters.CpuShares = &burstableCPUShares
	return nil
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

	// update the qos level cgroup settings for cpu shares
	if err := m.setCPUCgroupConfig(qosConfigs); err != nil {
		return err
	}

	for _, config := range qosConfigs {
		err := m.cgroupManager.Update(config)
		if err != nil {
			glog.V(2).Infof("[ContainerManager]: Failed updated QoS cgroup configuration")
			return err
		}
	}
	glog.V(2).Infof("[ContainerManager]: Updated QoS cgroup configuration")
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
