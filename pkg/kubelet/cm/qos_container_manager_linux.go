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

	"k8s.io/kubernetes/pkg/api/v1"
)

type QOSContainerManager interface {
	Start(*v1.Node, ActivePodsFunc) error
	GetQOSContainersInfo() QOSContainersInfo
	UpdateCgroups() error
}

type qosContainerManagerImpl struct {
	sync.Mutex
	nodeInfo          *v1.Node
	qosContainersInfo QOSContainersInfo
	subsystems        *CgroupSubsystems
	cgroupManager     CgroupManager
	activePods        ActivePodsFunc
	cgroupRoot        string
}

func NewQOSContainerManager(subsystems *CgroupSubsystems, cgroupRoot string, nodeConfig NodeConfig) (QOSContainerManager, error) {
	if !nodeConfig.CgroupsPerQOS {
		return &qosContainerManagerNoop{
			cgroupRoot: CgroupName(nodeConfig.CgroupRoot),
		}, nil
	}

	return &qosContainerManagerImpl{
		subsystems:    subsystems,
		cgroupManager: NewCgroupManager(subsystems, nodeConfig.CgroupDriver),
		cgroupRoot:    cgroupRoot,
	}, nil
}

func (m *qosContainerManagerImpl) GetQOSContainersInfo() QOSContainersInfo {
	return m.qosContainersInfo
}

func (m *qosContainerManagerImpl) Start(nodeInfo *v1.Node, activePods ActivePodsFunc) error {
	cm := m.cgroupManager
	rootContainer := m.cgroupRoot
	if !cm.Exists(CgroupName(rootContainer)) {
		return fmt.Errorf("root container %s doesn't exist", rootContainer)
	}

	// Top level for Qos containers are created only for Burstable
	// and Best Effort classes
	qosClasses := [2]v1.PodQOSClass{v1.PodQOSBurstable, v1.PodQOSBestEffort}

	// Create containers for both qos classes
	for _, qosClass := range qosClasses {
		// get the container's absolute name
		absoluteContainerName := CgroupName(path.Join(rootContainer, string(qosClass)))

		resourceParameters := &ResourceConfig{}
		// the BestEffort QoS class has a statically configured minShares value
		if qosClass == v1.PodQOSBestEffort {
			minShares := int64(MinShares)
			resourceParameters.CpuShares = &minShares
		}
		// containerConfig object stores the cgroup specifications
		containerConfig := &CgroupConfig{
			Name:               absoluteContainerName,
			ResourceParameters: resourceParameters,
		}
		// check if it exists
		if !cm.Exists(absoluteContainerName) {
			if err := cm.Create(containerConfig); err != nil {
				return fmt.Errorf("failed to create top level %v QOS cgroup : %v", qosClass, err)
			}
		} else {
			// to ensure we actually have the right state, we update the config on startup
			if err := cm.Update(containerConfig); err != nil {
				return fmt.Errorf("failed to update top level %v QOS cgroup : %v", qosClass, err)
			}
		}
	}
	// Store the top level qos container names
	m.qosContainersInfo = QOSContainersInfo{
		Guaranteed: rootContainer,
		Burstable:  path.Join(rootContainer, string(v1.PodQOSBurstable)),
		BestEffort: path.Join(rootContainer, string(v1.PodQOSBestEffort)),
	}
	m.nodeInfo = nodeInfo
	m.activePods = activePods

	return nil
}

func (m *qosContainerManagerImpl) UpdateCgroups() error {
	m.Lock()
	defer m.Unlock()

	// TODO: Update cgroups

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
