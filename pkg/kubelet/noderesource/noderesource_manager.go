/*
Copyright 2020 The Kubernetes Authors.

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

package noderesource

import (
	"fmt"
	"sync"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/server"
)

type Manager interface {
	Start()
	MachineInfo() <-chan *cadvisorapi.MachineInfo
	NodeCapacityDecreasedStatus() error
}

// Config represents Manager configuration
type Config struct {
	Host               server.HostInterface
	CAdvisor           cadvisor.Interface
	SyncNodeStatusFunc func()
}

type manager struct {
	// server kubelet methods
	host server.HostInterface
	// cadvisor client
	cadvisor cadvisor.Interface
	// channel of MachineInfo
	machineInfoChan chan *cadvisorapi.MachineInfo
	syncNodeStatus  func()

	machineInfoDecreaseMutex sync.Mutex
	machineInfoDecreased     bool

	// storing it here to report it in node event
	machineInfo       *cadvisorapi.MachineInfo
	cachedMachineInfo *cadvisorapi.MachineInfo
}

// managerStub is a fake node resource managerImpl.
type managerStub struct {
	// channel of MachineInfo
	machineInfoChan chan *cadvisorapi.MachineInfo
}

func NewNodeResourceManager(conf *Config) Manager {
	if !utilfeature.DefaultFeatureGate.Enabled(features.NodeResourceHotPlug) {
		return &managerStub{machineInfoChan: make(chan *cadvisorapi.MachineInfo)}
	}
	if conf == nil {
		return &managerStub{machineInfoChan: make(chan *cadvisorapi.MachineInfo)}
	}
	return &manager{
		host:            conf.Host,
		cadvisor:        conf.CAdvisor,
		machineInfoChan: make(chan *cadvisorapi.MachineInfo),
		syncNodeStatus:  conf.SyncNodeStatusFunc,
	}
}

func (m *manager) Start() {
	// starting to fetch machine info from cadvisor cache.
	klog.Info("Starting node resource manager")
	go wait.Forever(func() {
		klog.Info("Fetching machine info")
		var machineInfoDecreased bool
		machineInfo, err := m.cadvisor.MachineInfo()
		if err != nil {
			klog.ErrorS(err, "Error fetching machine info")
			return
		}
		m.machineInfo = machineInfo
		cachedMachineInfo, _ := m.host.GetCachedMachineInfo()
		m.cachedMachineInfo = cachedMachineInfo
		// Avoid collector collects it as a timestamped metric
		// See PR #95210 and #97006 for more details.
		machineInfo.Timestamp = time.Time{}

		if isNodeCapacityIncreased(cachedMachineInfo, machineInfo) {
			klog.Info("Node capacity increased")
			m.machineInfoChan <- machineInfo
		} else if isNodeCapacityDecreased(cachedMachineInfo, machineInfo) {
			klog.Info("Node capacity decreased, Setting node as not ready")
			// set node not ready
			machineInfoDecreased = true
		}
		// If the machine info decreased we need to set node to not ready
		// Once the node is set to not ready, Later again if machine info back to valid state
		// we should make node as ready.
		m.machineInfoDecreaseMutex.Lock()
		previousMachineState := m.machineInfoDecreased
		m.machineInfoDecreased = machineInfoDecreased
		m.machineInfoDecreaseMutex.Unlock()
		if previousMachineState || machineInfoDecreased {
			klog.Info("Updating node status from node resource manager")
			m.syncNodeStatus()
		}

		// cadvisor updates its cache in `update_machine_info_interval` defaulted to 5 minutes.
	}, 1*time.Second)
}

func (m *manager) MachineInfo() <-chan *cadvisorapi.MachineInfo {
	return m.machineInfoChan
}

// NodeCapacityDecreasedStatus will return an error if the node is capacity has decreased.
func (m *manager) NodeCapacityDecreasedStatus() error {
	m.machineInfoDecreaseMutex.Lock()
	defer m.machineInfoDecreaseMutex.Unlock()

	if m.machineInfoDecreased {
		errMessage := fmt.Errorf("expected CPU: %d Memory:%d Actual: CPU: %d Memory:%d",
			m.cachedMachineInfo.NumCores, m.cachedMachineInfo.MemoryCapacity,
			m.machineInfo.NumCores, m.machineInfo.MemoryCapacity)
		klog.ErrorS(errMessage, "Node capacity decreased")
		return fmt.Errorf("node capacity has decreased %s", errMessage)
	}
	return nil
}

func (m *managerStub) Start() {
	return
}

func (m *managerStub) MachineInfo() <-chan *cadvisorapi.MachineInfo {
	return m.machineInfoChan
}

func (m *managerStub) NodeCapacityDecreasedStatus() error {
	return nil
}

func isNodeCapacityDecreased(currentMachineInfo, newMachineInfo *cadvisorapi.MachineInfo) bool {
	if newMachineInfo.MemoryCapacity < currentMachineInfo.MemoryCapacity ||
		newMachineInfo.NumCores < currentMachineInfo.NumCores {
		return true
	}
	return false
}

func isNodeCapacityIncreased(currentMachineInfo, newMachineInfo *cadvisorapi.MachineInfo) bool {
	if newMachineInfo.MemoryCapacity > currentMachineInfo.MemoryCapacity ||
		newMachineInfo.NumCores > currentMachineInfo.NumCores {
		return true
	}
	return false
}
