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

package fake

import (
	"context"

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/utils/cpuset"
)

type (
	Manager        = cpumanager.Manager
	Policy         = cpumanager.Policy
	ActivePodsFunc = cpumanager.ActivePodsFunc
	RuntimeService = cpumanager.RuntimeService
)

type manager struct {
	logger logr.Logger
	state  state.State
}

var _ Manager = &manager{}

func (m *manager) Start(ctx context.Context, activePods ActivePodsFunc, sourcesReady config.SourcesReady, podStatusProvider status.PodStatusProvider, containerRuntime RuntimeService, initialContainers containermap.ContainerMap) error {
	m.logger.Info("Start()")
	return nil
}

func (m *manager) Policy() Policy {
	m.logger.Info("Policy()")
	pol, _ := cpumanager.NewNonePolicy(nil)
	return pol
}

func (m *manager) Allocate(pod *v1.Pod, container *v1.Container) error {
	m.logger.Info("Allocate", "pod", klog.KObj(pod), "containerName", container.Name)
	return nil
}

func (m *manager) AddContainer(logger logr.Logger, pod *v1.Pod, container *v1.Container, containerID string) {
	m.logger.Info("AddContainer", "pod", klog.KObj(pod), "containerName", container.Name, "containerID", containerID)
}

func (m *manager) RemoveContainer(logger logr.Logger, containerID string) error {
	m.logger.Info("RemoveContainer", "containerID", containerID)
	return nil
}

func (m *manager) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	m.logger.Info("Get container topology hints")
	return map[string][]topologymanager.TopologyHint{}
}

func (m *manager) GetPodTopologyHints(pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	m.logger.Info("Get pod topology hints")
	return map[string][]topologymanager.TopologyHint{}
}

func (m *manager) State() state.Reader {
	return m.state
}

func (m *manager) GetExclusiveCPUs(podUID, containerName string) cpuset.CPUSet {
	m.logger.Info("GetExclusiveCPUs", "podUID", podUID, "containerName", containerName)
	return cpuset.CPUSet{}
}

func (m *manager) GetAllocatableCPUs() cpuset.CPUSet {
	m.logger.Info("Get Allocatable CPUs")
	return cpuset.CPUSet{}
}

func (m *manager) GetCPUAffinity(podUID, containerName string) cpuset.CPUSet {
	m.logger.Info("GetCPUAffinity", "podUID", podUID, "containerName", containerName)
	return cpuset.CPUSet{}
}

func (m *manager) GetAllCPUs() cpuset.CPUSet {
	m.logger.Info("GetAllCPUs")
	return cpuset.CPUSet{}
}

// NewFakeManager creates empty/fake cpu manager
func NewFakeManager(logger logr.Logger) cpumanager.Manager {
	logger = klog.LoggerWithName(logger, "cpu.fake")
	return &manager{
		logger: logger,
		state:  state.NewMemoryState(logger),
	}
}
