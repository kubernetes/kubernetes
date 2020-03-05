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

package memorymanager

import (
	"sync"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// memoryManagerStateFileName is the file name where memory manager stores its state
const memoryManagerStateFileName = "memory_manager_state"

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

type runtimeService interface {
	UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error
}

type sourcesReadyStub struct{}

func (s *sourcesReadyStub) AddSource(source string) {}
func (s *sourcesReadyStub) AllReady() bool          { return true }

// Manager interface provides methods for Kubelet to manage pod memory.
type Manager interface {
	// Start is called during Kubelet initialization.
	Start(activePods ActivePodsFunc, sourcesReady config.SourcesReady, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService, initialContainers containermap.ContainerMap) error

	// AddContainer is called between container create and container start
	// so that initial memory affinity settings can be written through to the
	// container runtime before the first process begins to execute.
	AddContainer(p *v1.Pod, c *v1.Container, containerID string) error

	// Allocate is called to pre-allocate memory resources during Pod admission.
	// This must be called at some point prior to the AddContainer() call for a container, e.g. at pod admission time.
	Allocate(pod *v1.Pod, container *v1.Container) error

	// RemoveContainer is called after Kubelet decides to kill or delete a
	// container. After this call, the memory manager stops trying to reconcile
	// that container, and any memory allocated to the container are freed.
	RemoveContainer(containerID string) error

	// State returns a read-only interface to the internal memory manager state.
	State() state.Reader

	// GetTopologyHints implements the topologymanager.HintProvider Interface
	// and is consulted to achieve NUMA aware resource alignment among this
	// and other resource controllers.
	GetTopologyHints(*v1.Pod, *v1.Container) map[string][]topologymanager.TopologyHint

	// GetPodTopologyHints implements the topologymanager.HintProvider Interface
	// and is consulted to achieve NUMA aware resource alignment among this
	// and other resource controllers.
	GetPodTopologyHints(*v1.Pod) map[string][]topologymanager.TopologyHint
}

type manager struct {
	sync.Mutex
	policy Policy

	// state allows to restore information regarding memory allocation for guaranteed pods
	// in the case of the kubelet restart
	state state.State

	// containerRuntime is the container runtime service interface needed
	// to make UpdateContainerResources() calls against the containers.
	containerRuntime runtimeService

	// activePods is a method for listing active pods on the node
	// so all the containers can be updated in the reconciliation loop.
	activePods ActivePodsFunc

	// podStatusProvider provides a method for obtaining pod statuses
	// and the containerID of their containers
	podStatusProvider status.PodStatusProvider

	// containerMap provides a mapping from (pod, container) -> containerID
	// for all containers a pod
	containerMap containermap.ContainerMap

	nodeAllocatableReservation v1.ResourceList

	// sourcesReady provides the readiness of kubelet configuration sources such as apiserver update readiness.
	// We use it to determine when we can purge inactive pods from checkpointed state.
	sourcesReady config.SourcesReady

	// stateFileDirectory holds the directory where the state file for checkpoints is held.
	stateFileDirectory string
}

var _ Manager = &manager{}

// NewManager returns new instance of the memory manager
func NewManager(reconcilePeriod time.Duration, machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, stateFileDirectory string, affinity topologymanager.Store) (Manager, error) {

}

// Start starts the memory manager reconcile loop under the kubelet to keep state updated
func (m *manager) Start(activePods ActivePodsFunc, sourcesReady config.SourcesReady, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService, initialContainers containermap.ContainerMap) error {

}

// AddContainer saves the value of requested memory for the guranteed pod under the state and set memory affinity according to the topolgy manager
func (m *manager) AddContainer(p *v1.Pod, c *v1.Container, containerID string) error {

}

// Allocate is called to pre-allocate memory resources during Pod admission.
func (m *manager) Allocate(pod *v1.Pod, container *v1.Container) error {

}

// RemoveContainer removes the container from the state
func (m *manager) RemoveContainer(containerID string) error {

}

// State returns the state of the manager
func (m *manager) State() state.Reader {
	return m.state
}

// GetTopologyHints returns the topology hints for the topology manager
func (m *manager) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {

}
