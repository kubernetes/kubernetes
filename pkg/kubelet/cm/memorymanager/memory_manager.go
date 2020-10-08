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
	"fmt"
	"strconv"
	"strings"
	"sync"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// memoryManagerStateFileName is the file name where memory manager stores its state
const memoryManagerStateFileName = "memory_manager_state"

// ActivePodsFunc is a function that returns a list of active pods
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
	// container. After this call, any memory allocated to the container are freed.
	RemoveContainer(containerID string) error

	// State returns a read-only interface to the internal memory manager state.
	State() state.Reader

	// GetTopologyHints implements the topologymanager.HintProvider Interface
	// and is consulted to achieve NUMA aware resource alignment among this
	// and other resource controllers.
	GetTopologyHints(*v1.Pod, *v1.Container) map[string][]topologymanager.TopologyHint
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
func NewManager(policyName string, machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, reservedMemory map[int]map[v1.ResourceName]resource.Quantity, stateFileDirectory string, affinity topologymanager.Store) (Manager, error) {
	var policy Policy

	switch policyType(policyName) {

	case policyTypeNone:
		policy = NewPolicyNone()

	case policyTypeStatic:
		reserved, err := getReservedMemory(machineInfo, nodeAllocatableReservation, reservedMemory)
		if err != nil {
			return nil, err
		}

		policy, err = NewPolicyStatic(machineInfo, reserved, affinity)
		if err != nil {
			return nil, err
		}

	default:
		return nil, fmt.Errorf("unknown policy: \"%s\"", policyName)
	}

	manager := &manager{
		policy:                     policy,
		nodeAllocatableReservation: nodeAllocatableReservation,
		stateFileDirectory:         stateFileDirectory,
	}
	manager.sourcesReady = &sourcesReadyStub{}
	return manager, nil
}

// Start starts the memory manager under the kubelet and calls policy start
func (m *manager) Start(activePods ActivePodsFunc, sourcesReady config.SourcesReady, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService, initialContainers containermap.ContainerMap) error {
	klog.Infof("[memorymanager] starting with %s policy", m.policy.Name())
	m.sourcesReady = sourcesReady
	m.activePods = activePods
	m.podStatusProvider = podStatusProvider
	m.containerRuntime = containerRuntime
	m.containerMap = initialContainers

	stateImpl, err := state.NewCheckpointState(m.stateFileDirectory, memoryManagerStateFileName, m.policy.Name(), m.containerMap)
	if err != nil {
		klog.Errorf("[memorymanager] could not initialize checkpoint manager: %v, please drain node and remove policy state file", err)
		return err
	}
	m.state = stateImpl

	err = m.policy.Start(m.state)
	if err != nil {
		klog.Errorf("[memorymanager] policy start error: %v", err)
		return err
	}

	return nil
}

// AddContainer saves the value of requested memory for the guaranteed pod under the state and set memory affinity according to the topolgy manager
func (m *manager) AddContainer(pod *v1.Pod, container *v1.Container, containerID string) error {
	m.Lock()
	m.containerMap.Add(string(pod.UID), container.Name, containerID)
	m.Unlock()

	// Get NUMA node affinity of blocks assigned to the container during Allocate()
	var nodes []string
	for _, block := range m.state.GetMemoryBlocks(string(pod.UID), container.Name) {
		for _, nodeID := range block.NUMAAffinity {
			nodes = append(nodes, strconv.Itoa(nodeID))
		}
	}

	if len(nodes) < 1 {
		klog.V(5).Infof("[memorymanager] update container resources is skipped due to memory blocks are empty")
		return nil
	}

	affinity := strings.Join(nodes, ",")
	klog.Infof("[memorymanager] Set container %q cpuset.mems to %q", containerID, affinity)
	err := m.containerRuntime.UpdateContainerResources(containerID, &runtimeapi.LinuxContainerResources{CpusetMems: affinity})
	if err != nil {
		klog.Errorf("[memorymanager] AddContainer error: error updating cpuset.mems for container (pod: %s, container: %s, container id: %s, err: %v)", pod.Name, container.Name, containerID, err)

		m.Lock()
		err = m.policyRemoveContainerByRef(string(pod.UID), container.Name)
		if err != nil {
			klog.Errorf("[memorymanager] AddContainer rollback state error: %v", err)
		}
		m.Unlock()
	}
	return err
}

// Allocate is called to pre-allocate memory resources during Pod admission.
func (m *manager) Allocate(pod *v1.Pod, container *v1.Container) error {
	// Garbage collect any stranded resources before allocation
	m.removeStaleState()

	m.Lock()
	defer m.Unlock()

	// Call down into the policy to assign this container memory if required.
	if err := m.policy.Allocate(m.state, pod, container); err != nil {
		klog.Errorf("[memorymanager] Allocate error: %v", err)
		return err
	}
	return nil
}

// RemoveContainer removes the container from the state
func (m *manager) RemoveContainer(containerID string) error {
	m.Lock()
	defer m.Unlock()

	// if error appears it means container entry already does not exist under the container map
	podUID, containerName, err := m.containerMap.GetContainerRef(containerID)
	if err != nil {
		klog.Warningf("[memorymanager] Failed to get container %s from container map error: %v", containerID, err)
		return nil
	}

	err = m.policyRemoveContainerByRef(podUID, containerName)
	if err != nil {
		klog.Errorf("[memorymanager] RemoveContainer error: %v", err)
		return err
	}

	return nil
}

// State returns the state of the manager
func (m *manager) State() state.Reader {
	return m.state
}

// GetTopologyHints returns the topology hints for the topology manager
func (m *manager) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	// Garbage collect any stranded resources before providing TopologyHints
	m.removeStaleState()
	// Delegate to active policy
	return m.policy.GetTopologyHints(m.state, pod, container)
}

// TODO: consider to move this method to manager interface, the only difference between CPU manager is assignments, we can send it to the method
func (m *manager) removeStaleState() {
	// Only once all sources are ready do we attempt to remove any stale state.
	// This ensures that the call to `m.activePods()` below will succeed with
	// the actual active pods list.
	if !m.sourcesReady.AllReady() {
		return
	}

	// We grab the lock to ensure that no new containers will grab memory block while
	// executing the code below. Without this lock, its possible that we end up
	// removing state that is newly added by an asynchronous call to
	// AddContainer() during the execution of this code.
	m.Lock()
	defer m.Unlock()

	// Get the list of active pods.
	activePods := m.activePods()

	// Build a list of (podUID, containerName) pairs for all containers in all active Pods.
	activeContainers := make(map[string]map[string]struct{})
	for _, pod := range activePods {
		activeContainers[string(pod.UID)] = make(map[string]struct{})
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
			activeContainers[string(pod.UID)][container.Name] = struct{}{}
		}
	}

	// Loop through the MemoryManager state. Remove any state for containers not
	// in the `activeContainers` list built above.
	assignments := m.state.GetMemoryAssignments()
	for podUID := range assignments {
		for containerName := range assignments[podUID] {
			if _, ok := activeContainers[podUID][containerName]; !ok {
				klog.Infof("[memorymanager] removeStaleState: removing (pod %s, container: %s)", podUID, containerName)
				err := m.policyRemoveContainerByRef(podUID, containerName)
				if err != nil {
					klog.Errorf("[memorymanager] removeStaleState: failed to remove (pod %s, container %s), error: %v)", podUID, containerName, err)
				}
			}
		}
	}
}

func (m *manager) policyRemoveContainerByRef(podUID string, containerName string) error {
	err := m.policy.RemoveContainer(m.state, podUID, containerName)
	if err == nil {
		m.containerMap.RemoveByContainerRef(podUID, containerName)
	}

	return err
}

func validateReservedMemory(nodeAllocatableReservation v1.ResourceList, reservedMemory map[int]map[v1.ResourceName]resource.Quantity) error {
	// TODO: this will check equality of total reserved memory by node allocatable feature and total pre-reserved memory

	return nil
}

func getReservedMemory(machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, reservedMemory map[int]map[v1.ResourceName]resource.Quantity) (systemReservedMemory, error) {
	// TODO: we should add new kubelet parameter, and to get reserved memory per NUMA node from it
	// currently we use kube-reserved + system-reserved + eviction reserve for each NUMA node, that creates memory over-consumption
	// and no reservation for huge pages

	if err := validateReservedMemory(nodeAllocatableReservation, reservedMemory); err != nil {
		return nil, err
	}

	reserved := systemReservedMemory{}
	for _, node := range machineInfo.Topology {
		memory := nodeAllocatableReservation[v1.ResourceMemory]
		if memory.IsZero() {
			break
		}
		value, succeeded := memory.AsInt64()
		if !succeeded {
			return nil, fmt.Errorf("failed to represent reserved memory as int64")
		}

		reserved[node.Id] = map[v1.ResourceName]uint64{
			v1.ResourceMemory: uint64(value),
		}
	}
	return reserved, nil
}
