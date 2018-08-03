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

package cpumanager

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

type runtimeService interface {
	UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error
}

type policyName string

// cpuManagerStateFileName is the name file name where cpu manager stores it's state
const cpuManagerStateFileName = "cpu_manager_state"

// Manager interface provides methods for Kubelet to manage pod cpus.
type Manager interface {
	// Start is called during Kubelet initialization.
	Start(activePods ActivePodsFunc, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService)

	// AddContainer is called between container create and container start
	// so that initial CPU affinity settings can be written through to the
	// container runtime before the first process begins to execute.
	AddContainer(p *v1.Pod, c *v1.Container, containerID string) error

	// RemoveContainer is called after Kubelet decides to kill or delete a
	// container. After this call, the CPU manager stops trying to reconcile
	// that container and any CPUs dedicated to the container are freed.
	RemoveContainer(containerID string) error

	// State returns a read-only interface to the internal CPU manager state.
	State() state.Reader
}

// ReconcileFunc provides a way for policy implementations to trigger
// reconciliation of the current CPU manager state with real container
// configuration via the CRI.
type ReconcileFunc func()

type manager struct {
	sync.Mutex
	policy Policy

	// reconcilePeriod is the duration between calls to reconcileState.
	reconcilePeriod time.Duration

	// state allows pluggable CPU assignment policies while sharing a common
	// representation of state for the system to inspect and reconcile.
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

	machineInfo *cadvisorapi.MachineInfo

	nodeAllocatableReservation v1.ResourceList

	// map of pods to containers that require reconciliation.
	containersToReconcile map[*v1.Pod][]v1.Container

	// channel to signal periodic reconciliation for containers that
	// we previously failed to update.
	reconcileFailed chan struct{}

	// channel to signal reconciliation for all active containers.
	reconcileAll chan struct{}
}

var _ Manager = &manager{}

// NewManager creates new cpu manager based on provided policy
func NewManager(cpuPolicyName string, reconcilePeriod time.Duration, machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, stateFileDirectory string) (Manager, error) {
	var policy Policy

	switch policyName(cpuPolicyName) {

	case PolicyNone:
		policy = NewNonePolicy()

	case PolicyStatic:
		topo, err := topology.Discover(machineInfo)
		if err != nil {
			return nil, err
		}
		glog.Infof("[cpumanager] detected CPU topology: %v", topo)
		reservedCPUs, ok := nodeAllocatableReservation[v1.ResourceCPU]
		if !ok {
			// The static policy cannot initialize without this information.
			return nil, fmt.Errorf("[cpumanager] unable to determine reserved CPU resources for static policy")
		}
		if reservedCPUs.IsZero() {
			// The static policy requires this to be nonzero. Zero CPU reservation
			// would allow the shared pool to be completely exhausted. At that point
			// either we would violate our guarantee of exclusivity or need to evict
			// any pod that has at least one container that requires zero CPUs.
			// See the comments in policy_static.go for more details.
			return nil, fmt.Errorf("[cpumanager] the static policy requires systemreserved.cpu + kubereserved.cpu to be greater than zero")
		}

		// Take the ceiling of the reservation, since fractional CPUs cannot be
		// exclusively allocated.
		reservedCPUsFloat := float64(reservedCPUs.MilliValue()) / 1000
		numReservedCPUs := int(math.Ceil(reservedCPUsFloat))
		policy = NewStaticPolicy(topo, numReservedCPUs)

	default:
		glog.Errorf("[cpumanager] Unknown policy \"%s\", falling back to default policy \"%s\"", cpuPolicyName, PolicyNone)
		policy = NewNonePolicy()
	}

	stateImpl, err := state.NewCheckpointState(stateFileDirectory, cpuManagerStateFileName, policy.Name())
	if err != nil {
		return nil, fmt.Errorf("could not initialize checkpoint manager: %v", err)
	}

	manager := &manager{
		policy:                     policy,
		reconcilePeriod:            reconcilePeriod,
		state:                      stateImpl,
		machineInfo:                machineInfo,
		nodeAllocatableReservation: nodeAllocatableReservation,
		containersToReconcile:      make(map[*v1.Pod][]v1.Container),
		reconcileFailed:            make(chan struct{}), // unbuffered
		reconcileAll:               make(chan struct{}), // unbuffered
	}
	return manager, nil
}

func (m *manager) Start(activePods ActivePodsFunc, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService) {
	glog.Infof("[cpumanager] starting with %s policy", m.policy.Name())
	glog.Infof("[cpumanager] reconciling every %v", m.reconcilePeriod)

	m.activePods = activePods
	m.podStatusProvider = podStatusProvider
	m.containerRuntime = containerRuntime

	m.policy.Start(m.state, m.reconcileFunc)
	if m.policy.Name() == string(PolicyNone) {
		return
	}
	// Start continuous read from reconciliation channels
	go m.reconcile()
	// Periodically signal retries for failed reconciliation
	go wait.Until(func() { m.reconcileFailed <- struct{}{} }, m.reconcilePeriod, wait.NeverStop)
}

func (m *manager) AddContainer(p *v1.Pod, c *v1.Container, containerID string) error {
	m.Lock()
	err := m.policy.AddContainer(m.state, p, c, containerID)
	if err != nil {
		glog.Errorf("[cpumanager] AddContainer error: %v", err)
		m.Unlock()
		return err
	}
	cpus := m.state.GetCPUSetOrDefault(containerID)
	m.Unlock()

	if !cpus.IsEmpty() {
		err = m.updateContainerCPUSet(containerID, cpus)
		if err != nil {
			glog.Errorf("[cpumanager] AddContainer error: %v", err)
			return err
		}
	} else {
		glog.V(5).Infof("[cpumanager] update container resources is skipped due to cpu set is empty")
	}

	return nil
}

func (m *manager) RemoveContainer(containerID string) error {
	m.Lock()
	defer m.Unlock()

	err := m.policy.RemoveContainer(m.state, containerID)
	if err != nil {
		glog.Errorf("[cpumanager] RemoveContainer error: %v", err)
		return err
	}
	return nil
}

func (m *manager) State() state.Reader {
	return m.state
}

func (m *manager) reconcile() {
	// forever
	for {
		select {
		case <-m.reconcileAll:
			glog.V(5).Info("[cpumanager] reconciling all active containers")
			m.resetContainersToReconcile()
			m.containersToReconcile = m.doReconcile()
		case <-m.reconcileFailed:
			glog.V(5).Info("[cpumanager] reconciling containers that previously failed to reconcile")
			m.containersToReconcile = m.doReconcile()
		}
	}
}

// Implements ReconcileFunc; this function is passed to the Policy.
func (m *manager) reconcileFunc() {
	m.reconcileAll <- struct{}{}
}

func (m *manager) resetContainersToReconcile() {
	result := make(map[*v1.Pod][]v1.Container)
	for _, pod := range m.activePods() {
		allContainers := pod.Spec.InitContainers
		allContainers = append(allContainers, pod.Spec.Containers...)
		result[pod] = allContainers
	}
	m.containersToReconcile = result
}

// This logic is thread-safe by virtue of only being called from
// reconcile(), which is itself executed in only one goroutine.
func (m *manager) doReconcile() (failed map[*v1.Pod][]v1.Container) {
	failed = make(map[*v1.Pod][]v1.Container)

	// Adds the supplied pod and container to the list of failed
	// reconciliations.
	addFailed := func(pod *v1.Pod, container v1.Container) {
		if _, ok := failed[pod]; !ok {
			failed[pod] = make([]v1.Container, 1)
		}
		podContainers := failed[pod]
		failed[pod] = append(podContainers, container)
	}

	// Returns true if the supplied pod exists in the list of active pods.
	podExists := func(activePods []*v1.Pod, pod *v1.Pod) bool {
		for _, activePod := range activePods {
			if activePod.UID == pod.UID {
				return true
			}
		}
		return false
	}

	// Remove inactive pods from containersToReconcile.
	// After this operation, `m.containersToReconcile` contains:
	//
	// 1. containers queued for reconciliation and not yet processed
	// 2. containers that failed to update in a prior reconciliation pass
	activePods := m.activePods()
	for pod := range m.containersToReconcile {
		if !podExists(activePods, pod) {
			delete(m.containersToReconcile, pod)
		}
	}

	for pod, containers := range m.containersToReconcile {
		// Get pod status. Pod status may be missing for pods that
		// have been admitted but are not yet running.
		podStatus, ok := m.podStatusProvider.GetPodStatus(pod.UID)
		if !ok {
			glog.V(5).Infof("[cpumanager] reconcile: skipping pod; status not found (pod: %s)", pod.Name)
			for _, container := range containers {
				addFailed(pod, container)
			}
			continue
		}

		if podStatus.Phase == v1.PodRunning && pod.DeletionTimestamp != nil {
			// Pod's containers have already been removed from state
			// Skip the whole pod since it's not running and will be deleted soon
			glog.V(5).Infof("[cpumanager] reconcile: skipping pod since deletion timestamp is set (pod: %s)", pod.Name)
			continue
		}

		// Process containers in this pod.
		for _, container := range containers {
			// Skip unnamed containers (e.g. pause)
			if container.Name == "" {
				continue
			}

			// Look up container ID in the pod status. We need this in order to
			// both get the assigned CPU set from the state and also ask the
			// CRI to update the container resource config. The most likely
			// cause of missing container ID is that it is not yet running.
			containerID, err := findContainerIDByName(&podStatus, container.Name)
			if err != nil {
				glog.V(5).Infof("[cpumanager] reconcile: skipping container; ID not found in status (pod: %s, container: %s, error: %v)", pod.Name, container.Name, err)
				addFailed(pod, container)
				continue
			}

			// Update the container with the appropriate CPU based on the
			// current state.
			cset := m.state.GetCPUSetOrDefault(containerID)
			glog.V(4).Infof("[cpumanager] reconcile: updating container (pod: %s, container: %s, container id: %s, cpuset: \"%v\")", pod.Name, container.Name, containerID, cset)
			err = m.updateContainerCPUSet(containerID, cset)
			if err != nil {
				glog.Errorf("[cpumanager] reconcile: failed to update container (pod: %s, container: %s, container id: %s, cpuset: \"%v\", error: %v)", pod.Name, container.Name, containerID, cset, err)
				addFailed(pod, container)
				continue
			}
		}
	}
	return failed
}

func findContainerIDByName(status *v1.PodStatus, name string) (string, error) {
	for _, container := range status.ContainerStatuses {
		if container.Name == name && container.ContainerID != "" {
			cid := &kubecontainer.ContainerID{}
			err := cid.ParseString(container.ContainerID)
			if err != nil {
				return "", err
			}
			return cid.ID, nil
		}
	}
	return "", fmt.Errorf("unable to find ID for container with name %v in pod status (it may not be running)", name)
}

func (m *manager) updateContainerCPUSet(containerID string, cpus cpuset.CPUSet) error {
	// TODO: Consider adding a `ResourceConfigForContainer` helper in
	// helpers_linux.go similar to what exists for pods.
	// It would be better to pass the full container resources here instead of
	// this patch-like partial resources.
	if cpus.IsEmpty() {
		// NOTE: This should not happen outside of tests.
		glog.Infof("[cpumanager] skipping container update; cpuset is empty (containerID: %s)", containerID)
		return fmt.Errorf("empty cpuset")
	}

	return m.containerRuntime.UpdateContainerResources(
		containerID,
		&runtimeapi.LinuxContainerResources{
			CpusetCpus: cpus.String(),
		})
}
