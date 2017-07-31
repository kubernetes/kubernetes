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
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/eviction"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

type kletGetter interface {
	GetPods() []*v1.Pod
	GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error)
	GetNode() (*v1.Node, error)
}

type runtimeService interface {
	UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error
}

type policyName string

// Manager interface provides methods for kubelet to manage pod cpus
type Manager interface {
	Start()

	// RegisterContainer registers a container with the cpuset manager
	// resulting in synchronous setting on the container cpuset.  This
	// is called after RegisterContainer(), which creates the containerID, and
	// before StartContainer(), such that the cpuset is configured before
	// the container starts
	RegisterContainer(p *v1.Pod, c *v1.Container, containerID string) error

	// UnregisterContainer is called near UnregisterContainer() so that the manager
	// stops trying to update that container in the reconcilation loop and
	// any CPUs dedicated to the container are freed to the shared pool.
	UnregisterContainer(containerID string) error

	State() state.Reader

	IsUnderCPUPressure() bool

	lifecycle.PodAdmitHandler
}

type manager struct {
	sync.Mutex
	policy Policy
	state  state.State

	// containerRuntime is the container runtime service interface needed
	// to make UpdateContainerResources() calls against the containers.
	containerRuntime runtimeService

	// podLister provides a method for listing all the pods on the node
	// so all the containers can be updated in the reconciliation loop.
	kletGetter kletGetter

	// podStatusProvider provides a method for obtaining pod statuses
	// and the containerID of their containers
	podStatusProvider status.PodStatusProvider
}

var _ Manager = &manager{}

// NewManager creates new cpu manager based on provided policy
func NewManager(cpuPolicyName string, cr runtimeService, kletGetter kletGetter, statusProvider status.PodStatusProvider, capacityProvider eviction.CapacityProvider) (Manager, lifecycle.PodAdmitHandler, error) {
	var policy Policy

	switch policyName(cpuPolicyName) {
	case PolicyNoop:
		policy = NewNoopPolicy()
	case PolicyStatic:
		machineInfo, err := kletGetter.GetCachedMachineInfo()
		if err != nil {
			return nil, nil, err
		}
		topo, err := topology.Discover(machineInfo)
		if err != nil {
			return nil, nil, err
		}
		glog.Infof("[cpumanager] detected CPU topology: %v", topo)

		resources := capacityProvider.GetNodeAllocatableReservation()
		cpuResource, ok := resources[v1.ResourceCPU]
		if ok {
			reservedCores := int(cpuResource.Value())
			glog.Infof("[cpumanager] reserving %v cores due to kube/system reserved", reservedCores)
			topo.NumReservedCores = reservedCores
		}
		policy = NewStaticPolicy(topo)
	default:
		glog.Warningf("[cpumanager] Unknown policy (\"%s\"), falling back to \"%s\" policy (\"%s\")", cpuPolicyName, PolicyNoop)
		policy = NewNoopPolicy()
	}

	manager := &manager{
		policy:            policy,
		state:             state.NewMemoryState(),
		containerRuntime:  cr,
		kletGetter:        kletGetter,
		podStatusProvider: statusProvider,
	}
	return manager, manager, nil
}

func (m *manager) Start() {
	glog.Infof("[cpumanger] starting with %s policy", m.policy.Name())
	m.policy.Start(m.state)
	if m.policy.Name() == string(PolicyNoop) {
		return
	}
	go wait.Until(func() { m.reconcileState() }, time.Second, wait.NeverStop)
}

func (m *manager) RegisterContainer(p *v1.Pod, c *v1.Container, containerID string) error {
	m.Lock()
	defer m.Unlock()

	err := m.policy.RegisterContainer(m.state, p, c, containerID)
	if err != nil {
		glog.Errorf("[cpumanager] RegisterContainer error: %v", err)
		return err
	}
	cpuset := m.state.GetCPUSetOrDefault(containerID)
	err = m.containerRuntime.UpdateContainerResources(
		containerID,
		&runtimeapi.LinuxContainerResources{
			CpusetCpus: cpuset.String(),
		})
	if err != nil {
		glog.Errorf("[cpumanager] RegisterContainer error: %v", err)
		return err
	}
	return nil
}

func (m *manager) UnregisterContainer(containerID string) error {
	m.Lock()
	defer m.Unlock()

	err := m.policy.UnregisterContainer(m.state, containerID)
	if err != nil {
		glog.Errorf("[cpumanager] UnregisterContainer error: %v", err)
		return err
	}
	return nil
}

// CPU Manager state is read-only from outside this package.
func (m *manager) State() state.Reader {
	return m.state
}

type reconciledContainer struct {
	podName       string
	containerName string
	containerID   string
}

func (m *manager) reconcileState() (success []reconciledContainer, failure []reconciledContainer) {
	m.Lock()
	defer m.Unlock()

	success = []reconciledContainer{}
	failure = []reconciledContainer{}

	for _, pod := range m.kletGetter.GetPods() {
		for _, container := range pod.Spec.Containers {
			status, ok := m.podStatusProvider.GetPodStatus(pod.UID)
			if !ok {
				glog.Warningf("[cpumanager] reconcileState: skipping pod; status not found (pod: %s, container: %s)", pod.Name, container.Name)
				failure = append(failure, reconciledContainer{pod.Name, container.Name, ""})
				break
			}

			containerID, err := findContainerIDByName(&status, container.Name)
			if err != nil {
				glog.Warningf("[cpumanager] reconcileState: skipping container; ID not found in status (pod: %s, container: %s, error: %v)", pod.Name, container.Name, err)
				failure = append(failure, reconciledContainer{pod.Name, container.Name, ""})
				continue
			}

			cset := m.state.GetCPUSetOrDefault(containerID)
			if cset.IsEmpty() {
				glog.Infof("[cpumanager] reconcileState: skipping container; assigned cpuset is empty (pod: %s, container: %s)", pod.Name, container.Name)
				failure = append(failure, reconciledContainer{pod.Name, container.Name, containerID})
				continue
			}

			glog.Infof("[cpumanager] reconcileState: updating container (pod: %s, container: %s, container id: %s, cpuset: \"%v\")", pod.Name, container.Name, containerID, cset)
			err = m.containerRuntime.UpdateContainerResources(
				containerID,
				&runtimeapi.LinuxContainerResources{
					CpusetCpus: cset.String(),
				})
			if err != nil {
				glog.Errorf("[cpumanager] reconcileState: failed to update container (pod: %s, container: %s, container id: %s, cpuset: \"%v\", error: %v)", pod.Name, container.Name, containerID, cset, err)
				failure = append(failure, reconciledContainer{pod.Name, container.Name, containerID})
				continue
			}
			success = append(success, reconciledContainer{pod.Name, container.Name, containerID})
		}
	}
	return success, failure
}

func findContainerIDByName(status *v1.PodStatus, name string) (string, error) {
	for _, container := range status.ContainerStatuses {
		if container.Name == name && container.ContainerID != "" {
			// hack hack strip docker:// hack hack
			return container.ContainerID[9:], nil
		}
	}
	return "", fmt.Errorf("[cpumanager] unable to find ID for container with name %v in pod status (it may not be running)", name)
}

func (m *manager) IsUnderCPUPressure() bool {
	return m.policy.IsUnderPressure()
}

// Admit rejects a pod if its not safe to admit for node stability.
// Required for lifecycle.PodAdmitHandler interface
func (m *manager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if !m.policy.IsUnderPressure() {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	admit := true
	containers := attrs.Pod.Spec.Containers
	for _, c := range containers {
		cpu := c.Resources.Requests.Cpu()
		if cpu == nil || cpu.Value() == 0 {
			admit = false
			break
		}
	}
	if admit {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	// reject pods with no CPU requests when under CPU pressure
	glog.Warningf("Failed to admit pod without CPU resource request %s - node is experiencing CPU pressure", attrs.Pod)
	return lifecycle.PodAdmitResult{
		Admit:   false,
		Reason:  "InsufficientCPU",
		Message: "Node is experiencing CPU Pressure",
	}
}
