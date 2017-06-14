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
	"io/ioutil"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/topo"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

type podLister interface {
	GetPods() []*v1.Pod
}

type Manager interface {
	Start()

	Policy() Policy

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
}

func NewManager(p Policy, cr internalapi.RuntimeService, lister podLister, statusProvider status.PodStatusProvider) (Manager, error) {
	cpuInfoFile, err := ioutil.ReadFile("/proc/cpuinfo")
	if err != nil {
		return nil, err
	}
	topo, err := discoverTopology(cpuInfoFile)
	if err != nil {
		return nil, err
	}

	glog.Infof("detected CPU topology: %v", topo)

	return &manager{
		policy:            p,
		state:             state.NewMemoryState(topo),
		containerRuntime:  cr,
		podLister:         lister,
		podStatusProvider: statusProvider,
	}, nil
}

type manager struct {
	sync.Mutex
	policy Policy
	state  state.State

	// containerRuntime is the container runtime service interface needed
	// to make UpdateContainerResources() calls against the containers.
	containerRuntime internalapi.RuntimeService

	// podLister provides a method for listing all the pods on the node
	// so all the containers can be updated in the reconciliation loop.
	podLister podLister

	// podStatusProvider provides a method for obtaining pod statuses
	// and the containerID of their containers
	podStatusProvider status.PodStatusProvider
}

func (m *manager) Start() {
	glog.Infof("[cpumanger] starting (policy: \"%s\")", m.policy.Name())
	m.policy.Start(m.state)
	go wait.Until(m.reconcileState, time.Second, wait.NeverStop)
}

func (m *manager) Policy() Policy {
	return m.policy
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

var (
	processorRegExp = regexp.MustCompile(`^processor\s*:\s*([0-9]+)$`)
)

func discoverTopology(cpuinfo []byte) (*topo.CPUTopology, error) {
	cpus := 0
	for _, line := range strings.Split(string(cpuinfo), "\n") {
		matches := processorRegExp.FindSubmatch([]byte(line))
		if len(matches) == 2 {
			cpus++
		}
	}
	if cpus == 0 {
		return nil, fmt.Errorf("could not detect number of cpus")
	}
	return &topo.CPUTopology{
		NumCPUs: cpus,
	}, nil
}

func (m *manager) reconcileState() {
	m.Lock()
	defer m.Unlock()

	for _, pod := range m.podLister.GetPods() {
		for _, container := range pod.Spec.Containers {
			status, ok := m.podStatusProvider.GetPodStatus(pod.UID)
			if !ok {
				glog.Warningf("[cpumanager] reconcileState: skipping pod; status not found (pod: %s, container: %s)", pod.Name, container.Name)
				break
			}

			containerID, err := findContainerIDByName(&status, container.Name)
			if err != nil {
				glog.Warningf("[cpumanager] reconcileState: skipping container; ID not found in status (pod: %s, container: %s, error: %v)", pod.Name, container.Name, err)
				continue
			}

			cset := m.state.GetCPUSetOrDefault(containerID)
			if cset.IsEmpty() {
				glog.Info("[cpumanager] reconcileState: skipping container; assigned cpuset is empty (pod: %s, container: %s)", pod.Name, container.Name)
				continue
			}

			glog.Infof("[cpumanager] reconcileState: updating container (pod: %s, container: %s, container id: %s, cpuset: \"%v\")", pod.Name, container.Name, containerID, cset)
			err = m.containerRuntime.UpdateContainerResources(
				containerID,
				&runtimeapi.LinuxContainerResources{
					CpusetCpus: cset.String(),
				})
			if err != nil {
				glog.Errorf("[cpumanager] reconcileState: failed to update container (pod: %s, container: %s, container id: %s, cpuset: \"%v\", err: %v)", pod.Name, container.Name, containerID, cset, err)
			}
		}
	}
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
