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

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	v1qos "k8s.io/kubernetes/pkg/api/v1/helper/qos"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

type staticPolicy struct {
	sharedContainers map[string]struct{}
}

// NewStaticPolicy returns a cupset manager policy that does not change
// CPU assignments for exclusively pinned guaranteed containers after
// the main container process starts.
func NewStaticPolicy() Policy {
	return &staticPolicy{
		sharedContainers: map[string]struct{}{},
	}
}

func (p *staticPolicy) Name() string {
	return "static"
}

func (p *staticPolicy) Start(s state.State) {
	// Build the initial shared cpuset.
	// NB: Iteration starts at index `1` here because CPU `0` is reserved
	//     for infrastructure processes.
	// TODO(CD): Improve this to align with kube/system reserved resources.
	shared := cpuset.NewCPUSet()
	for cpuid := 1; cpuid < s.Topology().NumCPUs; cpuid++ {
		shared.Add(cpuid)
	}
	s.SetDefaultCPUSet(shared)
}

func (p *staticPolicy) RegisterContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error {
	glog.Infof("[cpumanager] static policy: RegisterContainer (pod: %s, container: %s, container id: %s)", pod.Name, container.Name, containerID)
	if numCPUs := guaranteedCPUs(pod, container); numCPUs != 0 {
		// container belongs in an exclusively allocated pool
		cpuset, err := p.allocateCPUs(s, numCPUs)
		if err != nil {
			glog.Errorf("[cpumanager] unable to allocate %d CPUs (container: (%v)", numCPUs, containerID, err)
			return err
		}
		s.SetCPUSet(containerID, cpuset)
	}
	// container belongs in the shared pool (nothing to do; use default cpuset)
	return nil
}

func (p *staticPolicy) UnregisterContainer(s state.State, containerID string) error {
	glog.Infof("[cpumanager] static policy: UnregisterContainer (container id: %s)", containerID)
	if toRelease, ok := s.GetCPUSet(containerID); ok {
		s.Delete(containerID)
		p.releaseCPUs(s, toRelease)
	}
	return nil
}

func (p *staticPolicy) allocateCPUs(s state.State, numCPUs int) (cpuset.CPUSet, error) {
	glog.Infof("[cpumanager] allocateCpus: (numCPUs: %d)", numCPUs)
	if numCPUs > s.GetDefaultCPUSet().Size() {
		return cpuset.NewCPUSet(), fmt.Errorf("not enough cpus available to satisfy request")
	}

	// TODO(CD): Acquire CPUs topologically instead of sequentially...
	sharedCPUs := s.GetDefaultCPUSet().AsSlice()
	resultCPUs := sharedCPUs[0:numCPUs]
	result := cpuset.NewCPUSet(resultCPUs...)

	// Remove allocated CPUs from the shared CPUSet.
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Difference(result))

	glog.Infof("[cpumanager] allocateCPUs: returning \"%v\"", result)
	return result, nil
}

func (p *staticPolicy) releaseCPUs(s state.State, release cpuset.CPUSet) {
	// mutate the shared pool, adding supplied cpus
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(release))
}

func guaranteedCPUs(pod *v1.Pod, container *v1.Container) int {
	if v1qos.GetPodQOS(pod) != v1.PodQOSGuaranteed {
		return 0
	}
	cpuQuantity := container.Resources.Requests[v1.ResourceCPU]
	glog.Infof("[cpumanager] guaranteedCpus (container: %s, cpu request: %v)", container.Name, cpuQuantity.MilliValue())
	if cpuQuantity.Value()*1000 != cpuQuantity.MilliValue() {
		return 0
	}
	// Safe downcast to do for all systems with < 2.1 billion CPUs.
	// Per the language spec, `int` is guaranteed to be at least 32 bits wide.
	// https://golang.org/ref/spec#Numeric_types
	return int(cpuQuantity.Value())
}
