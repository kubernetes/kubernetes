/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/google/cadvisor/info"
)

type FreeResourceCalculator interface {
	// MaxContainer() returns the max. available resource set on a given
	// machine. The resource set is reprecented in terms of
	// info.ContainerSpec
	MaxContainer(host string) (*info.ContainerSpec, error)
}

// machineSelector selects feasible machines based on some constraints.
type machineSelector struct {
	machineToPods    map[string][]*api.Pod
	feasibleMachines []string
	err              error
}

func containsPort(pod *api.Pod, port api.Port) bool {
	for _, container := range pod.DesiredState.Manifest.Containers {
		for _, podPort := range container.Ports {
			if podPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

func (self *machineSelector) PortsOpenFor(pod *api.Pod) *machineSelector {
	if self.err != nil {
		return self
	}
	newFeasibleMachines := make([]string, 0, len(self.feasibleMachines))
	for _, machine := range self.feasibleMachines {
		podFits := true
	P:
		for _, scheduledPod := range self.machineToPods[machine] {
			for _, container := range pod.DesiredState.Manifest.Containers {
				for _, port := range container.Ports {
					if containsPort(scheduledPod, port) {
						podFits = false
						break P
					}
				}
			}
		}
		if podFits {
			newFeasibleMachines = append(newFeasibleMachines, machine)
		}
	}
	self.feasibleMachines = newFeasibleMachines
	if len(self.feasibleMachines) == 0 {
		self.feasibleMachines = nil
		self.err = fmt.Errorf("failed to find fit for %#v: no ports available", pod)
	}
	return self
}

// Return true if all resources in a is larger than resources in b.
func isLargerThan(a, b *info.ContainerSpec) bool {
	if a.Cpu.Limit < b.Cpu.Limit {
		return false
	}
	if a.Memory.Limit < b.Memory.Limit {
		return false
	}
	return true
}

func (self *machineSelector) Fits(requirement *info.ContainerSpec, calc FreeResourceCalculator, pod *api.Pod) *machineSelector {
	if self.err != nil {
		return self
	}
	newFeasibleMachines := make([]string, 0, len(self.feasibleMachines))
	for _, machine := range self.feasibleMachines {
		maxContainer, err := calc.MaxContainer(machine)
		if err != nil {
			self.err = err
			return self
		}
		if isLargerThan(maxContainer, requirement) {
			newFeasibleMachines = append(newFeasibleMachines, machine)
		}
	}
	self.feasibleMachines = newFeasibleMachines
	if len(self.feasibleMachines) == 0 {
		self.feasibleMachines = nil
		self.err = fmt.Errorf("failed to find fit for %#v: no resources available", pod)
	}
	return self
}

func (self *machineSelector) FeasibleMachines() ([]string, error) {
	if self.err != nil {
		return nil, self.err
	}
	return self.feasibleMachines, nil
}

type resourceCalculatorBasedOnPercentiles struct {
	cpuPercentile       int
	memPercentile       int
	containerInfoGetter client.ContainerInfoGetter
}

func (self *resourceCalculatorBasedOnPercentiles) MaxContainer(host string) (*info.ContainerSpec, error) {
	machineSpec, err := self.containerInfoGetter.GetMachineSpec(host)
	if err != nil {
		return nil, err
	}
	maxContainer := &info.ContainerSpec{
		Cpu: &info.CpuSpec{
			Limit: uint64(machineSpec.NumCores * 1000),
		},
		Memory: &info.MemorySpec{
			Limit: uint64(machineSpec.MemoryCapacity),
		},
	}
}
