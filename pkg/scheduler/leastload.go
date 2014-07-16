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
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/google/cadvisor/info"
)

// Resource calculator used to calculate how much free resources are
// available on a given machine.
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

func NewMachineSelector(podLister PodLister, machines ...string) (*machineSelector, error) {
	machineToPods := make(map[string][]*api.Pod, len(machines))
	pods, err := podLister.ListPods(labels.Everything())
	if err != nil {
		return nil, err
	}
	for _, scheduledPod := range pods {
		host := scheduledPod.CurrentState.Host
		machineToPods[host] = append(machineToPods[host], &scheduledPod)
	}

	ret := &machineSelector{
		feasibleMachines: machines,
		machineToPods:    machineToPods,
	}
	return ret, nil
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

func (self *machineSelector) WithTimestampBefore(timestamps map[string]time.Time, t time.Time) *machineSelector {
	if self.err != nil {
		return self
	}
	newFeasibleMachines := make([]string, 0, len(self.feasibleMachines))
	for _, machine := range self.feasibleMachines {
		if machineTimestamp, ok := timestamps[machine]; ok {
			if machineTimestamp.Before(t) {
				newFeasibleMachines = append(newFeasibleMachines, machine)
			}
		} else {
			newFeasibleMachines = append(newFeasibleMachines, machine)
		}
	}
	self.feasibleMachines = newFeasibleMachines
	if len(self.feasibleMachines) == 0 {
		self.feasibleMachines = nil
		self.err = fmt.Errorf("failed to find fit: all machines' timestamps are later than %v", t)
	}
	return self
}

func (self *machineSelector) Remove(machines ...string) *machineSelector {
	if self.err != nil {
		return self
	}
	newFeasibleMachines := make([]string, 0, len(self.feasibleMachines))
	for _, machine := range self.feasibleMachines {
		include := true
		for _, rm := range machines {
			if rm == machine {
				include = false
				break
			}
		}
		if include {
			newFeasibleMachines = append(newFeasibleMachines, machine)
		}
	}
	self.feasibleMachines = newFeasibleMachines
	if len(self.feasibleMachines) == 0 {
		self.feasibleMachines = nil
		self.err = errors.New("failed to find fit: all machines are removed manually")
	}
	return self
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
	if b == nil {
		return true
	}
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
	req := &info.ContainerInfoRequest{
		NumStats:               1,
		NumSamples:             1,
		CpuUsagePercentiles:    []int{self.cpuPercentile},
		MemoryUsagePercentages: []int{self.memPercentile},
	}
	machineStats, err := self.containerInfoGetter.GetMachineInfo(host, req)
	if err != nil {
		return nil, err
	}

	cpuUsage := machineStats.StatsPercentiles.CpuUsagePercentiles[0].Value
	memUsage := machineStats.StatsPercentiles.MemoryUsagePercentiles[0].Value

	maxContainer := &info.ContainerSpec{
		Cpu: &info.CpuSpec{
			// Number of CPU nano seconds per second
			Limit: uint64(machineSpec.NumCores*1E9) - cpuUsage,
		},
		Memory: &info.MemorySpec{
			Limit: uint64(machineSpec.MemoryCapacity) - memUsage,
		},
	}

	return maxContainer, nil
}

func NewResourceCalculatorBasedOnPercentiles(cpuPercentile, memPercentile int, getter client.ContainerInfoGetter) FreeResourceCalculator {
	return &resourceCalculatorBasedOnPercentiles{
		cpuPercentile:       cpuPercentile,
		memPercentile:       memPercentile,
		containerInfoGetter: getter,
	}
}

type machineSetWithSemiorder struct {
	calc     FreeResourceCalculator
	machines []string
}

func (self *machineSetWithSemiorder) Len() int {
	return len(self.machines)
}

func (self *machineSetWithSemiorder) Swap(i, j int) {
	self.machines[i], self.machines[j] = self.machines[j], self.machines[i]
}

func (self *machineSetWithSemiorder) Less(i, j int) bool {
	maxContainerOnI, err := self.calc.MaxContainer(self.machines[i])
	if err != nil {
		// XXX(monnand): How could be report this error?

		// If the ith machine's info is not available, then it is
		// smaller than j.
		return true
	}
	maxContainerOnJ, err := self.calc.MaxContainer(self.machines[j])
	if err != nil {
		// If the jth machine's info is not available, then it is
		// smaller than i.
		return false
	}
	// TODO(monnand): Use some more advanced algorithm.
	return maxContainerOnI.Memory.Limit < maxContainerOnJ.Memory.Limit
}

type LeastLoadScheduler struct {
	// The time when the last job scheduled to each machine
	machineRecentScheduledTime map[string]time.Time

	// Resource calculator used to calculate how much free resources are
	// available on a given machine.
	calc         FreeResourceCalculator
	podLister    PodLister
	minContainer minContainerEstimator
}

type minContainerEstimator interface {
	MinContainer(pod *api.Pod) (*info.ContainerSpec, error)
}

func (self *LeastLoadScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	machines, err := minionLister.List()
	if err != nil {
		return "", err
	}

	selector, err := NewMachineSelector(self.podLister, machines...)
	if err != nil {
		return "", err
	}

	minContainer, err := self.minContainer.MinContainer(&pod)
	if err != nil {
		return "", err
	}
	// When possible, do not scheduler pods to the same machine within one minute.
	t := time.Now().Add(-time.Minute)
	selector = selector.WithTimestampBefore(self.machineRecentScheduledTime, t)
	if feasibleMachines, err := selector.FeasibleMachines(); err != nil || len(feasibleMachines) == 0 {
		// All machines are scheduled within one minute.
		// We could do nothing at this point.
		selector, err = NewMachineSelector(self.podLister, machines...)
		if err != nil {
			return "", err
		}
	}

	machines, err = selector.PortsOpenFor(&pod).Fits(minContainer, self.calc, &pod).FeasibleMachines()
	if err != nil {
		return "", err
	}

	semiorderSet := &machineSetWithSemiorder{
		machines: machines,
		calc:     self.calc,
	}
	// TODO(monnand): A heap would be better.
	sort.Sort(semiorderSet)
	selectedMachine := semiorderSet.machines[semiorderSet.Len()-1]
	self.machineRecentScheduledTime[selectedMachine] = time.Now()
	return selectedMachine, nil
}
