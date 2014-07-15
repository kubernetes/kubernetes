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

import "github.com/GoogleCloudPlatform/kubernetes/pkg/api"

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
	return self
}

func (self *machineSelector) FeasibleMachines() ([]string, error) {
	if self.err != nil {
		return nil, self.err
	}
	return self.feasibleMachines, nil
}
