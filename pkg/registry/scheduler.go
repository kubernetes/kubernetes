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

package registry

import (
	"fmt"
	"math/rand"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Scheduler is an interface implemented by things that know how to schedule pods onto machines.
type Scheduler interface {
	Schedule(api.Pod) (string, error)
}

// RandomScheduler choses machines uniformly at random.
type RandomScheduler struct {
	machines MinionRegistry
	random   rand.Rand
}

func MakeRandomScheduler(machines MinionRegistry, random rand.Rand) Scheduler {
	return &RandomScheduler{
		machines: machines,
		random:   random,
	}
}

func (s *RandomScheduler) Schedule(pod api.Pod) (string, error) {
	machines, err := s.machines.List()
	if err != nil {
		return "", err
	}
	return machines[s.random.Int()%len(machines)], nil
}

// RoundRobinScheduler chooses machines in order.
type RoundRobinScheduler struct {
	machines     MinionRegistry
	currentIndex int
}

func MakeRoundRobinScheduler(machines MinionRegistry) Scheduler {
	return &RoundRobinScheduler{
		machines:     machines,
		currentIndex: -1,
	}
}

func (s *RoundRobinScheduler) Schedule(pod api.Pod) (string, error) {
	machines, err := s.machines.List()
	if err != nil {
		return "", err
	}
	s.currentIndex = (s.currentIndex + 1) % len(machines)
	result := machines[s.currentIndex]
	return result, nil
}

type FirstFitScheduler struct {
	machines MinionRegistry
	registry PodRegistry
	random   *rand.Rand
}

func MakeFirstFitScheduler(machines MinionRegistry, registry PodRegistry, random *rand.Rand) Scheduler {
	return &FirstFitScheduler{
		machines: machines,
		registry: registry,
		random:   random,
	}
}

func (s *FirstFitScheduler) containsPort(pod api.Pod, port api.Port) bool {
	for _, container := range pod.DesiredState.Manifest.Containers {
		for _, podPort := range container.Ports {
			if podPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

func (s *FirstFitScheduler) Schedule(pod api.Pod) (string, error) {
	machines, err := s.machines.List()
	if err != nil {
		return "", err
	}
	machineToPods := map[string][]api.Pod{}
	pods, err := s.registry.ListPods(labels.Everything())
	if err != nil {
		return "", err
	}
	for _, scheduledPod := range pods {
		host := scheduledPod.CurrentState.Host
		machineToPods[host] = append(machineToPods[host], scheduledPod)
	}
	var machineOptions []string
	for _, machine := range machines {
		podFits := true
		for _, scheduledPod := range machineToPods[machine] {
			for _, container := range pod.DesiredState.Manifest.Containers {
				for _, port := range container.Ports {
					if s.containsPort(scheduledPod, port) {
						podFits = false
					}
				}
			}
		}
		if podFits {
			machineOptions = append(machineOptions, machine)
		}
	}
	if len(machineOptions) == 0 {
		return "", fmt.Errorf("failed to find fit for %#v", pod)
	} else {
		return machineOptions[s.random.Int()%len(machineOptions)], nil
	}
}
