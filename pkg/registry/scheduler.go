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

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Scheduler is an interface implemented by things that know how to schedule tasks onto machines.
type Scheduler interface {
	Schedule(Pod) (string, error)
}

// RandomScheduler choses machines uniformly at random.
type RandomScheduler struct {
	machines []string
	random   rand.Rand
}

func MakeRandomScheduler(machines []string, random rand.Rand) Scheduler {
	return &RandomScheduler{
		machines: machines,
		random:   random,
	}
}

func (s *RandomScheduler) Schedule(task Pod) (string, error) {
	return s.machines[s.random.Int()%len(s.machines)], nil
}

// RoundRobinScheduler chooses machines in order.
type RoundRobinScheduler struct {
	machines     []string
	currentIndex int
}

func MakeRoundRobinScheduler(machines []string) Scheduler {
	return &RoundRobinScheduler{
		machines:     machines,
		currentIndex: 0,
	}
}

func (s *RoundRobinScheduler) Schedule(task Pod) (string, error) {
	result := s.machines[s.currentIndex]
	s.currentIndex = (s.currentIndex + 1) % len(s.machines)
	return result, nil
}

type FirstFitScheduler struct {
	machines []string
	registry PodRegistry
}

func MakeFirstFitScheduler(machines []string, registry PodRegistry) Scheduler {
	return &FirstFitScheduler{
		machines: machines,
		registry: registry,
	}
}

func (s *FirstFitScheduler) containsPort(task Pod, port Port) bool {
	for _, container := range task.DesiredState.Manifest.Containers {
		for _, taskPort := range container.Ports {
			if taskPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

func (s *FirstFitScheduler) Schedule(task Pod) (string, error) {
	machineToTasks := map[string][]Pod{}
	tasks, err := s.registry.ListTasks(nil)
	if err != nil {
		return "", err
	}
	for _, scheduledTask := range tasks {
		host := scheduledTask.CurrentState.Host
		machineToTasks[host] = append(machineToTasks[host], scheduledTask)
	}
	for _, machine := range s.machines {
		taskFits := true
		for _, scheduledTask := range machineToTasks[machine] {
			for _, container := range task.DesiredState.Manifest.Containers {
				for _, port := range container.Ports {
					if s.containsPort(scheduledTask, port) {
						taskFits = false
					}
				}
			}
		}
		if taskFits {
			return machine, nil
		}
	}
	return "", fmt.Errorf("Failed to find fit for %#v", task)
}
