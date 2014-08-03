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
	"math/rand"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// RandomFitScheduler is a Scheduler which schedules a Pod on a random machine which matches its requirement.
type RandomFitScheduler struct {
	podLister  PodLister
	random     *rand.Rand
	randomLock sync.Mutex
}

func NewRandomFitScheduler(podLister PodLister, random *rand.Rand) Scheduler {
	return &RandomFitScheduler{
		podLister: podLister,
		random:    random,
	}
}

func (s *RandomFitScheduler) containsPort(pod api.Pod, port api.Port) bool {
	for _, container := range pod.DesiredState.Manifest.Containers {
		for _, podPort := range container.Ports {
			if podPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

// Schedule schedules a pod on a random machine which matches its requirement.
func (s *RandomFitScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	machines, err := minionLister.List()
	if err != nil {
		return "", err
	}
	machineToPods := map[string][]api.Pod{}
	// TODO: perform more targeted query...
	pods, err := s.podLister.ListPods(labels.Everything())
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
					if port.HostPort == 0 {
						continue
					}
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
	}
	s.randomLock.Lock()
	defer s.randomLock.Unlock()
	return machineOptions[s.random.Int()%len(machineOptions)], nil
}
