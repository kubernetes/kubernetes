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
	predicates []FitPredicate
	random     *rand.Rand
	randomLock sync.Mutex
}

// NewRandomFitScheduler creates a random fit scheduler with the default set of fit predicates
func NewRandomFitScheduler(podLister PodLister, random *rand.Rand) Scheduler {
	return NewRandomFitSchedulerWithPredicates(podLister, random, []FitPredicate{podFitsPorts})
}

// NewRandomFitScheduler creates a random fit scheduler with the specified set of fit predicates.
// All predicates must be true for the pod to be considered a fit.
func NewRandomFitSchedulerWithPredicates(podLister PodLister, random *rand.Rand, predicates []FitPredicate) Scheduler {
	return &RandomFitScheduler{
		podLister:  podLister,
		random:     random,
		predicates: predicates,
	}
}

func podFitsPorts(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	for _, scheduledPod := range existingPods {
		for _, container := range pod.DesiredState.Manifest.Containers {
			for _, port := range container.Ports {
				if port.HostPort == 0 {
					continue
				}
				if containsPort(scheduledPod, port) {
					return false, nil
				}
			}
		}
	}
	return true, nil
}

func containsPort(pod api.Pod, port api.Port) bool {
	for _, container := range pod.DesiredState.Manifest.Containers {
		for _, podPort := range container.Ports {
			if podPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

// MapPodsToMachines obtains a list of pods and pivots that list into a map where the keys are host names
// and the values are the list of pods running on that host.
func MapPodsToMachines(lister PodLister) (map[string][]api.Pod, error) {
	machineToPods := map[string][]api.Pod{}
	// TODO: perform more targeted query...
	pods, err := lister.ListPods(labels.Everything())
	if err != nil {
		return map[string][]api.Pod{}, err
	}
	for _, scheduledPod := range pods {
		host := scheduledPod.CurrentState.Host
		machineToPods[host] = append(machineToPods[host], scheduledPod)
	}
	return machineToPods, nil
}

// Schedule schedules a pod on a random machine which matches its requirement.
func (s *RandomFitScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	machines, err := minionLister.List()
	if err != nil {
		return "", err
	}
	machineToPods, err := MapPodsToMachines(s.podLister)
	if err != nil {
		return "", err
	}
	var machineOptions []string
	for _, machine := range machines {
		podFits := true
		for _, predicate := range s.predicates {
			fits, err := predicate(pod, machineToPods[machine], machine)
			if err != nil {
				return "", err
			}
			if !fits {
				podFits = false
				break
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
