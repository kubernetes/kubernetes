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
	"sort"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type genericScheduler struct {
	predicates  []FitPredicate
	prioritizer PriorityFunction
	pods        PodLister
	random      *rand.Rand
	randomLock  sync.Mutex
}

type listMinionLister struct {
	nodes []string
}

func (l *listMinionLister) List() ([]string, error) {
	return l.nodes, nil
}

func (g *genericScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	minions, err := minionLister.List()
	if err != nil {
		return "", err
	}
	filtered := []string{}
	machineToPods, err := MapPodsToMachines(g.pods)
	if err != nil {
		return "", err
	}
	for _, minion := range minions {
		fits := true
		for _, predicate := range g.predicates {
			fit, err := predicate(pod, machineToPods[minion], minion)
			if err != nil {
				return "", err
			}
			if !fit {
				fits = false
				break
			}
		}
		if fits {
			filtered = append(filtered, minion)
		}
	}
	priorityList, err := g.prioritizer(pod, g.pods, &listMinionLister{filtered})
	if err != nil {
		return "", err
	}
	if len(priorityList) == 0 {
		return "", fmt.Errorf("failed to find a fit for pod: %v", pod)
	}
	sort.Sort(priorityList)

	hosts := getMinHosts(priorityList)
	g.randomLock.Lock()
	defer g.randomLock.Unlock()

	ix := g.random.Int() % len(hosts)
	return hosts[ix], nil
}

func getMinHosts(list HostPriorityList) []string {
	result := []string{}
	for _, hostEntry := range list {
		if hostEntry.score == list[0].score {
			result = append(result, hostEntry.host)
		}
	}
	return result
}

func NewGenericScheduler(predicates []FitPredicate, prioritizer PriorityFunction, pods PodLister, random *rand.Rand) Scheduler {
	return &genericScheduler{
		predicates:  predicates,
		prioritizer: prioritizer,
		pods:        pods,
		random:      random,
	}
}
