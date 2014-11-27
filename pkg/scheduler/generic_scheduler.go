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

func (g *genericScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	minions, err := minionLister.List()
	if err != nil {
		return "", err
	}
	filteredNodes, err := findNodesThatFit(pod, g.pods, g.predicates, minions)
	if err != nil {
		return "", err
	}
	priorityList, err := g.prioritizer(pod, g.pods, FakeMinionLister(filteredNodes))
	if err != nil {
		return "", err
	}
	if len(priorityList) == 0 {
		return "", fmt.Errorf("failed to find a fit for pod: %v", pod)
	}
	return g.selectHost(priorityList)
}

func (g *genericScheduler) selectHost(priorityList HostPriorityList) (string, error) {
	if len(priorityList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}
	sort.Sort(priorityList)

	hosts := getMinHosts(priorityList)
	g.randomLock.Lock()
	defer g.randomLock.Unlock()

	ix := g.random.Int() % len(hosts)
	return hosts[ix], nil
}

func findNodesThatFit(pod api.Pod, podLister PodLister, predicates []FitPredicate, nodes api.MinionList) (api.MinionList, error) {
	filtered := []api.Minion{}
	machineToPods, err := MapPodsToMachines(podLister)
	if err != nil {
		return api.MinionList{}, err
	}
	for _, node := range nodes.Items {
		fits := true
		for _, predicate := range predicates {
			fit, err := predicate(pod, machineToPods[node.Name], node.Name)
			if err != nil {
				return api.MinionList{}, err
			}
			if !fit {
				fits = false
				break
			}
		}
		if fits {
			filtered = append(filtered, node)
		}
	}
	return api.MinionList{Items: filtered}, nil
}

func getMinHosts(list HostPriorityList) []string {
	result := []string{}
	for _, hostEntry := range list {
		if hostEntry.score == list[0].score {
			result = append(result, hostEntry.host)
		} else {
			break
		}
	}
	return result
}

// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
func EqualPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	nodes, err := minionLister.List()
	if err != nil {
		fmt.Errorf("failed to list nodes: %v", err)
		return []HostPriority{}, err
	}

	result := []HostPriority{}
	for _, minion := range nodes.Items {
		result = append(result, HostPriority{
			host:  minion.Name,
			score: 1,
		})
	}
	return result, nil
}

func NewGenericScheduler(predicates []FitPredicate, prioritizer PriorityFunction, pods PodLister, random *rand.Rand) Scheduler {
	return &genericScheduler{
		predicates:  predicates,
		prioritizer: prioritizer,
		pods:        pods,
		random:      random,
	}
}
