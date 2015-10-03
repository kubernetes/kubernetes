/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
)

type FailedPredicateMap map[string]sets.String

type FitError struct {
	Pod              *api.Pod
	FailedPredicates FailedPredicateMap
}

var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

// implementation of the error interface
func (f *FitError) Error() string {
	var reason string
	// We iterate over all nodes for logging purposes, even though we only return one reason from one node
	for node, predicateList := range f.FailedPredicates {
		glog.Infof("Failed to find fit for pod %v on node %s: %s", f.Pod.Name, node, strings.Join(predicateList.List(), ","))
		if len(reason) == 0 {
			reason, _ = predicateList.PopAny()
		}
	}
	return fmt.Sprintf("Failed for reason %s and possibly others", reason)
}

type genericScheduler struct {
	predicates   map[string]algorithm.FitPredicate
	prioritizers []algorithm.PriorityConfig
	pods         algorithm.PodLister
	random       *rand.Rand
	randomLock   sync.Mutex
}

func (g *genericScheduler) Schedule(pod *api.Pod, nodeLister algorithm.NodeLister) (string, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", ErrNoNodesAvailable
	}

	filteredNodes, failedPredicateMap, err := findNodesThatFit(pod, g.pods, g.predicates, nodes)
	if err != nil {
		return "", err
	}

	priorityList, err := PrioritizeNodes(pod, g.pods, g.prioritizers, algorithm.FakeNodeLister(filteredNodes))
	if err != nil {
		return "", err
	}
	if len(priorityList) == 0 {
		return "", &FitError{
			Pod:              pod,
			FailedPredicates: failedPredicateMap,
		}
	}

	return g.selectHost(priorityList)
}

// This method takes a prioritized list of nodes and sorts them in reverse order based on scores
// and then picks one randomly from the nodes that had the highest score
func (g *genericScheduler) selectHost(priorityList algorithm.HostPriorityList) (string, error) {
	if len(priorityList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}
	sort.Sort(sort.Reverse(priorityList))

	hosts := getBestHosts(priorityList)
	g.randomLock.Lock()
	defer g.randomLock.Unlock()

	ix := g.random.Int() % len(hosts)
	return hosts[ix], nil
}

// Filters the nodes to find the ones that fit based on the given predicate functions
// Each node is passed through the predicate functions to determine if it is a fit
func findNodesThatFit(pod *api.Pod, podLister algorithm.PodLister, predicateFuncs map[string]algorithm.FitPredicate, nodes api.NodeList) (api.NodeList, FailedPredicateMap, error) {
	filtered := []api.Node{}
	machineToPods, err := predicates.MapPodsToMachines(podLister)
	failedPredicateMap := FailedPredicateMap{}
	if err != nil {
		return api.NodeList{}, FailedPredicateMap{}, err
	}
	for _, node := range nodes.Items {
		fits := true
		for name, predicate := range predicateFuncs {
			predicates.FailedResourceType = ""
			fit, err := predicate(pod, machineToPods[node.Name], node.Name)
			if err != nil {
				return api.NodeList{}, FailedPredicateMap{}, err
			}
			if !fit {
				fits = false
				if _, found := failedPredicateMap[node.Name]; !found {
					failedPredicateMap[node.Name] = sets.String{}
				}
				if predicates.FailedResourceType != "" {
					failedPredicateMap[node.Name].Insert(predicates.FailedResourceType)
					break
				}
				failedPredicateMap[node.Name].Insert(name)
				break
			}
		}
		if fits {
			filtered = append(filtered, node)
		}
	}
	return api.NodeList{Items: filtered}, failedPredicateMap, nil
}

// Prioritizes the nodes by running the individual priority functions sequentially.
// Each priority function is expected to set a score of 0-10
// 0 is the lowest priority score (least preferred node) and 10 is the highest
// Each priority function can also have its own weight
// The node scores returned by the priority function are multiplied by the weights to get weighted scores
// All scores are finally combined (added) to get the total weighted scores of all nodes
func PrioritizeNodes(pod *api.Pod, podLister algorithm.PodLister, priorityConfigs []algorithm.PriorityConfig, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
	result := algorithm.HostPriorityList{}

	// If no priority configs are provided, then the EqualPriority function is applied
	// This is required to generate the priority list in the required format
	if len(priorityConfigs) == 0 {
		return EqualPriority(pod, podLister, nodeLister)
	}

	combinedScores := map[string]int{}
	for _, priorityConfig := range priorityConfigs {
		weight := priorityConfig.Weight
		// skip the priority function if the weight is specified as 0
		if weight == 0 {
			continue
		}
		priorityFunc := priorityConfig.Function
		prioritizedList, err := priorityFunc(pod, podLister, nodeLister)
		if err != nil {
			return algorithm.HostPriorityList{}, err
		}
		for _, hostEntry := range prioritizedList {
			combinedScores[hostEntry.Host] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		glog.V(10).Infof("Host %s Score %d", host, score)
		result = append(result, algorithm.HostPriority{Host: host, Score: score})
	}
	return result, nil
}

func getBestHosts(list algorithm.HostPriorityList) []string {
	result := []string{}
	for _, hostEntry := range list {
		if hostEntry.Score == list[0].Score {
			result = append(result, hostEntry.Host)
		} else {
			break
		}
	}
	return result
}

// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
func EqualPriority(_ *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		glog.Errorf("Failed to list nodes: %v", err)
		return []algorithm.HostPriority{}, err
	}

	result := []algorithm.HostPriority{}
	for _, node := range nodes.Items {
		result = append(result, algorithm.HostPriority{
			Host:  node.Name,
			Score: 1,
		})
	}
	return result, nil
}

func NewGenericScheduler(predicates map[string]algorithm.FitPredicate, prioritizers []algorithm.PriorityConfig, pods algorithm.PodLister, random *rand.Rand) algorithm.ScheduleAlgorithm {
	return &genericScheduler{
		predicates:   predicates,
		prioritizers: prioritizers,
		pods:         pods,
		random:       random,
	}
}
