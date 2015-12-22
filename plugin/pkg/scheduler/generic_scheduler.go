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
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

type FailedPredicateMap map[string]sets.String

type FitResult struct {
	Pod              *api.Pod
	Err              chan error
	FailedPredicates FailedPredicateMap
}

type PriorityResult struct {
	Err             chan error
	PrioritizeNodes schedulerapi.HostPriorityList
}

type ScheduleCache struct {
	//channel size
	Size int
	Sub  []chan *api.Node

	FResult FitResult
	PResult PriorityResult
}

func NewScheduleCache(nodeLen, priLen, extenderLen int) *ScheduleCache {
	sCache := &ScheduleCache{
		Size: priLen,
		FResult: FitResult{
			Err:              make(chan error),
			FailedPredicates: FailedPredicateMap{},
		},
		PResult: PriorityResult{
			Err:             make(chan error),
			PrioritizeNodes: schedulerapi.HostPriorityList{},
		},
	}

	sCache.FResult.FailedPredicates = FailedPredicateMap{}
	sCache.PResult.PrioritizeNodes = schedulerapi.HostPriorityList{}

	if extenderLen != 0 {
		sCache.Size += 1
	}

	if sCache.Size == 0 {
		sCache.Size = 1
	}

	for i := 0; i < sCache.Size; i++ {
		sCache.Sub = append(sCache.Sub, make(chan *api.Node, nodeLen))
	}

	return sCache
}

func (c *ScheduleCache) Insert(a *api.Node) {
	for _, s := range c.Sub {
		s <- a
	}
}

func (c *ScheduleCache) Close() {
	for _, s := range c.Sub {
		close(s)
	}
}

var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

// implementation of the error interface
func (f *FitResult) Error() string {
	var reason string
	// We iterate over all nodes for logging purposes, even though we only return one reason from one node
	for node, predicateList := range f.FailedPredicates {
		glog.V(2).Infof("Failed to find fit for pod %v on node %s: %s", f.Pod.Name, node, strings.Join(predicateList.List(), ","))
		if len(reason) == 0 {
			reason, _ = predicateList.PopAny()
		}
	}
	return fmt.Sprintf("Failed for reason %s and possibly others", reason)
}

type genericScheduler struct {
	predicates   map[string]algorithm.FitPredicate
	prioritizers []algorithm.PriorityConfig
	extenders    []algorithm.SchedulerExtender
	pods         algorithm.PodLister
	random       *rand.Rand
	randomLock   sync.Mutex
}

// Schedule tries to schedule the given pod to one of node in the node list.
// If it succeeds, it will return the name of the node.
// If it fails, it will return a Fiterror error with reasons.
func (g *genericScheduler) Schedule(pod *api.Pod, nodeLister algorithm.NodeLister) (string, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", ErrNoNodesAvailable
	}

	// TODO: we should compute this once and dynamically update it using Watch, not constantly re-compute.
	// But at least we're now only doing it in one place
	machinesToPods, err := predicates.MapPodsToMachines(g.pods)
	if err != nil {
		return "", err
	}

	c := NewScheduleCache(len(nodes.Items), len(g.prioritizers), len(g.extenders))

	go findNodesThatFit(pod, machinesToPods, g.predicates, nodes, g.extenders, c)

	go PrioritizeNodes(pod, machinesToPods, g.pods, g.prioritizers, g.extenders, c)

	select {
	case fErr := <-c.FResult.Err:
		if fErr != nil {
			return "", fErr
		}
	case pErr := <-c.PResult.Err:
		if pErr != nil {
			return "", pErr
		}
	}

	if len(c.PResult.PrioritizeNodes) == 0 {
		return "", &FitResult{
			Pod:              pod,
			FailedPredicates: c.FResult.FailedPredicates,
		}
	}

	return g.selectHost(c.PResult.PrioritizeNodes)
}

// This method takes a prioritized list of nodes and sorts them in reverse order based on scores
// and then picks one randomly from the nodes that had the highest score
func (g *genericScheduler) selectHost(priorityList schedulerapi.HostPriorityList) (string, error) {
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

func testNodeFit(pod *api.Pod, node *api.Node, machineToPods map[string][]*api.Pod, predicateFuncs map[string]algorithm.FitPredicate) (bool, sets.String, error) {
	for name, predicate := range predicateFuncs {
		predicates.FailedResourceType = ""
		fit, err := predicate(pod, machineToPods[node.Name], node.Name)
		if err != nil {
			return false, nil, err
		}
		if !fit {
			failedPredicates := sets.String{}
			if predicates.FailedResourceType != "" {
				failedPredicates.Insert(predicates.FailedResourceType)
				return false, failedPredicates, nil
			}
			failedPredicates.Insert(name)
			return false, failedPredicates, nil
		}
	}
	return true, nil, nil
}

// Filters the nodes to find the ones that fit based on the given predicate functions
// Each node is passed through the predicate functions to determine if it is a fit
func findNodesThatFit(pod *api.Pod, machineToPods map[string][]*api.Pod, predicateFuncs map[string]algorithm.FitPredicate, nodes api.NodeList, extenders []algorithm.SchedulerExtender, c *ScheduleCache) {
	if len(extenders) != 0 {
		for _, extender := range extenders {
			filteredList, err := extender.Filter(pod, &nodes)
			if err != nil {
				c.FResult.Err <- err
				return
			}
			if len(filteredList.Items) == 0 {
				c.Close()
				return
			}
			nodes = *filteredList
		}
	}

	wg := sync.WaitGroup{}
	wg.Add(len(nodes.Items))
	for ix := range nodes.Items {
		go func(node *api.Node) {
			defer wg.Done()
			fits, failedPredicates, err := testNodeFit(pod, node, machineToPods, predicateFuncs)
			if err != nil {
				c.FResult.Err <- err
				return
			}

			if fits {
				c.Insert(node)
			} else {
				c.FResult.FailedPredicates[node.Name] = failedPredicates
			}
		}(&nodes.Items[ix])
	}
	wg.Wait()
	c.Close()
}

// Prioritizes the nodes by running the individual priority functions in parallel.
// Each priority function is expected to set a score of 0-10
// 0 is the lowest priority score (least preferred node) and 10 is the highest
// Each priority function can also have its own weight
// The node scores returned by the priority function are multiplied by the weights to get weighted scores
// All scores are finally combined (added) to get the total weighted scores of all nodes
func PrioritizeNodes(pod *api.Pod, machinesToPods map[string][]*api.Pod, podLister algorithm.PodLister, priorityConfigs []algorithm.PriorityConfig, extenders []algorithm.SchedulerExtender, c *ScheduleCache) {

	// If no priority configs are provided, then the EqualPriority function is applied
	// This is required to generate the priority list in the required format
	if len(priorityConfigs) == 0 && len(extenders) == 0 {
		result, err := EqualPriority(pod, machinesToPods, podLister, c.Sub[0])
		c.PResult.PrioritizeNodes = result
		c.PResult.Err <- err
		return
	}

	combinedScores := map[string]int{}

	wg := sync.WaitGroup{}
	lock := sync.Mutex{}
	wg.Add(len(priorityConfigs) + len(extenders))
	if len(extenders) != 0 {
		go func() {
			var nodes api.NodeList
			for node := range c.Sub[len(priorityConfigs)] {
				nodes.Items = append(nodes.Items, *node)
			}

			for ix := range extenders {
				go func(extender algorithm.SchedulerExtender) {
					defer wg.Done()
					prioritizedList, weight, err := extender.Prioritize(pod, &nodes)
					if err != nil {
						// Prioritization errors from extender can be ignored, let k8s/other extenders determine the priorities
						return
					}

					lock.Lock()
					defer lock.Unlock()
					for _, hostEntry := range *prioritizedList {
						combinedScores[hostEntry.Host] += hostEntry.Score * weight
					}
				}(extenders[ix])
			}
		}()
	}

	for ix := range priorityConfigs {
		go func(ix int) {
			defer wg.Done()
			weight := priorityConfigs[ix].Weight
			// skip the priority function if the weight is specified as 0
			if weight == 0 {
				return
			}
			priorityFunc := priorityConfigs[ix].Function
			prioritizedList, err := priorityFunc(pod, machinesToPods, podLister, c.Sub[ix])
			if err != nil {
				c.PResult.Err <- err
				return
			}

			lock.Lock()
			defer lock.Unlock()
			for _, hostEntry := range prioritizedList {
				combinedScores[hostEntry.Host] += hostEntry.Score * weight
			}
		}(ix)
	}
	wg.Wait()

	for host, score := range combinedScores {
		glog.V(10).Infof("Host %s Score %d", host, score)
		c.PResult.PrioritizeNodes = append(c.PResult.PrioritizeNodes, schedulerapi.HostPriority{Host: host, Score: score})
	}

	c.PResult.Err <- nil
}

func getBestHosts(list schedulerapi.HostPriorityList) []string {
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
func EqualPriority(_ *api.Pod, machinesToPods map[string][]*api.Pod, podLister algorithm.PodLister, nodes chan *api.Node) (schedulerapi.HostPriorityList, error) {
	result := []schedulerapi.HostPriority{}

	for node := range nodes {
		result = append(result, schedulerapi.HostPriority{
			Host:  node.Name,
			Score: 1,
		})
	}
	return result, nil
}

func NewGenericScheduler(predicates map[string]algorithm.FitPredicate, prioritizers []algorithm.PriorityConfig, extenders []algorithm.SchedulerExtender, pods algorithm.PodLister, random *rand.Rand) algorithm.ScheduleAlgorithm {
	return &genericScheduler{
		predicates:   predicates,
		prioritizers: prioritizers,
		extenders:    extenders,
		pods:         pods,
		random:       random,
	}
}
