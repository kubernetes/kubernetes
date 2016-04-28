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
	"bytes"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type FailedPredicateMap map[string]string

type FitError struct {
	Pod              *api.Pod
	FailedPredicates FailedPredicateMap
}

var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

// Error returns detailed information of why the pod failed to fit on each node
func (f *FitError) Error() string {
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("pod (%s) failed to fit in any node\n", f.Pod.Name))
	for node, predicate := range f.FailedPredicates {
		reason := fmt.Sprintf("fit failure on node (%s): %s\n", node, predicate)
		buf.WriteString(reason)
	}
	return buf.String()
}

type genericScheduler struct {
	cache         schedulercache.Cache
	predicates    map[string]algorithm.FitPredicate
	prioritizers  []algorithm.PriorityConfig
	extenders     []algorithm.SchedulerExtender
	pods          algorithm.PodLister
	random        *rand.Rand
	randomLock    sync.Mutex
	lastNodeIndex uint64
}

// Schedule tries to schedule the given pod to one of node in the node list.
// If it succeeds, it will return the name of the node.
// If it fails, it will return a Fiterror error with reasons.
func (g *genericScheduler) Schedule(pod *api.Pod, nodeLister algorithm.NodeLister) (string, error) {
	var trace *util.Trace
	if pod != nil {
		trace = util.NewTrace(fmt.Sprintf("Scheduling %s/%s", pod.Namespace, pod.Name))
	} else {
		trace = util.NewTrace("Scheduling <nil> pod")
	}
	defer trace.LogIfLong(20 * time.Millisecond)

	nodes, err := nodeLister.List()
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", ErrNoNodesAvailable
	}

	// Used for all fit and priority funcs.
	nodeNameToInfo, err := g.cache.GetNodeNameToInfoMap()
	if err != nil {
		return "", err
	}

	trace.Step("Computing predicates")
	filteredNodes, failedPredicateMap, err := findNodesThatFit(pod, nodeNameToInfo, g.predicates, nodes, g.extenders)
	if err != nil {
		return "", err
	}

	if len(filteredNodes.Items) == 0 {
		return "", &FitError{
			Pod:              pod,
			FailedPredicates: failedPredicateMap,
		}
	}

	trace.Step("Prioritizing")
	priorityList, err := PrioritizeNodes(pod, nodeNameToInfo, g.prioritizers, algorithm.FakeNodeLister(filteredNodes), g.extenders)
	if err != nil {
		return "", err
	}

	trace.Step("Selecting host")
	return g.selectHost(priorityList)
}

// selectHost takes a prioritized list of nodes and then picks one
// randomly from the nodes that had the highest score.
func (g *genericScheduler) selectHost(priorityList schedulerapi.HostPriorityList) (string, error) {
	if len(priorityList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}

	sort.Sort(sort.Reverse(priorityList))
	maxScore := priorityList[0].Score
	firstAfterMaxScore := sort.Search(len(priorityList), func(i int) bool { return priorityList[i].Score < maxScore })

	g.randomLock.Lock()
	ix := int(g.lastNodeIndex % uint64(firstAfterMaxScore))
	g.lastNodeIndex++
	g.randomLock.Unlock()

	return priorityList[ix].Host, nil
}

// Filters the nodes to find the ones that fit based on the given predicate functions
// Each node is passed through the predicate functions to determine if it is a fit
func findNodesThatFit(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, predicateFuncs map[string]algorithm.FitPredicate, nodes api.NodeList, extenders []algorithm.SchedulerExtender) (api.NodeList, FailedPredicateMap, error) {
	predicateResultLock := sync.Mutex{}
	filtered := []api.Node{}
	failedPredicateMap := FailedPredicateMap{}
	errs := []error{}

	checkNode := func(i int) {
		nodeName := nodes.Items[i].Name
		fits, failedPredicate, err := podFitsOnNode(pod, nodeNameToInfo[nodeName], predicateFuncs)

		predicateResultLock.Lock()
		defer predicateResultLock.Unlock()
		if err != nil {
			errs = append(errs, err)
			return
		}
		if fits {
			filtered = append(filtered, nodes.Items[i])
		} else {
			failedPredicateMap[nodeName] = failedPredicate
		}
	}
	workqueue.Parallelize(16, len(nodes.Items), checkNode)
	if len(errs) > 0 {
		return api.NodeList{}, FailedPredicateMap{}, errors.NewAggregate(errs)
	}

	if len(filtered) > 0 && len(extenders) != 0 {
		for _, extender := range extenders {
			filteredList, err := extender.Filter(pod, &api.NodeList{Items: filtered})
			if err != nil {
				return api.NodeList{}, FailedPredicateMap{}, err
			}
			filtered = filteredList.Items
			if len(filtered) == 0 {
				break
			}
		}
	}
	return api.NodeList{Items: filtered}, failedPredicateMap, nil
}

// Checks whether node with a given name and NodeInfo satisfies all predicateFuncs.
func podFitsOnNode(pod *api.Pod, info *schedulercache.NodeInfo, predicateFuncs map[string]algorithm.FitPredicate) (bool, string, error) {
	for _, predicate := range predicateFuncs {
		fit, err := predicate(pod, info)
		if err != nil {
			switch e := err.(type) {
			case *predicates.InsufficientResourceError:
				if fit {
					err := fmt.Errorf("got InsufficientResourceError: %v, but also fit='true' which is unexpected", e)
					return false, "", err
				}
			case *predicates.PredicateFailureError:
				if fit {
					err := fmt.Errorf("got PredicateFailureError: %v, but also fit='true' which is unexpected", e)
					return false, "", err
				}
			default:
				return false, "", err
			}
		}
		if !fit {
			if re, ok := err.(*predicates.InsufficientResourceError); ok {
				return false, fmt.Sprintf("Insufficient %s", re.ResourceName), nil
			}
			if re, ok := err.(*predicates.PredicateFailureError); ok {
				return false, re.PredicateName, nil
			} else {
				err := fmt.Errorf("SchedulerPredicates failed due to %v, which is unexpected.", err)
				return false, "", err
			}
		}
	}
	return true, "", nil
}

// Prioritizes the nodes by running the individual priority functions in parallel.
// Each priority function is expected to set a score of 0-10
// 0 is the lowest priority score (least preferred node) and 10 is the highest
// Each priority function can also have its own weight
// The node scores returned by the priority function are multiplied by the weights to get weighted scores
// All scores are finally combined (added) to get the total weighted scores of all nodes
func PrioritizeNodes(
	pod *api.Pod,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
	priorityConfigs []algorithm.PriorityConfig,
	nodeLister algorithm.NodeLister,
	extenders []algorithm.SchedulerExtender,
) (schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}

	// If no priority configs are provided, then the EqualPriority function is applied
	// This is required to generate the priority list in the required format
	if len(priorityConfigs) == 0 && len(extenders) == 0 {
		return EqualPriority(pod, nodeNameToInfo, nodeLister)
	}

	var (
		mu             = sync.Mutex{}
		wg             = sync.WaitGroup{}
		combinedScores = map[string]int{}
		errs           []error
	)

	for _, priorityConfig := range priorityConfigs {
		// skip the priority function if the weight is specified as 0
		if priorityConfig.Weight == 0 {
			continue
		}

		wg.Add(1)
		go func(config algorithm.PriorityConfig) {
			defer wg.Done()
			weight := config.Weight
			priorityFunc := config.Function
			prioritizedList, err := priorityFunc(pod, nodeNameToInfo, nodeLister)

			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				return
			}
			for i := range prioritizedList {
				host, score := prioritizedList[i].Host, prioritizedList[i].Score
				combinedScores[host] += score * weight
			}
		}(priorityConfig)
	}
	if len(errs) != 0 {
		return schedulerapi.HostPriorityList{}, errors.NewAggregate(errs)
	}

	// wait for all go routines to finish
	wg.Wait()

	if len(extenders) != 0 && nodeLister != nil {
		nodes, err := nodeLister.List()
		if err != nil {
			return schedulerapi.HostPriorityList{}, err
		}
		for _, extender := range extenders {
			wg.Add(1)
			go func(ext algorithm.SchedulerExtender) {
				defer wg.Done()
				prioritizedList, weight, err := ext.Prioritize(pod, &nodes)
				if err != nil {
					// Prioritization errors from extender can be ignored, let k8s/other extenders determine the priorities
					return
				}
				mu.Lock()
				for i := range *prioritizedList {
					host, score := (*prioritizedList)[i].Host, (*prioritizedList)[i].Score
					combinedScores[host] += score * weight
				}
				mu.Unlock()
			}(extender)
		}
	}
	// wait for all go routines to finish
	wg.Wait()

	for host, score := range combinedScores {
		glog.V(10).Infof("Host %s Score %d", host, score)
		result = append(result, schedulerapi.HostPriority{Host: host, Score: score})
	}
	return result, nil
}

// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
func EqualPriority(_ *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		glog.Errorf("Failed to list nodes: %v", err)
		return []schedulerapi.HostPriority{}, err
	}

	result := []schedulerapi.HostPriority{}
	for _, node := range nodes.Items {
		result = append(result, schedulerapi.HostPriority{
			Host:  node.Name,
			Score: 1,
		})
	}
	return result, nil
}

func NewGenericScheduler(cache schedulercache.Cache, predicates map[string]algorithm.FitPredicate, prioritizers []algorithm.PriorityConfig, extenders []algorithm.SchedulerExtender, random *rand.Rand) algorithm.ScheduleAlgorithm {
	return &genericScheduler{
		cache:        cache,
		predicates:   predicates,
		prioritizers: prioritizers,
		extenders:    extenders,
		random:       random,
	}
}
