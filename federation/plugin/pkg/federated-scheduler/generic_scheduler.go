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
	"strings"
	"sync"

	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

type FailedPredicateMap map[string]sets.String

type FitError struct {
	Rc               *extensions.ReplicaSet
	FailedPredicates FailedPredicateMap
}

var ErrNoClustersAvailable = fmt.Errorf("no clusters available to schedule rcs")

// Error returns detailed information of why the replicaSet failed to fit on each cluster
func (f *FitError) Error() string {
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("replicaSet (%s) failed to fit in any cluster\n", f.Rc.Name))
	for cluster, predicateList := range f.FailedPredicates {
		reason := fmt.Sprintf("fit failure on cluster (%s): %s\n", cluster, strings.Join(predicateList.List(), ","))
		buf.WriteString(reason)
	}
	return buf.String()
}

type genericScheduler struct {
	cache            schedulercache.Cache
	predicates       map[string]algorithm.FitPredicate
	prioritizers     []algorithm.PriorityConfig
	random           *rand.Rand
	randomLock       sync.Mutex
	lastClusterIndex uint64
}

// Schedule tries to schedule the given replicaSet to one of cluster in the cluster list.
// If it succeeds, it will return the name of the cluster.
// If it fails, it will return a Fiterror error with reasons.
func (g *genericScheduler) Schedule(rc *extensions.ReplicaSet, clusterLister algorithm.ClusterLister) (string, error) {
	clusters, err := clusterLister.List()
	if err != nil {
		return "", err
	}
	if len(clusters.Items) == 0 {
		return "", ErrNoClustersAvailable
	}

	// Used for all fit and priority funcs.
	clusterNameToInfo, err := g.cache.GetClusterNameToInfoMap()
	if err != nil {
		return "", err
	}

	filteredClusters, failedPredicateMap, err := findClustersThatFit(rc, clusterNameToInfo, g.predicates, clusters)
	if err != nil {
		return "", err
	}

	if len(filteredClusters.Items) == 0 {
		return "", &FitError{
			Rc:               rc,
			FailedPredicates: failedPredicateMap,
		}
	}

	priorityList, err := PrioritizeClusters(rc, clusterNameToInfo, g.prioritizers, algorithm.FakeClusterLister(filteredClusters))
	if err != nil {
		return "", err
	}

	return g.selectClusters(priorityList)
}

// selectCluster takes a prioritized list of clusters and then picks one
// randomly from the clusters that had the highest score.
// in phase I, only one cluster will be picked and in later phase, the replicaset will be sp
func (g *genericScheduler) selectClusters(priorityList schedulerapi.ClusterPriorityList) (string, error) {
	if len(priorityList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}

	sort.Sort(sort.Reverse(priorityList))
	maxScore := priorityList[0].Score
	firstAfterMaxScore := sort.Search(len(priorityList), func(i int) bool { return priorityList[i].Score < maxScore })

	g.randomLock.Lock()
	ix := int(g.lastClusterIndex % uint64(firstAfterMaxScore))
	g.lastClusterIndex++
	g.randomLock.Unlock()

	return priorityList[ix].Cluster, nil
}

// Filters the clusters to find the ones that fit based on the given predicate functions
// Each clusters is passed through the predicate functions to determine if it is a fit
func findClustersThatFit(rc *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, predicateFuncs map[string]algorithm.FitPredicate, clusters federation.ClusterList) (federation.ClusterList, FailedPredicateMap, error) {
	filtered := []federation.Cluster{}
	failedPredicateMap := FailedPredicateMap{}

	for _, cluster := range clusters.Items {
		fits := true
		for name, predicate := range predicateFuncs {
			fit, err := predicate(rc, cluster.Name, clusterNameToInfo[cluster.Name])
			if err != nil {
				return federation.ClusterList{}, FailedPredicateMap{}, err
			}
			if !fit {
				fits = false
				if _, found := failedPredicateMap[cluster.Name]; !found {
					failedPredicateMap[cluster.Name] = sets.String{}
				}
				failedPredicateMap[cluster.Name].Insert(name)
				break
			}
		}
		if fits {
			filtered = append(filtered, cluster)
		}
	}

	return federation.ClusterList{Items: filtered}, failedPredicateMap, nil
}

// Prioritizes the clusters by running the individual priority functions in parallel.
// Each priority function is expected to set a score of 0-10
// 0 is the lowest priority score (least preferred clusters) and 10 is the highest
// Each priority function can also have its own weight
// The clusters scores returned by the priority function are multiplied by the weights to get weighted scores
// All scores are finally combined (added) to get the total weighted scores of all clusters
func PrioritizeClusters(
	rc *extensions.ReplicaSet,
	clusterNameToInfo map[string]*schedulercache.ClusterInfo,
	priorityConfigs []algorithm.PriorityConfig,
	clusterLister algorithm.ClusterLister,
) (schedulerapi.ClusterPriorityList, error) {
	result := schedulerapi.ClusterPriorityList{}

	// If no priority configs are provided, then the EqualPriority function is applied
	// This is required to generate the priority list in the required format
	if len(priorityConfigs) == 0 {
		return EqualPriority(rc, clusterNameToInfo, clusterLister)
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
			prioritizedList, err := priorityFunc(rc, clusterNameToInfo, clusterLister)
			if err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
				return
			}
			mu.Lock()
			for i := range prioritizedList {
				cluster, score := prioritizedList[i].Cluster, prioritizedList[i].Score
				combinedScores[cluster] += score * weight
			}
			mu.Unlock()
		}(priorityConfig)
	}
	if len(errs) != 0 {
		return schedulerapi.ClusterPriorityList{}, errors.NewAggregate(errs)
	}
	// wait for all go routines to finish
	wg.Wait()

	for cluster, score := range combinedScores {
		glog.V(10).Infof("Cluster %s Score %d", cluster, score)
		result = append(result, schedulerapi.ClusterPriority{Cluster: cluster, Score: score})
	}
	return result, nil
}

// EqualPriority is a prioritizer function that gives an equal weight of one to all clusters
func EqualPriority(_ *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	clusters, err := clusterLister.List()
	if err != nil {
		glog.Errorf("Failed to list clusters: %v", err)
		return []schedulerapi.ClusterPriority{}, err
	}

	result := []schedulerapi.ClusterPriority{}
	for _, cluster := range clusters.Items {
		result = append(result, schedulerapi.ClusterPriority{
			Cluster: cluster.Name,
			Score:   1,
		})
	}
	return result, nil
}

func NewGenericScheduler(cache schedulercache.Cache, predicates map[string]algorithm.FitPredicate, prioritizers []algorithm.PriorityConfig, random *rand.Rand) algorithm.ScheduleAlgorithm {
	return &genericScheduler{
		cache:        cache,
		predicates:   predicates,
		prioritizers: prioritizers,
		random:       random,
	}
}
