/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/scheduler/testing"
)

type fitPredicate func(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error)
type priorityFunc func(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*schedulerapi.ClusterPriorityList, error)

type priorityConfig struct {
	function priorityFunc
	weight   int
}

func errorPredicateExtender(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error) {
	return false, fmt.Errorf("Some error")
}

func falsePredicateExtender(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error) {
	return false, nil
}

func truePredicateExtender(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error) {
	return true, nil
}

func cluster1PredicateExtender(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error) {
	if cluster.Name == "cluster1" {
		return true, nil
	}
	return false, nil
}

func cluster2PredicateExtender(replicaSet *extensions.ReplicaSet, cluster *federation.Cluster) (bool, error) {
	if cluster.Name == "cluster2" {
		return true, nil
	}
	return false, nil
}

func errorPrioritizerExtender(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*schedulerapi.ClusterPriorityList, error) {
	return &schedulerapi.ClusterPriorityList{}, fmt.Errorf("Some error")
}

func cluster1PrioritizerExtender(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*schedulerapi.ClusterPriorityList, error) {
	result := schedulerapi.ClusterPriorityList{}
	for _, cluster := range clusters.Items {
		score := 1
		if cluster.Name == "cluster1" {
			score = 10
		}
		result = append(result, schedulerapi.ClusterPriority{cluster.Name, score})
	}
	return &result, nil
}

func cluster2PrioritizerExtender(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*schedulerapi.ClusterPriorityList, error) {
	result := schedulerapi.ClusterPriorityList{}
	for _, cluster := range clusters.Items {
		score := 1
		if cluster.Name == "cluster2" {
			score = 10
		}
		result = append(result, schedulerapi.ClusterPriority{cluster.Name, score})
	}
	return &result, nil
}

func cluster2Prioritizer(_ *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	clusters, err := clusterLister.List()
	if err != nil {
		return []schedulerapi.ClusterPriority{}, err
	}

	result := []schedulerapi.ClusterPriority{}
	for _, cluster := range clusters.Items {
		score := 1
		if cluster.Name == "cluster2" {
			score = 10
		}
		result = append(result, schedulerapi.ClusterPriority{cluster.Name, score})
	}
	return result, nil
}

type FakeExtender struct {
	predicates   []fitPredicate
	prioritizers []priorityConfig
	weight       int
}

func (f *FakeExtender) Filter(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*federation.ClusterList, error) {
	filtered := []federation.Cluster{}
	for _, cluster := range clusters.Items {
		fits := true
		for _, predicate := range f.predicates {
			fit, err := predicate(replicaSet, &cluster)
			if err != nil {
				return &federation.ClusterList{}, err
			}
			if !fit {
				fits = false
				break
			}
		}
		if fits {
			filtered = append(filtered, cluster)
		}
	}
	return &federation.ClusterList{Items: filtered}, nil
}

func (f *FakeExtender) Prioritize(replicaSet *extensions.ReplicaSet, clusters *federation.ClusterList) (*schedulerapi.ClusterPriorityList, int, error) {
	result := schedulerapi.ClusterPriorityList{}
	combinedScores := map[string]int{}
	for _, prioritizer := range f.prioritizers {
		weight := prioritizer.weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.function
		prioritizedList, err := priorityFunc(replicaSet, clusters)
		if err != nil {
			return &schedulerapi.ClusterPriorityList{}, 0, err
		}
		for _, clusterEntry := range *prioritizedList {
			combinedScores[clusterEntry.Cluster] += clusterEntry.Score * weight
		}
	}
	for cluster, score := range combinedScores {
		result = append(result, schedulerapi.ClusterPriority{Cluster: cluster, Score: score})
	}
	return &result, f.weight, nil
}

func TestGenericSchedulerWithExtenders(t *testing.T) {
	tests := []struct {
		name                 string
		predicates           map[string]algorithm.FitPredicate
		prioritizers         []algorithm.PriorityConfig
		extenders            []FakeExtender
		extenderPredicates   []fitPredicate
		extenderPrioritizers []priorityConfig
		clusters                []string
		replicaSet                  *extensions.ReplicaSet
		replicaSets                 []*extensions.ReplicaSet
		expectedHost         string
		expectsErr           bool
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{errorPredicateExtender},
				},
			},
			clusters:      []string{"cluster1", "cluster2"},
			expectsErr: true,
			name:       "test 1",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{falsePredicateExtender},
				},
			},
			clusters:      []string{"cluster1", "cluster2"},
			expectsErr: true,
			name:       "test 2",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{cluster1PredicateExtender},
				},
			},
			clusters:        []string{"cluster1", "cluster2"},
			expectedHost: "cluster1",
			name:         "test 3",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{cluster2PredicateExtender},
				},
				{
					predicates: []fitPredicate{cluster1PredicateExtender},
				},
			},
			clusters:      []string{"cluster1", "cluster2"},
			expectsErr: true,
			name:       "test 4",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{errorPrioritizerExtender, 10}},
					weight:       1,
				},
			},
			clusters:        []string{"cluster1"},
			expectedHost: "cluster1",
			name:         "test 5",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{EqualPriority, 1}},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{cluster1PrioritizerExtender, 10}},
					weight:       1,
				},
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{cluster2PrioritizerExtender, 10}},
					weight:       5,
				},
			},
			clusters:        []string{"cluster1", "cluster2"},
			expectedHost: "cluster2",
			name:         "test 6",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{cluster2Prioritizer, 20}},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{cluster1PrioritizerExtender, 10}},
					weight:       1,
				},
			},
			clusters:        []string{"cluster1", "cluster2"},
			expectedHost: "cluster2", // cluster2 has higher score
			name:         "test 7",
		},
	}

	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		extenders := []algorithm.SchedulerExtender{}
		for ii := range test.extenders {
			extenders = append(extenders, &test.extenders[ii])
		}
		scheduler := NewGenericScheduler(schedulertesting.ReplicaSetsToCache(test.replicaSets), test.predicates, test.prioritizers, extenders, random)
		cluster, err := scheduler.Schedule(test.replicaSet, algorithm.FakeClusterLister(makeClusterList(test.clusters)))
		if test.expectsErr {
			if err == nil {
				t.Errorf("Unexpected non-error for %s, cluster %s", test.name, cluster)
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if test.expectedHost != cluster {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedHost, cluster)
			}
		}
	}
}
