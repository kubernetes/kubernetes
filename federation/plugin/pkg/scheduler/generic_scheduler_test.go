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
	"math"
	"math/rand"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/federation/apis/federation"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/scheduler/testing"
)

func falsePredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return false, nil
}

func truePredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return true, nil
}

func matchesPredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return replicaSet.Name == clusterName, nil
}

func hasNoReplicaSetsPredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return len(clusterInfo.ReplicaSets()) == 0, nil
}

func numericPriority(replicaSet *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	clusters, err := clusterLister.List()
	result := []schedulerapi.ClusterPriority{}

	if err != nil {
		return nil, fmt.Errorf("failed to list clusters: %v", err)
	}
	for _, cluster := range clusters.Items {
		score, err := strconv.Atoi(cluster.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, schedulerapi.ClusterPriority{
			Cluster:  cluster.Name,
			Score: score,
		})
	}
	return result, nil
}

func reverseNumericPriority(replicaSet *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []schedulerapi.ClusterPriority{}
	result, err := numericPriority(replicaSet, clusterNameToInfo, clusterLister)
	if err != nil {
		return nil, err
	}

	for _, hostPriority := range result {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for _, hostPriority := range result {
		reverseResult = append(reverseResult, schedulerapi.ClusterPriority{
			Cluster:  hostPriority.Cluster,
			Score: int(maxScore + minScore - float64(hostPriority.Score)),
		})
	}

	return reverseResult, nil
}

func makeClusterList(clusterNames []string) federation.ClusterList {
	result := federation.ClusterList{
		Items: make([]federation.Cluster, len(clusterNames)),
	}
	for ix := range clusterNames {
		result.Items[ix].Name = clusterNames[ix]
	}
	return result
}

func TestSelectCluster(t *testing.T) {
	scheduler := genericScheduler{random: rand.New(rand.NewSource(0))}
	tests := []struct {
		list          schedulerapi.ClusterPriorityList
		possibleClusters sets.String
		expectsErr    bool
	}{
		{
			list: []schedulerapi.ClusterPriority{
				{Cluster: "cluster1.1", Score: 1},
				{Cluster: "cluster2.1", Score: 2},
			},
			possibleClusters: sets.NewString("cluster2.1"),
			expectsErr:    false,
		},
		// equal scores
		{
			list: []schedulerapi.ClusterPriority{
				{Cluster: "cluster1.1", Score: 1},
				{Cluster: "cluster1.2", Score: 2},
				{Cluster: "cluster1.3", Score: 2},
				{Cluster: "cluster2.1", Score: 2},
			},
			possibleClusters: sets.NewString("cluster1.2", "cluster1.3", "cluster2.1"),
			expectsErr:    false,
		},
		// out of order scores
		{
			list: []schedulerapi.ClusterPriority{
				{Cluster: "cluster1.1", Score: 3},
				{Cluster: "cluster1.2", Score: 3},
				{Cluster: "cluster2.1", Score: 2},
				{Cluster: "cluster3.1", Score: 1},
				{Cluster: "cluster1.3", Score: 3},
			},
			possibleClusters: sets.NewString("cluster1.1", "cluster1.2", "cluster1.3"),
			expectsErr:    false,
		},
		// empty priorityList
		{
			list:          []schedulerapi.ClusterPriority{},
			possibleClusters: sets.NewString(),
			expectsErr:    true,
		},
	}

	for _, test := range tests {
		// increase the randomness
		for i := 0; i < 10; i++ {
			got, err := scheduler.selectClusters(test.list)
			if test.expectsErr {
				if err == nil {
					t.Error("Unexpected non-error")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if !test.possibleClusters.Has(got) {
					t.Errorf("got %s is not in the possible map %v", got, test.possibleClusters)
				}
			}
		}
	}
}

func TestGenericScheduler(t *testing.T) {
	tests := []struct {
		name          string
		predicates    map[string]algorithm.FitPredicate
		prioritizers  []algorithm.PriorityConfig
		clusters         []string
		replicaSet           *extensions.ReplicaSet
		replicaSets          []*extensions.ReplicaSet
		expectedClusters sets.String
		expectsErr    bool
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			clusters:        []string{"cluster1", "cluster2"},
			expectsErr:   true,
			name:         "test 1",
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			clusters:         []string{"cluster1", "cluster2"},
			expectedClusters: sets.NewString("cluster1", "cluster2"),
			name:          "test 2",
		},
		{
			// Fits on a cluster where the replicaSet ID matches the cluster name
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			clusters:         []string{"cluster1", "cluster2"},
			replicaSet:           &extensions.ReplicaSet{ObjectMeta: api.ObjectMeta{Name: "cluster2"}},
			expectedClusters: sets.NewString("cluster2"),
			name:          "test 3",
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			clusters:         []string{"3", "2", "1"},
			expectedClusters: sets.NewString("3"),
			name:          "test 4",
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			clusters:         []string{"3", "2", "1"},
			replicaSet:           &extensions.ReplicaSet{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedClusters: sets.NewString("2"),
			name:          "test 5",
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			clusters:         []string{"3", "2", "1"},
			replicaSet:           &extensions.ReplicaSet{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedClusters: sets.NewString("1"),
			name:          "test 6",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			clusters:        []string{"3", "2", "1"},
			expectsErr:   true,
			name:         "test 7",
		},
		{
			predicates: map[string]algorithm.FitPredicate{
				"noreplicaSets":  hasNoReplicaSetsPredicate,
				"matches": matchesPredicate,
			},
			replicaSets: []*extensions.ReplicaSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "2"},
					Spec: extensions.ReplicaSetSpec{
					},
				},
			},
			replicaSet: &extensions.ReplicaSet{ObjectMeta: api.ObjectMeta{Name: "2"}},

			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			clusters:        []string{"1", "2"},
			expectsErr:   true,
			name:         "test 8",
		},
	}

	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		scheduler := NewGenericScheduler(schedulertesting.ReplicaSetsToCache(test.replicaSets), test.predicates, test.prioritizers, []algorithm.SchedulerExtender{}, random)
		cluster, err := scheduler.Schedule(test.replicaSet, algorithm.FakeClusterLister(makeClusterList(test.clusters)))
		if test.expectsErr {
			if err == nil {
				t.Error("Unexpected non-error")
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !test.expectedClusters.Has(cluster) {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedClusters, cluster)
			}
		}
	}

}
