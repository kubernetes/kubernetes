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
	"math/rand"
	"testing"

	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/testing"
)

func falsePredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return false, nil
}

func truePredicate(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return true, nil
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
	}

	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		scheduler := NewGenericScheduler(schedulertesting.ReplicaSetsToCache(test.replicaSets), test.predicates, test.prioritizers, random)
		cluster, err := scheduler.Schedule(test.replicaSet, algorithm.FakeClusterLister(makeClusterList(test.clusters)))
		if test.expectsErr {
			if err == nil {
				t.Error("Unexpected non-error for %s, cluster %s", test.name, cluster)
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v, cluster %s", err, cluster)
			}
			if !test.expectedClusters.Has(cluster) {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedClusters, cluster)
			}
		}
	}

}
