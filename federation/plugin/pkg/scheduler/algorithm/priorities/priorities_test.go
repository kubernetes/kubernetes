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
//TODO: to be changed later
package priorities

import (
	"reflect"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func makeCluster(Cluster string, milliCPU, memory int64) federation.Cluster {
	return federation.Cluster{
		ObjectMeta: api.ObjectMeta{Name: Cluster},
		Status: federation.ClusterStatus{
			Capacity: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
			Allocatable: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

func TestZeroRequest(t *testing.T) {
	// A replicaSet with no resources. We expect spreading to count it as having the default resources.
	noResources := extensions.ReplicaSet{
		Spec: extensions.ReplicaSetSpec{
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{},
					},
				},
			},
		},
	}
	// A replicaSet with the same resources as a 0-request replicaSet gets by default as its resources (for spreading).
	small := extensions.ReplicaSet{
		Spec: extensions.ReplicaSetSpec{
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									"cpu": resource.MustParse(
										strconv.FormatInt(priorityutil.DefaultMilliCpuRequest, 10) + "m"),
									"memory": resource.MustParse(
										strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
								},
							},
						},
					},
				},
			},
		},
	}
	// A larger replicaSet.
	large := extensions.ReplicaSet{
		Spec: extensions.ReplicaSetSpec{
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									"cpu": resource.MustParse(
										strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*3, 10) + "m"),
									"memory": resource.MustParse(
										strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
								},
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		replicaSet   *extensions.ReplicaSet
		replicaSets  []*extensions.ReplicaSet
		Clusters []federation.Cluster
		test  string
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request replicaSet
		// as for a replicaSet with the defaults requests, both when the zero-request replicaSet is already on the cluster
		// and when the zero-request replicaSet is the one being scheduled.
		{
			replicaSet:   &extensions.ReplicaSet{Spec: noResources},
			Clusters: []federation.Cluster{makeCluster("cluster1", 1000, priorityutil.DefaultMemoryRequest*10), makeCluster("cluster2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of zero-request replicaSet with cluster with zero-request replicaSet",
			replicaSets: []*extensions.ReplicaSet{
				{Spec: large}, {Spec: noResources},
			},
		},
		{
			replicaSet:   &extensions.ReplicaSet{Spec: small},
			Clusters: []federation.Cluster{makeCluster("cluster1", 1000, priorityutil.DefaultMemoryRequest*10), makeCluster("cluster2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of nonzero-request replicaSet with cluster with zero-request replicaSet",
			replicaSets: []*extensions.ReplicaSet{
				{Spec: small}, {Spec: noResources},
			},
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			replicaSet:   &extensions.ReplicaSet{Spec: large},
			Clusters: []federation.Cluster{makeCluster("cluster1", 1000, priorityutil.DefaultMemoryRequest*10), makeCluster("cluster2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of larger replicaSet with cluster with zero-request replicaSet",
			replicaSets: []*extensions.ReplicaSet{
				{Spec: large}, {Spec: large},

			},
		},
	}

	const expectedPriority int = 25
	for _, test := range tests {
		ClusterNameToInfo := schedulercache.CreateClusterNameToInfoMap(test.replicaSets)
		list, err := scheduler.PrioritizeClusters(
			test.replicaSet,
			ClusterNameToInfo,
			// This should match the configuration in defaultPriorities() in
			// plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go if you want
			// to test what's actually in production.
			[]algorithm.PriorityConfig{{Function: LeastRequestedPriority, Weight: 1}},
			algorithm.FakeClusterLister(federation.ClusterList{Items: test.Clusters}), []algorithm.SchedulerExtender{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, hp := range list {
			if test.test == "test priority of larger replicaSet with cluster with zero-request replicaSet" {
				if hp.Score == expectedPriority {
					t.Errorf("%s: expected non-%d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			} else {
				if hp.Score != expectedPriority {
					t.Errorf("%s: expected %d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			}
		}
	}
}

func TestLeastRequested(t *testing.T) {
	//TODO waiting for the final call of how prioritizing func
	//random or based on least requests
}