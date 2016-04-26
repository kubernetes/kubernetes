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
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func makeCluster(Cluster string) federation.Cluster {
	return federation.Cluster{
		ObjectMeta: v1.ObjectMeta{Name: Cluster},
		Status: federation.ClusterStatus{
		},
	}
}

func getAssumedSubRS(replicaSets []*extensions.ReplicaSet) ([]*federation.SubReplicaSet) {

	result := []*federation.SubReplicaSet{}
	for _, rs := range replicaSets {
		clone, _ := conversion.NewCloner().DeepCopy(rs)
		rsTemp := clone.(*extensions.ReplicaSet)
		subRS := &federation.SubReplicaSet{}
		subRS.TypeMeta = rsTemp.TypeMeta
		subRS.ObjectMeta = rsTemp.ObjectMeta
		subRS.Spec = rsTemp.Spec
		subRS.Status = rsTemp.Status
		meta := &api.ObjectMeta{}
		//&subRS.ObjectMeta
		meta.GenerateName = subRS.Name + "-"
		api.GenerateName(api.SimpleNameGenerator, meta)
		subRS.Name = meta.Name
		result = append(result, subRS)
	}
	return result
}

func TestRandomChoosePriority(t *testing.T) {
	//random
	replicaSet := extensions.ReplicaSetSpec{
		Template: v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{},
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
			replicaSet:   &extensions.ReplicaSet{Spec: replicaSet},
			Clusters: []federation.Cluster{makeCluster("cluster1"), makeCluster("cluster2")},
			test:  "test priority of random choose",
			replicaSets: []*extensions.ReplicaSet{
				{Spec: replicaSet},
			},
		},
	}
	const expectedPriority int = 100
	for _, test := range tests {
		ClusterNameToInfo := schedulercache.CreateClusterNameToInfoMap(test.replicaSets)
		list, err := scheduler.PrioritizeClusters(
			test.replicaSet,
			ClusterNameToInfo,
			// This should match the configuration in defaultPriorities() in
			// plugin/pkg/federated-scheduler/algorithmprovider/defaults/defaults.go if you want
			// to test what's actually in production.
			[]algorithm.PriorityConfig{{Function: NewRandomChoosePriority(algorithm.FakeClusterLister(federation.ClusterList{Items: test.Clusters})), Weight: 1}},
			algorithm.FakeClusterLister(federation.ClusterList{Items: test.Clusters}))
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
