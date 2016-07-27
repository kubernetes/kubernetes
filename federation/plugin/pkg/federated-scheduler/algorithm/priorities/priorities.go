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

package priorities

import (
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"

	"math/rand"
)

type ClusterRandom struct {
	clusterLister algorithm.ClusterLister
}

func NewRandomChoosePriority(clusterLister algorithm.ClusterLister) algorithm.PriorityFunction {
	clusterRandom := &ClusterRandom{
		clusterLister: clusterLister,
	}
	return clusterRandom.CalculateRandomPriority
}

// RandomChoosePriority is a priority function that randomly choose target from candidate.
// it get a random index between 0 and len(clusters) - 1
func (c *ClusterRandom) CalculateRandomPriority(rc *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	chosenPriority := 100
	clusters, err := clusterLister.List()
	if err != nil {
		return schedulerapi.ClusterPriorityList{}, err
	}
	index := rand.Intn(len(clusters.Items))
	list := schedulerapi.ClusterPriorityList{}
	cluster := clusters.Items[index]
	list = append(list,
		schedulerapi.ClusterPriority{
			Cluster: cluster.Name,
			Score:   chosenPriority,
		},
	)
	return list, nil
}
