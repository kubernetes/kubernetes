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

package algorithm

import (
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// FitPredicate is a function that indicates if a replicaset fits into an existing cluster.
type FitPredicate func(rc *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error)

type PriorityFunction func(rc *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister ClusterLister) (schedulerapi.ClusterPriorityList, error)

type PriorityConfig struct {
	Function PriorityFunction
	Weight   int
}
