/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// FitPredicate is a function that indicates if a pod fits into an existing node.
// The failure information is given by the error.
// TODO: Change interface{} to a specific type.
type FitPredicate func(pod *api.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (bool, []PredicateFailureReason, error)

// PriorityMapFunction is a function that computes per-node results for a given node.
// TODO: Figure out the exact API of this method.
// TODO: Change interface{} to a specific type.
type PriorityMapFunction func(pod *api.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error)

// PriorityReduceFunction is a function that aggregated per-node results and computes
// final scores for all nodes.
// TODO: Figure out the exact API of this method.
// TODO: Change interface{} to a specific type.
type PriorityReduceFunction func(pod *api.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error

// MetdataProducer is a function that computes metadata for a given pod.
type MetadataProducer func(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo) interface{}

// DEPRECATED
// Use Map-Reduce pattern for priority functions.
type PriorityFunction func(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*api.Node) (schedulerapi.HostPriorityList, error)

type PriorityConfig struct {
	Map    PriorityMapFunction
	Reduce PriorityReduceFunction
	// TODO: Remove it after migrating all functions to
	// Map-Reduce pattern.
	Function PriorityFunction
	Weight   int
}

// EmptyMetadataProducer returns a no-op MetadataProducer type.
func EmptyMetadataProducer(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo) interface{} {
	return nil
}

type PredicateFailureReason interface {
	GetReason() string
}

type GetEquivalencePodFunc func(pod *api.Pod) interface{}
