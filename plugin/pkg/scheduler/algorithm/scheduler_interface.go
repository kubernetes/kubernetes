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
	"k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// SchedulerExtender is an interface for external processes to influence scheduling
// decisions made by Kubernetes. This is typically needed for resources not directly
// managed by Kubernetes.
type SchedulerExtender interface {
	// Filter based on extender-implemented predicate functions. The filtered list is
	// expected to be a subset of the supplied list. failedNodesMap optionally contains
	// the list of failed nodes and failure reasons.
	Filter(pod *v1.Pod, nodes []*v1.Node, nodeNameToInfo map[string]*schedulercache.NodeInfo) (filteredNodes []*v1.Node, failedNodesMap schedulerapi.FailedNodesMap, err error)

	// Prioritize based on extender-implemented priority functions. The returned scores & weight
	// are used to compute the weighted score for an extender. The weighted scores are added to
	// the scores computed  by Kubernetes scheduler. The total scores are used to do the host selection.
	Prioritize(pod *v1.Pod, nodes []*v1.Node) (hostPriorities *schedulerapi.HostPriorityList, weight int, err error)

	// Bind delegates the action of binding a pod to a node to the extender.
	Bind(binding *v1.Binding) error

	// IsBinder returns whether this extender is configured for the Bind method.
	IsBinder() bool
}

// ScheduleAlgorithm is an interface implemented by things that know how to schedule pods
// onto machines.
type ScheduleAlgorithm interface {
	Schedule(*v1.Pod, NodeLister) (selectedMachine string, err error)
	// Predicates() returns a pointer to a map of predicate functions. This is
	// exposed for testing.
	Predicates() map[string]FitPredicate
	// Prioritizers returns a slice of priority config. This is exposed for
	// testing.
	Prioritizers() []PriorityConfig
}
