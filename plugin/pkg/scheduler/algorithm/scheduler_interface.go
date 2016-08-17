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
)

// SchedulerExtender is an interface for external processes to influence scheduling
// decisions made by Kubernetes. This is typically needed for resources not directly
// managed by Kubernetes.
type SchedulerExtender interface {
	// Filter based on extender-implemented predicate functions. The filtered list is
	// expected to be a subset of the supplied list. failedNodesMap optionally contains
	// the list of failed nodes and failure reasons.
	Filter(pod *api.Pod, nodes []*api.Node) (filteredNodes []*api.Node, failedNodesMap schedulerapi.FailedNodesMap, err error)

	// Prioritize based on extender-implemented priority functions. The returned scores & weight
	// are used to compute the weighted score for an extender. The weighted scores are added to
	// the scores computed  by Kubernetes scheduler. The total scores are used to do the host selection.
	Prioritize(pod *api.Pod, nodes []*api.Node) (hostPriorities *schedulerapi.HostPriorityList, weight int, err error)
}

// ScheduleAlgorithm is an interface implemented by things that know how to schedule pods
// onto machines.
type ScheduleAlgorithm interface {
	Schedule(*api.Pod, NodeLister) (selectedMachine string, err error)
}
