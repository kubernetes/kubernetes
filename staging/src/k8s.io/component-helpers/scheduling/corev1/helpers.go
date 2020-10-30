/*
Copyright 2020 The Kubernetes Authors.

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

package corev1

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
)

// PodPriority returns priority of the given pod.
func PodPriority(pod *v1.Pod) int32 {
	if pod.Spec.Priority != nil {
		return *pod.Spec.Priority
	}
	// When priority of a running pod is nil, it means it was created at a time
	// that there was no global default priority class and the priority class
	// name of the pod was empty. So, we resolve to the static default priority.
	return 0
}

// MatchNodeSelectorTerms checks whether the node labels and fields match node selector terms in ORed;
// nil or empty term matches no objects.
func MatchNodeSelectorTerms(
	node *v1.Node,
	nodeSelector *v1.NodeSelector,
) (bool, error) {
	if node == nil {
		return false, nil
	}
	return nodeaffinity.NewLazyErrorNodeSelector(nodeSelector).Match(node)
}
