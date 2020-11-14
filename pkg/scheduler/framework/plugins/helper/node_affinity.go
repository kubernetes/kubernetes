/*
Copyright 2019 The Kubernetes Authors.

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

package helper

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/component-helpers/scheduling/corev1"
)

// PodMatchesNodeSelectorAndAffinityTerms checks whether the pod is schedulable onto nodes according to
// the requirements in both NodeAffinity and nodeSelector.
func PodMatchesNodeSelectorAndAffinityTerms(pod *v1.Pod, node *v1.Node) bool {
	// Check if node.Labels match pod.Spec.NodeSelector.
	if len(pod.Spec.NodeSelector) > 0 {
		selector := labels.SelectorFromSet(pod.Spec.NodeSelector)
		if !selector.Matches(labels.Set(node.Labels)) {
			return false
		}
	}
	if pod.Spec.Affinity == nil {
		return true
	}
	return NodeMatchesNodeAffinity(pod.Spec.Affinity.NodeAffinity, node)
}

// NodeMatchesNodeAffinity checks whether the Node satisfy the given NodeAffinity.
func NodeMatchesNodeAffinity(affinity *v1.NodeAffinity, node *v1.Node) bool {
	// 1. nil NodeSelector matches all nodes (i.e. does not filter out any nodes)
	// 2. nil []NodeSelectorTerm (equivalent to non-nil empty NodeSelector) matches no nodes
	// 3. zero-length non-nil []NodeSelectorTerm matches no nodes also, just for simplicity
	// 4. nil []NodeSelectorRequirement (equivalent to non-nil empty NodeSelectorTerm) matches no nodes
	// 5. zero-length non-nil []NodeSelectorRequirement matches no nodes also, just for simplicity
	// 6. non-nil empty NodeSelectorRequirement is not allowed
	if affinity == nil {
		return true
	}
	// Match node selector for requiredDuringSchedulingRequiredDuringExecution.
	// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
	// if affinity.RequiredDuringSchedulingRequiredDuringExecution != nil && !nodeMatchesNodeSelector(node, affinity.RequiredDuringSchedulingRequiredDuringExecution) {
	// 	return false
	// }

	// Match node selector for requiredDuringSchedulingIgnoredDuringExecution.
	if affinity.RequiredDuringSchedulingIgnoredDuringExecution != nil && !nodeMatchesNodeSelector(node, affinity.RequiredDuringSchedulingIgnoredDuringExecution) {
		return false
	}
	return true
}

// nodeMatchesNodeSelector checks if a node's labels satisfy a list of node selector terms,
// terms are ORed, and an empty list of terms will match nothing.
func nodeMatchesNodeSelector(node *v1.Node, nodeSelector *v1.NodeSelector) bool {
	// TODO(#96164): parse this error earlier in the plugin so we only need to do it once per Pod.
	matches, _ := corev1.MatchNodeSelectorTerms(node, nodeSelector)
	return matches
}
