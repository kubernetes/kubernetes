/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/klog"
)

// GetNamespacesFromPodAffinityTerm returns a list of names
// according to the namespaces indicated in podAffinityTerm.
// If namespaces is empty it considers the given pod's namespace.
func GetNamespacesFromPodAffinityTerm(pod *v1.Pod, podAffinityTerm *v1.PodAffinityTerm) []string {
	if len(podAffinityTerm.Namespaces) > 0 {
		return podAffinityTerm.Namespaces
	}
	return []string{pod.Namespace}
}

// PodMatchesTermsNamespaceAndSelector returns true if the given <pod>
// matches the namespace and selector defined by <affinityPod>`s <term>.
func PodMatchesTermsNamespaceAndSelector(pod *v1.Pod, namespaces []string, selector *metav1.LabelSelector) bool {
	for _, ns := range namespaces {
		if pod.Namespace == ns {
			return LabelsMatchLabelSelector(labels.Set(pod.Labels), selector)
		}
	}

	return false
}

// LabelsMatchLabelSelector returns true if the given <labels> matches the labelSelector
func LabelsMatchLabelSelector(labels labels.Labels, labelSelector *metav1.LabelSelector) bool {
	if labelSelector == nil {
		return false
	}
	for k, v := range labelSelector.MatchLabels {
		if !matchLabels(labels, k, v) {
			return false
		}
	}
	for _, expr := range labelSelector.MatchExpressions {
		if !matchExpression(labels, expr) {
			return false
		}
	}
	return true
}

func matchLabels(labels labels.Labels, key string, value string) bool {
	if !labels.Has(key) {
		return false
	}
	return value == labels.Get(key)
}

func expressionHasValue(expr metav1.LabelSelectorRequirement, value string) bool {
	for _, v := range expr.Values {
		if value == v {
			return true
		}
	}
	return false
}

func matchExpression(labels labels.Labels, expr metav1.LabelSelectorRequirement) bool {
	switch expr.Operator {
	case metav1.LabelSelectorOpIn:
		if !labels.Has(expr.Key) {
			return false
		}
		return expressionHasValue(expr, labels.Get(expr.Key))
	case metav1.LabelSelectorOpNotIn:
		if !labels.Has(expr.Key) {
			return true
		}
		return !expressionHasValue(expr, labels.Get(expr.Key))
	case metav1.LabelSelectorOpExists:
		return labels.Has(expr.Key)
	case metav1.LabelSelectorOpDoesNotExist:
		return !labels.Has(expr.Key)
	default:
		klog.V(5).Infof("%q is not a valid pod selector operator", expr.Operator)
		return false
	}
}

// NodesHaveSameTopologyKey checks if nodeA and nodeB have same label value with given topologyKey as label key.
// Returns false if topologyKey is empty.
func NodesHaveSameTopologyKey(nodeA, nodeB *v1.Node, topologyKey string) bool {
	if len(topologyKey) == 0 {
		return false
	}

	if nodeA.Labels == nil || nodeB.Labels == nil {
		return false
	}

	nodeALabel, okA := nodeA.Labels[topologyKey]
	nodeBLabel, okB := nodeB.Labels[topologyKey]

	// If found label in both nodes, check the label
	if okB && okA {
		return nodeALabel == nodeBLabel
	}

	return false
}

// Topologies contains topologies information of nodes.
type Topologies struct {
	DefaultKeys []string
}

// NodesHaveSameTopologyKey checks if nodeA and nodeB have same label value with given topologyKey as label key.
func (tps *Topologies) NodesHaveSameTopologyKey(nodeA, nodeB *v1.Node, topologyKey string) bool {
	return NodesHaveSameTopologyKey(nodeA, nodeB, topologyKey)
}
