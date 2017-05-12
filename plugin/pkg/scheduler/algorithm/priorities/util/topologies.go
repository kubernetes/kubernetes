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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/features"
)

// NamespaceLister helps list Namespaces.
type namespaceLister interface {
	// List lists all Namespaces in the indexer.
	List(selector labels.Selector) (ret []*v1.Namespace, err error)
}

// GetNamespacesFromPodAffinityTerm returns a set of names
// according to the NamespaceSelector indicated in podAffinityTerm.
// If NamespaceSelector is not provided, podAffinityTerm.Namespaces will be used and if it is empty,
// pod's namespace will be returned.
// TODO We will remove the filed term.Namespaces in 1.8, then we need to remove using the term.Namespaces here
func GetNamespacesFromPodAffinityTerm(nsLister namespaceLister, pod *v1.Pod, term *v1.PodAffinityTerm) sets.String {
	namespaces := sets.String{}
	namespaceSelector, err := metav1.LabelSelectorAsSelector(term.NamespaceSelector)
	if err != nil {
		return namespaces
	}
	namespaceList, err := nsLister.List(namespaceSelector)
	if err != nil {
		return namespaces
	}

	for _, ns := range namespaceList {
		namespaces.Insert(ns.Name)
	}
	if len(namespaces) == 0 {
		if len(term.Namespaces) == 0 {
			namespaces.Insert(pod.Namespace)
		} else {
			namespaces.Insert(term.Namespaces...)
		}
	}
	return namespaces
}

// PodMatchesTermsNamespaceAndSelector returns true if the given <pod>
// matches the namespace and selector defined by <affinityPod>`s <term>.
func PodMatchesTermsNamespaceAndSelector(pod *v1.Pod, namespaces sets.String, podSelector labels.Selector) bool {
	if !namespaces.Has(pod.Namespace) {
		return false
	}

	if !podSelector.Matches(labels.Set(pod.Labels)) {
		return false
	}
	return true
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

type Topologies struct {
	DefaultKeys []string
}

// NodesHaveSameTopologyKey checks if nodeA and nodeB have same label value with given topologyKey as label key.
// If the topologyKey is empty, check if the two nodes have any of the default topologyKeys, and have same corresponding label value.
func (tps *Topologies) NodesHaveSameTopologyKey(nodeA, nodeB *v1.Node, topologyKey string) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.AffinityInAnnotations) && len(topologyKey) == 0 {
		// assumes this is allowed only for PreferredDuringScheduling pod anti-affinity (ensured by api/validation)
		for _, defaultKey := range tps.DefaultKeys {
			if NodesHaveSameTopologyKey(nodeA, nodeB, defaultKey) {
				return true
			}
		}
		return false
	} else {
		return NodesHaveSameTopologyKey(nodeA, nodeB, topologyKey)
	}
}
