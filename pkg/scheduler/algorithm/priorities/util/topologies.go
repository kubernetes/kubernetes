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
	"sync"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
)

// NamespaceLister helps list Namespaces.
type namespaceLister interface {
	// List lists all Namespaces in the indexer.
	List(selector labels.Selector) (ret []*v1.Namespace, err error)
}

var (
	once     sync.Once
	nsLister namespaceLister
)

// InitNamespaceLister initialize the namespace lister
func InitNamespaceLister(namespaceListerLister namespaceLister) {
	once.Do(func() {
		nsLister = namespaceListerLister
	})
}

// GetNamespacesFromPodAffinityTerm return the namespaces from the pod affinity item
func GetNamespacesFromPodAffinityTerm(pod *v1.Pod, podAffinityTerm *v1.PodAffinityTerm) sets.String {
	namespaces := sets.String{}
	if podAffinityTerm.NamespaceSelector != nil {
		namespaceSelector, err := metav1.LabelSelectorAsSelector(podAffinityTerm.NamespaceSelector)
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
	} else {
		if len(podAffinityTerm.Namespaces) == 0 {
			namespaces.Insert(pod.Namespace)
		} else {
			namespaces.Insert(podAffinityTerm.Namespaces...)
		}
	}
	return namespaces
}

// PodMatchesTermsNamespaceAndSelector returns true if the given <pod>
// matches the namespace and selector defined by <affinityPod>`s <term>.
func PodMatchesTermsNamespaceAndSelector(pod *v1.Pod, namespaces sets.String, selector labels.Selector) bool {
	if !namespaces.Has(pod.Namespace) {
		return false
	}

	if !selector.Matches(labels.Set(pod.Labels)) {
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

// Topologies contains topologies information of nodes.
type Topologies struct {
	DefaultKeys []string
}

// NodesHaveSameTopologyKey checks if nodeA and nodeB have same label value with given topologyKey as label key.
func (tps *Topologies) NodesHaveSameTopologyKey(nodeA, nodeB *v1.Node, topologyKey string) bool {
	return NodesHaveSameTopologyKey(nodeA, nodeB, topologyKey)
}
