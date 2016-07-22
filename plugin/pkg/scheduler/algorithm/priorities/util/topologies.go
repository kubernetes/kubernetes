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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

// GetNamespacesFromPodAffinityTerm returns a set of names
// according to the namespaces indicated in podAffinityTerm.
// 1. If the namespaces is nil considers the given pod's namespace
// 2. If the namespaces is empty list then considers all the namespaces
func getNamespacesFromPodAffinityTerm(pod *api.Pod, podAffinityTerm api.PodAffinityTerm) sets.String {
	names := sets.String{}
	if podAffinityTerm.Namespaces == nil {
		names.Insert(pod.Namespace)
	} else if len(podAffinityTerm.Namespaces) != 0 {
		names.Insert(podAffinityTerm.Namespaces...)
	}
	return names
}

// PodMatchesTermsNamespaceAndSelector returns true if the given <pod>
// matches the namespace and selector defined by <affinityPod>`s <term>.
func PodMatchesTermsNamespaceAndSelector(pod *api.Pod, affinityPod *api.Pod, term *api.PodAffinityTerm) (bool, error) {
	namespaces := getNamespacesFromPodAffinityTerm(affinityPod, *term)
	if len(namespaces) != 0 && !namespaces.Has(pod.Namespace) {
		return false, nil
	}

	selector, err := unversioned.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil || !selector.Matches(labels.Set(pod.Labels)) {
		return false, err
	}
	return true, nil
}

// nodesHaveSameTopologyKeyInternal checks if nodeA and nodeB have same label value with given topologyKey as label key.
func nodesHaveSameTopologyKeyInternal(nodeA, nodeB *api.Node, topologyKey string) bool {
	return nodeA.Labels != nil && nodeB.Labels != nil && len(nodeA.Labels[topologyKey]) > 0 && nodeA.Labels[topologyKey] == nodeB.Labels[topologyKey]
}

type Topologies struct {
	DefaultKeys []string
}

// NodesHaveSameTopologyKey checks if nodeA and nodeB have same label value with given topologyKey as label key.
// If the topologyKey is nil/empty, check if the two nodes have any of the default topologyKeys, and have same corresponding label value.
func (tps *Topologies) NodesHaveSameTopologyKey(nodeA, nodeB *api.Node, topologyKey string) bool {
	if len(topologyKey) == 0 {
		// assumes this is allowed only for PreferredDuringScheduling pod anti-affinity (ensured by api/validation)
		for _, defaultKey := range tps.DefaultKeys {
			if nodesHaveSameTopologyKeyInternal(nodeA, nodeB, defaultKey) {
				return true
			}
		}
		return false
	} else {
		return nodesHaveSameTopologyKeyInternal(nodeA, nodeB, topologyKey)
	}
}
