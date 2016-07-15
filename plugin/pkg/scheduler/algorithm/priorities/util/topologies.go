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

// FilterPodsByNameSpaces filters the pods based the given list of namespaces,
// empty set of namespaces means all namespaces.
func FilterPodsByNameSpaces(names sets.String, pods []*api.Pod) []*api.Pod {
	if len(pods) == 0 || len(names) == 0 {
		return pods
	}
	result := []*api.Pod{}
	for _, pod := range pods {
		if names.Has(pod.Namespace) {
			result = append(result, pod)
		}
	}
	return result
}

// GetNamespacesFromPodAffinityTerm returns a set of names
// according to the namespaces indicated in podAffinityTerm.
// if the NameSpaces is nil considers the given pod's namespace
// if the Namespaces is empty list then considers all the namespaces
func GetNamespacesFromPodAffinityTerm(pod *api.Pod, podAffinityTerm api.PodAffinityTerm) sets.String {
	names := sets.String{}
	if podAffinityTerm.Namespaces == nil {
		names.Insert(pod.Namespace)
	} else if len(podAffinityTerm.Namespaces) != 0 {
		names.Insert(podAffinityTerm.Namespaces...)
	}
	return names
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

// CheckIfPodMatchPodAffinityTerm checks if podB's affinity request is compatible with podA
// TODO: Get rid this method. We should avoid computing Namespaces and selectors multiple times
// and check them on higher levels and then use NodesHaveSameTopologyKey method.
func (tps *Topologies) CheckIfPodMatchPodAffinityTerm(podA *api.Pod, nodeA, nodeB *api.Node, podB *api.Pod, podBAffinityTerm api.PodAffinityTerm) (bool, error) {
	names := GetNamespacesFromPodAffinityTerm(podB, podBAffinityTerm)
	if len(names) != 0 && !names.Has(podA.Namespace) {
		return false, nil
	}

	labelSelector, err := unversioned.LabelSelectorAsSelector(podBAffinityTerm.LabelSelector)
	if err != nil || !labelSelector.Matches(labels.Set(podA.Labels)) {
		return false, err
	}

	return tps.NodesHaveSameTopologyKey(nodeA, nodeB, podBAffinityTerm.TopologyKey), nil
}
