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

package predicates

import "k8s.io/kubernetes/pkg/labels"
import "k8s.io/kubernetes/pkg/api"

// findLabelsInSet gets as many constraints as possible from the pod itself.
func findLabelsInSet(labelsToKeep []string, nodeSelector labels.Set) map[string]string {
	aL := make(map[string]string)
	for _, l := range labelsToKeep {
		if nodeSelector.Has(l) {
			aL[l] = nodeSelector.Get(l)
		}
		// No need to flag missing labels: if the length of this map < labelsToKeep,
		// we know that we need to introspect nodes to come up with a complet set of
		// homogenous labelling constraints.
	}
	return aL
}

// addUnsetLabelsToMap backfills missing constraints with constraints we find in a map.
// Context: This function was originally made to support coming up with a constraint set for affinity services
// by introspecting labels being used by already existing pods.
func addUnsetLabelsToMap(aL map[string]string, labelsToAdd []string, labelSet labels.Set) {
	for _, l := range labelsToAdd {
		// if the label is already there, dont overwrite it.
		if _, exists := aL[l]; exists {
			continue
		}
		// otherwise, backfill this label.
		if labelSet.Has(l) {
			aL[l] = labelSet.Get(l)
		}
	}
}

// filterPodsByNamespace filters pods outside a namespace from the given list.
func filterPodsByNamespace(pods []*api.Pod, ns string) []*api.Pod {
	filtered := []*api.Pod{}
	for _, nsPod := range pods {
		if nsPod.Namespace == ns {
			filtered = append(filtered, nsPod)
		}
	}
	return filtered
}

// createSelectorFromLabels is used to define the final selector which we will use to decide wether a pod
// does, or does not, match a given node.
func createSelectorFromLabels(aL map[string]string) labels.Selector {
	if aL == nil || len(aL) == 0 {
		return labels.Everything()
	}
	return labels.Set(aL).AsSelector()
}
