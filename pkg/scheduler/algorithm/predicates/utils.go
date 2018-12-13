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

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// FindLabelsInSet gets as many key/value pairs as possible out of a label set.
func FindLabelsInSet(labelsToKeep []string, selector labels.Set) map[string]string {
	aL := make(map[string]string)
	for _, l := range labelsToKeep {
		if selector.Has(l) {
			aL[l] = selector.Get(l)
		}
	}
	return aL
}

// AddUnsetLabelsToMap backfills missing values with values we find in a map.
func AddUnsetLabelsToMap(aL map[string]string, labelsToAdd []string, labelSet labels.Set) {
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

// FilterPodsByNamespace filters pods outside a namespace from the given list.
func FilterPodsByNamespace(pods []*v1.Pod, ns string) []*v1.Pod {
	filtered := []*v1.Pod{}
	for _, nsPod := range pods {
		if nsPod.Namespace == ns {
			filtered = append(filtered, nsPod)
		}
	}
	return filtered
}

// CreateSelectorFromLabels is used to define a selector that corresponds to the keys in a map.
func CreateSelectorFromLabels(aL map[string]string) labels.Selector {
	if aL == nil || len(aL) == 0 {
		return labels.Everything()
	}
	return labels.Set(aL).AsSelector()
}

// portsConflict check whether existingPorts and wantPorts conflict with each other
// return true if we have a conflict
func portsConflict(existingPorts schedulernodeinfo.HostPortInfo, wantPorts []*v1.ContainerPort) bool {
	for _, cp := range wantPorts {
		if existingPorts.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
			return true
		}
	}

	return false
}

// Intersect returns a new set which includes the items in ALL "strsets"
// For example:
// input: [{a1, a2}, {a2, a3}, {a2, a4}]
// output: {a2}
func Intersect(strsets []sets.String) sets.String {
	length := len(strsets)
	if length == 0 {
		return sets.String{}
	}
	if length == 1 {
		return strsets[0]
	}
	if length == 2 {
		return strsets[0].Intersection(strsets[1])
	}
	return Intersect(strsets[:length/2]).Intersection(Intersect(strsets[length/2:]))
}

// Union returns a new set which includes all non-duplicate items in "strsets"
// For example:
// input: [{a1, a2}, {a2, a3}, {a4}]
// output: {a1, a2, a3, a4}
func Union(strsets []sets.String) sets.String {
	length := len(strsets)
	if length == 0 {
		return sets.String{}
	}
	if length == 1 {
		return strsets[0]
	}
	if length == 2 {
		return strsets[0].Union(strsets[1])
	}
	return Union(strsets[:length/2]).Union(Union(strsets[length/2:]))
}
