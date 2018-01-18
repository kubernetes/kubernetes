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
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
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

// EquivalencePodGenerator is a generator of equivalence class for pod with consideration of PVC info.
type EquivalencePodGenerator struct {
	pvcInfo PersistentVolumeClaimInfo
}

// NewEquivalencePodGenerator returns a getEquivalencePod method with consideration of PVC info.
func NewEquivalencePodGenerator(pvcInfo PersistentVolumeClaimInfo) algorithm.GetEquivalencePodFunc {
	g := &EquivalencePodGenerator{
		pvcInfo: pvcInfo,
	}
	return g.getEquivalencePod
}

// GetEquivalencePod returns a equivalencePod which contains a group of pod attributes which can be reused.
func (e *EquivalencePodGenerator) getEquivalencePod(pod *v1.Pod) interface{} {
	// For equivalence hash to be formally correct, lists and maps
	// in the equivalencePod should be normalized. (e.g. by sorting
	// them) However, the vast majority of equivalent pod classes
	// are expected to be created from a single pod template, so
	// they will all have the same ordering.
	return &equivalencePod{
		Labels:         pod.Labels,
		Affinity:       pod.Spec.Affinity,
		Containers:     pod.Spec.Containers,
		InitContainers: pod.Spec.InitContainers,
		NodeName:       &pod.Spec.NodeName,
		NodeSelector:   pod.Spec.NodeSelector,
		Tolerations:    pod.Spec.Tolerations,
		Volumes:        pod.Spec.Volumes,
	}
}

// equivalencePod is a group of pod attributes which can be reused as equivalence to schedule other pods.
type equivalencePod struct {
	Labels         map[string]string
	Affinity       *v1.Affinity
	Containers     []v1.Container // note about ordering
	InitContainers []v1.Container // note about ordering
	NodeName       *string
	NodeSelector   map[string]string
	Tolerations    []v1.Toleration
	Volumes        []v1.Volume // note about ordering
}

// portsConflict check whether existingPorts and wantPorts conflict with each other
// return true if we have a conflict
func portsConflict(existingPorts schedutil.HostPortInfo, wantPorts []*v1.ContainerPort) bool {
	for _, cp := range wantPorts {
		if existingPorts.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
			return true
		}
	}

	return false
}
