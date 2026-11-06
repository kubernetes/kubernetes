/*
Copyright 2017 The Kubernetes Authors.

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

package pods

import (
	"fmt"
	"iter"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/fieldpath"
)

// TODO: Move ContainerType, ContainerIter and ContainerPairs back to pkg/api/pod (which aliases
// them) once pkg/apis/core/validation, which uses them, no longer has to avoid importing
// pkg/api/pod, which imports it.

// ContainerType signifies container type
type ContainerType int

const (
	// Containers is for normal containers
	Containers ContainerType = 1 << iota
	// InitContainers is for init containers
	InitContainers
	// EphemeralContainers is for ephemeral containers
	EphemeralContainers
)

// AllContainers specifies that all containers be visited
const AllContainers ContainerType = (InitContainers | Containers | EphemeralContainers)

// ContainerIter returns an iterator over all containers in the given pod spec with a masked type.
// The iteration order is InitContainers, then main Containers, then EphemeralContainers.
func ContainerIter(podSpec *api.PodSpec, mask ContainerType) iter.Seq2[*api.Container, ContainerType] {
	return func(yield func(*api.Container, ContainerType) bool) {
		if mask&InitContainers != 0 {
			for i := range podSpec.InitContainers {
				if !yield(&podSpec.InitContainers[i], InitContainers) {
					return
				}
			}
		}
		if mask&Containers != 0 {
			for i := range podSpec.Containers {
				if !yield(&podSpec.Containers[i], Containers) {
					return
				}
			}
		}
		if mask&EphemeralContainers != 0 {
			for i := range podSpec.EphemeralContainers {
				if !yield((*api.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon), EphemeralContainers) {
					return
				}
			}
		}
	}
}

// ContainerPair is a container in two versions of a pod, matched by name. Old is nil for a
// container that is only in the new pod (added), and New is nil for one that is only in the old
// pod (removed).
type ContainerPair struct {
	New, Old *api.Container
}

// ContainerPairs yields the containers of pod and oldPod of the types in mask, paired by name,
// with their type. Containers are only matched within their type, and are assumed not to be
// reordered, which validation enforces. The iteration order is InitContainers, then main
// Containers, then EphemeralContainers.
func ContainerPairs(pod, oldPod *api.Pod, mask ContainerType) iter.Seq2[ContainerPair, ContainerType] {
	return func(yield func(ContainerPair, ContainerType) bool) {
		for _, containerType := range []ContainerType{InitContainers, Containers, EphemeralContainers} {
			if mask&containerType == 0 {
				continue
			}
			if !pairContainers(containersOfType(&pod.Spec, containerType), containersOfType(&oldPod.Spec, containerType), containerType, yield) {
				return
			}
		}
	}
}

func containersOfType(podSpec *api.PodSpec, containerType ContainerType) []*api.Container {
	var containers []*api.Container
	for c := range ContainerIter(podSpec, containerType) {
		containers = append(containers, c)
	}
	return containers
}

// pairContainers walks two container lists in parallel, yielding matched, added and removed
// containers in order. It returns false if yield did.
func pairContainers(containers, oldContainers []*api.Container, containerType ContainerType, yield func(ContainerPair, ContainerType) bool) bool {
	i, j := 0, 0
	for i < len(containers) && j < len(oldContainers) {
		c := containers[i]
		if c.Name == oldContainers[j].Name {
			if !yield(ContainerPair{New: c, Old: oldContainers[j]}, containerType) {
				return false
			}
			i++
			j++
			continue
		}
		// Containers aren't reordered, so if c isn't further on in the old list it was added,
		// and otherwise everything before it there was removed.
		k := j + 1
		for k < len(oldContainers) && oldContainers[k].Name != c.Name {
			k++
		}
		if k == len(oldContainers) {
			if !yield(ContainerPair{New: c}, containerType) {
				return false
			}
			i++
			continue
		}
		for ; j < k; j++ {
			if !yield(ContainerPair{Old: oldContainers[j]}, containerType) {
				return false
			}
		}
	}
	for ; i < len(containers); i++ {
		if !yield(ContainerPair{New: containers[i]}, containerType) {
			return false
		}
	}
	for ; j < len(oldContainers); j++ {
		if !yield(ContainerPair{Old: oldContainers[j]}, containerType) {
			return false
		}
	}
	return true
}

// ContainerVisitorWithPath is called with each container and the field.Path to that container,
// and returns true if visiting should continue.
type ContainerVisitorWithPath func(container *api.Container, path *field.Path) bool

// VisitContainersWithPath invokes the visitor function with a pointer to the spec
// of every container in the given pod spec and the field.Path to that container.
// If visitor returns false, visiting is short-circuited. VisitContainersWithPath returns true if visiting completes,
// false if visiting was short-circuited.
func VisitContainersWithPath(podSpec *api.PodSpec, specPath *field.Path, visitor ContainerVisitorWithPath) bool {
	fldPath := specPath.Child("initContainers")
	for i := range podSpec.InitContainers {
		if !visitor(&podSpec.InitContainers[i], fldPath.Index(i)) {
			return false
		}
	}
	fldPath = specPath.Child("containers")
	for i := range podSpec.Containers {
		if !visitor(&podSpec.Containers[i], fldPath.Index(i)) {
			return false
		}
	}
	fldPath = specPath.Child("ephemeralContainers")
	for i := range podSpec.EphemeralContainers {
		if !visitor((*api.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon), fldPath.Index(i)) {
			return false
		}
	}
	return true
}

// ConvertDownwardAPIFieldLabel converts the specified downward API field label
// and its value in the pod of the specified version to the internal version,
// and returns the converted label and value. This function returns an error if
// the conversion fails.
func ConvertDownwardAPIFieldLabel(version, label, value string) (string, string, error) {
	if version != "v1" {
		return "", "", fmt.Errorf("unsupported pod version: %s", version)
	}

	if path, _, ok := fieldpath.SplitMaybeSubscriptedPath(label); ok {
		switch path {
		case "metadata.annotations", "metadata.labels":
			return label, value, nil
		default:
			return "", "", fmt.Errorf("field label does not support subscript: %s", label)
		}
	}

	switch label {
	case "metadata.annotations",
		"metadata.labels",
		"metadata.name",
		"metadata.namespace",
		"metadata.uid",
		"spec.nodeName",
		"spec.restartPolicy",
		"spec.serviceAccountName",
		"spec.schedulerName",
		"status.phase",
		"status.hostIP",
		"status.hostIPs",
		"status.podIP",
		"status.podIPs":
		return label, value, nil
	// This is for backwards compatibility with old v1 clients which send spec.host
	case "spec.host":
		return "spec.nodeName", value, nil
	default:
		return "", "", fmt.Errorf("field label not supported: %s", label)
	}
}
