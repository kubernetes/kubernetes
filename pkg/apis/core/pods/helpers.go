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
	"k8s.io/kubernetes/pkg/util/slice"
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
type ContainerPair = slice.Pair[api.Container]

// ContainerPairs yields the containers of pod and oldPod of the types in mask, paired by name,
// with their type. Containers are only matched within their type, and are assumed not to be
// reordered, which validation enforces. The iteration order is InitContainers, then main
// Containers, then EphemeralContainers.
func ContainerPairs(pod, oldPod *api.Pod, mask ContainerType) iter.Seq2[ContainerPair, ContainerType] {
	return func(yield func(ContainerPair, ContainerType) bool) {
		if mask&InitContainers != 0 && !slice.PairByKey(pod.Spec.InitContainers, oldPod.Spec.InitContainers, identityContainer, containerName, func(p ContainerPair) bool {
			return yield(p, InitContainers)
		}) {
			return
		}
		if mask&Containers != 0 && !slice.PairByKey(pod.Spec.Containers, oldPod.Spec.Containers, identityContainer, containerName, func(p ContainerPair) bool {
			return yield(p, Containers)
		}) {
			return
		}
		if mask&EphemeralContainers != 0 {
			slice.PairByKey(pod.Spec.EphemeralContainers, oldPod.Spec.EphemeralContainers, ephemeralAsContainer, containerName, func(p ContainerPair) bool {
				return yield(p, EphemeralContainers)
			})
		}
	}
}

func identityContainer(c *api.Container) *api.Container {
	return c
}

func ephemeralAsContainer(ec *api.EphemeralContainer) *api.Container {
	return (*api.Container)(&ec.EphemeralContainerCommon)
}

func containerName(c *api.Container) string {
	return c.Name
}

// VolumePair is a volume in two versions of a pod, matched by name. Old is nil for a volume
// that is only in the new pod (added), and New is nil for one that is only in the old pod
// (removed).
type VolumePair = slice.Pair[api.Volume]

// VolumePairs yields the volumes of pod and oldPod, paired by name. Volumes are assumed not to
// be reordered, which validation enforces; a reordered volume is yielded once as removed
// (New == nil) and once as added (Old == nil).
func VolumePairs(pod, oldPod *api.Pod) iter.Seq[VolumePair] {
	return slice.PairsByKey(pod.Spec.Volumes, oldPod.Spec.Volumes, volumeName)
}

func volumeName(v *api.Volume) string {
	return v.Name
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
