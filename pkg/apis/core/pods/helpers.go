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

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/fieldpath"
)

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
		"status.podIPs",
		"spec.terminationGracePeriodSeconds":
		return label, value, nil
	// This is for backwards compatibility with old v1 clients which send spec.host
	case "spec.host":
		return "spec.nodeName", value, nil
	default:
		return "", "", fmt.Errorf("field label not supported: %s", label)
	}
}
