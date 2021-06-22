/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ContainerVisitorWithPath is called with each container and the field.Path to that container
type ContainerVisitorWithPath func(container *corev1.Container, path *field.Path)

// visitContainersWithPath invokes the visitor function with a pointer to the spec
// of every container in the given pod spec and the field.Path to that container.
func visitContainersWithPath(podSpec *corev1.PodSpec, specPath *field.Path, visitor ContainerVisitorWithPath) {
	fldPath := specPath.Child("initContainers")
	for i := range podSpec.InitContainers {
		visitor(&podSpec.InitContainers[i], fldPath.Index(i))
	}
	fldPath = specPath.Child("containers")
	for i := range podSpec.Containers {
		visitor(&podSpec.Containers[i], fldPath.Index(i))
	}
	fldPath = specPath.Child("ephemeralContainers")
	for i := range podSpec.EphemeralContainers {
		visitor((*corev1.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon), fldPath.Index(i))
	}
}
