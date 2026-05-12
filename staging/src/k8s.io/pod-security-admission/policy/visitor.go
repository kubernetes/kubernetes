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
)

// ContainerVisitor is called with each container and the field.Path to that container
type ContainerVisitor func(container *corev1.Container)

// visitContainers invokes the visitor function for every container in the given pod spec
func visitContainers(podSpec *corev1.PodSpec, visitor ContainerVisitor) {
	for i := range podSpec.InitContainers {
		visitor(&podSpec.InitContainers[i])
	}
	for i := range podSpec.Containers {
		visitor(&podSpec.Containers[i])
	}
	for i := range podSpec.EphemeralContainers {
		visitor((*corev1.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon))
	}
}
