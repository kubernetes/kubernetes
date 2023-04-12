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

// ContainerVisitor is called with each container and the pathFn to that container.
type ContainerVisitor func(container *corev1.Container, pathFn PathFn)

// visitContainers invokes the visitor function with a pointer to the spec
// of every container in the given pod spec.
func visitContainers(podSpec *corev1.PodSpec, opts options, visitor ContainerVisitor) {
	for i := range podSpec.InitContainers {
		var pathFn PathFn
		if opts.withFieldErrors {
			pathFn = initContainersFldPath.index(i)
		}
		visitor(&podSpec.InitContainers[i], pathFn)
	}
	for i := range podSpec.Containers {
		var pathFn PathFn
		if opts.withFieldErrors {
			pathFn = containersFldPath.index(i)
		}
		visitor(&podSpec.Containers[i], pathFn)
	}
	for i := range podSpec.EphemeralContainers {
		var pathFn PathFn
		if opts.withFieldErrors {
			pathFn = ephemeralContainersFldPath.index(i)
		}
		visitor((*corev1.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon), pathFn)
	}
}
