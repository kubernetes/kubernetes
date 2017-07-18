/*
Copyright 2015 The Kubernetes Authors.

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

package container

import (
	"fmt"

	"k8s.io/api/core/v1"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/kubernetes/pkg/api"
)

var ImplicitContainerPrefix string = "implicitly required container "

// GenerateContainerRef returns an *v1.ObjectReference which references the given container
// within the given pod. Returns an error if the reference can't be constructed or the
// container doesn't actually belong to the pod.
//
// This function will return an error if the provided Pod does not have a selfLink,
// but we expect selfLink to be populated at all call sites for the function.
func GenerateContainerRef(pod *v1.Pod, container *v1.Container) (*v1.ObjectReference, error) {
	fieldPath, err := fieldPath(pod, container)
	if err != nil {
		// TODO: figure out intelligent way to refer to containers that we implicitly
		// start (like the pod infra container). This is not a good way, ugh.
		fieldPath = ImplicitContainerPrefix + container.Name
	}
	ref, err := ref.GetPartialReference(api.Scheme, pod, fieldPath)
	if err != nil {
		return nil, err
	}
	return ref, nil
}

// fieldPath returns a fieldPath locating container within pod.
// Returns an error if the container isn't part of the pod.
func fieldPath(pod *v1.Pod, container *v1.Container) (string, error) {
	for i := range pod.Spec.Containers {
		here := &pod.Spec.Containers[i]
		if here.Name == container.Name {
			if here.Name == "" {
				return fmt.Sprintf("spec.containers[%d]", i), nil
			} else {
				return fmt.Sprintf("spec.containers{%s}", here.Name), nil
			}
		}
	}
	for i := range pod.Spec.InitContainers {
		here := &pod.Spec.InitContainers[i]
		if here.Name == container.Name {
			if here.Name == "" {
				return fmt.Sprintf("spec.initContainers[%d]", i), nil
			} else {
				return fmt.Sprintf("spec.initContainers{%s}", here.Name), nil
			}
		}
	}
	return "", fmt.Errorf("container %#v not found in pod %#v", container, pod)
}
