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

package podcmd

import (
	"fmt"
	"io"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// DefaultContainerAnnotationName is an annotation name that can be used to preselect the interesting container
// from a pod when running kubectl.
const DefaultContainerAnnotationName = "kubectl.kubernetes.io/default-container"

// FindContainerByName selects the named container from the spec of
// the provided pod or return nil if no such container exists.
func FindContainerByName(pod *v1.Pod, name string) (*v1.Container, string) {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			return &pod.Spec.Containers[i], fmt.Sprintf("spec.containers{%s}", name)
		}
	}
	for i := range pod.Spec.InitContainers {
		if pod.Spec.InitContainers[i].Name == name {
			return &pod.Spec.InitContainers[i], fmt.Sprintf("spec.initContainers{%s}", name)
		}
	}
	for i := range pod.Spec.EphemeralContainers {
		if pod.Spec.EphemeralContainers[i].Name == name {
			return (*v1.Container)(&pod.Spec.EphemeralContainers[i].EphemeralContainerCommon), fmt.Sprintf("spec.ephemeralContainers{%s}", name)
		}
	}
	return nil, ""
}

// FindOrDefaultContainerByName defaults a container for a pod to the first container if any
// exists, or returns an error. It will print a message to the user indicating a default was
// selected if there was more than one container.
func FindOrDefaultContainerByName(pod *v1.Pod, name string, quiet bool, warn io.Writer) (*v1.Container, error) {
	// this should never happen, but just in case
	if len(pod.Spec.Containers) == 0 {
		return nil, fmt.Errorf("pod %s/%s does not have any containers", pod.Namespace, pod.Name)
	}

	// if no container name was specified read the default container from the annotation as per
	// https://github.com/kubernetes/enhancements/tree/master/keps/sig-cli/2227-kubectl-default-container
	if len(name) == 0 {
		name = pod.Annotations[DefaultContainerAnnotationName]
		if len(name) > 0 {
			klog.V(4).Infof("Attempting default container name from annotation %s", name)
		}
	}

	if len(name) > 0 {
		container, _ := FindContainerByName(pod, name)
		if container == nil {
			return nil, fmt.Errorf("container %s is not valid for pod %s out of: %s", name, pod.Name, AllContainerNames(pod))
		}
		return container, nil
	}

	// fallback to pick the first container to maintain existing behavior
	container := &pod.Spec.Containers[0]
	if !quiet && (len(pod.Spec.Containers) > 1 || len(pod.Spec.InitContainers) > 0 || len(pod.Spec.EphemeralContainers) > 0) {
		fmt.Fprintf(warn, "Defaulted container %q out of: %s\n", container.Name, AllContainerNames(pod)) // nolint:errcheck
	}
	klog.V(4).Infof("Using the first container in a pod: %s", container.Name)
	return container, nil
}

// AllContainerNames returns a comma-separated list of all container names in the pod,
// including regular containers, ephemeral containers (marked with "ephem"), and init
// containers (marked with "init") in that order.
func AllContainerNames(pod *v1.Pod) string {
	var containers []string
	for _, container := range pod.Spec.Containers {
		containers = append(containers, container.Name)
	}
	for _, container := range pod.Spec.EphemeralContainers {
		containers = append(containers, fmt.Sprintf("%s (ephem)", container.Name))
	}
	for _, container := range pod.Spec.InitContainers {
		containers = append(containers, fmt.Sprintf("%s (init)", container.Name))
	}
	return strings.Join(containers, ", ")
}
