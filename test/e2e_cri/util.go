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

package e2e_cri

import (
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

var (
	defaultUid      string = "e2e-cri-uid"
	defaulNamespace string = "e2e-cri-namespace"
	defaulAttempt   uint32 = 2
)

// sandboxFound returns whether PodSandbox is found
func sandboxFound(podsandboxs []*runtimeapi.PodSandbox, podId string) bool {
	if len(podsandboxs) == 1 && podsandboxs[0].GetId() == podId {
		return true
	}
	return false
}

// containerFound returns whether containers is found
func containerFound(containers []*runtimeapi.Container, containerID string) bool {
	if len(containers) == 1 && containers[0].GetId() == containerID {
		return true
	}
	return false
}

// buildPodSandboxMetadata builds default PodSandboxMetadata with podName
func buildPodSandboxMetadata(podName *string) *runtimeapi.PodSandboxMetadata {
	return &runtimeapi.PodSandboxMetadata{
		Name:      podName,
		Uid:       &defaultUid,
		Namespace: &defaulNamespace,
		Attempt:   &defaulAttempt,
	}
}

// buildContainerMetadata builds default PodSandboxMetadata with containerName
func buildContainerMetadata(containerName *string) *runtimeapi.ContainerMetadata {
	return &runtimeapi.ContainerMetadata{
		Name:    containerName,
		Attempt: &defaulAttempt,
	}
}
