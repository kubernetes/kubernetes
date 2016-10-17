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

package testing

import (
	"fmt"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

func BuildContainerName(metadata *runtimeApi.ContainerMetadata, sandboxID string) string {
	// include the sandbox ID to make the container ID unique.
	return fmt.Sprintf("%s_%s_%d", sandboxID, metadata.GetName(), metadata.GetAttempt())
}

func BuildSandboxName(metadata *runtimeApi.PodSandboxMetadata) string {
	return fmt.Sprintf("%s_%s_%s_%d", metadata.GetName(), metadata.GetNamespace(), metadata.GetUid(), metadata.GetAttempt())
}

func filterInLabels(filter, labels map[string]string) bool {
	for k, v := range filter {
		if value, ok := labels[k]; ok {
			if value != v {
				return false
			}
		} else {
			return false
		}
	}

	return true
}
