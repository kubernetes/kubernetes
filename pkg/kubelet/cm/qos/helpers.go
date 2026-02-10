/*
Copyright 2025 The Kubernetes Authors.

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

package qos

import (
	v1 "k8s.io/api/core/v1"
)

// IsContainerEquivalentQOSGuaranteed checks if a container is considered Guaranteed
// from the perspective of resource managers that allocate exclusive resources.
// This requires the container to have both requests and limits set and equal
// for both CPU and Memory.
func IsContainerEquivalentQOSGuaranteed(container *v1.Container) bool {
	// Check container-level CPU requests and limits are equal.
	cpuRequest := container.Resources.Requests[v1.ResourceCPU]
	cpuLimit := container.Resources.Limits[v1.ResourceCPU]
	if cpuRequest.IsZero() || !cpuRequest.Equal(cpuLimit) {
		return false
	}

	// Check container-level Memory requests and limits are equal.
	memRequest := container.Resources.Requests[v1.ResourceMemory]
	memLimit := container.Resources.Limits[v1.ResourceMemory]
	if memRequest.IsZero() || !memRequest.Equal(memLimit) {
		return false
	}

	return true
}
