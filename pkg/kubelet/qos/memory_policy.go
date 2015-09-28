/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
)

const (
	PodInfraOOMAdj       int = -999
	KubeletOOMScoreAdj   int = -999
	KubeProxyOOMScoreAdj int = -999
)

// isMemoryBestEffort returns true if the container's memory requirements are best-effort.
func isMemoryBestEffort(container *api.Container) bool {
	// A container is memory best-effort if its memory request is unspecified or 0.
	// If a request is specified, then the user expects some kind of resource guarantee.
	return container.Resources.Requests.Memory().Value() == 0
}

// isMemoryGuaranteed returns true if the container's memory requirements are Guaranteed.
func isMemoryGuaranteed(container *api.Container) bool {
	// A container is memory guaranteed if its memory request == memory limit.
	// If memory request == memory limit, the user is very confident of resource consumption.
	memoryRequest := container.Resources.Requests.Memory()
	memoryLimit := container.Resources.Limits.Memory()
	return (*memoryRequest).Cmp(*memoryLimit) == 0 && memoryRequest.Value() != 0
}

// GetContainerOOMAdjust returns the amount by which the OOM score of all processes in the
// container should be adjusted. The OOM score of a process is the percentage of memory it consumes
// multiplied by 100 (barring exceptional cases) + a configurable quantity which is between -1000
// and 1000. Containers with higher OOM scores are killed if the system runs out of memory.
// See https://lwn.net/Articles/391222/ for more information.
func GetContainerOOMScoreAdjust(container *api.Container, memoryCapacity int64) int {
	if isMemoryGuaranteed(container) {
		// Memory guaranteed containers should be the last to get killed.
		return -999
	} else if isMemoryBestEffort(container) {
		// Memory best-effort containers should be the first to be killed.
		return 1000
	} else {
		// Burstable containers are a middle tier, between Guaranteed and Best-Effort. Ideally,
		// we want to protect Burstable containers that consume less memory than requested.
		// The formula below is a heuristic. A container requesting for 10% of a system's
		// memory will have an oom score adjust of 900. If a process in container Y
		// uses over 10% of memory, its OOM score will be 1000. The idea is that containers
		// which use more than their request will have an OOM score of 1000 and will be prime
		// targets for OOM kills.
		// Note that this is a heuristic, it won't work if a container has many small processes.
		memoryRequest := container.Resources.Requests.Memory().Value()
		oomScoreAdjust := 1000 - (1000*memoryRequest)/memoryCapacity
		// A memory guaranteed container using 100% of memory can have an OOM score of 1. Ensure
		// that memory burstable containers have a higher OOM score.
		if oomScoreAdjust < 2 {
			return 2
		}
		return int(oomScoreAdjust)
	}
}
