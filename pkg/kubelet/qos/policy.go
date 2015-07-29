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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type Tier int

const (
	PodInfraOomAdj int  = -999
	BestEffort     Tier = iota
	Burstable
	Guaranteed
)

// IsBestEffort returns true if the pod is Best-Effort.
func IsBestEffort(podSpec *api.PodSpec) bool {
	// A pod is best-effort if all resource requests were unspecified or 0.
	// If a resource is specified, then the user expects some kind of resource guarantee.
	for _, container := range podSpec.Containers {
		requests := container.Resources.Requests
		if requests.Memory().Value() != 0 || requests.Cpu().MilliValue() != 0 {
			return false
		}
	}
	return true
}

// IsGuaranteed returns true if the pod is Guaranteed.
func IsGuaranteed(podSpec *api.PodSpec) bool {
	// A pod is guaranteed if for all containers, 0 != memory limit == memory request.
	// Ideally the guaranteed tier would not be necessary, but
	// 1. The kubelet might commit more burstable/guaranteed pods than it can handle
	// 2. Upstream kernel does not have mechanisms to represent tiers without races.
	// So we prioritize pods that are very confident in their memory footprint. We don't consider
	// CPU requests & limits for 2 reasons:
	// 1. CPU is compressible, so it can be hard to fix a request that a pod needs.
	// 2. CPU limits don't always work as expected, so users will often disable it.
	for _, container := range podSpec.Containers {
		requests := container.Resources.Requests
		memoryRequest := requests.Memory().Value()
		limits := container.Resources.Limits
		if memoryRequest == 0 || memoryRequest != limits.Memory().Value() {
			return false
		}
	}
	return true
}

// IsGuaranteed returns the pod's tier (Best-Effort, Burstable, Guaranteed).
func GetPodTier(podSpec *api.PodSpec) Tier {
	if IsBestEffort(podSpec) {
		return BestEffort
	} else if IsGuaranteed(podSpec) {
		return Guaranteed
	} else {
		return Burstable
	}
}

func podMemoryRequest(podSpec *api.PodSpec) (memoryRequest int64) {
	for _, container := range podSpec.Containers {
		memoryRequest += container.Resources.Requests.Memory().Value()
	}
	return
}

// GetPodOomAdjust returns the amount by which the OOM score of all processes in the pod should
// be adjusted. The OOM score of a process is the percentage of memory it consumes multiplied by
// 100 (barring exceptional cases). Pods with lower OOM scores are less likely to be killed if the
// system runs out of memory.
func GetPodOomAdjust(podSpec *api.PodSpec, memoryCapacity int64) int {
	podTier := GetPodTier(podSpec)
	switch podTier {
	case Guaranteed:
		return -999
	case Burstable:
		oomAdjust := 995 - (1000*podMemoryRequest(podSpec))/memoryCapacity
		if oomAdjust < 2 {
			return -999
		} else {
			return int(oomAdjust)
		}
	}
	return 1000
}
