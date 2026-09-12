//go:build linux

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

package lifecycle

import (
	"fmt"

	libcontainercgroups "github.com/opencontainers/cgroups"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
)

func isPodLevelResourcesSupported(pod *v1.Pod) PodAdmitResult {
	podLevelResourcesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)
	if resource.IsPodLevelResourcesSet(pod) && !podLevelResourcesEnabled {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PodLevelResourcesNotAdmittedReason,
			Message: "PodLevelResources feature gate is disabled",
		}
	}
	return PodAdmitResult{Admit: true}
}

// Valid range for spec.resources.limits.pids.
// Keep in sync with minPIDLimit/maxPIDLimit in pkg/apis/core/validation.
// The kubelet revalidates the range because static pods bypass apiserver
// validation entirely, and an apiserver with the gate disabled never
// validates the field at all.
const (
	minPodPIDLimit = 128
	maxPodPIDLimit = 16384
)

func isPerPodPIDLimitSupported(pod *v1.Pod) PodAdmitResult {
	if pod.Spec.Resources == nil {
		return PodAdmitResult{Admit: true}
	}
	pidQuantity, ok := pod.Spec.Resources.Limits[v1.ResourcePID]
	if !ok {
		return PodAdmitResult{Admit: true}
	}
	// Fail closed: a pod that requested a PID limit must not run without it.
	// This mirrors isPodLevelResourcesSupported above.
	if !utilfeature.DefaultFeatureGate.Enabled(features.PerPodPIDLimit) {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PerPodPIDLimitNotAdmittedReason,
			Message: fmt.Sprintf("pod %s specifies spec.resources.limits.pids, but the PerPodPIDLimit feature gate is disabled on this node", pod.Name),
		}
	}
	if pidValue := pidQuantity.Value(); pidValue < minPodPIDLimit || pidValue > maxPodPIDLimit {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PerPodPIDLimitNotAdmittedReason,
			Message: fmt.Sprintf("pod %s specifies spec.resources.limits.pids=%d, which is outside the allowed range [%d, %d]", pod.Name, pidValue, minPodPIDLimit, maxPodPIDLimit),
		}
	}
	if !libcontainercgroups.IsCgroup2UnifiedMode() {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PerPodPIDLimitNotAdmittedReason,
			Message: fmt.Sprintf("per-pod PID limit requires cgroupsv2, but this node is running cgroupsv1; pod %s specifies spec.resources.limits.pids", pod.Name),
		}
	}
	return PodAdmitResult{Admit: true}
}
