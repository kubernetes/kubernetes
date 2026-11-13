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
	coreresource "k8s.io/kubernetes/pkg/api/v1/resource"
	"k8s.io/kubernetes/pkg/features"
)

// isCgroupV2 is a package-level variable that defaults to the real cgroup check.
// Tests can override it to simulate cgroupsv1 environments.
var isCgroupV2 = libcontainercgroups.IsCgroup2UnifiedMode

func isPodLevelResourcesSupported(pod *v1.Pod) PodAdmitResult {
	podLevelResourcesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)
	if resource.IsPodLevelResourcesSet(pod) && !podLevelResourcesEnabled {
		return PodAdmitResult{
			Admit:         false,
			Reason:        PodLevelResourcesNotAdmittedReason,
			Message:       "PodLevelResources feature gate is disabled",
			RejectionType: RejectionInfeasible,
		}
	}
	return PodAdmitResult{Admit: true}
}

// isPerPodPIDLimitSupported revalidates the PID range as defense against
// version skew between this node and the API server (e.g. a kubelet binary
// with different MinPodPIDLimit/MaxPodPIDLimit bounds), and enforces the
// PerPodPIDLimit gate when it is disabled on this node even though the API
// server admitted the pod. Static pods already run through the same
// validation.ValidatePodCreate path (see pkg/kubelet/config/common.go).
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
	if pidValue := pidQuantity.Value(); pidValue < coreresource.MinPodPIDLimit || pidValue > coreresource.MaxPodPIDLimit {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PerPodPIDLimitNotAdmittedReason,
			Message: fmt.Sprintf("pod %s specifies spec.resources.limits.pids=%d, which is outside the allowed range [%d, %d]", pod.Name, pidValue, coreresource.MinPodPIDLimit, coreresource.MaxPodPIDLimit),
		}
	}
	if !isCgroupV2() {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PerPodPIDLimitNotAdmittedReason,
			Message: fmt.Sprintf("per-pod PID limit requires cgroupsv2, but this node is running cgroupsv1; pod %s specifies spec.resources.limits.pids", pod.Name),
		}
	}
	return PodAdmitResult{Admit: true}
}
