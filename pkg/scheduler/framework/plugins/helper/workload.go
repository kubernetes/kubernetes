/*
Copyright The Kubernetes Authors.

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

package helper

import (
	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
)

// MatchingWorkloadReference returns true if two pods belong to the same workload, including their pod group and replica key.
func MatchingWorkloadReference(pod1, pod2 *v1.Pod) bool {
	return pod1.Spec.WorkloadRef != nil && pod2.Spec.WorkloadRef != nil && pod1.Namespace == pod2.Namespace && *pod1.Spec.WorkloadRef == *pod2.Spec.WorkloadRef
}

// PodGroupPolicy is a helper to find the policy for a specific pod group name in a workload.
func PodGroupPolicy(workload *schedulingapi.Workload, podGroupName string) (schedulingapi.PodGroupPolicy, bool) {
	for _, podGroup := range workload.Spec.PodGroups {
		if podGroup.Name == podGroupName {
			return podGroup.Policy, true
		}
	}
	return schedulingapi.PodGroupPolicy{}, false
}
