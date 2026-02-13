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
)

// MatchingSchedulingGroup returns true if two pods belong to the same scheduling group.
func MatchingSchedulingGroup(pod1, pod2 *v1.Pod) bool {
	return pod1.Namespace == pod2.Namespace && pod1.Spec.SchedulingGroup != nil && pod2.Spec.SchedulingGroup != nil && *pod1.Spec.SchedulingGroup.PodGroupName == *pod2.Spec.SchedulingGroup.PodGroupName
}
