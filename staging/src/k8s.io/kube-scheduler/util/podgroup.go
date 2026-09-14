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

package util

import (
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
)

// PodGroupPriority returns priority of a given pod group.
func PodGroupPriority(pg *schedulingv1beta1.PodGroup) int32 {
	if pg.Spec.Priority != nil {
		return *pg.Spec.Priority
	}
	// When priority of a pod group is nil, it means it was created at a time
	// that there was no global default priority class and the priority class
	// name of the pod group was empty. So, we resolve to the static default priority.
	return 0
}

// CompositePodGroupPriority returns priority of a given composite pod group.
func CompositePodGroupPriority(cpg *schedulingv1alpha3.CompositePodGroup) int32 {
	if cpg.Spec.Priority != nil {
		return *cpg.Spec.Priority
	}
	// When priority of a composite pod group is nil, it means it was created
	// at a time that there was no global default priority class and the priority
	// class name of the composite pod group was empty. So, we resolve to the
	// static default priority.
	return 0
}
