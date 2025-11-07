/*
Copyright 2022 The Kubernetes Authors.

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

package scheduling

import (
	"fmt"

	"k8s.io/kubernetes/pkg/apis/scheduling"
)

func GetWarningsForWorkload(workload *scheduling.Workload) []string {
	var warnings []string

	if workload != nil && len(workload.Spec.PodGroups) > 0 {
		for _, podGroup := range workload.Spec.PodGroups {
			if podGroup.Policy.Gang != nil && podGroup.Policy.Gang.MinCount <= 0 {
				warnings = append(warnings, fmt.Sprintf("podGroup.policy.gang.minCount: must be greater than 0"))
			}
		}
	}
	return warnings
}
