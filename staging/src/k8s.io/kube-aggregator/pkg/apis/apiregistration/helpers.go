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

package apiregistration

import (
	"sort"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/version"
)

// SortedByGroupAndVersion sorts APIServices into their different groups, and then sorts them based on their versions.
// For example, the first element of the first array contains the APIService with the highest version number, in the
// group with the highest priority; while the last element of the last array contains the APIService with the lowest
// version number, in the group with the lowest priority.
func SortedByGroupAndVersion(servers []*APIService) [][]*APIService {
	serversByGroupPriorityMinimum := ByGroupPriorityMinimum(servers)
	sort.Sort(serversByGroupPriorityMinimum)

	ret := [][]*APIService{}
	for _, curr := range serversByGroupPriorityMinimum {
		// check to see if we already have an entry for this group
		existingIndex := -1
		for j, groupInReturn := range ret {
			if groupInReturn[0].Spec.Group == curr.Spec.Group {
				existingIndex = j
				break
			}
		}

		if existingIndex >= 0 {
			ret[existingIndex] = append(ret[existingIndex], curr)
			sort.Sort(ByVersionPriority(ret[existingIndex]))
			continue
		}

		ret = append(ret, []*APIService{curr})
	}

	return ret
}

// ByGroupPriorityMinimum sorts with the highest group number first, then by name.
// This is not a simple reverse, because we want the name sorting to be alpha, not
// reverse alpha.
type ByGroupPriorityMinimum []*APIService

func (s ByGroupPriorityMinimum) Len() int      { return len(s) }
func (s ByGroupPriorityMinimum) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ByGroupPriorityMinimum) Less(i, j int) bool {
	if s[i].Spec.GroupPriorityMinimum != s[j].Spec.GroupPriorityMinimum {
		return s[i].Spec.GroupPriorityMinimum > s[j].Spec.GroupPriorityMinimum
	}
	return s[i].Name < s[j].Name
}

// ByVersionPriority sorts with the highest version number first, then by name.
// This is not a simple reverse, because we want the name sorting to be alpha, not
// reverse alpha.
type ByVersionPriority []*APIService

func (s ByVersionPriority) Len() int      { return len(s) }
func (s ByVersionPriority) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ByVersionPriority) Less(i, j int) bool {
	if s[i].Spec.VersionPriority != s[j].Spec.VersionPriority {
		return s[i].Spec.VersionPriority > s[j].Spec.VersionPriority
	}
	return version.CompareKubeAwareVersionStrings(s[i].Spec.Version, s[j].Spec.Version) > 0
}

// NewLocalAvailableAPIServiceCondition returns a condition for an available local APIService.
func NewLocalAvailableAPIServiceCondition() APIServiceCondition {
	return APIServiceCondition{
		Type:               Available,
		Status:             ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "Local",
		Message:            "Local APIServices are always available",
	}
}

// GetAPIServiceConditionByType gets an *APIServiceCondition by APIServiceConditionType if present
func GetAPIServiceConditionByType(apiService *APIService, conditionType APIServiceConditionType) *APIServiceCondition {
	for i := range apiService.Status.Conditions {
		if apiService.Status.Conditions[i].Type == conditionType {
			return &apiService.Status.Conditions[i]
		}
	}
	return nil
}

// SetAPIServiceCondition sets the status condition.  It either overwrites the existing one or
// creates a new one
func SetAPIServiceCondition(apiService *APIService, newCondition APIServiceCondition) {
	existingCondition := GetAPIServiceConditionByType(apiService, newCondition.Type)
	if existingCondition == nil {
		apiService.Status.Conditions = append(apiService.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// IsAPIServiceConditionTrue indicates if the condition is present and strictly true
func IsAPIServiceConditionTrue(apiService *APIService, conditionType APIServiceConditionType) bool {
	condition := GetAPIServiceConditionByType(apiService, conditionType)
	return condition != nil && condition.Status == ConditionTrue
}
