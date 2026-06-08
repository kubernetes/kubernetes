/*
Copyright 2020 The Kubernetes Authors.

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

package meta

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SetStatusCondition sets the corresponding condition in conditions to newCondition and returns true
// if the conditions are changed by this call.
// conditions must be non-nil.
//  1. if the condition of the specified type already exists (all fields of the existing condition are updated to
//     newCondition, LastTransitionTime is set to now if the new status differs from the old status)
//  2. if a condition of the specified type does not exist (LastTransitionTime is set to now() if unset, and newCondition is appended)
func SetStatusCondition(conditions *[]metav1.Condition, newCondition metav1.Condition) (changed bool) {
	if conditions == nil {
		return false
	}
	existingCondition := FindStatusCondition(*conditions, newCondition.Type)
	if existingCondition == nil {
		if newCondition.LastTransitionTime.IsZero() {
			newCondition.LastTransitionTime = metav1.NewTime(time.Now())
		}
		*conditions = append(*conditions, newCondition)
		return true
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		if !newCondition.LastTransitionTime.IsZero() {
			existingCondition.LastTransitionTime = newCondition.LastTransitionTime
		} else {
			existingCondition.LastTransitionTime = metav1.NewTime(time.Now())
		}
		changed = true
	}

	if existingCondition.Reason != newCondition.Reason {
		existingCondition.Reason = newCondition.Reason
		changed = true
	}
	if existingCondition.Message != newCondition.Message {
		existingCondition.Message = newCondition.Message
		changed = true
	}
	if existingCondition.ObservedGeneration != newCondition.ObservedGeneration {
		existingCondition.ObservedGeneration = newCondition.ObservedGeneration
		changed = true
	}

	return changed
}

// RemoveStatusCondition removes the corresponding conditionType from conditions if present. Returns
// true if it was present and got removed.
// conditions must be non-nil.
func RemoveStatusCondition(conditions *[]metav1.Condition, conditionType string) (removed bool) {
	if conditions == nil || len(*conditions) == 0 {
		return false
	}
	newConditions := make([]metav1.Condition, 0, len(*conditions)-1)
	for _, condition := range *conditions {
		if condition.Type != conditionType {
			newConditions = append(newConditions, condition)
		}
	}

	removed = len(*conditions) != len(newConditions)
	*conditions = newConditions

	return removed
}

// FindStatusCondition finds the conditionType in conditions.
func FindStatusCondition(conditions []metav1.Condition, conditionType string) *metav1.Condition {
	for i := range conditions {
		if conditions[i].Type == conditionType {
			return &conditions[i]
		}
	}

	return nil
}

// IsStatusConditionTrue returns true when the conditionType is present and set to `metav1.ConditionTrue`
func IsStatusConditionTrue(conditions []metav1.Condition, conditionType string) bool {
	return IsStatusConditionPresentAndEqual(conditions, conditionType, metav1.ConditionTrue)
}

// IsStatusConditionFalse returns true when the conditionType is present and set to `metav1.ConditionFalse`
func IsStatusConditionFalse(conditions []metav1.Condition, conditionType string) bool {
	return IsStatusConditionPresentAndEqual(conditions, conditionType, metav1.ConditionFalse)
}

// IsStatusConditionPresentAndEqual returns true when conditionType is present and equal to status.
func IsStatusConditionPresentAndEqual(conditions []metav1.Condition, conditionType string, status metav1.ConditionStatus) bool {
	for _, condition := range conditions {
		if condition.Type == conditionType {
			return condition.Status == status
		}
	}
	return false
}
