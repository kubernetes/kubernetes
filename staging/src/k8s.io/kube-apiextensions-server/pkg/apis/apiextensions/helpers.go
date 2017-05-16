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

package apiextensions

// SetCRDCondition sets the status condition.  It either overwrites the existing one or
// creates a new one
func SetCRDCondition(customResourceDefinition *CustomResourceDefinition, newCondition CustomResourceDefinitionCondition) {
	existingCondition := FindCRDCondition(customResourceDefinition, newCondition.Type)
	if existingCondition == nil {
		customResourceDefinition.Status.Conditions = append(customResourceDefinition.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// FindCRDCondition returns the condition you're looking for or nil
func FindCRDCondition(customResourceDefinition *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) *CustomResourceDefinitionCondition {
	for i := range customResourceDefinition.Status.Conditions {
		if customResourceDefinition.Status.Conditions[i].Type == conditionType {
			return &customResourceDefinition.Status.Conditions[i]
		}
	}

	return nil
}

// IsCRDConditionTrue indicates if the condition is present and strictly true
func IsCRDConditionTrue(customResourceDefinition *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(customResourceDefinition, conditionType, ConditionTrue)
}

// IsCRDConditionFalse indicates if the condition is present and false true
func IsCRDConditionFalse(customResourceDefinition *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(customResourceDefinition, conditionType, ConditionFalse)
}

// IsCRDConditionPresentAndEqual indicates if the condition is present and equal to the arg
func IsCRDConditionPresentAndEqual(customResourceDefinition *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType, status ConditionStatus) bool {
	for _, condition := range customResourceDefinition.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status == status
		}
	}
	return false
}

// IsCRDConditionEquivalent returns true if the lhs and rhs are equivalent except for times
func IsCRDConditionEquivalent(lhs, rhs *CustomResourceDefinitionCondition) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}

	return lhs.Message == rhs.Message && lhs.Reason == rhs.Reason && lhs.Status == rhs.Status && lhs.Type == rhs.Type
}
