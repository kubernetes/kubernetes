/*
Copyright 2017 The Kubernetes Authors.

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

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// SetCRDCondition sets the status condition. It either overwrites the existing one or creates a new one.
func SetCRDCondition(crd *CustomResourceDefinition, newCondition CustomResourceDefinitionCondition) {
	newCondition.LastTransitionTime = metav1.NewTime(time.Now())

	existingCondition := findCRDCondition(crd, newCondition.Type)
	if existingCondition == nil {
		crd.Status.Conditions = append(crd.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status || existingCondition.LastTransitionTime.IsZero() {
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Status = newCondition.Status
	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// findCRDCondition returns the condition you're looking for or nil.
func findCRDCondition(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) *CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

// IsCRDConditionTrue indicates if the condition is present and strictly true.
func IsCRDConditionTrue(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, ConditionTrue)
}

// IsCRDConditionPresentAndEqual indicates if the condition is present and equal to the given status.
func IsCRDConditionPresentAndEqual(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType, status ConditionStatus) bool {
	for _, condition := range crd.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status == status
		}
	}
	return false
}

// IsStoredVersion returns whether the given version is the storage version of the CRD.
func IsStoredVersion(crd *CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}
