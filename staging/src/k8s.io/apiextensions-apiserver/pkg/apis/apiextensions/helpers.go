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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SetCRDCondition sets the status condition.  It either overwrites the existing one or
// creates a new one
func SetCRDCondition(crd *CustomResourceDefinition, newCondition CustomResourceDefinitionCondition) {
	existingCondition := FindCRDCondition(crd, newCondition.Type)
	if existingCondition == nil {
		newCondition.LastTransitionTime = metav1.NewTime(time.Now())
		crd.Status.Conditions = append(crd.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// RemoveCRDCondition removes the status condition.
func RemoveCRDCondition(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) {
	newConditions := []CustomResourceDefinitionCondition{}
	for _, condition := range crd.Status.Conditions {
		if condition.Type != conditionType {
			newConditions = append(newConditions, condition)
		}
	}
	crd.Status.Conditions = newConditions
}

// FindCRDCondition returns the condition you're looking for or nil
func FindCRDCondition(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) *CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

// IsCRDConditionTrue indicates if the condition is present and strictly true
func IsCRDConditionTrue(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, ConditionTrue)
}

// IsCRDConditionFalse indicates if the condition is present and false true
func IsCRDConditionFalse(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, ConditionFalse)
}

// IsCRDConditionPresentAndEqual indicates if the condition is present and equal to the arg
func IsCRDConditionPresentAndEqual(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType, status ConditionStatus) bool {
	for _, condition := range crd.Status.Conditions {
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

// CRDHasFinalizer returns true if the finalizer is in the list
func CRDHasFinalizer(crd *CustomResourceDefinition, needle string) bool {
	for _, finalizer := range crd.Finalizers {
		if finalizer == needle {
			return true
		}
	}

	return false
}

// CRDRemoveFinalizer removes the finalizer if present
func CRDRemoveFinalizer(crd *CustomResourceDefinition, needle string) {
	newFinalizers := []string{}
	for _, finalizer := range crd.Finalizers {
		if finalizer != needle {
			newFinalizers = append(newFinalizers, finalizer)
		}
	}
	crd.Finalizers = newFinalizers
}

// HasServedCRDVersion returns true if `version` is in the list of CRD's versions and the Served flag is set.
func HasServedCRDVersion(crd *CustomResourceDefinition, version string) bool {
	for _, v := range crd.Spec.Versions {
		if v.Name == version {
			return v.Served
		}
	}
	return false
}

// GetCRDStorageVersion returns the storage version for given CRD.
func GetCRDStorageVersion(crd *CustomResourceDefinition) (string, error) {
	for _, v := range crd.Spec.Versions {
		if v.Storage {
			return v.Name, nil
		}
	}
	// This should not happened if crd is valid
	return "", fmt.Errorf("invalid CustomResourceDefinition, no storage version")
}

func IsStoredVersion(crd *CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}
