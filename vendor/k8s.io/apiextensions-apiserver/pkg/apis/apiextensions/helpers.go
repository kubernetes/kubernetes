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

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// SetCRDCondition sets the status condition. It either overwrites the existing one or creates a new one.
func SetCRDCondition(crd *CustomResourceDefinition, newCondition CustomResourceDefinitionCondition) {
	newCondition.LastTransitionTime = metav1.NewTime(time.Now())

	existingCondition := FindCRDCondition(crd, newCondition.Type)
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

// FindCRDCondition returns the condition you're looking for or nil.
func FindCRDCondition(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) *CustomResourceDefinitionCondition {
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

// IsCRDConditionFalse indicates if the condition is present and false.
func IsCRDConditionFalse(crd *CustomResourceDefinition, conditionType CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, ConditionFalse)
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

// IsCRDConditionEquivalent returns true if the lhs and rhs are equivalent except for times.
func IsCRDConditionEquivalent(lhs, rhs *CustomResourceDefinitionCondition) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}

	return lhs.Message == rhs.Message && lhs.Reason == rhs.Reason && lhs.Status == rhs.Status && lhs.Type == rhs.Type
}

// CRDHasFinalizer returns true if the finalizer is in the list.
func CRDHasFinalizer(crd *CustomResourceDefinition, needle string) bool {
	for _, finalizer := range crd.Finalizers {
		if finalizer == needle {
			return true
		}
	}

	return false
}

// CRDRemoveFinalizer removes the finalizer if present.
func CRDRemoveFinalizer(crd *CustomResourceDefinition, needle string) {
	newFinalizers := []string{}
	for _, finalizer := range crd.Finalizers {
		if finalizer != needle {
			newFinalizers = append(newFinalizers, finalizer)
		}
	}
	crd.Finalizers = newFinalizers
}

// HasServedCRDVersion returns true if the given version is in the list of CRD's versions and the Served flag is set.
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

// IsStoredVersion returns whether the given version is the storage version of the CRD.
func IsStoredVersion(crd *CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}

// GetSchemaForVersion returns the validation schema for the given version or nil.
func GetSchemaForVersion(crd *CustomResourceDefinition, version string) (*CustomResourceValidation, error) {
	if !HasPerVersionSchema(crd.Spec.Versions) {
		return crd.Spec.Validation, nil
	}
	if crd.Spec.Validation != nil {
		return nil, fmt.Errorf("malformed CustomResourceDefinition %s version %s: top-level and per-version schemas must be mutual exclusive", crd.Name, version)
	}
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return v.Schema, nil
		}
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinition: %v", version, crd.Name)
}

// GetSubresourcesForVersion returns the subresources for given version or nil.
func GetSubresourcesForVersion(crd *CustomResourceDefinition, version string) (*CustomResourceSubresources, error) {
	if !HasPerVersionSubresources(crd.Spec.Versions) {
		return crd.Spec.Subresources, nil
	}
	if crd.Spec.Subresources != nil {
		return nil, fmt.Errorf("malformed CustomResourceDefinition %s version %s: top-level and per-version subresources must be mutual exclusive", crd.Name, version)
	}
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return v.Subresources, nil
		}
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinition: %v", version, crd.Name)
}

// GetColumnsForVersion returns the columns for given version or nil.
// NOTE: the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func GetColumnsForVersion(crd *CustomResourceDefinition, version string) ([]CustomResourceColumnDefinition, error) {
	if !HasPerVersionColumns(crd.Spec.Versions) {
		return serveDefaultColumnsIfEmpty(crd.Spec.AdditionalPrinterColumns), nil
	}
	if len(crd.Spec.AdditionalPrinterColumns) > 0 {
		return nil, fmt.Errorf("malformed CustomResourceDefinition %s version %s: top-level and per-version additionalPrinterColumns must be mutual exclusive", crd.Name, version)
	}
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return serveDefaultColumnsIfEmpty(v.AdditionalPrinterColumns), nil
		}
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinition: %v", version, crd.Name)
}

// HasPerVersionSchema returns true if a CRD uses per-version schema.
func HasPerVersionSchema(versions []CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Schema != nil {
			return true
		}
	}
	return false
}

// HasPerVersionSubresources returns true if a CRD uses per-version subresources.
func HasPerVersionSubresources(versions []CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Subresources != nil {
			return true
		}
	}
	return false
}

// HasPerVersionColumns returns true if a CRD uses per-version columns.
func HasPerVersionColumns(versions []CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if len(v.AdditionalPrinterColumns) > 0 {
			return true
		}
	}
	return false
}

// serveDefaultColumnsIfEmpty applies logically defaulting to columns, if the input columns is empty.
// NOTE: in this way, the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func serveDefaultColumnsIfEmpty(columns []CustomResourceColumnDefinition) []CustomResourceColumnDefinition {
	if len(columns) > 0 {
		return columns
	}
	return []CustomResourceColumnDefinition{
		{Name: "Age", Type: "date", Description: swaggerMetadataDescriptions["creationTimestamp"], JSONPath: ".metadata.creationTimestamp"},
	}
}

// HasVersionServed returns true if given CRD has given version served.
func HasVersionServed(crd *CustomResourceDefinition, version string) bool {
	for _, v := range crd.Spec.Versions {
		if !v.Served || v.Name != version {
			continue
		}
		return true
	}
	return false
}
