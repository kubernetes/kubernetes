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

package helpers

import (
	"fmt"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SwaggerMetadataDescriptions is a swagger doc for object meta information
var SwaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// SetCRDCondition sets the status condition.  It either overwrites the existing one or
// creates a new one
func SetCRDCondition(crd *apiextensions.CustomResourceDefinition, newCondition apiextensions.CustomResourceDefinitionCondition) {
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
func RemoveCRDCondition(crd *apiextensions.CustomResourceDefinition, conditionType apiextensions.CustomResourceDefinitionConditionType) {
	newConditions := []apiextensions.CustomResourceDefinitionCondition{}
	for _, condition := range crd.Status.Conditions {
		if condition.Type != conditionType {
			newConditions = append(newConditions, condition)
		}
	}
	crd.Status.Conditions = newConditions
}

// FindCRDCondition returns the condition you're looking for or nil
func FindCRDCondition(crd *apiextensions.CustomResourceDefinition, conditionType apiextensions.CustomResourceDefinitionConditionType) *apiextensions.CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

// IsCRDConditionTrue indicates if the condition is present and strictly true
func IsCRDConditionTrue(crd *apiextensions.CustomResourceDefinition, conditionType apiextensions.CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, apiextensions.ConditionTrue)
}

// IsCRDConditionFalse indicates if the condition is present and false true
func IsCRDConditionFalse(crd *apiextensions.CustomResourceDefinition, conditionType apiextensions.CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, apiextensions.ConditionFalse)
}

// IsCRDConditionPresentAndEqual indicates if the condition is present and equal to the arg
func IsCRDConditionPresentAndEqual(crd *apiextensions.CustomResourceDefinition, conditionType apiextensions.CustomResourceDefinitionConditionType, status apiextensions.ConditionStatus) bool {
	for _, condition := range crd.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status == status
		}
	}
	return false
}

// IsCRDConditionEquivalent returns true if the lhs and rhs are equivalent except for times
func IsCRDConditionEquivalent(lhs, rhs *apiextensions.CustomResourceDefinitionCondition) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}

	return lhs.Message == rhs.Message && lhs.Reason == rhs.Reason && lhs.Status == rhs.Status && lhs.Type == rhs.Type
}

// CRDHasFinalizer returns true if the finalizer is in the list
func CRDHasFinalizer(crd *apiextensions.CustomResourceDefinition, needle string) bool {
	for _, finalizer := range crd.Finalizers {
		if finalizer == needle {
			return true
		}
	}

	return false
}

// CRDRemoveFinalizer removes the finalizer if present
func CRDRemoveFinalizer(crd *apiextensions.CustomResourceDefinition, needle string) {
	newFinalizers := []string{}
	for _, finalizer := range crd.Finalizers {
		if finalizer != needle {
			newFinalizers = append(newFinalizers, finalizer)
		}
	}
	crd.Finalizers = newFinalizers
}

// IsStoredVersion returns true if the version matches CRD spec's storage version
func IsStoredVersion(crd *apiextensions.CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}

// GetSchemaForVersion returns the validation schema for given version in given CRD.
func GetSchemaForVersion(crd *apiextensions.CustomResourceDefinition, version string) (*apiextensions.CustomResourceValidation, error) {
	if !hasPerVersionSchema(crd.Spec.Versions) {
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

// GetSubresourcesForVersion returns the subresources for given version in given CRD.
func GetSubresourcesForVersion(crd *apiextensions.CustomResourceDefinition, version string) (*apiextensions.CustomResourceSubresources, error) {
	if !hasPerVersionSubresources(crd.Spec.Versions) {
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

// GetColumnsForVersion returns the columns for given version in given CRD.
// NOTE: the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func GetColumnsForVersion(crd *apiextensions.CustomResourceDefinition, version string) ([]apiextensions.CustomResourceColumnDefinition, error) {
	if !hasPerVersionColumns(crd.Spec.Versions) {
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

// serveDefaultColumnsIfEmpty applies logically defaulting to columns, if the input columns is empty.
// NOTE: in this way, the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func serveDefaultColumnsIfEmpty(columns []apiextensions.CustomResourceColumnDefinition) []apiextensions.CustomResourceColumnDefinition {
	if len(columns) > 0 {
		return columns
	}
	return []apiextensions.CustomResourceColumnDefinition{
		{Name: "Age", Type: "date", Description: SwaggerMetadataDescriptions["creationTimestamp"], JSONPath: ".metadata.creationTimestamp"},
	}
}

// hasPerVersionSchema returns true if a CRD uses per-version schema.
func hasPerVersionSchema(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Schema != nil {
			return true
		}
	}
	return false
}

// hasPerVersionSubresources returns true if a CRD uses per-version subresources.
func hasPerVersionSubresources(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Subresources != nil {
			return true
		}
	}
	return false
}

// hasPerVersionColumns returns true if a CRD uses per-version columns.
func hasPerVersionColumns(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if len(v.AdditionalPrinterColumns) > 0 {
			return true
		}
	}
	return false
}
