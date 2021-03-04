/*
Copyright 2019 The Kubernetes Authors.

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

package apihelpers

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// IsProtectedCommunityGroup returns whether or not a group specified for a CRD is protected for the community and needs
// to have the v1beta1.KubeAPIApprovalAnnotation set.
func IsProtectedCommunityGroup(group string) bool {
	switch {
	case group == "k8s.io" || strings.HasSuffix(group, ".k8s.io"):
		return true
	case group == "kubernetes.io" || strings.HasSuffix(group, ".kubernetes.io"):
		return true
	default:
		return false
	}

}

// APIApprovalState covers the various options for API approval annotation states
type APIApprovalState int

const (
	// APIApprovalInvalid means the annotation doesn't have an expected value
	APIApprovalInvalid APIApprovalState = iota
	// APIApproved if the annotation has a URL (this means the API is approved)
	APIApproved
	// APIApprovalBypassed if the annotation starts with "unapproved" indicating that for whatever reason the API isn't approved, but we should allow its creation
	APIApprovalBypassed
	// APIApprovalMissing means the annotation is empty
	APIApprovalMissing
)

// GetAPIApprovalState returns the state of the API approval and reason for that state
func GetAPIApprovalState(annotations map[string]string) (state APIApprovalState, reason string) {
	annotation := annotations[apiextensionsv1.KubeAPIApprovedAnnotation]

	// we use the result of this parsing in the switch/case below
	url, annotationURLParseErr := url.ParseRequestURI(annotation)
	switch {
	case len(annotation) == 0:
		return APIApprovalMissing, fmt.Sprintf("protected groups must have approval annotation %q, see https://github.com/kubernetes/enhancements/pull/1111", apiextensionsv1.KubeAPIApprovedAnnotation)
	case strings.HasPrefix(annotation, "unapproved"):
		return APIApprovalBypassed, fmt.Sprintf("not approved: %q", annotation)
	case annotationURLParseErr == nil && url != nil && len(url.Host) > 0 && len(url.Scheme) > 0:
		return APIApproved, fmt.Sprintf("approved in %v", annotation)
	default:
		return APIApprovalInvalid, fmt.Sprintf("protected groups must have approval annotation %q with either a URL or a reason starting with \"unapproved\", see https://github.com/kubernetes/enhancements/pull/1111", apiextensionsv1.KubeAPIApprovedAnnotation)
	}
}

// SetCRDCondition sets the status condition. It either overwrites the existing one or creates a new one.
func SetCRDCondition(crd *apiextensionsv1.CustomResourceDefinition, newCondition apiextensionsv1.CustomResourceDefinitionCondition) {
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
func RemoveCRDCondition(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType) {
	newConditions := []apiextensionsv1.CustomResourceDefinitionCondition{}
	for _, condition := range crd.Status.Conditions {
		if condition.Type != conditionType {
			newConditions = append(newConditions, condition)
		}
	}
	crd.Status.Conditions = newConditions
}

// FindCRDCondition returns the condition you're looking for or nil.
func FindCRDCondition(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType) *apiextensionsv1.CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

// IsCRDConditionTrue indicates if the condition is present and strictly true.
func IsCRDConditionTrue(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, apiextensionsv1.ConditionTrue)
}

// IsCRDConditionFalse indicates if the condition is present and false.
func IsCRDConditionFalse(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType) bool {
	return IsCRDConditionPresentAndEqual(crd, conditionType, apiextensionsv1.ConditionFalse)
}

// IsCRDConditionPresentAndEqual indicates if the condition is present and equal to the given status.
func IsCRDConditionPresentAndEqual(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType, status apiextensionsv1.ConditionStatus) bool {
	for _, condition := range crd.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status == status
		}
	}
	return false
}

// IsCRDConditionEquivalent returns true if the lhs and rhs are equivalent except for times.
func IsCRDConditionEquivalent(lhs, rhs *apiextensionsv1.CustomResourceDefinitionCondition) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}

	return lhs.Message == rhs.Message && lhs.Reason == rhs.Reason && lhs.Status == rhs.Status && lhs.Type == rhs.Type
}

// CRDHasFinalizer returns true if the finalizer is in the list.
func CRDHasFinalizer(crd *apiextensionsv1.CustomResourceDefinition, needle string) bool {
	for _, finalizer := range crd.Finalizers {
		if finalizer == needle {
			return true
		}
	}

	return false
}

// CRDRemoveFinalizer removes the finalizer if present.
func CRDRemoveFinalizer(crd *apiextensionsv1.CustomResourceDefinition, needle string) {
	newFinalizers := []string{}
	for _, finalizer := range crd.Finalizers {
		if finalizer != needle {
			newFinalizers = append(newFinalizers, finalizer)
		}
	}
	crd.Finalizers = newFinalizers
}

// HasServedCRDVersion returns true if the given version is in the list of CRD's versions and the Served flag is set.
func HasServedCRDVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) bool {
	for _, v := range crd.Spec.Versions {
		if v.Name == version {
			return v.Served
		}
	}
	return false
}

// GetCRDStorageVersion returns the storage version for given CRD.
func GetCRDStorageVersion(crd *apiextensionsv1.CustomResourceDefinition) (string, error) {
	for _, v := range crd.Spec.Versions {
		if v.Storage {
			return v.Name, nil
		}
	}
	// This should not happened if crd is valid
	return "", fmt.Errorf("invalid apiextensionsv1.CustomResourceDefinition, no storage version")
}

// IsStoredVersion returns whether the given version is the storage version of the CRD.
func IsStoredVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}

// GetSchemaForVersion returns the validation schema for the given version or nil.
func GetSchemaForVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) (*apiextensionsv1.CustomResourceValidation, error) {
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return v.Schema, nil
		}
	}
	return nil, fmt.Errorf("version %s not found in apiextensionsv1.CustomResourceDefinition: %v", version, crd.Name)
}

// GetSubresourcesForVersion returns the subresources for given version or nil.
func GetSubresourcesForVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) (*apiextensionsv1.CustomResourceSubresources, error) {
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return v.Subresources, nil
		}
	}
	return nil, fmt.Errorf("version %s not found in apiextensionsv1.CustomResourceDefinition: %v", version, crd.Name)
}

// HasPerVersionSchema returns true if a CRD uses per-version schema.
func HasPerVersionSchema(versions []apiextensionsv1.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Schema != nil {
			return true
		}
	}
	return false
}

// HasPerVersionSubresources returns true if a CRD uses per-version subresources.
func HasPerVersionSubresources(versions []apiextensionsv1.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Subresources != nil {
			return true
		}
	}
	return false
}

// HasPerVersionColumns returns true if a CRD uses per-version columns.
func HasPerVersionColumns(versions []apiextensionsv1.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if len(v.AdditionalPrinterColumns) > 0 {
			return true
		}
	}
	return false
}

// HasVersionServed returns true if given CRD has given version served.
func HasVersionServed(crd *apiextensionsv1.CustomResourceDefinition, version string) bool {
	for _, v := range crd.Spec.Versions {
		if !v.Served || v.Name != version {
			continue
		}
		return true
	}
	return false
}
