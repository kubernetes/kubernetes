/*
Copyright 2015 The Kubernetes Authors.

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

package validation

import (
	"fmt"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

// ValidateSubjectAccessReviewSpec validates a SubjectAccessReviewSpec and returns an
// ErrorList with any errors.
func ValidateSubjectAccessReviewSpec(spec authorizationapi.SubjectAccessReviewSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.ResourceAttributes != nil && spec.NonResourceAttributes != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceAttributes"), spec.NonResourceAttributes, `cannot be specified in combination with resourceAttributes`))
	}
	if spec.ResourceAttributes == nil && spec.NonResourceAttributes == nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceAttributes"), spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`))
	}
	if len(spec.User) == 0 && len(spec.Groups) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("user"), spec.User, `at least one of user or group must be specified`))
	}
	allErrs = append(allErrs, validateResourceAttributes(spec.ResourceAttributes, field.NewPath("spec.resourceAttributes"))...)

	return allErrs
}

// ValidateSelfSubjectAccessReviewSpec validates a SelfSubjectAccessReviewSpec and returns an
// ErrorList with any errors.
func ValidateSelfSubjectAccessReviewSpec(spec authorizationapi.SelfSubjectAccessReviewSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.ResourceAttributes != nil && spec.NonResourceAttributes != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceAttributes"), spec.NonResourceAttributes, `cannot be specified in combination with resourceAttributes`))
	}
	if spec.ResourceAttributes == nil && spec.NonResourceAttributes == nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceAttributes"), spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`))
	}
	allErrs = append(allErrs, validateResourceAttributes(spec.ResourceAttributes, field.NewPath("spec.resourceAttributes"))...)

	return allErrs
}

// ValidateSubjectAccessReview validates a SubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateSubjectAccessReview(sar *authorizationapi.SubjectAccessReview) field.ErrorList {
	allErrs := ValidateSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))
	objectMetaShallowCopy := sar.ObjectMeta
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), sar.ObjectMeta, `must be empty`))
	}
	return allErrs
}

// ValidateSelfSubjectAccessReview validates a SelfSubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateSelfSubjectAccessReview(sar *authorizationapi.SelfSubjectAccessReview) field.ErrorList {
	allErrs := ValidateSelfSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))
	objectMetaShallowCopy := sar.ObjectMeta
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), sar.ObjectMeta, `must be empty`))
	}
	return allErrs
}

// ValidateLocalSubjectAccessReview validates a LocalSubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateLocalSubjectAccessReview(sar *authorizationapi.LocalSubjectAccessReview) field.ErrorList {
	allErrs := ValidateSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))

	objectMetaShallowCopy := sar.ObjectMeta
	objectMetaShallowCopy.Namespace = ""
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), sar.ObjectMeta, `must be empty except for namespace`))
	}

	if sar.Spec.ResourceAttributes != nil && sar.Spec.ResourceAttributes.Namespace != sar.Namespace {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec.resourceAttributes.namespace"), sar.Spec.ResourceAttributes.Namespace, `must match metadata.namespace`))
	}
	if sar.Spec.NonResourceAttributes != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec.nonResourceAttributes"), sar.Spec.NonResourceAttributes, `disallowed on this kind of request`))
	}

	return allErrs
}

func validateResourceAttributes(resourceAttributes *authorizationapi.ResourceAttributes, fldPath *field.Path) field.ErrorList {
	if resourceAttributes == nil {
		return nil
	}
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validateFieldSelectorAttributes(resourceAttributes.FieldSelector, fldPath.Child("fieldSelector"))...)
	allErrs = append(allErrs, validateLabelSelectorAttributes(resourceAttributes.LabelSelector, fldPath.Child("labelSelector"))...)

	return allErrs
}

func validateFieldSelectorAttributes(selector *authorizationapi.FieldSelectorAttributes, fldPath *field.Path) field.ErrorList {
	if selector == nil {
		return nil
	}
	allErrs := field.ErrorList{}

	if len(selector.RawSelector) > 0 && len(selector.Requirements) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("rawSelector"), selector.RawSelector, "may not specified at the same time as requirements"))
	}
	if len(selector.RawSelector) == 0 && len(selector.Requirements) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("requirements"), fmt.Sprintf("when %s is specified, requirements or rawSelector is required", fldPath)))
	}

	// AllowUnknownOperatorInRequirement enables *SubjectAccessReview requests from newer skewed clients which understand operators kube-apiserver does not know about to be authorized.
	validationOptions := metav1validation.FieldSelectorValidationOptions{AllowUnknownOperatorInRequirement: true}
	for i, requirement := range selector.Requirements {
		allErrs = append(allErrs, metav1validation.ValidateFieldSelectorRequirement(requirement, validationOptions, fldPath.Child("requirements").Index(i))...)
	}

	return allErrs
}

func validateLabelSelectorAttributes(selector *authorizationapi.LabelSelectorAttributes, fldPath *field.Path) field.ErrorList {
	if selector == nil {
		return nil
	}
	allErrs := field.ErrorList{}

	if len(selector.RawSelector) > 0 && len(selector.Requirements) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("rawSelector"), selector.RawSelector, "may not specified at the same time as requirements"))
	}
	if len(selector.RawSelector) == 0 && len(selector.Requirements) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("requirements"), fmt.Sprintf("when %s is specified, requirements or rawSelector is required", fldPath)))
	}

	// AllowUnknownOperatorInRequirement enables *SubjectAccessReview requests from newer skewed clients which understand operators kube-apiserver does not know about to be authorized.
	validationOptions := metav1validation.LabelSelectorValidationOptions{AllowUnknownOperatorInRequirement: true}
	for i, requirement := range selector.Requirements {
		allErrs = append(allErrs, metav1validation.ValidateLabelSelectorRequirement(requirement, validationOptions, fldPath.Child("requirements").Index(i))...)
	}

	return allErrs
}
