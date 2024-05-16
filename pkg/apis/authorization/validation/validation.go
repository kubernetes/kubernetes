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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
