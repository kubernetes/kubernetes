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
	"context"
	"fmt"
	"slices"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"

	apiservervalidation "k8s.io/apiserver/pkg/apis/authorization/validation"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationinternalv1 "k8s.io/kubernetes/pkg/apis/authorization/v1"
)

var omittedV1beta1SARPaths = []string{
	"spec.authorizationOptions",
	"status.conditionalDecision",
}

func OmittedFieldPaths() map[schema.GroupVersionKind][]string {
	return map[schema.GroupVersionKind][]string{
		authorizationv1beta1.SchemeGroupVersion.WithKind("SubjectAccessReview"):      slices.Clone(omittedV1beta1SARPaths),
		authorizationv1beta1.SchemeGroupVersion.WithKind("SelfSubjectAccessReview"):  slices.Clone(omittedV1beta1SARPaths),
		authorizationv1beta1.SchemeGroupVersion.WithKind("LocalSubjectAccessReview"): slices.Clone(omittedV1beta1SARPaths),
	}
}

// ValidateSubjectAccessReviewCreate is the single composition of handwritten and declarative
// SubjectAccessReview validation.
func ValidateSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.SubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	versionedSAR := &authorizationv1.SubjectAccessReview{}

	// Call the conversion function directly, as we know it exactly. It is known to be fast as the internal package and v1 is byte-identical.
	// conversion.Scope is known to be unused in this specific case and thus left nil. We know it in practice never errors.
	if err := authorizationinternalv1.Convert_authorization_SubjectAccessReview_To_v1_SubjectAccessReview(sar, versionedSAR, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal SubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateSubjectAccessReview(versionedSAR)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, apiservervalidation.DeclarativeValidationConfig())
}

// ValidateSelfSubjectAccessReviewCreate is the single composition of handwritten and declarative
// SelfSubjectAccessReview validation.
func ValidateSelfSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.SelfSubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	versionedSAR := &authorizationv1.SelfSubjectAccessReview{}

	// Call the conversion function directly, as we know it exactly. It is known to be fast as the internal package and v1 is byte-identical.
	// conversion.Scope is known to be unused in this specific case and thus left nil. We know it in practice never errors.
	if err := authorizationinternalv1.Convert_authorization_SelfSubjectAccessReview_To_v1_SelfSubjectAccessReview(sar, versionedSAR, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal SelfSubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateSelfSubjectAccessReview(versionedSAR)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, apiservervalidation.DeclarativeValidationConfig())
}

// ValidateLocalSubjectAccessReviewCreate is the single composition of handwritten and declarative
// LocalSubjectAccessReview validation.
func ValidateLocalSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.LocalSubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	versionedSAR := &authorizationv1.LocalSubjectAccessReview{}

	// Call the conversion function directly, as we know it exactly. It is known to be fast as the internal package and v1 is byte-identical.
	// conversion.Scope is known to be unused in this specific case and thus left nil. We know it in practice never errors.
	if err := authorizationinternalv1.Convert_authorization_LocalSubjectAccessReview_To_v1_LocalSubjectAccessReview(sar, versionedSAR, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal LocalSubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateLocalSubjectAccessReview(versionedSAR)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, apiservervalidation.DeclarativeValidationConfig())
}

// ValidateAuthorizationConditionsReviewCreate is the single composition of handwritten and declarative
// AuthorizationConditionsReview validation.
func ValidateAuthorizationConditionsReviewCreate(ctx context.Context, scheme *runtime.Scheme, acr *authorizationapi.AuthorizationConditionsReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	versionedACR := &authorizationv1alpha1.AuthorizationConditionsReview{}
	if err := scheme.Convert(acr, versionedACR, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal AuthorizationConditionsReview to v1alpha1: %w", err))}
	}

	errs := apiservervalidation.ValidateAuthorizationConditionsReview(versionedACR)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, acr, nil, errs, operation.Create, apiservervalidation.DeclarativeValidationConfig())
}
