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
	"strings"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"

	apiservervalidation "k8s.io/apiserver/pkg/apis/authorization/validation"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationinternalv1 "k8s.io/kubernetes/pkg/apis/authorization/v1"
)

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

const (
	authorizationV1      = "authorization.k8s.io/v1"
	authorizationV1beta1 = "authorization.k8s.io/v1beta1"
)

var v1OnlyFieldPaths = []string{
	"spec.authorizationOptions",
	"status.conditionalDecision",
}

// MapV1ToV1beta1ErrorLists makes the cross-version validation equivalence sweep tolerate
// the fields that only exist in authorization.k8s.io/v1. When a v1 error list is compared
// against a v1beta1 one, errors under the v1-only paths are dropped from the v1 side,
// since the corresponding fields do not survive the conversion and v1beta1 therefore has
// nothing to report. Comparisons that do not involve exactly this pair are left alone.
func MapV1ToV1beta1ErrorLists(gvLeft, gvRight string, errListLeft, errListRight field.ErrorList) (field.ErrorList, field.ErrorList) {
	switch {
	case gvLeft == authorizationV1 && gvRight == authorizationV1beta1:
		return dropV1OnlyFieldErrors(errListLeft), errListRight
	case gvLeft == authorizationV1beta1 && gvRight == authorizationV1:
		return errListLeft, dropV1OnlyFieldErrors(errListRight)
	default:
		return errListLeft, errListRight
	}
}

func dropV1OnlyFieldErrors(errs field.ErrorList) field.ErrorList {
	kept := make(field.ErrorList, 0, len(errs))
	for _, err := range errs {
		if !isV1OnlyFieldPath(err.Field) {
			kept = append(kept, err)
		}
	}
	return kept
}

// isV1OnlyFieldPath reports whether path is one of the v1-only paths or nested under one.
// Matching the separators explicitly keeps a sibling such as "status.conditionalDecisions"
// from being swallowed by the "status.conditionalDecision" entry.
func isV1OnlyFieldPath(path string) bool {
	for _, v1Only := range v1OnlyFieldPaths {
		if path == v1Only || strings.HasPrefix(path, v1Only+".") || strings.HasPrefix(path, v1Only+"[") {
			return true
		}
	}
	return false
}
