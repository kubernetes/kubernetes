/*
Copyright The Kubernetes Authors.

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

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"

	apiservervalidation "k8s.io/apiserver/pkg/apis/authorization/validation"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

// ValidateSubjectAccessReviewCreate is the single composition of handwritten and declarative
// SubjectAccessReview validation.
func ValidateSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.SubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	sarV1 := &authorizationv1.SubjectAccessReview{}
	if err := scheme.Convert(sar, sarV1, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal SubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateSubjectAccessReview(sarV1)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, sarValidationConfig())
}

// ValidateSelfSubjectAccessReviewCreate is the single composition of handwritten and declarative
// SelfSubjectAccessReview validation.
func ValidateSelfSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.SelfSubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	sarV1 := &authorizationv1.SelfSubjectAccessReview{}
	if err := scheme.Convert(sar, sarV1, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal SelfSubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateSelfSubjectAccessReview(sarV1)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, sarValidationConfig())
}

// ValidateLocalSubjectAccessReviewCreate is the single composition of handwritten and declarative
// LocalSubjectAccessReview validation.
func ValidateLocalSubjectAccessReviewCreate(ctx context.Context, scheme *runtime.Scheme, sar *authorizationapi.LocalSubjectAccessReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	sarV1 := &authorizationv1.LocalSubjectAccessReview{}
	if err := scheme.Convert(sar, sarV1, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal LocalSubjectAccessReview to v1: %w", err))}
	}

	errs := apiservervalidation.ValidateLocalSubjectAccessReview(sarV1)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, sar, nil, errs, operation.Create, sarValidationConfig())
}

// ValidateAuthorizationConditionsReviewCreate is the single composition of handwritten and declarative
// AuthorizationConditionsReview validation.
func ValidateAuthorizationConditionsReviewCreate(ctx context.Context, scheme *runtime.Scheme, acr *authorizationapi.AuthorizationConditionsReview) field.ErrorList {
	// The hand-written validations are written only once, for the most recent external API version, so that also k8s.io/apiserver
	// importers can make use of the validations.
	acrV1 := &authorizationv1alpha1.AuthorizationConditionsReview{}
	if err := scheme.Convert(acr, acrV1, nil); err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected, could not convert internal AuthorizationConditionsReview to v1alpha1: %w", err))}
	}

	errs := apiservervalidation.ValidateAuthorizationConditionsReview(acrV1)
	dv := rest.DeclarativeValidation{Scheme: scheme}
	return dv.ValidateDeclaratively(ctx, acr, nil, errs, operation.Create, sarValidationConfig())
}

// sarValidationConfig returns the declarative validation config to use for
// SubjectAccessReview-family create validation. It enables the
// "ConditionalAuthorization" option when the corresponding feature gate is
// enabled, so that the +k8s:ifDisabled("ConditionalAuthorization")=+k8s:forbidden
// tag on spec.conditionalAuthorization does not reject the field.
func sarValidationConfig() rest.DeclarativeValidationConfig {
	return rest.DeclarativeValidationConfig{
		Options: map[string]bool{
			string(genericfeatures.ConditionalAuthorization): utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization),
		},
	}
}
