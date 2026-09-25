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

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// ValidateSubjectAccessReviewSpec validates a SubjectAccessReviewSpec and returns an
// ErrorList with any errors.
func ValidateSubjectAccessReviewSpec(spec authorizationv1.SubjectAccessReviewSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.ResourceAttributes != nil && spec.NonResourceAttributes != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`).WithOrigin("union").MarkCoveredByDeclarative())
	}
	if spec.ResourceAttributes == nil && spec.NonResourceAttributes == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`).WithOrigin("union").MarkCoveredByDeclarative())
	}
	if len(spec.User) == 0 && len(spec.Groups) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("user"), spec.User, `at least one of user or group must be specified`))
	}
	allErrs = append(allErrs, validateResourceAttributes(spec.ResourceAttributes, field.NewPath("spec.resourceAttributes"))...)

	if spec.AuthorizationOptions != nil {
		allErrs = append(allErrs, validateAuthorizationOptions(spec.AuthorizationOptions, fldPath.Child("authorizationOptions"))...)
	}
	return allErrs
}

// ValidateSelfSubjectAccessReviewSpec validates a SelfSubjectAccessReviewSpec and returns an
// ErrorList with any errors.
func ValidateSelfSubjectAccessReviewSpec(spec authorizationv1.SelfSubjectAccessReviewSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.ResourceAttributes != nil && spec.NonResourceAttributes != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`).WithOrigin("union").MarkCoveredByDeclarative())
	}
	if spec.ResourceAttributes == nil && spec.NonResourceAttributes == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.NonResourceAttributes, `exactly one of nonResourceAttributes or resourceAttributes must be specified`).WithOrigin("union").MarkCoveredByDeclarative())
	}
	allErrs = append(allErrs, validateResourceAttributes(spec.ResourceAttributes, field.NewPath("spec.resourceAttributes"))...)

	if spec.AuthorizationOptions != nil {
		allErrs = append(allErrs, validateAuthorizationOptions(spec.AuthorizationOptions, fldPath.Child("authorizationOptions"))...)
	}
	return allErrs
}

// validateAuthorizationOptions validates a AuthorizationOptions and returns an
// ErrorList with any errors.
func validateAuthorizationOptions(ao *authorizationv1.AuthorizationOptions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// Only run the validation for HandledDecisionTypes when it is set, declarative validation already covers the "handledDecisionTypes is required case"
	if 0 < len(ao.HandledDecisionTypes) && len(ao.HandledDecisionTypes) <= 32 {
		if !sets.New(ao.HandledDecisionTypes...).IsSuperset(authorizationv1.UnconditionalAuthorizationDecisionTypes()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("handledDecisionTypes"), ao.HandledDecisionTypes, "set must at least contain {Allow, Deny, NoOpinion}"))
		}
	}

	return allErrs
}

// ValidateSubjectAccessReviewStatus validates a SubjectAccessReviewSpec and returns an
// ErrorList with any errors.
func ValidateSubjectAccessReviewStatus(status authorizationv1.SubjectAccessReviewStatus, clientIsConditionsAware bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if status.Allowed && status.Denied {
		allErrs = append(allErrs, field.Invalid(fldPath, authorizationv1.SubjectAccessReviewStatus{Allowed: status.Allowed, Denied: status.Denied},
			"allowed and denied are mutually exclusive"))
	}

	if status.ConditionalDecision != nil {

		// If status.ConditionalDecision is set, but the client did _not_ opt-into conditions, the server did not respect the wish of the client to be unconditional-only
		if !clientIsConditionsAware {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("conditionalDecision"), "can only be set when the client opted into conditions-awareness"))
		}

		switch status.ConditionalDecision.Type {
		// Avoid confusion; don't make it possible to specify [Allow, Deny, NoOpinion] top-level using status.conditionalDecision
		case authorizationv1.ConditionsAwareDecisionTypeDeny, authorizationv1.ConditionsAwareDecisionTypeAllow, authorizationv1.ConditionsAwareDecisionTypeNoOpinion:
			allErrs = append(allErrs, field.Invalid(fldPath.Child("conditionalDecision", "type"), status.ConditionalDecision.Type,
				"cannot be one of [Allow, Deny, NoOpinion], these decisions must be expressed using status.allowed and status.denied"))

		// Enforce status.allowed=false && status.denied=false for a conditional decision
		case authorizationv1.ConditionsAwareDecisionTypeConditionsMap, authorizationv1.ConditionsAwareDecisionTypeUnion:
			if status.Allowed {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("allowed"),
					fmt.Sprintf("must be false when status.conditionalDecision.type=%s", status.ConditionalDecision.Type)))
			}
			if status.Denied {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("denied"),
					fmt.Sprintf("must be false when status.conditionalDecision.type=%s", status.ConditionalDecision.Type)))
			}
			if len(status.EvaluationError) != 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("evaluationError"),
					fmt.Sprintf("must be empty when status.conditionalDecision.type=%s", status.ConditionalDecision.Type)))
			}
			if len(status.Reason) != 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("reason"),
					fmt.Sprintf("must be empty when status.conditionalDecision.type=%s", status.ConditionalDecision.Type)))
			}
			// unrecognized modes are covered by declarative validation
		}
	}

	return allErrs
}

// ValidateSubjectAccessReview validates a SubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateSubjectAccessReview(sar *authorizationv1.SubjectAccessReview) field.ErrorList {
	allErrs := ValidateSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))
	allErrs = append(allErrs, ValidateSubjectAccessReviewStatus(sar.Status, sar.Spec.AuthorizationOptions.SupportsConditionalAuthorization(), field.NewPath("status"))...)

	objectMetaShallowCopy := sar.ObjectMeta
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), sar.ObjectMeta, `must be empty`))
	}
	return allErrs
}

// ValidateSelfSubjectAccessReview validates a SelfSubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateSelfSubjectAccessReview(sar *authorizationv1.SelfSubjectAccessReview) field.ErrorList {
	allErrs := ValidateSelfSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))
	allErrs = append(allErrs, ValidateSubjectAccessReviewStatus(sar.Status, sar.Spec.AuthorizationOptions.SupportsConditionalAuthorization(), field.NewPath("status"))...)
	objectMetaShallowCopy := sar.ObjectMeta
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), sar.ObjectMeta, `must be empty`))
	}
	return allErrs
}

// ValidateLocalSubjectAccessReview validates a LocalSubjectAccessReview and returns an
// ErrorList with any errors.
func ValidateLocalSubjectAccessReview(sar *authorizationv1.LocalSubjectAccessReview) field.ErrorList {
	allErrs := ValidateSubjectAccessReviewSpec(sar.Spec, field.NewPath("spec"))
	allErrs = append(allErrs, ValidateSubjectAccessReviewStatus(sar.Status, sar.Spec.AuthorizationOptions.SupportsConditionalAuthorization(), field.NewPath("status"))...)

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

func validateResourceAttributes(resourceAttributes *authorizationv1.ResourceAttributes, fldPath *field.Path) field.ErrorList {
	if resourceAttributes == nil {
		return nil
	}
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validateFieldSelectorAttributes(resourceAttributes.FieldSelector, fldPath.Child("fieldSelector"))...)
	allErrs = append(allErrs, validateLabelSelectorAttributes(resourceAttributes.LabelSelector, fldPath.Child("labelSelector"))...)

	return allErrs
}

func validateFieldSelectorAttributes(selector *authorizationv1.FieldSelectorAttributes, fldPath *field.Path) field.ErrorList {
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

func validateLabelSelectorAttributes(selector *authorizationv1.LabelSelectorAttributes, fldPath *field.Path) field.ErrorList {
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

// ValidateAuthorizationConditionsReview validates a AuthorizationConditionsReview and returns an
// ErrorList with any errors.
func ValidateAuthorizationConditionsReview(acr *authorizationv1alpha1.AuthorizationConditionsReview) field.ErrorList {
	allErrs := field.ErrorList{}
	if acr.Request != nil {
		allErrs = append(allErrs, ValidateAuthorizationConditionsRequest(acr.Request, field.NewPath("request"))...)
	}
	if acr.Response != nil {
		allErrs = append(allErrs, ValidateAuthorizationConditionsResponse(acr.Response, field.NewPath("response"))...)
	}

	objectMetaShallowCopy := acr.ObjectMeta
	objectMetaShallowCopy.ManagedFields = nil
	if !apiequality.Semantic.DeepEqual(metav1.ObjectMeta{}, objectMetaShallowCopy) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata"), acr.ObjectMeta, `must be empty`))
	}
	return allErrs
}

// ValidateAuthorizationConditionsRequest validates a AuthorizationConditionsRequest and returns an
// ErrorList with any errors.
func ValidateAuthorizationConditionsRequest(req *authorizationv1alpha1.AuthorizationConditionsRequest, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	// conditionalDecision does not need any handwritten validation.
	// That a ConditionsMap has between 1 and 128 conditions is enforced by authorizer.ConditionsAwareDecisionConditionsMap(...)

	// Note: One could consider validating request.admissionRequest here, either declaratively or manually.
	// However, the original AdmissionRequest does not have any validation.
	return allErrs
}

// ValidateAuthorizationConditionsResponse validates a AuthorizationConditionsResponse and returns an
// ErrorList with any errors.
func ValidateAuthorizationConditionsResponse(resp *authorizationv1alpha1.AuthorizationConditionsResponse, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Declarative validation covers type being required, only validate if set
	if len(resp.Decision.Type) != 0 {
		switch resp.Decision.Type {
		case authorizationv1.ConditionsAwareDecisionTypeDeny,
			authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
			authorizationv1.ConditionsAwareDecisionTypeAllow:
			// ok
		default:
			allErrs = append(allErrs, field.Invalid(fldPath.Child("decision", "type"), resp.Decision.Type, "currently must evaluate to an unconditional decision"))
		}
	}

	return allErrs
}

// GetDeclarativeValidationOptions returns the options used in the authorization.k8s.io API group
// DeclarativeValidationConfig returns the declarative validation config for the
// authorization.k8s.io API group.
func DeclarativeValidationConfig() rest.DeclarativeValidationConfig {
	return rest.DeclarativeValidationConfig{
		Options: map[string]bool{
			string(genericfeatures.ConditionalAuthorization): utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization),
		},
	}
}

// CombinedValidateSubjectAccessReviewCreate calls both the handwritten and declarative validations for SubjectAccessReview.
func CombinedValidateSubjectAccessReviewCreate(ctx context.Context, sar *authorizationv1.SubjectAccessReview) (errs field.ErrorList) {
	defer func() {
		if r := recover(); r != nil {
			errs = append(errs, field.InternalError(nil, fmt.Errorf("panic during SAR validation: %v", r)))
		}
	}()
	errs = ValidateSubjectAccessReview(sar)

	op := operation.Operation{
		Type:    operation.Create,
		Options: DeclarativeValidationConfig().Options,
	}
	declarativeErrs := authorizationv1.Validate_SubjectAccessReview(ctx, op, nil /* fldPath */, sar, nil)
	errs = append(errs, declarativeErrs...)
	return errs
}

// CombinedValidateAuthorizationConditionsReviewCreate calls both the handwritten and declarative validations for AuthorizationConditionsReview.
func CombinedValidateAuthorizationConditionsReviewCreate(ctx context.Context, acr *authorizationv1alpha1.AuthorizationConditionsReview) (errs field.ErrorList) {
	defer func() {
		if r := recover(); r != nil {
			errs = append(errs, field.InternalError(nil, fmt.Errorf("panic during ACR validation: %v", r)))
		}
	}()
	errs = ValidateAuthorizationConditionsReview(acr)

	op := operation.Operation{
		Type:    operation.Create,
		Options: DeclarativeValidationConfig().Options,
	}
	declarativeErrs := authorizationv1alpha1.Validate_AuthorizationConditionsReview(ctx, op, nil /* fldPath */, acr, nil)
	errs = append(errs, declarativeErrs...)
	return errs
}
