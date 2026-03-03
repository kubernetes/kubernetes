/*
Copyright 2016 The Kubernetes Authors.

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

package authorizationconditionsreview

import (
	"context"
	"fmt"
	"iter"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/rest"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationvalidation "k8s.io/kubernetes/pkg/apis/authorization/validation"
)

type REST struct {
	authorizer  authorizer.Authorizer
	jsonDecoder runtime.Decoder
}

func NewREST(authorizer authorizer.Authorizer, negotiatedSerializer runtime.NegotiatedSerializer) (*REST, error) {
	jsonInfo, ok := runtime.SerializerInfoForMediaType(negotiatedSerializer.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || jsonInfo.Serializer == nil {
		return nil, fmt.Errorf("could not find JSON serializer")
	}
	return &REST{
		authorizer:  authorizer,
		jsonDecoder: negotiatedSerializer.DecoderToVersion(jsonInfo.Serializer, runtime.InternalGroupVersioner),
	}, nil
}

func (r *REST) NamespaceScoped() bool {
	return false
}

func (r *REST) New() runtime.Object {
	return &authorizationapi.AuthorizationConditionsReview{}
}

// Destroy cleans up resources on shutdown.
func (r *REST) Destroy() {
	// Given no underlying store, we don't destroy anything
	// here explicitly.
}

var _ rest.SingularNameProvider = &REST{}

func (r *REST) GetSingularName() string {
	return "authorizationconditionsreview"
}

func (r *REST) Create(ctx context.Context, acr runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	authorizationConditionsReview, ok := acr.(*authorizationapi.AuthorizationConditionsReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a AuthorizationConditionsReview: %#v", acr))
	}
	if errs := authorizationvalidation.ValidateAuthorizationConditionsReview(authorizationConditionsReview); len(errs) > 0 {
		return nil, apierrors.NewInvalid(authorizationapi.Kind(authorizationConditionsReview.Kind), "", errs)
	}

	// Make the safety measure of being able to construct a Conditional response only when ConditionsMode != "" pass
	fakeAttrsWithConditionsSupport := &authorizer.AttributesRecord{
		ConditionsMode: authorizer.ConditionsModeOptimized, // doesn't matter which mode is specified here, just != ""
	}
	unevaluatedDecision, errs := deserializeDecision(fakeAttrsWithConditionsSupport, authorizationConditionsReview.Request.Decision, field.NewPath("request", "decision"))
	if len(errs) > 0 {
		return nil, apierrors.NewInvalid(authorizationapi.Kind(authorizationConditionsReview.Kind), "", errs)
	}

	if createValidation != nil {
		if err := createValidation(ctx, acr.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	data, err := r.toConditionsData(authorizationConditionsReview.Request)
	if err != nil {
		allErrs := field.ErrorList{}
		allErrs = append(allErrs, field.Invalid(field.NewPath("request"), authorizationConditionsReview.Request, err.Error()))
		return nil, apierrors.NewInvalid(authorizationapi.Kind(authorizationConditionsReview.Kind), "", allErrs)
	}

	evaluatedDecision, err := r.authorizer.EvaluateConditions(ctx, unevaluatedDecision, data)
	if err != nil {
		// TODO: How to handle this error?
		return nil, err
	}

	// TODO: Should we set acr.Request to nil, or keep it? Keeping it yields more data to write back unnecessarily.
	serializedDecision := serializeDecision(evaluatedDecision)
	authorizationConditionsReview.Response = &authorizationapi.AuthorizationConditionsResponse{
		SubjectAccessReviewAuthorizationDecision: serializedDecision,
	}

	return authorizationConditionsReview, nil
}

func (r *REST) toConditionsData(req *authorizationapi.AuthorizationConditionsRequest) (authorizer.ConditionData, error) {
	if req.WriteRequest == nil {
		return nil, fmt.Errorf("unsupported conditions data: request.writeRequest == nil")
	}

	wr := &conditionsDataWriteRequest{
		operation: string(req.WriteRequest.Operation),
	}

	var err error
	if len(req.WriteRequest.Object.Raw) != 0 {
		wr.object, err = r.decodeObject(req.WriteRequest.Object.Raw)
		if err != nil {
			return nil, err
		}
	}

	if len(req.WriteRequest.OldObject.Raw) != 0 {
		wr.oldObject, err = r.decodeObject(req.WriteRequest.OldObject.Raw)
		if err != nil {
			return nil, err
		}
	}

	// TODO: How to decode options?
	return &conditionsData{writeReq: wr}, nil
}

// decodeObject tries to decode the raw bytes using the known scheme serializer first.
// If the type is not registered (e.g., objects from aggregated API servers), it falls
// back to decoding as unstructured.
func (r *REST) decodeObject(raw []byte) (runtime.Object, error) {
	obj, _, err := r.jsonDecoder.Decode(raw, nil, nil)
	if err == nil {
		return obj, nil
	}
	// Fall back to unstructured for types not registered in the scheme
	// (e.g., objects from aggregated API servers).
	if !runtime.IsNotRegisteredError(err) {
		return nil, err
	}
	obj, _, err = unstructured.UnstructuredJSONScheme.Decode(raw, nil, nil)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

var _ authorizer.ConditionData = &conditionsData{}

type conditionsData struct {
	writeReq *conditionsDataWriteRequest
}

func (d *conditionsData) WriteRequest() authorizer.WriteRequestConditionData {
	if d.writeReq == nil {
		return nil
	}
	return d.writeReq
}

func (d *conditionsData) ImpersonationRequest() authorizer.ImpersonationRequestConditionData {
	return nil
}

var _ authorizer.WriteRequestConditionData = &conditionsDataWriteRequest{}

type conditionsDataWriteRequest struct {
	object    runtime.Object
	oldObject runtime.Object
	options   runtime.Object // TODO: how are these encoded?
	operation string
}

func (r *conditionsDataWriteRequest) GetOperation() string                { return r.operation }
func (r *conditionsDataWriteRequest) GetOperationOptions() runtime.Object { return r.options }
func (r *conditionsDataWriteRequest) GetObject() runtime.Object           { return r.object }
func (r *conditionsDataWriteRequest) GetOldObject() runtime.Object        { return r.oldObject }

// TODO: Figure out how to de-duplicate this logic with the webhook authorizer
func toAuthorizerConditions(conditionList []authorizationapi.SubjectAccessReviewCondition) iter.Seq2[string, authorizer.Condition] {
	return func(yield func(string, authorizer.Condition) bool) {
		for _, condition := range conditionList {
			cond := authorizer.Condition{
				Effect:      authorizer.ConditionEffect(condition.Effect),
				Condition:   condition.Condition,
				Description: condition.Description,
			}
			if !yield(condition.ID, cond) {
				return
			}
		}
	}
}

func deserializeDecision(attrs authorizer.Attributes, serializedDecision authorizationapi.SubjectAccessReviewAuthorizationDecision, fldPath *field.Path) (authorizer.Decision, field.ErrorList) {
	var allErrs field.ErrorList

	hasConditionSet := len(serializedDecision.Conditions) != 0
	hasDecisionChain := len(serializedDecision.ConditionalDecisionChain) != 0

	if serializedDecision.Denied && serializedDecision.Allowed {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Denied and Allowed are both specified"))
		return authorizer.DecisionDeny(serializedDecision.Reason), allErrs
	}

	// check all newly-introduced mutual exclusion possibilities
	// this function is only ever called when the conditional authorization feature gate is enabled
	if serializedDecision.Denied && hasConditionSet {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Denied and Conditions are both specified"))
		return authorizer.DecisionDeny(), allErrs
	}
	if serializedDecision.Denied && hasDecisionChain {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Denied and ConditionalDecisionChain are both specified"))
		return authorizer.DecisionDeny(), allErrs
	}
	if serializedDecision.Allowed && hasConditionSet {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Allowed and Conditions are both specified"))
		return authorizer.DecisionDeny(), allErrs
	}
	if serializedDecision.Allowed && hasDecisionChain {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Allowed and ConditionalDecisionChain are both specified"))
		return authorizer.DecisionDeny(), allErrs
	}
	if hasConditionSet && hasDecisionChain {
		allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, "mutually exclusive Conditions and ConditionalDecisionChain are both specified"))
		return authorizer.DecisionDeny(), allErrs
	}

	if serializedDecision.Denied {
		return authorizer.DecisionDeny(serializedDecision.Reason), nil
	}

	if serializedDecision.Allowed {
		return authorizer.DecisionAllow(serializedDecision.Reason), nil
	}

	if hasConditionSet {
		ct := authorizer.ConditionType(serializedDecision.ConditionsType)
		condResp, err := authorizer.DecisionConditional(attrs, ct, toAuthorizerConditions(serializedDecision.Conditions))
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, serializedDecision, err.Error()))
		}
		return condResp, allErrs
	}

	if hasDecisionChain {
		subDecisions := make([]authorizer.Decision, 0, len(serializedDecision.ConditionalDecisionChain))
		for i, serializedSubDecision := range serializedDecision.ConditionalDecisionChain {
			subDecision, err := deserializeDecision(attrs, serializedSubDecision, fldPath.Child("conditionalDecisionChain").Index(i))
			if err != nil {
				return authorizer.DecisionDeny(), err
			}
			subDecisions = append(subDecisions, subDecision)
		}
		return authorizer.DecisionConditionalChain(subDecisions...), nil
	}

	return authorizer.DecisionNoOpinion(serializedDecision.Reason), nil
}

func conditionSetToInternalAPIDecision(conditionSet *authorizer.ConditionSet) authorizationapi.SubjectAccessReviewAuthorizationDecision {
	if conditionSet == nil {
		return authorizationapi.SubjectAccessReviewAuthorizationDecision{} // NoOpinion
	}
	conds := []authorizationapi.SubjectAccessReviewCondition{}
	for id, condition := range conditionSet.Conditions() {
		conds = append(conds, authorizationapi.SubjectAccessReviewCondition{
			ID:          id,
			Effect:      authorizationapi.SubjectAccessReviewConditionEffect(condition.Effect),
			Condition:   condition.Condition,
			Description: condition.Description,
		})
	}

	return authorizationapi.SubjectAccessReviewAuthorizationDecision{
		Conditions:     conds,
		ConditionsType: string(conditionSet.Type()),
	}
}

func serializeDecision(decision authorizer.Decision) authorizationapi.SubjectAccessReviewAuthorizationDecision {
	if decision.IsAllowed() {
		return authorizationapi.SubjectAccessReviewAuthorizationDecision{Allowed: true, Reason: decision.Reason()}
	}
	if decision.IsDenied() {
		return authorizationapi.SubjectAccessReviewAuthorizationDecision{Denied: true, Reason: decision.Reason()}
	}

	if decision.IsConditional() {
		d := conditionSetToInternalAPIDecision(decision.ConditionSet())
		d.Reason = decision.Reason()
		return d
	}
	if decision.IsConditionalChain() {
		subDecisions := make([]authorizationapi.SubjectAccessReviewAuthorizationDecision, 0, len(decision.ConditionalChain()))
		for _, subDecision := range decision.ConditionalChain() {
			subDecisions = append(subDecisions, serializeDecision(subDecision))
		}
		return authorizationapi.SubjectAccessReviewAuthorizationDecision{
			ConditionalDecisionChain: subDecisions,
			Reason:                   decision.Reason(),
		}
	}
	// no opinion
	return authorizationapi.SubjectAccessReviewAuthorizationDecision{Reason: decision.Reason()}
}
