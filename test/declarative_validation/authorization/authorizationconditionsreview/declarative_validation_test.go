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

// Package authorizationconditionsreview exercises declarative validation for the
// authorization.k8s.io AuthorizationConditionsReview resource.
package authorizationconditionsreview

import (
	"context"
	"strconv"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/admission"
	"k8s.io/kubernetes/pkg/apis/authorization"
	authorizationvalidation "k8s.io/kubernetes/pkg/apis/authorization/validation"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testDeclarativeValidate(t, v)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{APIGroup: "authorization.k8s.io", APIVersion: apiVersion, Resource: "authorizationconditionsreviews"})

	// response.decision.type carries a handwritten constraint on top of the declarative
	// rules: evaluating conditions can only ever produce Allow, Deny or NoOpinion, so a
	// conditional decision is rejected. Cases that put a ConditionsMap or a Union in the
	// response therefore expect this error alongside whatever they are exercising. It has
	// no declarative counterpart, hence MarkFromImperative.
	responseNotUnconditionalErr := field.Invalid(field.NewPath("response", "decision", "type"), "", "").MarkFromImperative()

	testCases := map[string]struct {
		obj          authorization.AuthorizationConditionsReview
		expectedErrs field.ErrorList
	}{
		"valid": {
			obj:          mkACR(),
			expectedErrs: nil,
		},
		"request.admissionRequest required": {
			obj: mkACR(clearRequestAdmissionRequest()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("request", "admissionRequest"), ""),
			},
		},
		"response.uid required": {
			obj: mkACR(clearResponseUID()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("response", "uid"), ""),
			},
		},
		"decision.type required (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{}),
				setResponseDecision(authorization.ConditionsAwareDecision{}),
			),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("request", "decision", "type"), ""),
				field.Required(field.NewPath("response", "decision", "type"), ""),
			},
		},
		"decision.type not supported (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{Type: "BogusType"}),
				setResponseDecision(authorization.ConditionsAwareDecision{Type: "BogusType"}),
			),
			expectedErrs: field.ErrorList{
				field.NotSupported[authorization.ConditionsAwareDecisionType](field.NewPath("request", "decision", "type"), authorization.ConditionsAwareDecisionType("BogusType"), nil),
				field.NotSupported[authorization.ConditionsAwareDecisionType](field.NewPath("response", "decision", "type"), authorization.ConditionsAwareDecisionType("BogusType"), nil),
				// An unrecognized type is not unconditional either.
				responseNotUnconditionalErr,
			},
		},
		"decision.conditionsMap[deny|noOpinion|allow]Conditions[*].id required (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorization.ConditionsMap{
						DenyConditions:      []authorization.Condition{{ID: ""}},
						NoOpinionConditions: []authorization.Condition{{ID: ""}},
						AllowConditions:     []authorization.Condition{{ID: ""}},
					},
				}),
				setResponseDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorization.ConditionsMap{
						DenyConditions:      []authorization.Condition{{ID: ""}},
						NoOpinionConditions: []authorization.Condition{{ID: ""}},
						AllowConditions:     []authorization.Condition{{ID: ""}},
					},
				}),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Required(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(0).Child("id"), ""),
			},
		},
		"decision.conditionsMap[deny|noOpinion|allow]Conditions[*] duplicate (request+response)": {
			obj: mkACR(
				setRequestDecision(duplicateConditionsMapDecision()),
				setResponseDecision(duplicateConditionsMapDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Duplicate(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(1), nil),
				field.Duplicate(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(1), nil),
				field.Duplicate(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(1), nil),
				field.Duplicate(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(1), nil),
				field.Duplicate(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(1), nil),
				field.Duplicate(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(1), nil),
			},
		},
		"decision.union[*].authorizerName required (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "", Decision: validNoOpinionDecision()},
					},
				}),
				setResponseDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "", Decision: validNoOpinionDecision()},
					},
				}),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Required(field.NewPath("request", "decision", "union").Index(0).Child("authorizerName"), ""),
				field.Required(field.NewPath("response", "decision", "union").Index(0).Child("authorizerName"), ""),
			},
		},
		"decision.union[*] duplicate (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
						{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
					},
				}),
				setResponseDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
						{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
					},
				}),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Duplicate(field.NewPath("request", "decision", "union").Index(1), nil),
				field.Duplicate(field.NewPath("response", "decision", "union").Index(1), nil),
			},
		},
		"decision.union[*].authorizerName invalid subdomain (request+response)": {
			obj: mkACR(
				setRequestDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "not a valid label", Decision: validNoOpinionDecision()},
					},
				}),
				setResponseDecision(authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{AuthorizerName: "not a valid label", Decision: validNoOpinionDecision()},
					},
				}),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Invalid(field.NewPath("request", "decision", "union").Index(0).Child("authorizerName"), "", "").WithOrigin("format=k8s-long-name"),
				field.Invalid(field.NewPath("response", "decision", "union").Index(0).Child("authorizerName"), "", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"decision.conditionsMap[deny|noOpinion|allow]Conditions too many (request+response)": {
			obj: mkACR(
				setRequestDecision(tooManyConditionsDecision()),
				setResponseDecision(tooManyConditionsDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.TooMany(field.NewPath("request", "decision", "conditionsMap", "denyConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("request", "decision", "conditionsMap", "allowConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("response", "decision", "conditionsMap", "denyConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("response", "decision", "conditionsMap", "allowConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
			},
		},
		"decision.conditionsMap[*]Conditions[*].id invalid label key (request+response)": {
			obj: mkACR(
				setRequestDecision(invalidIDConditionsDecision()),
				setResponseDecision(invalidIDConditionsDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"decision.conditionsMap[*]Conditions[*].type invalid label key (request+response)": {
			obj: mkACR(
				setRequestDecision(invalidTypeConditionsDecision()),
				setResponseDecision(invalidTypeConditionsDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"decision.conditionsMap[*]Conditions[*].condition too long (request+response)": {
			obj: mkACR(
				setRequestDecision(tooLongConditionDecision()),
				setResponseDecision(tooLongConditionDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
			},
		},
		"decision.conditionsMap[*]Conditions[*].description too long (request+response)": {
			obj: mkACR(
				setRequestDecision(tooLongDescriptionDecision()),
				setResponseDecision(tooLongDescriptionDecision()),
			),
			expectedErrs: field.ErrorList{
				responseNotUnconditionalErr,
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "denyConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("request", "decision", "conditionsMap", "allowConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "denyConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "noOpinionConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("response", "decision", "conditionsMap", "allowConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalenceFunc(t, ctx, &tc.obj, func(ctx context.Context, obj runtime.Object) field.ErrorList {
				acr := obj.(*authorization.AuthorizationConditionsReview)
				// TODO(luxas): This should probably just call declarative validation, not both? Ofc both is good
				// TODO(luxas): We might want to register coverage rules for the hand-written stuff in a hand-written init() func
				return authorizationvalidation.ValidateAuthorizationConditionsReviewCreate(ctx, legacyscheme.Scheme, acr)
			}, tc.expectedErrs)
		})
	}
}

func mkACR(tweaks ...func(*authorization.AuthorizationConditionsReview)) authorization.AuthorizationConditionsReview {
	acr := authorization.AuthorizationConditionsReview{
		ObjectMeta: metav1.ObjectMeta{},
		Request: &authorization.AuthorizationConditionsRequest{
			Decision:         validNoOpinionDecision(),
			AdmissionRequest: &admission.AdmissionRequest{UID: "test-uid"},
		},
		Response: &authorization.AuthorizationConditionsResponse{
			UID:      "test-uid",
			Decision: validNoOpinionDecision(),
		},
	}
	for _, tweak := range tweaks {
		tweak(&acr)
	}
	return acr
}

func setRequestDecision(d authorization.ConditionsAwareDecision) func(*authorization.AuthorizationConditionsReview) {
	return func(acr *authorization.AuthorizationConditionsReview) {
		acr.Request.Decision = d
	}
}

func clearResponseUID() func(*authorization.AuthorizationConditionsReview) {
	return func(acr *authorization.AuthorizationConditionsReview) {
		acr.Response.UID = ""
	}
}

func clearRequestAdmissionRequest() func(*authorization.AuthorizationConditionsReview) {
	return func(acr *authorization.AuthorizationConditionsReview) {
		acr.Request.AdmissionRequest = nil
	}
}

func setResponseDecision(d authorization.ConditionsAwareDecision) func(*authorization.AuthorizationConditionsReview) {
	return func(acr *authorization.AuthorizationConditionsReview) {
		acr.Response.Decision = d
	}
}

// validNoOpinionDecision returns a minimally-valid ConditionsAwareDecision for
// nesting inside union members: type=NoOpinion with the NoOpinion field set.
func validNoOpinionDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type:      authorization.ConditionsAwareDecisionTypeNoOpinion,
		NoOpinion: &authorization.UnconditionalDecision{},
	}
}

func duplicateConditionsMapDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions: []authorization.Condition{
				{ID: "example.com/dup"},
				{ID: "example.com/dup"},
			},
			NoOpinionConditions: []authorization.Condition{
				{ID: "example.com/dup"},
				{ID: "example.com/dup"},
			},
			AllowConditions: []authorization.Condition{
				{ID: "example.com/dup"},
				{ID: "example.com/dup"},
			},
		},
	}
}

// makeConditions produces n Conditions with unique domain-prefixed IDs so that only the
// slice-length (maxItems) rule fires, not per-item id/type validation.
func makeConditions(n int) []authorization.Condition {
	out := make([]authorization.Condition, n)
	for i := range out {
		out[i].ID = "example.com/cond-" + strconv.Itoa(i)
	}
	return out
}

func tooManyConditionsDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions:      makeConditions(authorizer.MaxConditionsPerMap + 1),
			NoOpinionConditions: makeConditions(authorizer.MaxConditionsPerMap + 1),
			AllowConditions:     makeConditions(authorizer.MaxConditionsPerMap + 1),
		},
	}
}

func invalidIDConditionsDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions:      []authorization.Condition{{ID: "not a valid label"}},
			NoOpinionConditions: []authorization.Condition{{ID: "not a valid label"}},
			AllowConditions:     []authorization.Condition{{ID: "not a valid label"}},
		},
	}
}

func invalidTypeConditionsDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions:      []authorization.Condition{{ID: "example.com/d", Type: "not a valid label"}},
			NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Type: "not a valid label"}},
			AllowConditions:     []authorization.Condition{{ID: "example.com/a", Type: "not a valid label"}},
		},
	}
}

func tooLongConditionDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions:      []authorization.Condition{{ID: "example.com/d", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
			NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
			AllowConditions:     []authorization.Condition{{ID: "example.com/a", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
		},
	}
}

func tooLongDescriptionDecision() authorization.ConditionsAwareDecision {
	return authorization.ConditionsAwareDecision{
		Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorization.ConditionsMap{
			DenyConditions:      []authorization.Condition{{ID: "example.com/d", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
			NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
			AllowConditions:     []authorization.Condition{{ID: "example.com/a", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
		},
	}
}
