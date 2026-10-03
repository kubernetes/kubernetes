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

// Package subjectaccessreview exercises declarative validation for the
// authorization.k8s.io SubjectAccessReview resource.
package subjectaccessreview

import (
	"context"
	"strconv"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apiserverauthorizationvalidation "k8s.io/apiserver/pkg/apis/authorization/validation"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
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
		&genericapirequest.RequestInfo{APIGroup: "authorization.k8s.io", APIVersion: apiVersion, Resource: "subjectaccessreviews"})

	testCases := map[string]struct {
		obj                            authorization.SubjectAccessReview
		enableConditionalAuthorization bool
		// v1Only marks a case that exercises fields only present in v1
		// (authorizationOptions on the spec, conditionalDecision on the
		// status). v1beta1 dropped those fields, so the case is skipped
		// when apiVersion != "v1".
		v1Only       bool
		expectedErrs field.ErrorList
	}{
		"valid": {
			obj: mkSAR(),
		},
		"the server should not fail on unrecognized handledDecisionTypes": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalAuthorization(&authorization.AuthorizationOptions{
				HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
					authorization.ConditionsAwareDecisionTypeAllow,
					authorization.ConditionsAwareDecisionTypeDeny,
					authorization.ConditionsAwareDecisionTypeNoOpinion,
					authorization.ConditionsAwareDecisionType("IntroducedInAFutureVersion"),
				},
			})),
		},
		"neither": {
			obj:          mkSAR(clearResourceAttributes()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec"), "", "").WithOrigin("union").MarkAlpha()},
		},
		"both": {
			obj: mkSAR(setNonResourceAttributes()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "", "").WithOrigin("union").MarkAlpha(),
			},
		},
		"spec.authorizationOptions forbidden when feature gate disabled": {
			v1Only: true,
			// Use a conditional HandledDecisionTypes set (includes ConditionsMap/Union) so the
			// v1 -> v1beta1 conversion fails closed; that lets the equivalence sweep skip
			// v1beta1 via WithIgnoreObjectConversionErrors instead of comparing this v1-only
			// Forbidden against v1beta1's empty result.
			obj: mkSAR(setConditionalAuthorization(&authorization.AuthorizationOptions{
				HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
					authorization.ConditionsAwareDecisionTypeAllow,
					authorization.ConditionsAwareDecisionTypeDeny,
					authorization.ConditionsAwareDecisionTypeNoOpinion,
					authorization.ConditionsAwareDecisionTypeConditionsMap,
					authorization.ConditionsAwareDecisionTypeUnion,
				},
			})),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "authorizationOptions"), ""),
			},
		},
		"spec.authorizationOptions.handledDecisionTypes required when feature gate enabled and empty": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj:                            mkSAR(setConditionalAuthorization(&authorization.AuthorizationOptions{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "authorizationOptions", "handledDecisionTypes"), ""),
			},
		},
		"spec.authorizationOptions.handledDecisionTypes duplicate": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalAuthorization(&authorization.AuthorizationOptions{
				HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
					authorization.ConditionsAwareDecisionTypeAllow,
					authorization.ConditionsAwareDecisionTypeAllow,
					authorization.ConditionsAwareDecisionTypeDeny,
					authorization.ConditionsAwareDecisionTypeNoOpinion,
					authorization.ConditionsAwareDecisionTypeConditionsMap,
					authorization.ConditionsAwareDecisionTypeUnion,
				},
			})),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "authorizationOptions", "handledDecisionTypes").Index(1), authorization.ConditionsAwareDecisionTypeAllow),
			},
		},
		"spec.authorizationOptions.handledDecisionTypes too many": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalAuthorization(&authorization.AuthorizationOptions{
				HandledDecisionTypes: tooManyHandledDecisionTypes(),
			})),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "authorizationOptions", "handledDecisionTypes"), maxHandledDecisionTypes+1, maxHandledDecisionTypes).WithOrigin("maxItems"),
			},
		},
		"status.conditionalDecision forbidden when feature gate disabled": {
			v1Only: true,
			obj:    mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{})),
			// Two Forbidden errors are emitted at status.conditionalDecision: one from
			// declarative validation (+k8s:ifDisabled=ConditionalAuthorization) and one
			// from the imperative "client did not opt into conditions-awareness" check.
			// TODO(luxas): Should we fix this to only report once?
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "conditionalDecision"), ""),
				field.Forbidden(field.NewPath("status", "conditionalDecision"), "").MarkFromImperative(),
			},
		},
		"status.conditionalDecision.type required": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj:                            mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("status", "conditionalDecision", "type"), ""),
			},
		},
		"status.conditionalDecision.type not supported": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj:                            mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{Type: "BogusType"})),
			expectedErrs: field.ErrorList{
				field.NotSupported[authorization.ConditionsAwareDecisionType](field.NewPath("status", "conditionalDecision", "type"), authorization.ConditionsAwareDecisionType("BogusType"), nil),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*].id required": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      []authorization.Condition{{ID: ""}},
					NoOpinionConditions: []authorization.Condition{{ID: ""}},
					AllowConditions:     []authorization.Condition{{ID: ""}},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), ""),
				field.Required(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(0).Child("id"), ""),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*] duplicate": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
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
			})),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(1), nil),
				field.Duplicate(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(1), nil),
				field.Duplicate(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(1), nil),
			},
		},
		"status.conditionalDecision.union[*].authorizerName required": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeUnion,
				Union: []authorization.NamedConditionsAwareDecision{
					{AuthorizerName: "", Decision: validNoOpinionDecision()},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("status", "conditionalDecision", "union").Index(0).Child("authorizerName"), ""),
			},
		},
		"status.conditionalDecision.union[*] duplicate": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeUnion,
				Union: []authorization.NamedConditionsAwareDecision{
					{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
					{AuthorizerName: "dup.example.com", Decision: validNoOpinionDecision()},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "conditionalDecision", "union").Index(1), nil),
			},
		},
		"status.conditionalDecision.union[*].authorizerName invalid subdomain": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeUnion,
				Union: []authorization.NamedConditionsAwareDecision{
					{AuthorizerName: "not a valid label", Decision: validNoOpinionDecision()},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditionalDecision", "union").Index(0).Child("authorizerName"), "", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions too many": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      makeConditions(authorizer.MaxConditionsPerMap + 1),
					NoOpinionConditions: makeConditions(authorizer.MaxConditionsPerMap + 1),
					AllowConditions:     makeConditions(authorizer.MaxConditionsPerMap + 1),
				},
			})),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
				field.TooMany(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions"), authorizer.MaxConditionsPerMap+1, authorizer.MaxConditionsPerMap).WithOrigin("maxItems"),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*].id invalid label key": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      []authorization.Condition{{ID: "not a valid label"}},
					NoOpinionConditions: []authorization.Condition{{ID: "not a valid label"}},
					AllowConditions:     []authorization.Condition{{ID: "not a valid label"}},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(0).Child("id"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*].type invalid label key": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      []authorization.Condition{{ID: "example.com/d", Type: "not a valid label"}},
					NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Type: "not a valid label"}},
					AllowConditions:     []authorization.Condition{{ID: "example.com/a", Type: "not a valid label"}},
				},
			})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
				field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(0).Child("type"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*].condition too long": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      []authorization.Condition{{ID: "example.com/d", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
					NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
					AllowConditions:     []authorization.Condition{{ID: "example.com/a", Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1)}},
				},
			})),
			expectedErrs: field.ErrorList{
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(0).Child("condition"), "", authorizer.MaxConditionBytes).WithOrigin("maxBytes"),
			},
		},
		"status.conditionalDecision.conditionsMap[deny|noOpinion|allow]Conditions[*].description too long": {
			v1Only:                         true,
			enableConditionalAuthorization: true,
			obj: mkSAR(setConditionalDecision(&authorization.ConditionsAwareDecision{
				Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorization.ConditionsMap{
					DenyConditions:      []authorization.Condition{{ID: "example.com/d", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
					NoOpinionConditions: []authorization.Condition{{ID: "example.com/n", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
					AllowConditions:     []authorization.Condition{{ID: "example.com/a", Description: strings.Repeat("a", authorizer.MaxConditionDescriptionBytes+1)}},
				},
			})),
			expectedErrs: field.ErrorList{
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "noOpinionConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
				field.TooLong(field.NewPath("status", "conditionalDecision", "conditionsMap", "allowConditions").Index(0).Child("description"), "", authorizer.MaxConditionDescriptionBytes).WithOrigin("maxBytes"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			if tc.v1Only && apiVersion != "v1" {
				t.Skipf("case exercises authorization.k8s.io/v1-only fields; skipping for apiVersion=%q", apiVersion)
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, tc.enableConditionalAuthorization)
			// Cases that exercise status.conditionalDecision validation assume the client
			// opted into conditions-awareness; otherwise the validator emits an unrelated
			// Forbidden error at status.conditionalDecision. Set a valid AuthorizationOptions
			// when the feature gate is enabled and no opt-in has been configured explicitly.
			if tc.enableConditionalAuthorization && tc.obj.Status.ConditionalDecision != nil && tc.obj.Spec.AuthorizationOptions == nil {
				tc.obj.Spec.AuthorizationOptions = &authorization.AuthorizationOptions{
					HandledDecisionTypes: []authorization.ConditionsAwareDecisionType{
						authorization.ConditionsAwareDecisionTypeAllow,
						authorization.ConditionsAwareDecisionTypeDeny,
						authorization.ConditionsAwareDecisionTypeNoOpinion,
						authorization.ConditionsAwareDecisionTypeConditionsMap,
						authorization.ConditionsAwareDecisionTypeUnion,
					},
				}
			}

			apitesting.VerifyValidationEquivalenceFunc(t, ctx, &tc.obj, func(ctx context.Context, obj runtime.Object) field.ErrorList {
				sar := obj.(*authorization.SubjectAccessReview)
				return authorizationvalidation.ValidateSubjectAccessReviewCreate(ctx, legacyscheme.Scheme, sar)
			}, tc.expectedErrs,
				apitesting.WithOptions(apiserverauthorizationvalidation.DeclarativeValidationConfig().Options),
				// authorization.k8s.io/v1beta1 has no authorizationOptions or
				// conditionalDecision, so converting to it drops those fields and that
				// version cannot report errors under them. Only errors on those paths are
				// excused; every field the two versions share is still compared.
				apitesting.WithMapErrorListsFuncs(authorizationvalidation.MapV1ToV1beta1ErrorLists),
			)
		})
	}
}

func mkSAR(tweaks ...func(*authorization.SubjectAccessReview)) authorization.SubjectAccessReview {
	sar := authorization.SubjectAccessReview{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: authorization.SubjectAccessReviewSpec{
			ResourceAttributes: &authorization.ResourceAttributes{
				Namespace: "default",
				Verb:      "get",
				Resource:  "pods",
			},
			User: "admin",
		},
	}
	for _, tweak := range tweaks {
		tweak(&sar)
	}
	return sar
}

func clearResourceAttributes() func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Spec.ResourceAttributes = nil
	}
}

func setNonResourceAttributes() func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Spec.NonResourceAttributes = &authorization.NonResourceAttributes{}
	}
}

func setConditionalAuthorization(opts *authorization.AuthorizationOptions) func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Spec.AuthorizationOptions = opts
	}
}

func setConditionalDecision(d *authorization.ConditionsAwareDecision) func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Status.ConditionalDecision = d
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

// makeConditions produces n Conditions with unique domain-prefixed IDs so that only the
// slice-length (maxItems) rule fires, not per-item id/type validation.
func makeConditions(n int) []authorization.Condition {
	out := make([]authorization.Condition, n)
	for i := range out {
		out[i].ID = "example.com/cond-" + strconv.Itoa(i)
	}
	return out
}

// maxHandledDecisionTypes mirrors the +k8s:maxItems marker on
// AuthorizationOptions.HandledDecisionTypes.
const maxHandledDecisionTypes = 32

// tooManyHandledDecisionTypes returns one entry more than the maximum. The entries are
// distinct because the field is a set, so that only the maxItems rule fires and not the
// duplicate rule. Their values are arbitrary: the field is opaque, so unrecognized
// decision types are not themselves validated.
func tooManyHandledDecisionTypes() []authorization.ConditionsAwareDecisionType {
	out := make([]authorization.ConditionsAwareDecisionType, 0, maxHandledDecisionTypes+1)
	for i := 0; i <= maxHandledDecisionTypes; i++ {
		out = append(out, authorization.ConditionsAwareDecisionType("Type"+strconv.Itoa(i)))
	}
	return out
}
