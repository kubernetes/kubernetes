/*
Copyright 2017 The Kubernetes Authors.

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

package subjectaccessreview

import (
	"context"
	"errors"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
)

// unconditionalDecisionTypes is the client-handled decision type set for a
// conditions-unaware client.
var unconditionalDecisionTypes = []authorizationapi.ConditionsAwareDecisionType{
	authorizationapi.ConditionsAwareDecisionTypeAllow,
	authorizationapi.ConditionsAwareDecisionTypeDeny,
	authorizationapi.ConditionsAwareDecisionTypeNoOpinion,
}

// conditionalDecisionTypes is the client-handled decision type set for a
// fully conditions-aware client.
var conditionalDecisionTypes = []authorizationapi.ConditionsAwareDecisionType{
	authorizationapi.ConditionsAwareDecisionTypeAllow,
	authorizationapi.ConditionsAwareDecisionTypeDeny,
	authorizationapi.ConditionsAwareDecisionTypeNoOpinion,
	authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
	authorizationapi.ConditionsAwareDecisionTypeUnion,
}

// conditionsAwareFakeAuthorizer lets tests inject arbitrary ConditionsAwareDecision
// values so both the unconditional and conditional REST branches can be exercised.
// A conditions-unaware Authorize call folds the decision back to its (Decision,
// reason, error) triple; if the decision is still conditional, it fails closed
// with a fixed reason so callers can assert the fallback behavior.
type conditionsAwareFakeAuthorizer struct {
	attrs        authorizer.Attributes
	makeDecision func() authorizer.ConditionsAwareDecision
}

func (f *conditionsAwareFakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	f.attrs = a
	d := f.makeDecision()
	switch {
	case d.IsAllow():
		return authorizer.DecisionAllow, d.Reason(), d.Error()
	case d.IsDeny():
		return authorizer.DecisionDeny, d.Reason(), d.Error()
	case d.IsNoOpinion():
		return authorizer.DecisionNoOpinion, d.Reason(), d.Error()
	default:
		return d.FailureDecision(), "failed closed: tried to return conditional decision to conditions-unaware authorizer", nil
	}
}

func (f *conditionsAwareFakeAuthorizer) ConditionsAwareAuthorize(_ context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	f.attrs = a
	return f.makeDecision()
}

func (*conditionsAwareFakeAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

// assertions captures the expected outcome for a single (case × feature-gate)
// combination. A zero-valued assertions means "expect no error, status is
// zero-value, no attribute check".
type assertions struct {
	// err, when non-empty, expects Create() to return an error whose full
	// message equals this string; attrs and status are ignored in that case.
	err string
	// attrs, when non-nil, is compared against the authorizer.Attributes the
	// REST handler passed into the authorizer.
	attrs authorizer.Attributes
	// status is the expected SAR status after a successful Create().
	status authorizationapi.SubjectAccessReviewStatus
	// dropSelectorErr clears FieldSelectorParsingErr / LabelSelectorParsingErr
	// on the captured attributes before comparing against attrs — used by the
	// "invalid selectors" case where the parse-error struct is not directly
	// comparable via cmp.Diff.
	dropSelectorErr bool
}

// testcase is one row of the table. Every case runs against both feature-gate
// values; the expected outcome is declared via exactly one of:
//   - assertFeatureAgnostic: same result on both sides of the gate.
//   - assertFeatureOn AND assertFeatureOff: result differs by gate value.
//
// The runner fails loudly (t.Fatalf) if a case violates that invariant.
type testcase struct {
	spec         authorizationapi.SubjectAccessReviewSpec
	makeDecision func() authorizer.ConditionsAwareDecision

	assertFeatureAgnostic *assertions
	assertFeatureOn       *assertions
	assertFeatureOff      *assertions
}

func TestCreate(t *testing.T) {
	// Shared builders / expected values for cases that reference them more than once.
	allowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionAllow("allowed", nil)
	}
	denyDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionDeny("", nil)
	}
	condMapAllowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			nil, nil,
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/allow-cond", Condition: "true", Type: "example.com/cel"}},
		)
	}
	// baseResourceSpec returns a spec that is minimally valid so cases can
	// focus on the AuthorizationOptions/decision-type dimensions.
	baseResourceSpec := func() authorizationapi.SubjectAccessReviewSpec {
		return authorizationapi.SubjectAccessReviewSpec{
			User: "bob",
			ResourceAttributes: &authorizationapi.ResourceAttributes{
				Namespace: "myns",
				Verb:      "get",
				Resource:  "pods",
			},
		}
	}
	baseResourceAttrs := authorizer.AttributesRecord{
		User:            &user.DefaultInfo{Name: "bob"},
		Namespace:       "myns",
		Verb:            "get",
		Resource:        "pods",
		APIVersion:      "*",
		ResourceRequest: true,
	}
	unconditionalAllowStatus := authorizationapi.SubjectAccessReviewStatus{
		Allowed: true,
		Reason:  "allowed",
	}
	conditionalAllowStatus := authorizationapi.SubjectAccessReviewStatus{
		ConditionalDecision: &authorizationapi.ConditionsAwareDecision{
			Type: authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationapi.ConditionsMap{
				DenyConditions:      []authorizationapi.Condition{},
				NoOpinionConditions: []authorizationapi.Condition{},
				AllowConditions: []authorizationapi.Condition{
					{ID: "example.com/allow-cond", Condition: "true", Type: "example.com/cel"},
				},
			},
		},
	}
	// When the authorizer returns a conditional decision but the REST handler
	// takes the unconditional path, the fake fails closed to NoOpinion(FailureDecision)
	// with this fixed reason. Allowed/Denied are both false → default NoOpinion.
	unconditionalFallbackStatus := authorizationapi.SubjectAccessReviewStatus{
		Reason: "failed closed: tried to return conditional decision to conditions-unaware authorizer",
	}

	testcases := map[string]testcase{
		// ------------------------------------------------------------------
		// Pre-existing TestCreate cases — none set AuthorizationOptions, so
		// they take the unconditional path on both sides of the gate.
		// ------------------------------------------------------------------
		"empty": {
			// No spec fields set: validation rejects with union error.
			makeDecision: allowDecision,
			assertFeatureAgnostic: &assertions{
				err: `.authorization.k8s.io "" is invalid: [spec: Invalid value: null: exactly one of nonResourceAttributes or resourceAttributes must be specified, spec.user: Invalid value: "": at least one of user or group must be specified]`,
			},
		},
		"nonresource rejected": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User:                  "bob",
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
			},
			makeDecision: func() authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionNoOpinion("myreason", errors.New("myerror"))
			},
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					Verb:            "get",
					Path:            "/mypath",
					ResourceRequest: false,
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Reason:          "myreason",
					EvaluationError: "myerror",
				},
			},
		},
		"nonresource allowed": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User:                  "bob",
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
			},
			makeDecision: allowDecision,
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					Verb:            "get",
					Path:            "/mypath",
					ResourceRequest: false,
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Allowed: true,
					Reason:  "allowed",
				},
			},
		},
		"resource rejected": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					Namespace:   "myns",
					Verb:        "create",
					Group:       "extensions",
					Version:     "v1beta1",
					Resource:    "deployments",
					Subresource: "scale",
					Name:        "mydeployment",
				},
			},
			makeDecision: func() authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionNoOpinion("myreason", errors.New("myerror"))
			},
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					Namespace:       "myns",
					Verb:            "create",
					APIGroup:        "extensions",
					APIVersion:      "v1beta1",
					Resource:        "deployments",
					Subresource:     "scale",
					Name:            "mydeployment",
					ResourceRequest: true,
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Reason:          "myreason",
					EvaluationError: "myerror",
				},
			},
		},
		"resource allowed": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					Namespace:   "myns",
					Verb:        "create",
					Group:       "extensions",
					Version:     "v1beta1",
					Resource:    "deployments",
					Subresource: "scale",
					Name:        "mydeployment",
				},
			},
			makeDecision: allowDecision,
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					Namespace:       "myns",
					Verb:            "create",
					APIGroup:        "extensions",
					APIVersion:      "v1beta1",
					Resource:        "deployments",
					Subresource:     "scale",
					Name:            "mydeployment",
					ResourceRequest: true,
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Allowed: true,
					Reason:  "allowed",
				},
			},
		},
		"resource denied": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User:               "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{},
			},
			makeDecision: denyDecision,
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					ResourceRequest: true,
					APIVersion:      "*",
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Denied: true,
				},
			},
		},
		"resource denied, valid selectors": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					FieldSelector: &authorizationapi.FieldSelectorAttributes{RawSelector: "foo=bar"},
					LabelSelector: &authorizationapi.LabelSelectorAttributes{RawSelector: "key=value"},
				},
			},
			makeDecision: denyDecision,
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:                      &user.DefaultInfo{Name: "bob"},
					ResourceRequest:           true,
					APIVersion:                "*",
					FieldSelectorRequirements: fields.Requirements{{Operator: "=", Field: "foo", Value: "bar"}},
					LabelSelectorRequirements: mustParse("key=value"),
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Denied: true,
				},
			},
		},
		"resource denied, invalid selectors": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					FieldSelector: &authorizationapi.FieldSelectorAttributes{RawSelector: "key in value"},
					LabelSelector: &authorizationapi.LabelSelectorAttributes{RawSelector: "&"},
				},
			},
			makeDecision: denyDecision,
			assertFeatureAgnostic: &assertions{
				attrs: authorizer.AttributesRecord{
					User:            &user.DefaultInfo{Name: "bob"},
					ResourceRequest: true,
					APIVersion:      "*",
				},
				status: authorizationapi.SubjectAccessReviewStatus{
					Denied:          true,
					EvaluationError: `spec.resourceAttributes.fieldSelector ignored due to parse error; spec.resourceAttributes.labelSelector ignored due to parse error`,
				},
				dropSelectorErr: true,
			},
		},

		// ------------------------------------------------------------------
		// AuthorizationOptions cases where the intersection is unconditional
		// on both sides of the gate — the outcome is identical.
		// ------------------------------------------------------------------
		"handledTypes=[Allow,Deny,NoOpinion] + unconditional Allow": {
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{HandledDecisionTypes: unconditionalDecisionTypes}
				return s
			}(),
			makeDecision: allowDecision,
			assertFeatureAgnostic: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalAllowStatus,
			},
		},
		"handledTypes=[Allow,Deny,NoOpinion] + ConditionsMap decision → unconditional fallback": {
			// Gate off: options cleared, unconditional path.
			// Gate on: intersection is exactly the unconditional set, same path.
			// Either way, the fake folds the ConditionsMap decision to NoOpinion.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{HandledDecisionTypes: unconditionalDecisionTypes}
				return s
			}(),
			makeDecision: condMapAllowDecision,
			assertFeatureAgnostic: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalFallbackStatus,
			},
		},
		"handledTypes=conditional + unconditional Allow decision": {
			// Gate off: options cleared → unconditional path → Allowed:true.
			// Gate on: conditional path, but unconditional decision still
			//          surfaces on the top-level Allowed field.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{HandledDecisionTypes: conditionalDecisionTypes}
				return s
			}(),
			makeDecision: allowDecision,
			assertFeatureAgnostic: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalAllowStatus,
			},
		},

		// ------------------------------------------------------------------
		// AuthorizationOptions cases where the outcome differs by gate value.
		// ------------------------------------------------------------------
		"handledTypes=conditional + ConditionsMap decision": {
			// Gate off: options cleared → unconditional fallback (NoOpinion).
			// Gate on: conditional path → status.ConditionalDecision set.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{HandledDecisionTypes: conditionalDecisionTypes}
				return s
			}(),
			makeDecision: condMapAllowDecision,
			assertFeatureOff: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalFallbackStatus,
			},
			assertFeatureOn: &assertions{
				attrs:  baseResourceAttrs,
				status: conditionalAllowStatus,
			},
		},
		"handledTypes=[Allow] only": {
			// Gate off: options cleared → default unconditional → Allowed:true.
			// Gate on: imperative validation rejects a set that does not include
			// {Allow, Deny, NoOpinion} before REST's own dispatch can fall
			// through to its "unsupported client-handled decision types" branch.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{
					HandledDecisionTypes: []authorizationapi.ConditionsAwareDecisionType{
						authorizationapi.ConditionsAwareDecisionTypeAllow,
					},
				}
				return s
			}(),
			makeDecision: allowDecision,
			assertFeatureOff: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalAllowStatus,
			},
			assertFeatureOn: &assertions{err: `.authorization.k8s.io "" is invalid: spec.authorizationOptions.handledDecisionTypes: Invalid value: ["Allow"]: set must at least contain {Allow, Deny, NoOpinion}`},
		},
		"handledTypes=[Allow,ConditionsMap] partial conditional set": {
			// Gate off: options cleared → default unconditional → Allowed:true.
			// Gate on: imperative validation rejects a set that does not include
			// {Allow, Deny, NoOpinion} before REST's own dispatch can fall
			// through to its "unsupported client-handled decision types" branch.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{
					HandledDecisionTypes: []authorizationapi.ConditionsAwareDecisionType{
						authorizationapi.ConditionsAwareDecisionTypeAllow,
						authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
					},
				}
				return s
			}(),
			makeDecision: allowDecision,
			assertFeatureOff: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalAllowStatus,
			},
			assertFeatureOn: &assertions{err: `.authorization.k8s.io "" is invalid: spec.authorizationOptions.handledDecisionTypes: Invalid value: ["Allow","ConditionsMap"]: set must at least contain {Allow, Deny, NoOpinion}`},
		},
		"handledTypes=[] (empty slice)": {
			// Gate off: options cleared before validation → succeeds → Allowed:true.
			// Gate on: declarative RequiredSlice rejects with a message that
			//          references the handledDecisionTypes field.
			spec: func() authorizationapi.SubjectAccessReviewSpec {
				s := baseResourceSpec()
				s.AuthorizationOptions = &authorizationapi.AuthorizationOptions{
					HandledDecisionTypes: []authorizationapi.ConditionsAwareDecisionType{},
				}
				return s
			}(),
			makeDecision: allowDecision,
			assertFeatureOff: &assertions{
				attrs:  baseResourceAttrs,
				status: unconditionalAllowStatus,
			},
			assertFeatureOn: &assertions{
				err: `.authorization.k8s.io "" is invalid: spec.authorizationOptions.handledDecisionTypes: Required value`,
			},
		},
	}

	// pickAssertion returns the assertion to check for the given (case × gate) combination.
	pickAssertion := func(tc testcase, gate bool) *assertions {
		if tc.assertFeatureAgnostic != nil {
			return tc.assertFeatureAgnostic
		}
		if gate {
			return tc.assertFeatureOn
		}
		return tc.assertFeatureOff
	}

	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewContext(),
		&genericapirequest.RequestInfo{
			APIGroup:          "authorization.k8s.io",
			APIVersion:        "v1",
			Resource:          "subjectaccessreviews",
			IsResourceRequest: true,
			Verb:              "create",
		},
	)

	for name, tc := range testcases {
		// Enforce the "exactly one style" invariant per case, once, before
		// running any subtests. This makes accidental misconfiguration
		// (setting neither, both, or only one half of the per-gate pair) fail
		// loudly with a helpful message.
		hasAgnostic := tc.assertFeatureAgnostic != nil
		hasPerGate := tc.assertFeatureOn != nil || tc.assertFeatureOff != nil
		if hasAgnostic == hasPerGate {
			t.Fatalf("case %q must set exactly one of {assertFeatureAgnostic} or {assertFeatureOn+assertFeatureOff}", name)
		}
		if hasPerGate && (tc.assertFeatureOn == nil || tc.assertFeatureOff == nil) {
			t.Fatalf("case %q with per-gate assertions must set BOTH assertFeatureOn and assertFeatureOff", name)
		}

		for _, gate := range []bool{false, true} {
			t.Run(name+"/gate="+strconv.FormatBool(gate), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, gate)
				want := pickAssertion(tc, gate)

				auth := &conditionsAwareFakeAuthorizer{makeDecision: tc.makeDecision}
				storage := NewREST(auth, legacyscheme.Scheme)
				result, err := storage.Create(ctx, &authorizationapi.SubjectAccessReview{Spec: tc.spec}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})

				if want.err != "" {
					if err == nil {
						t.Fatalf("expected error %q, got nil (result=%+v)", want.err, result)
					}
					if err.Error() != want.err {
						t.Errorf("error mismatch\ngot:  %q\nwant: %q", err.Error(), want.err)
					}
					return
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				if want.attrs != nil {
					gotAttrs := auth.attrs.(authorizer.AttributesRecord)
					if want.dropSelectorErr {
						gotAttrs.FieldSelectorParsingErr = nil
						gotAttrs.LabelSelectorParsingErr = nil
					}
					if diff := cmp.Diff(want.attrs, gotAttrs); diff != "" {
						t.Errorf("attrs mismatch (-want +got):\n%s", diff)
					}
				}
				got := result.(*authorizationapi.SubjectAccessReview).Status
				if diff := cmp.Diff(want.status, got); diff != "" {
					t.Errorf("status mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

func mustParse(s string) labels.Requirements {
	selector, err := labels.Parse(s)
	if err != nil {
		panic(err)
	}
	reqs, _ := selector.Requirements()
	return reqs
}
