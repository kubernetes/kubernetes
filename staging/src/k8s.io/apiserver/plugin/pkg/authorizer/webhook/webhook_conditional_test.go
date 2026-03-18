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

package webhook

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authorizationcel "k8s.io/apiserver/pkg/authorization/cel"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	v1 "k8s.io/client-go/tools/clientcmd/api/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// fakeSubjectAccessReviewer implements subjectAccessReviewer for testing.
type fakeSubjectAccessReviewer struct {
	response *authorizationv1.SubjectAccessReview
	err      error
	received *authorizationv1.SubjectAccessReview
}

func (f *fakeSubjectAccessReviewer) Create(_ context.Context, sar *authorizationv1.SubjectAccessReview, _ metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, int, error) {
	f.received = sar
	if f.err != nil {
		return nil, 0, f.err
	}
	return f.response, 200, nil
}

// fakeAuthorizationConditionsReviewer implements authorizationConditionsReviewer for testing.
type fakeAuthorizationConditionsReviewer struct {
	response *authorizationv1alpha1.AuthorizationConditionsReview
	err      error
	received *authorizationv1alpha1.AuthorizationConditionsReview
}

func (f *fakeAuthorizationConditionsReviewer) Create(_ context.Context, acr *authorizationv1alpha1.AuthorizationConditionsReview, _ metav1.CreateOptions) (*authorizationv1alpha1.AuthorizationConditionsReview, int, error) {
	f.received = acr
	if f.err != nil {
		return nil, 0, f.err
	}
	return f.response, 200, nil
}

// newTestWebhookAuthorizer creates a WebhookAuthorizer with fake clients for testing.
func newTestWebhookAuthorizer(
	sarReviewer subjectAccessReviewer,
	acrReviewer authorizationConditionsReviewer,
	decisionOnError authorizer.Decision,
	builtinConditionsEvaluator builtinConditionsEvaluator,
) *WebhookAuthorizer {
	wh, err := newWithBackoff(
		sarReviewer,
		0, // authorizedTTL
		0, // unauthorizedTTL
		testRetryBackoff,
		decisionOnError,
		nil, // matchConditions
		noopAuthorizerMetrics(),
		authorizationcel.NewDefaultCompiler(),
		"test",
		acrReviewer,
		builtinConditionsEvaluator,
	)
	if err != nil {
		panic(fmt.Sprintf("newWithBackoff failed: %v", err))
	}
	return wh
}

var testAttr = authorizer.AttributesRecord{
	User:            &user.DefaultInfo{Name: "alice"},
	Verb:            "create",
	Namespace:       "default",
	Resource:        "configmaps",
	ResourceRequest: true,
}

// TestConditionsAwareAuthorize tests the ConditionsAwareAuthorize method
// including all decision types and error cases.
func TestConditionsAwareAuthorize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	tests := []struct {
		name            string
		sarStatus       authorizationv1.SubjectAccessReviewStatus
		sarErr          error
		decisionOnError authorizer.Decision
		wantDecision    string // expected decision.String()
	}{
		{
			name: "conditions map with allow condition",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						Conditions: []authorizationv1.Condition{
							{
								ID:          "allow-safe-prefix",
								Effect:      authorizationv1.ConditionEffectAllow,
								Condition:   `object.metadata.name.startsWith("safe-")`,
								Type:        "opaque-cel",
								Description: "only allow objects with safe- prefix",
							},
						},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    "ConditionsMap(len=1)",
		},
		{
			name: "allow with reason",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "admin access",
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    `Allow(reason="admin access")`,
		},
		{
			name: "deny with reason",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Allowed: false,
				Denied:  true,
				Reason:  "access denied",
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(reason="access denied")`,
		},
		{
			name: "no opinion",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Allowed: false,
				Denied:  false,
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    "NoOpinion",
		},
		{
			name:            "webhook error with failurePolicy=NoOpinion",
			sarErr:          fmt.Errorf("webhook server unavailable"),
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    `NoOpinion(err="webhook server unavailable")`,
		},
		{
			name:            "webhook error with failurePolicy=Deny",
			sarErr:          fmt.Errorf("webhook server unavailable"),
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(err="webhook server unavailable")`,
		},
		{
			name: "both Allowed and Denied",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Allowed: true,
				Denied:  true,
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(err="webhook subject access review returned both allow and deny response")`,
		},
		{
			name: "both conditional and Denied",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Denied: true,
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(err="webhook subject access review returned both conditional and deny response")`,
		},
		{
			name: "both conditional and Allowed",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Allowed: true,
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(err="webhook subject access review returned both conditional and allow response")`,
		},
		{
			// ConditionsMap must come first in the Union sub-list so that
			// ConditionsAwareDecisionUnion does not simplify away the Union:
			// if Allow were first, it would be the first non-NoOpinion element
			// and the Union would collapse to just Allow.
			name: "union decision",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
					Union: []authorizationv1.ConditionsAwareDecision{
						{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								Conditions: []authorizationv1.Condition{
									{
										ID:     "check-label",
										Effect: authorizationv1.ConditionEffectDeny,
										Type:   "opaque",
									},
								},
							},
						},
						{
							Type:   authorizationv1.ConditionsAwareDecisionTypeAllow,
							Reason: "sub-allow",
						},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    `Union[ConditionsMap(len=1), Allow(reason="sub-allow")]`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sarReviewer := &fakeSubjectAccessReviewer{
				response: &authorizationv1.SubjectAccessReview{Status: tc.sarStatus},
				err:      tc.sarErr,
			}

			wh := newTestWebhookAuthorizer(sarReviewer, nil, tc.decisionOnError, nil)
			decision := wh.ConditionsAwareAuthorize(context.Background(), testAttr)

			if got := decision.String(); got != tc.wantDecision {
				t.Errorf("expected decision %s, got %s", tc.wantDecision, got)
			}
			if sarReviewer.received == nil {
				t.Error("expected SAR to be called")
			} else if sarReviewer.received.Spec.ConditionalAuthorization == nil {
				t.Error("expected ConditionalAuthorization to be set in the outgoing SAR")
			} else if !sarReviewer.received.Spec.ConditionalAuthorization.Enabled {
				t.Error("expected ConditionalAuthorization.Enabled=true in the outgoing SAR")
			}
		})
	}
}

// TestAuthorize_FoldDown tests that the conditions-unaware Authorize method
// folds conditional decisions down to their safe unconditional equivalents.
func TestAuthorize_FoldDown(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	tests := []struct {
		name               string
		serializedDecision authorizationv1.ConditionsAwareDecision
		wantDecision       authorizer.Decision
	}{
		{
			name: "ConditionsMap with Deny condition folds to Deny",
			serializedDecision: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					Conditions: []authorizationv1.Condition{
						{ID: "deny-all", Effect: authorizationv1.ConditionEffectDeny, Type: "opaque"},
					},
				},
			},
			wantDecision: authorizer.DecisionDeny,
		},
		{
			name: "ConditionsMap with a transitive Deny condition folds to Deny",
			serializedDecision: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.ConditionsAwareDecision{
					{
						Type: authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
					},
					{
						Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
						Union: []authorizationv1.ConditionsAwareDecision{
							{
								Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorizationv1.ConditionsMap{
									Conditions: []authorizationv1.Condition{
										{ID: "deny-sth", Effect: authorizationv1.ConditionEffectDeny, Type: "opaque"},
									},
								},
							},
						},
					},
				},
			},
			wantDecision: authorizer.DecisionDeny,
		},
		{
			// Cannot grant unconditional access based on unresolved conditions;
			// fold to NoOpinion so that the next authorizer has a chance to decide.
			name: "ConditionsMap with Allow-only condition folds to NoOpinion",
			serializedDecision: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					Conditions: []authorizationv1.Condition{
						{ID: "allow-all", Effect: authorizationv1.ConditionEffectAllow, Type: "opaque"},
					},
				},
			},
			wantDecision: authorizer.DecisionNoOpinion,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sarReviewer := &fakeSubjectAccessReviewer{
				response: &authorizationv1.SubjectAccessReview{
					Status: authorizationv1.SubjectAccessReviewStatus{
						ConditionalDecision: &tc.serializedDecision,
					},
				},
			}

			wh := newTestWebhookAuthorizer(sarReviewer, nil, authorizer.DecisionNoOpinion, nil)
			d, _, _ := wh.Authorize(context.Background(), testAttr)
			if d != tc.wantDecision {
				t.Errorf("expected %v, got %v", tc.wantDecision, d)
			}
		})
	}
}

// TestEvaluateConditions tests the EvaluateConditions method in a table-driven
// fashion, covering unconditional pass-through, reviewer errors, webhook
// responses, and request content verification.
func TestEvaluateConditions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	tests := []struct {
		name            string
		decision        authorizer.ConditionsAwareDecision
		acrResponse     *authorizationv1alpha1.AuthorizationConditionsReview
		acrErr          error
		noACRReviewer   bool
		decisionOnError authorizer.Decision
		wantDecision    authorizer.Decision
		wantReason      string
		wantErr         bool
		wantErrContains string
		// verifyACR is called after EvaluateConditions with the ACR request that
		// was received by the fake reviewer. Useful for inspecting serialized fields.
		verifyACR                  func(*testing.T, *authorizationv1alpha1.AuthorizationConditionsReview)
		builtinConditionsEvaluator builtinConditionsEvaluator
	}{
		// Unconditional decisions short-circuit without calling the ACR reviewer.
		{
			name:          "unconditional allow passes through",
			decision:      authorizer.ConditionsAwareDecisionAllow("allowed by admin", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionAllow,
			wantReason:    "allowed by admin",
		},
		{
			name:          "unconditional deny passes through",
			decision:      authorizer.ConditionsAwareDecisionDeny("denied by policy", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionDeny,
			wantReason:    "denied by policy",
		},
		{
			name:          "unconditional no-opinion passes through",
			decision:      authorizer.ConditionsAwareDecisionNoOpinion("no opinion", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionNoOpinion,
			wantReason:    "no opinion",
		},
		// No ACR reviewer configured: must fail closed.
		{
			name: "no reviewer, Deny condition, failurePolicy=Deny",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "deny-all", Effect: authorizer.ConditionEffectDeny, Type: "opaque"},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		{
			name: "no reviewer, Allow-only condition, failurePolicy=NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "allow-all", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		// ACR webhook returns various decision types.
		{
			name: "webhook returns allow",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{
					ID: "allow-safe-prefix", Effect: authorizer.ConditionEffectAllow,
					Condition: `object.metadata.name.startsWith("safe-")`, Type: "opaque-cel",
				},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
						Reason: "condition matched",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionAllow,
			wantReason:      "condition matched",
		},
		{
			name: "webhook returns deny",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{
					ID: "deny-restricted", Effect: authorizer.ConditionEffectDeny,
					Condition: `has(object.metadata.labels.restricted)`, Type: "opaque-cel",
				},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeDeny,
						Reason: "restricted label found",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "restricted label found",
		},
		{
			name: "webhook returns no opinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "some-condition", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion,
						Reason: "no matching condition",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "no matching condition",
		},
		// ACR webhook error: fail closed.
		{
			name: "webhook error, failurePolicy=Deny",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Type: "opaque"},
			),
			acrErr:          fmt.Errorf("conditions review webhook unavailable"),
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		{
			name: "webhook error, failurePolicy=NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "a", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			acrErr: fmt.Errorf("conditions review webhook unavailable"),
			// even though DecisionOnError is Deny, there were no Deny conditions, so there authorizer would never have produced a Deny in any case, so this is "ignored" on purpose
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		// Nil Response field in the ACR response.
		{
			name: "nil Response field returns NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			acrResponse:     &authorizationv1alpha1.AuthorizationConditionsReview{},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "",
		},
		// ACR request must contain the serialized conditions.
		{
			name: "ACR request contains conditions from ConditionsMap",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{
					ID:        "allow-by-name",
					Effect:    authorizer.ConditionEffectAllow,
					Condition: `object.metadata.name.startsWith("safe-")`,
					Type:      "opaque-cel",
				},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type: authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionAllow,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				t.Helper()
				if acr == nil {
					t.Fatal("expected ACR to be called")
				}
				req := acr.Request
				if req == nil {
					t.Fatal("expected Request to be non-nil in ACR")
				}
				if req.Decision.Type != authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap {
					t.Errorf("expected ConditionsMap decision type in ACR request, got %q", req.Decision.Type)
				}
				if req.Decision.ConditionsMap == nil {
					t.Fatal("expected ConditionsMap to be non-nil in ACR request")
				}
				if len(req.Decision.ConditionsMap.Conditions) != 1 {
					t.Fatalf("expected 1 condition in ACR request, got %d", len(req.Decision.ConditionsMap.Conditions))
				}
				cond := req.Decision.ConditionsMap.Conditions[0]
				if cond.ID != "allow-by-name" {
					t.Errorf("expected condition ID %q, got %q", "allow-by-name", cond.ID)
				}
				if cond.Type != "opaque-cel" {
					t.Errorf("expected condition type %q, got %q", "opaque-cel", cond.Type)
				}
			},
		},
		// EvaluationError in the response is surfaced as a non-nil error alongside
		// the (still-valid) decision.
		{
			name: "evaluation error alongside allow decision",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:            authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
						Reason:          "partial allow",
						EvaluationError: "condition 'c' evaluation had a warning",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionAllow,
			wantReason:      "partial allow",
			wantErr:         true,
			wantErrContains: "condition 'c' evaluation had a warning",
		},
		{
			name: "full builtin evaluation of one ConditionsMap => Deny",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent", Description: "all ok"},
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent", Description: "very bad"},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			}),
			wantDecision: authorizer.DecisionDeny,
			wantReason:   `condition "d" denied the request with description "very bad"`,
		},
		{
			name: "full builtin evaluation of one ConditionsMap => NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent", Description: "all ok"},
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent", Description: "very bad"},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(false)
			}),
			wantDecision: authorizer.DecisionNoOpinion,
			wantReason:   `no conditions matched`,
		},
		{
			name: "full builtin evaluation of one ConditionsMap => Allow",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent", Description: "all ok"},
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent", Description: "very bad"},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			}),
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `condition "c" allowed the request with description "all ok"`,
		},
		{
			name: "partial builtin evaluation of one ConditionsMap => Allow",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "opaque", Description: "all ok"},       // needs a webhook due to opaque type
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent", Description: "very bad"}, // simplified in-process
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
						Reason: "webhook's allow reason",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			}),
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `webhook's allow reason`,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				wantACRDecision := authorizationv1alpha1.ConditionsAwareDecision{
					Type: authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1alpha1.ConditionsMap{
						Conditions: []authorizationv1alpha1.Condition{
							{
								ID:          "c",
								Effect:      authorizationv1alpha1.ConditionEffectAllow,
								Condition:   "c",
								Type:        "opaque",
								Description: "all ok",
							},
						},
					},
				}
				if acr == nil || acr.Request == nil {
					t.Fatalf("wanted non-nil Request")
				}
				if diff := cmp.Diff(acr.Request.Decision, wantACRDecision); diff != "" {
					t.Errorf("File contents: got=%s, want=%s, diff=%s", acr.Request.Decision, wantACRDecision, diff)
				}
			},
		},
		{
			name: "builtin error does not affect the outcome",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent", Description: "all ok"},
				authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent", Description: "very bad"},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
						Reason: "webhook's allow reason",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultError(fmt.Errorf("unexpected error"))
			}),
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `webhook's allow reason`,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				wantACRDecision := authorizationv1alpha1.ConditionsAwareDecision{
					Type: authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1alpha1.ConditionsMap{
						Conditions: []authorizationv1alpha1.Condition{
							{ // Deny conditions are ordered before Allow ones in the ConditionsMap.Conditions() iterator
								ID:          "d",
								Effect:      authorizationv1alpha1.ConditionEffectDeny,
								Condition:   "d",
								Type:        "transparent",
								Description: "very bad",
							},
							{
								ID:          "c",
								Effect:      authorizationv1alpha1.ConditionEffectAllow,
								Condition:   "c",
								Type:        "transparent",
								Description: "all ok",
							},
						},
					},
				}
				if acr == nil || acr.Request == nil {
					t.Fatalf("wanted non-nil Request")
				}
				if diff := cmp.Diff(wantACRDecision, acr.Request.Decision); diff != "" {
					t.Errorf("File contents: got=%s, want=%s, diff=%s", acr.Request.Decision, wantACRDecision, diff)
				}
			},
		},
		{
			name: "builtin evaluation of union succeeds => Allow",
			decision: authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "a", Effect: authorizer.ConditionEffectAllow, Condition: "a", Type: "transparent"},
					authorizer.GenericCondition{ID: "b", Effect: authorizer.ConditionEffectDeny, Condition: "b", Type: "transparent"},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent"},
					authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent"},
				),
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			}),
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `condition "c" allowed the request`,
		},
		{
			name: "builtin evaluation of union succeeds => Deny",
			decision: authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "a", Effect: authorizer.ConditionEffectAllow, Condition: "a", Type: "transparent"},
					authorizer.GenericCondition{ID: "b", Effect: authorizer.ConditionEffectDeny, Condition: "b", Type: "transparent"},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent"},
					authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent"},
				),
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			}),
			wantDecision: authorizer.DecisionDeny,
			wantReason:   `condition "d" denied the request`,
		},
		{
			name: "first conditionsmap cannot be simplified fully",
			decision: authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "a", Effect: authorizer.ConditionEffectAllow, Condition: "a", Type: "opaque"},
					authorizer.GenericCondition{ID: "b", Effect: authorizer.ConditionEffectDeny, Condition: "b", Type: "transparent"},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent"},
					authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "transparent"},
				),
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion,
						Reason: "webhook's noopinion reason",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			}),
			wantDecision: authorizer.DecisionNoOpinion,
			wantReason:   `webhook's noopinion reason`,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				wantACRDecision := authorizationv1alpha1.ConditionsAwareDecision{
					Type: authorizationv1alpha1.ConditionsAwareDecisionTypeUnion,
					Union: []authorizationv1alpha1.ConditionsAwareDecision{
						{
							Type: authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1alpha1.ConditionsMap{
								Conditions: []authorizationv1alpha1.Condition{
									{
										ID:        "a",
										Effect:    authorizationv1alpha1.ConditionEffectAllow,
										Condition: "a",
										Type:      "opaque",
									},
								},
							},
						},
						{
							Type: authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1alpha1.ConditionsMap{
								Conditions: []authorizationv1alpha1.Condition{
									{
										ID:        "d",
										Effect:    authorizationv1alpha1.ConditionEffectDeny,
										Condition: "d",
										Type:      "transparent",
									},
									{
										ID:        "c",
										Effect:    authorizationv1alpha1.ConditionEffectAllow,
										Condition: "c",
										Type:      "transparent",
									},
								},
							},
						},
					},
				}
				if acr == nil || acr.Request == nil {
					t.Fatalf("wanted non-nil Request")
				}
				if diff := cmp.Diff(wantACRDecision, acr.Request.Decision); diff != "" {
					t.Errorf("File contents: got=%s, want=%s, diff=%s", acr.Request.Decision, wantACRDecision, diff)
				}
			},
		},
		{
			name: "first conditionsmap can be simplified fully, but not second",
			decision: authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "a", Effect: authorizer.ConditionEffectAllow, Condition: "a", Type: "transparent"},
					authorizer.GenericCondition{ID: "b", Effect: authorizer.ConditionEffectDeny, Condition: "b", Type: "transparent"},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Condition: "c", Type: "transparent"},
					authorizer.GenericCondition{ID: "d", Effect: authorizer.ConditionEffectDeny, Condition: "d", Type: "opaque"},
				),
				authorizer.ConditionsAwareDecisionDeny("something later denies", nil),
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
						Reason: "webhook's second authorizer allows",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			builtinConditionsEvaluator: conditionsEvaluationFunc(func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			}),
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `webhook's second authorizer allows`,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				wantACRDecision := authorizationv1alpha1.ConditionsAwareDecision{
					Type: authorizationv1alpha1.ConditionsAwareDecisionTypeUnion,
					Union: []authorizationv1alpha1.ConditionsAwareDecision{
						{
							Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion,
							Reason: "no conditions matched",
						},
						{
							Type: authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1alpha1.ConditionsMap{
								Conditions: []authorizationv1alpha1.Condition{
									{
										ID:        "d",
										Effect:    authorizationv1alpha1.ConditionEffectDeny,
										Condition: "d",
										Type:      "opaque",
									},
									{
										ID:        "c",
										Effect:    authorizationv1alpha1.ConditionEffectAllow,
										Condition: "c",
										Type:      "transparent",
									},
								},
							},
						},
						{
							Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeDeny,
							Reason: "something later denies",
						},
					},
				}
				if acr == nil || acr.Request == nil {
					t.Fatalf("wanted non-nil Request")
				}
				if diff := cmp.Diff(wantACRDecision, acr.Request.Decision); diff != "" {
					t.Errorf("File contents: got=%s, want=%s, diff=%s", acr.Request.Decision, wantACRDecision, diff)
				}
			},
		},
		// Unknown response type must fail closed.
		{
			name: "unknown response type fails closed, deny condition",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectDeny, Type: "opaque"},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type: "UnknownDecisionType",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion, // ignored on purpose, Deny condition is stronger
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		{
			name: "unknown response type fails closed, allow condition",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				authorizer.GenericCondition{ID: "c", Effect: authorizer.ConditionEffectAllow, Type: "opaque"},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type: "UnknownDecisionType",
					},
				},
			},
			decisionOnError: authorizer.DecisionDeny, // ignored on purpose, Deny condition is stronger
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         true,
		},
		// Union decision is serialized into the ACR request and its response is honored.
		{
			// ConditionsMap must come first so ConditionsAwareDecisionUnion
			// does not simplify the Union away (an Allow as first non-NoOpinion
			// would collapse to just Allow).
			name: "union decision serialized to ACR request",
			decision: authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "check-label", Effect: authorizer.ConditionEffectDeny, Type: "opaque"},
				),
				authorizer.ConditionsAwareDecisionAllow("first-allow", nil),
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1alpha1.ConditionsAwareDecision{
						Type: authorizationv1alpha1.ConditionsAwareDecisionTypeDeny,
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			verifyACR: func(t *testing.T, acr *authorizationv1alpha1.AuthorizationConditionsReview) {
				t.Helper()
				if acr == nil {
					t.Fatal("expected ACR to be called")
				}
				if acr.Request.Decision.Type != authorizationv1alpha1.ConditionsAwareDecisionTypeUnion {
					t.Errorf("expected Union type in ACR request, got %q", acr.Request.Decision.Type)
				}
				if len(acr.Request.Decision.Union) != 2 {
					t.Errorf("expected 2 sub-decisions in Union, got %d", len(acr.Request.Decision.Union))
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var acrReviewer authorizationConditionsReviewer
			var fakeACR *fakeAuthorizationConditionsReviewer
			if !tc.noACRReviewer {
				fakeACR = &fakeAuthorizationConditionsReviewer{
					response: tc.acrResponse,
					err:      tc.acrErr,
				}
				acrReviewer = fakeACR
			}

			wh := newTestWebhookAuthorizer(&fakeSubjectAccessReviewer{}, acrReviewer, tc.decisionOnError, tc.builtinConditionsEvaluator)
			d, reason, err := wh.EvaluateConditions(context.Background(), tc.decision, authorizer.ConditionsData{})

			if (err != nil) != tc.wantErr {
				t.Errorf("wantErr=%v, got err=%v", tc.wantErr, err)
			}
			if tc.wantErrContains != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErrContains)) {
				t.Errorf("expected error containing %q, got %v", tc.wantErrContains, err)
			}
			if d != tc.wantDecision {
				t.Errorf("expected decision %v, got %v", tc.wantDecision, d)
			}
			if reason != tc.wantReason {
				t.Errorf("expected reason %q, got %q", tc.wantReason, reason)
			}
			if tc.verifyACR != nil {
				var received *authorizationv1alpha1.AuthorizationConditionsReview
				if fakeACR != nil {
					received = fakeACR.received
				}
				tc.verifyACR(t, received)
			}
		})
	}
}

type conditionsEvaluationFunc func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult

func (f conditionsEvaluationFunc) EvaluateCondition(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
	return f(ctx, condition, data)
}

// TestConditionsAwareAuthorize_EndToEnd tests a full round-trip using an HTTP
// test server to simulate the webhook, verifying that ConditionsAwareAuthorize
// correctly sends ConditionalAuthorization in the spec and deserializes the
// conditional response.
func TestConditionsAwareAuthorize_EndToEnd(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	expectedDecision := &authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			Conditions: []authorizationv1.Condition{
				{
					ID:          "allow-prefix",
					Effect:      authorizationv1.ConditionEffectAllow,
					Condition:   `object.metadata.name.startsWith("ok-")`,
					Type:        "opaque-cel",
					Description: "only allow objects starting with ok-",
				},
			},
		},
	}

	var receivedConditionalAuth bool

	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var sar authorizationv1.SubjectAccessReview
		if err := json.Unmarshal(body, &sar); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		receivedConditionalAuth = sar.Spec.ConditionalAuthorization != nil && sar.Spec.ConditionalAuthorization.Enabled

		resp := authorizationv1.SubjectAccessReview{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "authorization.k8s.io/v1",
				Kind:       "SubjectAccessReview",
			},
			Status: authorizationv1.SubjectAccessReviewStatus{
				ConditionalDecision: expectedDecision,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	serverCAPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
	wh, err := newV1Authorizer(server.URL, nil, nil, serverCAPEM, 0, noopAuthorizerMetrics(), authorizationcel.NewDefaultCompiler(), nil, "test")
	if err != nil {
		t.Fatalf("failed to create authorizer: %v", err)
	}

	decision := wh.ConditionsAwareAuthorize(context.Background(), testAttr)

	if !receivedConditionalAuth {
		t.Error("expected ConditionalAuthorization to be sent in the SAR request")
	}
	if got := decision.String(); got != "ConditionsMap(len=1)" {
		t.Errorf("expected ConditionsMap(len=1), got: %s", got)
	}
}

// TestEvaluateConditions_EndToEnd tests that EvaluateConditions correctly
// sends the serialized decision to an ACR webhook and processes the response.
func TestEvaluateConditions_EndToEnd(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
	var receivedACR *authorizationv1alpha1.AuthorizationConditionsReview

	acrServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var acr authorizationv1alpha1.AuthorizationConditionsReview
		if err := json.Unmarshal(body, &acr); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		receivedACR = &acr

		resp := authorizationv1alpha1.AuthorizationConditionsReview{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "authorization.k8s.io/v1alpha1",
				Kind:       "AuthorizationConditionsReview",
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				Decision: authorizationv1alpha1.ConditionsAwareDecision{
					Type:   authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
					Reason: "webhook allowed",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer acrServer.Close()

	condDecision := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{
			ID:        "allow-safe",
			Effect:    authorizer.ConditionEffectAllow,
			Condition: `object.metadata.name.startsWith("safe-")`,
			Type:      "opaque-cel",
		},
	)

	acrConfig, err := conditionsReviewConfigFromTLSServer(acrServer, "v1alpha1")
	if err != nil {
		t.Fatalf("failed to build conditions review config: %v", err)
	}

	acrReviewer, err := authorizationConditionsReviewInterfaceFromConfig(acrConfig, "v1alpha1", testRetryBackoff) //nolint:staticcheck
	if err != nil {
		t.Fatalf("failed to build ACR reviewer: %v", err)
	}

	wh := newTestWebhookAuthorizer(&fakeSubjectAccessReviewer{}, acrReviewer, authorizer.DecisionNoOpinion, nil)
	d, reason, err := wh.EvaluateConditions(context.Background(), condDecision, authorizer.ConditionsData{})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if d != authorizer.DecisionAllow {
		t.Errorf("expected Allow, got %v", d)
	}
	if reason != "webhook allowed" {
		t.Errorf("expected reason %q, got %q", "webhook allowed", reason)
	}

	if receivedACR == nil {
		t.Fatal("expected ACR to be called")
	}
	if receivedACR.Request == nil {
		t.Fatal("expected Request to be non-nil")
	}
	if receivedACR.Request.Decision.Type != authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap {
		t.Errorf("expected ConditionsMap in ACR request, got %q", receivedACR.Request.Decision.Type)
	}
}

// conditionsReviewConfigFromTLSServer creates a rest.Config pointing to the
// given TLS test server for use with authorizationConditionsReviewInterfaceFromConfig.
func conditionsReviewConfigFromTLSServer(server *httptest.Server, _ string) (*rest.Config, error) {
	tempfile, err := os.CreateTemp("", "acr-kubeconfig-")
	if err != nil {
		return nil, err
	}
	defer func() { _ = os.Remove(tempfile.Name()) }()

	caCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
	config := v1.Config{
		Clusters: []v1.NamedCluster{{
			Cluster: v1.Cluster{
				Server:                   server.URL,
				CertificateAuthorityData: caCertPEM,
			},
		}},
		AuthInfos: []v1.NamedAuthInfo{{}},
	}
	if err := json.NewEncoder(tempfile).Encode(config); err != nil {
		return nil, err
	}
	return webhookutil.LoadKubeconfig(tempfile.Name(), nil)
}
