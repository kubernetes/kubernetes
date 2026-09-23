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
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authorizationcel "k8s.io/apiserver/pkg/authorization/cel"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	v1 "k8s.io/client-go/tools/clientcmd/api/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// fakeSubjectAccessReviewer implements subjectAccessReviewer for testing.
// calls counts invocations, so a test can tell a cached answer from a second trip to the
// webhook.
type fakeSubjectAccessReviewer struct {
	response *authorizationv1.SubjectAccessReview
	err      error
	received *authorizationv1.SubjectAccessReview
	calls    int
}

func (f *fakeSubjectAccessReviewer) Create(_ context.Context, sar *authorizationv1.SubjectAccessReview, _ metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, int, error) {
	f.received = sar
	f.calls++
	if f.err != nil {
		return nil, 0, f.err
	}
	return f.response, 200, nil
}

// fakeAuthorizationConditionsReviewer implements authorizationConditionsReviewer for testing.
// It mirrors real webhook behavior by echoing the request's AdmissionRequest.UID back
// into the response, so the caller's UID correlation check succeeds.
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
	if f.response != nil && f.response.Response != nil && acr.Request != nil && acr.Request.AdmissionRequest != nil {
		f.response.Response.UID = acr.Request.AdmissionRequest.UID
	}
	return f.response, 200, nil
}

// fakeConditionsData is a minimal authorizer.ConditionsData used by EvaluateConditions
// tests. It carries just enough to satisfy the interface; individual tests do not
// depend on any particular field value.
type fakeConditionsData struct{}

func (fakeConditionsData) GetName() string      { return "obj" }
func (fakeConditionsData) GetNamespace() string { return "default" }
func (fakeConditionsData) GetResource() schema.GroupVersionResource {
	return schema.GroupVersionResource{}
}
func (fakeConditionsData) GetSubresource() string { return "" }
func (fakeConditionsData) GetOperation() authorizer.AdmissionOperation {
	return authorizer.AdmissionOperation("CREATE")
}
func (fakeConditionsData) GetOperationOptions() runtime.Object { return nil }
func (fakeConditionsData) IsDryRun() bool                      { return false }
func (fakeConditionsData) GetObject() runtime.Object           { return nil }
func (fakeConditionsData) GetOldObject() runtime.Object        { return nil }
func (fakeConditionsData) GetKind() schema.GroupVersionKind    { return schema.GroupVersionKind{} }
func (fakeConditionsData) GetUserInfo() user.Info {
	return &user.DefaultInfo{Name: "alice"}
}

// newTestWebhookAuthorizer creates a WebhookAuthorizer with fake clients for testing.
func newTestWebhookAuthorizer(
	sarReviewer subjectAccessReviewer,
	acrReviewer authorizationConditionsReviewer,
	decisionOnError authorizer.Decision,
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

// testCtx is a context that models what the API server request chain always
// installs: a RequestInfo describing the in-flight request. Downstream
// declarative validation (invoked from the webhook authorizer's SAR/ACR
// response-validation step) reads the API version off this RequestInfo, so
// tests that call the authorizer methods directly must supply one.
var testCtx = genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
	IsResourceRequest: true,
	APIGroup:          "",
	APIVersion:        "v1",
	Resource:          "configmaps",
	Verb:              "create",
})

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
						AllowConditions: []authorizationv1.Condition{
							{
								ID:          "example.com/allow-safe-prefix",
								Condition:   `object.metadata.name.startsWith("safe-")`,
								Type:        "example.com/opaque-cel",
								Description: "only allow objects with safe- prefix",
							},
						},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    "ConditionsMap(allows=1)",
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
			// The failure path in conditionsAwareFailureDecision tags the decision
			// with reason="failed closed" so operators can distinguish a "the
			// webhook itself failed" outcome from an explicit authorizer answer.
			wantDecision: `NoOpinion(reason="failed closed", err="webhook server unavailable")`,
		},
		{
			name:            "webhook error with failurePolicy=Deny",
			sarErr:          fmt.Errorf("webhook server unavailable"),
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(reason="failed closed", err="webhook server unavailable")`,
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
			// The declarative-validation layer (ValidateSubjectAccessReviewCreate) emits an
			// aggregate of two errors when both status.denied and status.conditionalDecision
			// are set with an empty conditionsMap: the union violation and the required-when-
			// type=ConditionsMap violation.
			name: "both conditional and Denied",
			sarStatus: authorizationv1.SubjectAccessReviewStatus{
				Denied: true,
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    `Deny(reason="failed closed", err="[status.denied: Forbidden: must be false when status.conditionalDecision.type=ConditionsMap, status.conditionalDecision.conditionsMap: Invalid value: \"\": must be specified when ` + "`type`" + ` is \"ConditionsMap\"]")`,
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
			wantDecision:    `Deny(reason="failed closed", err="[status.allowed: Forbidden: must be false when status.conditionalDecision.type=ConditionsMap, status.conditionalDecision.conditionsMap: Invalid value: \"\": must be specified when ` + "`type`" + ` is \"ConditionsMap\"]")`,
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
					Union: []authorizationv1.NamedConditionsAwareDecision{
						{
							AuthorizerName: "cm",
							Decision: authorizationv1.ConditionsAwareDecision{
								Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorizationv1.ConditionsMap{
									DenyConditions: []authorizationv1.Condition{
										{
											ID:   "example.com/check-label",
											Type: "example.com/opaque",
										},
									},
								},
							},
						},
						{
							AuthorizerName: "allow",
							Decision: authorizationv1.ConditionsAwareDecision{
								Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
								Allow: &authorizationv1.UnconditionalDecision{Reason: "sub-allow"},
							},
						},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    `Union[cm: ConditionsMap(denies=1), allow: Allow(reason="sub-allow")]`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sarReviewer := &fakeSubjectAccessReviewer{
				response: &authorizationv1.SubjectAccessReview{Status: tc.sarStatus},
				err:      tc.sarErr,
			}

			wh := newTestWebhookAuthorizer(sarReviewer, nil, tc.decisionOnError)
			decision := wh.ConditionsAwareAuthorize(testCtx, testAttr)

			if got := decision.String(); got != tc.wantDecision {
				t.Errorf("expected decision %s, got %s", tc.wantDecision, got)
			}
			if sarReviewer.received == nil {
				t.Error("expected SAR to be called")
			} else if !sarReviewer.received.Spec.AuthorizationOptions.SupportsConditionalAuthorization() {
				t.Error("expected ConditionalAuthorization to be enabled in the outgoing SAR")
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
					DenyConditions: []authorizationv1.Condition{
						{ID: "example.com/deny-all", Type: "example.com/opaque"},
					},
				},
			},
			wantDecision: authorizer.DecisionDeny,
		},
		{
			name: "ConditionsMap with a transitive Deny condition folds to Deny",
			serializedDecision: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.NamedConditionsAwareDecision{
					{
						AuthorizerName: "outer-noop",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
							NoOpinion: &authorizationv1.UnconditionalDecision{},
						},
					},
					{
						AuthorizerName: "inner-union",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
							Union: []authorizationv1.NamedConditionsAwareDecision{
								{
									AuthorizerName: "inner-cm",
									Decision: authorizationv1.ConditionsAwareDecision{
										Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
										ConditionsMap: &authorizationv1.ConditionsMap{
											DenyConditions: []authorizationv1.Condition{
												{ID: "example.com/deny-sth", Type: "example.com/opaque"},
											},
										},
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
					AllowConditions: []authorizationv1.Condition{
						{ID: "example.com/allow-all", Type: "example.com/opaque"},
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

			wh := newTestWebhookAuthorizer(sarReviewer, nil, authorizer.DecisionNoOpinion)
			d, _, _ := wh.Authorize(testCtx, testAttr)
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
		// wantErr is the exact expected error text; the empty string means no error.
		wantErr string
		// verifyACR is called after EvaluateConditions with the ACR request that
		// was received by the fake reviewer. Useful for inspecting serialized fields.
		verifyACR func(*testing.T, *authorizationv1alpha1.AuthorizationConditionsReview)
	}{
		// Unconditional decisions are a programmer error at this API layer:
		// EvaluateConditions is only ever supposed to be called on a conditional
		// decision. It fails closed to the sub-decision's FailureDecision.
		{
			name:          "unconditional allow rejected: FailureDecision is NoOpinion",
			decision:      authorizer.ConditionsAwareDecisionAllow("allowed by admin", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionNoOpinion,
			wantReason:    "failed closed",
			wantErr:       "got unconditional decisionToEvaluate in EvaluateConditions",
		},
		{
			name:          "unconditional deny rejected: FailureDecision is Deny",
			decision:      authorizer.ConditionsAwareDecisionDeny("denied by policy", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionDeny,
			wantReason:    "failed closed",
			wantErr:       "got unconditional decisionToEvaluate in EvaluateConditions",
		},
		{
			name:          "unconditional no-opinion rejected: FailureDecision is NoOpinion",
			decision:      authorizer.ConditionsAwareDecisionNoOpinion("no opinion", nil),
			noACRReviewer: true,
			wantDecision:  authorizer.DecisionNoOpinion,
			wantReason:    "failed closed",
			wantErr:       "got unconditional decisionToEvaluate in EvaluateConditions",
		},
		// No ACR reviewer configured: must fail closed.
		{
			name: "no reviewer, Deny condition, failurePolicy=Deny",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/deny-all", Type: "example.com/opaque"}},
				nil, nil,
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         "no authorization conditions review client configured for the webhook authorizer, cannot evaluate conditions",
		},
		{
			name: "no reviewer, Allow-only condition, failurePolicy=NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/allow-all", Type: "example.com/opaque"}},
			),
			noACRReviewer:   true,
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         "no authorization conditions review client configured for the webhook authorizer, cannot evaluate conditions",
		},
		// ACR webhook returns various decision types.
		{
			name: "webhook returns allow",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{
					ID:        "example.com/allow-safe-prefix",
					Condition: `object.metadata.name.startsWith("safe-")`,
					Type:      "example.com/opaque-cel",
				}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
						Allow: &authorizationv1.UnconditionalDecision{Reason: "condition matched"},
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
				[]authorizer.Condition{authorizer.GenericCondition{
					ID:        "example.com/deny-restricted",
					Condition: `has(object.metadata.labels.restricted)`,
					Type:      "example.com/opaque-cel",
				}},
				nil, nil,
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
						Deny: &authorizationv1.UnconditionalDecision{Reason: "restricted label found"},
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
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/some-condition", Type: "example.com/opaque"}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
						NoOpinion: &authorizationv1.UnconditionalDecision{Reason: "no matching condition"},
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
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/d", Type: "example.com/opaque"}},
				nil, nil,
			),
			acrErr:          fmt.Errorf("conditions review webhook unavailable"),
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         "conditions review webhook unavailable",
		},
		{
			name: "webhook error, failurePolicy=NoOpinion",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/a", Type: "example.com/opaque"}},
			),
			acrErr: fmt.Errorf("conditions review webhook unavailable"),
			// even though DecisionOnError is Deny, there were no Deny conditions, so there authorizer would never have produced a Deny in any case, so this is "ignored" on purpose
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         "conditions review webhook unavailable",
		},
		// nil response field in the ACR response.
		{
			name: "nil response returns FailureDecision (Deny)",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c", Type: "example.com/opaque"}},
				nil, nil,
			),
			acrResponse:     &authorizationv1alpha1.AuthorizationConditionsReview{},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         "response: Required value: must be set in AuthorizationConditionsReview responses",
		},
		{
			name: "nil response returns FailureDecision (NoOpinion)",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c", Type: "example.com/opaque"}},
			),
			acrResponse:     &authorizationv1alpha1.AuthorizationConditionsReview{},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         "response: Required value: must be set in AuthorizationConditionsReview responses",
		},
		// ACR request must contain the serialized conditions.
		{
			name: "ACR request contains conditions from ConditionsMap",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{
					ID:        "example.com/allow-by-name",
					Condition: `object.metadata.name.startsWith("safe-")`,
					Type:      "example.com/opaque-cel",
				}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
						Allow: &authorizationv1.UnconditionalDecision{},
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
				if req.Decision.Type != authorizationv1.ConditionsAwareDecisionTypeConditionsMap {
					t.Errorf("expected ConditionsMap decision type in ACR request, got %q", req.Decision.Type)
				}
				if req.Decision.ConditionsMap == nil {
					t.Fatal("expected ConditionsMap to be non-nil in ACR request")
				}
				if len(req.Decision.ConditionsMap.AllowConditions) != 1 {
					t.Fatalf("expected 1 allow condition in ACR request, got %d", len(req.Decision.ConditionsMap.AllowConditions))
				}
				cond := req.Decision.ConditionsMap.AllowConditions[0]
				if cond.ID != "example.com/allow-by-name" {
					t.Errorf("expected condition ID %q, got %q", "example.com/allow-by-name", cond.ID)
				}
				if cond.Type != "example.com/opaque-cel" {
					t.Errorf("expected condition type %q, got %q", "example.com/opaque-cel", cond.Type)
				}
			},
		},
		// EvaluationError in the response is surfaced as a non-nil error alongside
		// the (still-valid) decision.
		{
			name: "evaluation error alongside allow decision",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c", Type: "example.com/opaque"}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeAllow,
						Allow: &authorizationv1.UnconditionalDecision{
							Reason:          "partial allow",
							EvaluationError: "condition 'c' evaluation had a warning",
						},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionAllow,
			wantReason:      "partial allow",
			wantErr:         "condition 'c' evaluation had a warning",
		},
		// Unknown response type must fail closed.
		{
			name: "unknown response type fails closed, deny condition",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c", Type: "example.com/opaque"}},
				nil, nil,
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: "UnknownDecisionType",
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion, // ignored on purpose, Deny condition is stronger
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         `unrecognized decisionToEvaluate type "UnknownDecisionType"`,
		},
		{
			name: "unknown response type fails closed, allow condition",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c", Type: "example.com/opaque"}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: "UnknownDecisionType",
					},
				},
			},
			decisionOnError: authorizer.DecisionDeny, // ignored on purpose, Deny condition is stronger
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         `unrecognized decisionToEvaluate type "UnknownDecisionType"`,
		},
		// Union decision is serialized into the ACR request and its response is honored.
		{
			// ConditionsMap must come first so ConditionsAwareDecisionUnion
			// does not simplify the Union away (an Allow as first non-NoOpinion
			// would collapse to just Allow).
			name: "union decision serialized to ACR request",
			decision: func() authorizer.ConditionsAwareDecision {
				var u authorizer.ConditionsAwareDecisionUnion
				u.Add("cm", authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/check-label", Type: "example.com/opaque"}},
					nil, nil,
				))
				u.Add("allow", authorizer.ConditionsAwareDecisionAllow("first-allow", nil))
				return u.ToDecision()
			}(),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
						Deny: &authorizationv1.UnconditionalDecision{},
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
				if acr.Request.Decision.Type != authorizationv1.ConditionsAwareDecisionTypeUnion {
					t.Errorf("expected Union type in ACR request, got %q", acr.Request.Decision.Type)
				}
				if len(acr.Request.Decision.Union) != 2 {
					t.Errorf("expected 2 sub-decisions in Union, got %d", len(acr.Request.Decision.Union))
				}
			},
		},
		// The returned decision must be a subset of decision.PossibleDecisions();
		// otherwise the webhook is treated as misbehaving and we fail closed to
		// decision.FailureDecision(). Each case below hits a different arm of the
		// switch that enforces this in EvaluateConditions. IDs and types must be
		// domain-prefixed or ConditionsAwareDecisionConditionsMap will collapse
		// the decision to an unconditional failure.
		{
			// PossibleDecisions of a Deny-only ConditionsMap is {NoOpinion, Deny},
			// so returning Allow is out-of-band. FailureDecision() is Deny because
			// Deny is a possible outcome.
			name: "deny-only conditions map: webhook returns Allow fails closed",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/deny"}},
				nil, nil,
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
						Allow: &authorizationv1.UnconditionalDecision{Reason: "should not be honored"},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         "webhook authorizer tried to return Allow from EvaluateConditions, but the possible outcomes were [Deny NoOpinion]",
		},
		{
			// PossibleDecisions of an Allow-only ConditionsMap is {NoOpinion, Allow},
			// so returning Deny is out-of-band. FailureDecision() is NoOpinion because
			// Deny is not a possible outcome.
			name: "allow-only conditions map: webhook returns Deny fails closed",
			decision: authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/allow"}},
			),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
						Deny: &authorizationv1.UnconditionalDecision{Reason: "should not be honored"},
					},
				},
			},
			decisionOnError: authorizer.DecisionDeny,
			wantDecision:    authorizer.DecisionNoOpinion,
			wantReason:      "failed closed",
			wantErr:         "webhook authorizer tried to return Deny from EvaluateConditions, but the possible outcomes were [Allow NoOpinion]",
		},
		{
			// A Union that contains any unconditional Allow/Deny leaf drops NoOpinion
			// from PossibleDecisions, so a webhook returning NoOpinion is out-of-band.
			// This is the only way to exercise the NoOpinion arm of the check with a
			// decision that is still conditional (unconditional decisions short-circuit
			// earlier via IsUnconditional).
			name: "union with unconditional allow leaf: webhook returns NoOpinion fails closed",
			decision: func() authorizer.ConditionsAwareDecision {
				var u authorizer.ConditionsAwareDecisionUnion
				u.Add("cm", authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/deny"}},
					nil, nil,
				))
				u.Add("allow", authorizer.ConditionsAwareDecisionAllow("unconditional allow", nil))
				return u.ToDecision()
			}(),
			acrResponse: &authorizationv1alpha1.AuthorizationConditionsReview{
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					Decision: authorizationv1.ConditionsAwareDecision{
						Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
						NoOpinion: &authorizationv1.UnconditionalDecision{Reason: "should not be honored"},
					},
				},
			},
			decisionOnError: authorizer.DecisionNoOpinion,
			wantDecision:    authorizer.DecisionDeny,
			wantReason:      "failed closed",
			wantErr:         "webhook authorizer tried to return NoOpinion from EvaluateConditions, but the possible outcomes were [Deny Allow]",
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

			wh := newTestWebhookAuthorizer(&fakeSubjectAccessReviewer{}, acrReviewer, tc.decisionOnError)
			d, reason, err := wh.EvaluateConditions(testCtx, tc.decision, fakeConditionsData{})

			var gotErr string
			if err != nil {
				gotErr = err.Error()
			}
			if gotErr != tc.wantErr {
				t.Errorf("expected error %q, got %q", tc.wantErr, gotErr)
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

// TestConditionsAwareAuthorize_EndToEnd tests a full round-trip using an HTTP
// test server to simulate the webhook, verifying that ConditionsAwareAuthorize
// correctly sends ConditionalAuthorization in the spec and deserializes the
// conditional response.
func TestConditionsAwareAuthorize_EndToEnd(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	expectedDecision := &authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			AllowConditions: []authorizationv1.Condition{
				{
					ID:          "example.com/allow-prefix",
					Condition:   `object.metadata.name.startsWith("ok-")`,
					Type:        "example.com/opaque-cel",
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

		receivedConditionalAuth = sar.Spec.AuthorizationOptions.SupportsConditionalAuthorization()

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

	decision := wh.ConditionsAwareAuthorize(testCtx, testAttr)

	if !receivedConditionalAuth {
		t.Error("expected ConditionalAuthorization to be sent in the SAR request")
	}
	if got := decision.String(); got != "ConditionsMap(allows=1)" {
		t.Errorf("expected ConditionsMap(allows=1), got: %s", got)
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
				UID: acr.Request.AdmissionRequest.UID,
				Decision: authorizationv1.ConditionsAwareDecision{
					Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
					Allow: &authorizationv1.UnconditionalDecision{Reason: "webhook allowed"},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer acrServer.Close()

	condDecision := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{authorizer.GenericCondition{
			ID:        "example.com/allow-safe",
			Condition: `object.metadata.name.startsWith("safe-")`,
			Type:      "example.com/opaque-cel",
		}},
	)

	acrConfig, err := conditionsReviewConfigFromTLSServer(acrServer, "v1alpha1")
	if err != nil {
		t.Fatalf("failed to build conditions review config: %v", err)
	}

	acrReviewer, err := authorizationConditionsReviewInterfaceFromConfig(acrConfig, "v1alpha1", testRetryBackoff) //nolint:staticcheck
	if err != nil {
		t.Fatalf("failed to build ACR reviewer: %v", err)
	}

	wh := newTestWebhookAuthorizer(&fakeSubjectAccessReviewer{}, acrReviewer, authorizer.DecisionNoOpinion)
	d, reason, err := wh.EvaluateConditions(testCtx, condDecision, fakeConditionsData{})

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
	if receivedACR.Request.Decision.Type != authorizationv1.ConditionsAwareDecisionTypeConditionsMap {
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

// TestConditionsAwareAuthorize_V1beta1Downgrade covers ConditionsAwareAuthorize
// being called against a webhook configured for v1beta1, which can only express
// unconditional authorization.
//
// The v1beta1 SubjectAccessReview has neither spec.authorizationOptions nor
// status.conditionalDecision, so sending the conditional opt-in to such a webhook
// silently downgrades to unconditional mode. Two properties must hold:
//
//  1. The opt-in is not sent on the wire. The webhook sees a plain SAR, identical
//     to what a pre-conditions API server would have sent, so existing v1beta1
//     webhooks keep working when a caller starts using the conditions-aware path.
//  2. The decision handed back is always unconditional, even if the webhook tries
//     to answer with conditions. v1beta1 cannot carry them, so the response decodes
//     without them rather than smuggling an unvalidated conditional decision into
//     the authorizer.
func TestConditionsAwareAuthorize_V1beta1Downgrade(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	tests := []struct {
		name string
		// sendStatusJSON is the raw JSON for the v1beta1 status the webhook answers
		// with. It is raw so that a response can attempt to carry a conditional
		// decision that the v1beta1 schema has no field for.
		sendStatusJSON string
		wantDecision   string
	}{
		{
			name:           "allow",
			sendStatusJSON: `{"allowed":true,"reason":"beta allowed"}`,
			wantDecision:   `Allow(reason="beta allowed")`,
		},
		{
			name:           "deny",
			sendStatusJSON: `{"allowed":false,"denied":true,"reason":"beta denied"}`,
			wantDecision:   `Deny(reason="beta denied")`,
		},
		{
			name:           "no opinion",
			sendStatusJSON: `{"allowed":false}`,
			wantDecision:   "NoOpinion",
		},
		{
			name:           "evaluation error is preserved",
			sendStatusJSON: `{"allowed":false,"reason":"beta errored","evaluationError":"backend lookup failed"}`,
			wantDecision:   `NoOpinion(reason="beta errored", err="backend lookup failed")`,
		},
		{
			// A v1beta1 webhook has no conditionalDecision field to populate. If one
			// shows up anyway it must be dropped on decode, leaving the unconditional
			// allowed/denied fields as the only signal.
			// Note: This is _invalid_ behavior of the SAR server, which must never return conditions when none were asked for,
			// but here we just show what happens if the SAR server behaves wrongly.
			name: "conditional decision in a v1beta1 response is dropped",
			sendStatusJSON: `{"allowed":false,"reason":"beta cannot condition",` +
				`"conditionalDecision":{"type":"ConditionsMap","conditionsMap":{"allowConditions":[` +
				`{"id":"example.com/allow-all","type":"example.com/opaque"}]}}}`,
			wantDecision: `NoOpinion(reason="beta cannot condition")`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var rawBody []byte
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rawBody, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"apiVersion":"authorization.k8s.io/v1beta1","kind":"SubjectAccessReview","status":%s}`, tc.sendStatusJSON)
			}))
			defer server.Close()

			serverCAPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
			wh, err := newV1beta1Authorizer(server.URL, nil, nil, serverCAPEM, 0)
			if err != nil {
				t.Fatalf("failed to create v1beta1 authorizer: %v", err)
			}

			decision := wh.ConditionsAwareAuthorize(testCtx, testAttr)
			if got := decision.String(); got != tc.wantDecision {
				t.Errorf("expected decision %s, got %s", tc.wantDecision, got)
			}
			// The downgraded decision must be usable by a conditions-unaware caller.
			if !decision.IsUnconditional() {
				t.Errorf("expected an unconditional decision from a v1beta1 webhook, got %s", decision.String())
			}

			// Verify the request was serialized as v1beta1 without the conditional opt-in.
			if rawBody == nil {
				t.Fatal("expected the webhook to be called")
			}
			var sent struct {
				APIVersion string                     `json:"apiVersion"`
				Kind       string                     `json:"kind"`
				Spec       map[string]json.RawMessage `json:"spec"`
			}
			if err := json.Unmarshal(rawBody, &sent); err != nil {
				t.Fatalf("failed to unmarshal the outgoing request: %v", err)
			}
			if sent.APIVersion != "authorization.k8s.io/v1beta1" {
				t.Errorf("expected apiVersion %q, got %q", "authorization.k8s.io/v1beta1", sent.APIVersion)
			}
			if sent.Kind != "SubjectAccessReview" {
				t.Errorf("expected kind %q, got %q", "SubjectAccessReview", sent.Kind)
			}
			// authorizationOptions is v1-only. Sending it to a v1beta1 webhook would
			// advertise conditional support that the response schema cannot express.
			if _, ok := sent.Spec["authorizationOptions"]; ok {
				t.Errorf("expected no authorizationOptions in the outgoing v1beta1 spec, got spec %v", sent.Spec)
			}
		})
	}
}

// TestConditionalResponseCaching covers which TTL a conditional response is cached under.
// A ConditionsMap is neither an allow nor a deny, so the choice depends on the caller: to
// one that can evaluate conditions the response is as useful as an allow and is cached
// under authorizedTTL, while to one that cannot it is not actionable and keeps the
// shorter unauthorizedTTL.
func TestConditionalResponseCaching(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	conditionalStatus := authorizationv1.SubjectAccessReviewStatus{
		ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				AllowConditions: []authorizationv1.Condition{{ID: "example.com/allow", Type: "example.com/opaque"}},
			},
		},
	}

	// A long authorizedTTL against a zero unauthorizedTTL makes the branch observable: an
	// entry cached as authorized answers the second call, an unauthorized one has already
	// expired and the webhook is consulted again.
	newAuthorizer := func(t *testing.T, sarReviewer subjectAccessReviewer) *WebhookAuthorizer {
		t.Helper()
		wh, err := newWithBackoff(sarReviewer, time.Hour, 0, testRetryBackoff,
			authorizer.DecisionNoOpinion, nil, noopAuthorizerMetrics(),
			authorizationcel.NewDefaultCompiler(), "test", nil)
		if err != nil {
			t.Fatalf("newWithBackoff failed: %v", err)
		}
		return wh
	}

	t.Run("conditions-aware caller caches a conditional response as authorized", func(t *testing.T) {
		reviewer := &fakeSubjectAccessReviewer{
			response: &authorizationv1.SubjectAccessReview{Status: conditionalStatus},
		}
		wh := newAuthorizer(t, reviewer)

		for i := 1; i <= 2; i++ {
			if got, want := wh.ConditionsAwareAuthorize(testCtx, testAttr).String(), "ConditionsMap(allows=1)"; got != want {
				t.Errorf("call %d: expected decision %s, got %s", i, want, got)
			}
		}

		if reviewer.calls != 1 {
			t.Errorf("expected the second call to be served from the cache, got %d webhook calls", reviewer.calls)
		}
	})

	t.Run("conditions-unaware caller does not cache a conditional response as authorized", func(t *testing.T) {
		reviewer := &fakeSubjectAccessReviewer{
			response: &authorizationv1.SubjectAccessReview{Status: conditionalStatus},
		}
		wh := newAuthorizer(t, reviewer)

		for i := 1; i <= 2; i++ {
			// Authorize cannot evaluate conditions, so it folds the decision down and
			// reports the response as unrequested. Only the caching matters here.
			if _, _, err := wh.Authorize(testCtx, testAttr); err == nil {
				t.Errorf("call %d: expected an error for an unrequested conditional decision", i)
			}
		}

		if reviewer.calls != 2 {
			t.Errorf("expected the entry to have expired and the webhook to be called again, got %d webhook calls", reviewer.calls)
		}
	})

	t.Run("conditions-aware caller still caches a denial as unauthorized", func(t *testing.T) {
		reviewer := &fakeSubjectAccessReviewer{
			response: &authorizationv1.SubjectAccessReview{
				Status: authorizationv1.SubjectAccessReviewStatus{Denied: true, Reason: "denied"},
			},
		}
		wh := newAuthorizer(t, reviewer)

		for i := 1; i <= 2; i++ {
			if got, want := wh.ConditionsAwareAuthorize(testCtx, testAttr).String(), `Deny(reason="denied")`; got != want {
				t.Errorf("call %d: expected decision %s, got %s", i, want, got)
			}
		}

		if reviewer.calls != 2 {
			t.Errorf("expected the entry to have expired and the webhook to be called again, got %d webhook calls", reviewer.calls)
		}
	})
}
