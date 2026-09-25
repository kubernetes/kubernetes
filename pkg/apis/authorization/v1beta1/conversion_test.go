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

package v1beta1

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	authorization "k8s.io/kubernetes/pkg/apis/authorization"
)

// TestConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus
// covers folding a conditional decision down to v1beta1, which has no
// conditionalDecision field to put it in.
//
// The fold is driven by the decision's own FailureDecision(): a decision that could
// have produced a Deny becomes denied=true, and one that could not becomes neither
// allowed nor denied, so a later authorizer still gets a say. allowed is never set,
// because an unevaluated condition must never be reported as a grant.
//
// Only ConditionsMap and Union are valid conditionalDecision types (validation
// rejects Allow/Deny/NoOpinion there, and requires the other status fields to be
// empty). The malformed cases below therefore document defensive behavior for input
// that validation would already have rejected.
func TestConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(t *testing.T) {
	denyCondition := authorization.Condition{ID: "example.com/deny-restricted", Type: "example.com/opaque"}
	allowCondition := authorization.Condition{ID: "example.com/allow-safe", Type: "example.com/opaque"}
	noOpinionCondition := authorization.Condition{ID: "example.com/abstain", Type: "example.com/opaque"}

	tests := []struct {
		name string
		in   authorization.SubjectAccessReviewStatus
		want authorizationv1beta1.SubjectAccessReviewStatus
	}{
		{
			name: "unconditional allow propagates",
			in: authorization.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "rbac: role/x allowed",
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "rbac: role/x allowed",
			},
		},
		{
			name: "unconditional deny propagates with EvaluationError",
			in: authorization.SubjectAccessReviewStatus{
				Denied:          true,
				Reason:          "webhook: denied",
				EvaluationError: "flaky evaluator",
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Denied:          true,
				Reason:          "webhook: denied",
				EvaluationError: "flaky evaluator",
			},
		},
		{
			// A Deny condition could have fired, so the safe fold is an outright deny.
			name: "ConditionsMap with both Deny and Allow conditions folds to denied",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorization.ConditionsMap{
						DenyConditions:  []authorization.Condition{denyCondition},
						AllowConditions: []authorization.Condition{allowCondition},
					},
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Denied: true,
			},
		},
		{
			// No Deny condition could have fired, so nothing stronger than "no opinion"
			// is justified: the Allow cannot be granted without evaluating the condition.
			name: "ConditionsMap with only Allow conditions folds to no opinion",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorization.ConditionsMap{
						AllowConditions: []authorization.Condition{allowCondition},
					},
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{},
		},
		{
			// NoOpinion-only conditions can never reach Allow or Deny, so the decision
			// collapses to an unconditional NoOpinion that carries its own reason.
			name: "ConditionsMap with only NoOpinion conditions folds to no opinion with a reason",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorization.ConditionsMap{
						NoOpinionConditions: []authorization.Condition{noOpinionCondition},
					},
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Reason: "only NoOpinion conditions always evaluate to NoOpinion",
			},
		},
		{
			name: "Union containing a nested Deny condition folds to denied",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{
							AuthorizerName: "cm",
							Decision: authorization.ConditionsAwareDecision{
								Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorization.ConditionsMap{
									DenyConditions: []authorization.Condition{denyCondition},
								},
							},
						},
						{
							AuthorizerName: "allow",
							Decision: authorization.ConditionsAwareDecision{
								Type:  authorization.ConditionsAwareDecisionTypeAllow,
								Allow: &authorization.UnconditionalDecision{Reason: "sub-allow"},
							},
						},
					},
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Denied: true,
			},
		},
		{
			name: "Union with no Deny anywhere folds to no opinion",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeUnion,
					Union: []authorization.NamedConditionsAwareDecision{
						{
							AuthorizerName: "cm",
							Decision: authorization.ConditionsAwareDecision{
								Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorization.ConditionsMap{
									AllowConditions: []authorization.Condition{allowCondition},
								},
							},
						},
					},
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{},
		},
		{
			// Defensive: the type claims a ConditionsMap but none is present. There are
			// no Deny conditions that could have fired, so this stays at no opinion.
			name: "ConditionsMap type with a nil map fails closed to no opinion",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Reason:          "no conditions",
				EvaluationError: "at least one condition must be passed to ConditionsAwareDecisionConditionsMap(), got none",
			},
		},
		{
			// Defensive: an unrecognized type cannot be interpreted at all, so it must
			// not be allowed to pass as anything weaker than a deny.
			name: "unrecognized type fails closed to denied",
			in: authorization.SubjectAccessReviewStatus{
				ConditionalDecision: &authorization.ConditionsAwareDecision{
					Type: "SomeFutureType",
				},
			},
			want: authorizationv1beta1.SubjectAccessReviewStatus{
				Denied:          true,
				Reason:          "failed closed",
				EvaluationError: `couldn't deserialize decision: unrecognized ConditionsAwareDecision.type="SomeFutureType"`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got authorizationv1beta1.SubjectAccessReviewStatus
			if err := Convert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(&tt.in, &got, nil); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected out (-want +got):\n%s", diff)
			}
			// An unevaluated conditional decision must never surface as a grant.
			if tt.in.ConditionalDecision != nil && got.Allowed {
				t.Error("a conditional decision must never fold to allowed=true")
			}
		})
	}
}

// TestConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus_DoesNotMutateInput
// guards the DeepCopy in the conversion: folding reads the input through a second
// conversion, and callers hand in a status they may still own.
func TestConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus_DoesNotMutateInput(t *testing.T) {
	in := authorization.SubjectAccessReviewStatus{
		ConditionalDecision: &authorization.ConditionsAwareDecision{
			Type: authorization.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorization.ConditionsMap{
				DenyConditions: []authorization.Condition{{ID: "example.com/deny", Type: "example.com/opaque"}},
			},
		},
	}
	want := in.DeepCopy()

	var got authorizationv1beta1.SubjectAccessReviewStatus
	if err := Convert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(&in, &got, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if diff := cmp.Diff(want, &in); diff != "" {
		t.Errorf("the conversion mutated its input (-want +got):\n%s", diff)
	}
}
