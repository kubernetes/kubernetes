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

package v1

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestSerializeConditionsAwareDecision(t *testing.T) {
	denyCond := authorizer.GenericCondition{ID: "example.com/deny", Condition: "cond-d", Type: "example.com/t", Description: "deny-desc"}
	noOpCond := authorizer.GenericCondition{ID: "example.com/noop", Condition: "cond-n", Type: "example.com/t", Description: "noop-desc"}
	allowCond := authorizer.GenericCondition{ID: "example.com/allow", Condition: "cond-a", Type: "example.com/t", Description: "allow-desc"}

	tests := []struct {
		name string
		in   authorizer.ConditionsAwareDecision
		want authorizationv1.ConditionsAwareDecision
	}{
		{
			name: "Allow with reason and error",
			in:   authorizer.ConditionsAwareDecisionAllow("allowed", errors.New("eval error")),
			want: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeAllow,
				Allow: &authorizationv1.UnconditionalDecision{
					Reason:          "allowed",
					EvaluationError: "eval error",
				},
			},
		},
		{
			name: "Deny with reason and no error",
			in:   authorizer.ConditionsAwareDecisionDeny("denied", nil),
			want: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
				Deny: &authorizationv1.UnconditionalDecision{
					Reason: "denied",
				},
			},
		},
		{
			name: "NoOpinion with empty reason",
			in:   authorizer.ConditionsAwareDecisionNoOpinion("", nil),
			want: authorizationv1.ConditionsAwareDecision{
				Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
				NoOpinion: &authorizationv1.UnconditionalDecision{},
			},
		},
		{
			name: "ConditionsMap propagates all three buckets",
			in: authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{denyCond},
				[]authorizer.Condition{noOpCond},
				[]authorizer.Condition{allowCond},
			),
			want: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					DenyConditions: []authorizationv1.Condition{
						{ID: "example.com/deny", Condition: "cond-d", Type: "example.com/t", Description: "deny-desc"},
					},
					NoOpinionConditions: []authorizationv1.Condition{
						{ID: "example.com/noop", Condition: "cond-n", Type: "example.com/t", Description: "noop-desc"},
					},
					AllowConditions: []authorizationv1.Condition{
						{ID: "example.com/allow", Condition: "cond-a", Type: "example.com/t", Description: "allow-desc"},
					},
				},
			},
		},
		{
			name: "Union serializes sub-decisions recursively in insertion order",
			in: func() authorizer.ConditionsAwareDecision {
				var u authorizer.ConditionsAwareDecisionUnion
				u.Add("a.example.com", authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil, []authorizer.Condition{allowCond},
				))
				u.Add("b.example.com", authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{denyCond}, nil, nil,
				))
				return u.ToDecision()
			}(),
			want: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.NamedConditionsAwareDecision{
					{
						AuthorizerName: "a.example.com",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								DenyConditions:      []authorizationv1.Condition{},
								NoOpinionConditions: []authorizationv1.Condition{},
								AllowConditions: []authorizationv1.Condition{
									{ID: "example.com/allow", Condition: "cond-a", Type: "example.com/t", Description: "allow-desc"},
								},
							},
						},
					},
					{
						AuthorizerName: "b.example.com",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								DenyConditions: []authorizationv1.Condition{
									{ID: "example.com/deny", Condition: "cond-d", Type: "example.com/t", Description: "deny-desc"},
								},
								NoOpinionConditions: []authorizationv1.Condition{},
								AllowConditions:     []authorizationv1.Condition{},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SerializeConditionsAwareDecision(tt.in)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected serialization (-want +got):\n%s", diff)
			}
		})
	}
}
