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

func TestDeserializeReason(t *testing.T) {
	tests := []struct {
		name string
		in   *authorizationv1.UnconditionalDecision
		want string
	}{
		{name: "nil returns empty string", in: nil, want: ""},
		{name: "empty reason returns empty string", in: &authorizationv1.UnconditionalDecision{}, want: ""},
		{name: "non-empty reason propagates", in: &authorizationv1.UnconditionalDecision{Reason: "hello"}, want: "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DeserializeReason(tt.in); got != tt.want {
				t.Errorf("DeserializeReason = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDeserializeEvaluationError(t *testing.T) {
	tests := []struct {
		name    string
		in      *authorizationv1.UnconditionalDecision
		wantErr string // "" means nil error
	}{
		{name: "nil returns nil", in: nil, wantErr: ""},
		{name: "empty EvaluationError returns nil", in: &authorizationv1.UnconditionalDecision{}, wantErr: ""},
		{name: "non-empty EvaluationError becomes an error with that message", in: &authorizationv1.UnconditionalDecision{EvaluationError: "boom"}, wantErr: "boom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DeserializeEvaluationError(tt.in)
			switch {
			case tt.wantErr == "" && got != nil:
				t.Errorf("expected nil error, got %v", got)
			case tt.wantErr != "" && got == nil:
				t.Errorf("expected error %q, got nil", tt.wantErr)
			case tt.wantErr != "" && got.Error() != tt.wantErr:
				t.Errorf("expected error %q, got %q", tt.wantErr, got.Error())
			}
		})
	}
}

func TestDeserializeConditionsAwareDecision(t *testing.T) {
	failClosedSentinel := authorizer.ConditionsAwareDecisionDeny("failClosed called", nil)
	failClosed := func(err error) authorizer.ConditionsAwareDecision {
		return failClosedSentinel
	}

	tests := []struct {
		name  string
		in    authorizationv1.ConditionsAwareDecision
		check func(t *testing.T, got authorizer.ConditionsAwareDecision)
	}{
		{
			name: "Allow with reason and error",
			in: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeAllow,
				Allow: &authorizationv1.UnconditionalDecision{
					Reason:          "allowed",
					EvaluationError: "eval error",
				},
			},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if !got.IsAllow() {
					t.Errorf("expected Allow, got %s", got.String())
				}
				if got.Reason() != "allowed" {
					t.Errorf("Reason = %q, want %q", got.Reason(), "allowed")
				}
				if got.Error() == nil || got.Error().Error() != "eval error" {
					t.Errorf("Error = %v, want %q", got.Error(), "eval error")
				}
			},
		},
		{
			name: "Deny with nil Deny sub-object",
			in:   authorizationv1.ConditionsAwareDecision{Type: authorizationv1.ConditionsAwareDecisionTypeDeny},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if !got.IsDeny() {
					t.Errorf("expected Deny, got %s", got.String())
				}
				if got.Reason() != "" {
					t.Errorf("Reason = %q, want empty", got.Reason())
				}
				if got.Error() != nil {
					t.Errorf("Error = %v, want nil", got.Error())
				}
			},
		},
		{
			name: "NoOpinion propagates reason",
			in: authorizationv1.ConditionsAwareDecision{
				Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
				NoOpinion: &authorizationv1.UnconditionalDecision{Reason: "idk"},
			},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if !got.IsNoOpinion() {
					t.Errorf("expected NoOpinion, got %s", got.String())
				}
				if got.Reason() != "idk" {
					t.Errorf("Reason = %q, want %q", got.Reason(), "idk")
				}
			},
		},
		{
			name: "ConditionsMap with a deny condition",
			in: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					DenyConditions: []authorizationv1.Condition{
						{ID: "example.com/deny", Condition: "cond-d", Type: "example.com/t", Description: "deny-desc"},
					},
				},
			},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if !got.IsConditionsMap() {
					t.Fatalf("expected ConditionsMap, got %s", got.String())
				}
				var count int
				for cond := range got.ConditionsMap().DenyConditions() {
					count++
					if cond.GetID() != "example.com/deny" {
						t.Errorf("condition ID = %q, want %q", cond.GetID(), "example.com/deny")
					}
					if cond.GetCondition() != "cond-d" {
						t.Errorf("condition = %q, want %q", cond.GetCondition(), "cond-d")
					}
				}
				if count != 1 {
					t.Errorf("expected 1 deny condition, got %d", count)
				}
			},
		},
		{
			name: "ConditionsMap with nil ConditionsMap field falls back to empty",
			in: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				// Empty conditions produce NoOpinion per ConditionsAwareDecisionConditionsMap.
				if !got.IsNoOpinion() {
					t.Errorf("expected NoOpinion (empty conditions folded), got %s", got.String())
				}
			},
		},
		{
			name: "Union with two leaves",
			in: authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.NamedConditionsAwareDecision{
					{
						AuthorizerName: "a.example.com",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								AllowConditions: []authorizationv1.Condition{
									{ID: "example.com/allow"},
								},
							},
						},
					},
					{
						AuthorizerName: "b.example.com",
						Decision: authorizationv1.ConditionsAwareDecision{
							Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
							ConditionsMap: &authorizationv1.ConditionsMap{
								AllowConditions: []authorizationv1.Condition{
									{ID: "example.com/allow-b"},
								},
							},
						},
					},
				},
			},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if !got.IsUnion() {
					t.Fatalf("expected Union, got %s", got.String())
				}
				var names []string
				for name := range got.UnionedDecisions() {
					names = append(names, name)
				}
				want := []string{"a.example.com", "b.example.com"}
				if diff := cmp.Diff(want, names); diff != "" {
					t.Errorf("union order mismatch (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "unknown Type falls back to failClosed",
			in:   authorizationv1.ConditionsAwareDecision{Type: "BogusType"},
			check: func(t *testing.T, got authorizer.ConditionsAwareDecision) {
				if got.Reason() != failClosedSentinel.Reason() {
					t.Errorf("expected failClosed sentinel; Reason = %q, want %q", got.Reason(), failClosedSentinel.Reason())
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DeserializeConditionsAwareDecision(tt.in, failClosed)
			tt.check(t, got)
		})
	}
}

func TestSerializeDeserializeRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		in   authorizer.ConditionsAwareDecision
		want string // decision.String() for comparison; opaque but stable
	}{
		{
			name: "Allow with reason",
			in:   authorizer.ConditionsAwareDecisionAllow("ok", nil),
		},
		{
			name: "Deny with reason and error",
			in:   authorizer.ConditionsAwareDecisionDeny("no", errors.New("bad")),
		},
		{
			name: "NoOpinion no reason",
			in:   authorizer.ConditionsAwareDecisionNoOpinion("", nil),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			serialized := SerializeConditionsAwareDecision(tt.in)
			deserialized := DeserializeConditionsAwareDecision(serialized, func(err error) authorizer.ConditionsAwareDecision {
				t.Fatalf("unexpected failClosed call: %v", err)
				return authorizer.ConditionsAwareDecisionDeny("", err)
			})
			if deserialized.String() != tt.in.String() {
				t.Errorf("round-trip mismatch:\ngot:  %s\nwant: %s", deserialized.String(), tt.in.String())
			}
			// Error content survives round-trip
			var gotErr, wantErr string
			if e := tt.in.Error(); e != nil {
				wantErr = e.Error()
			}
			if e := deserialized.Error(); e != nil {
				gotErr = e.Error()
			}
			if gotErr != wantErr {
				t.Errorf("Error = %q, want %q", gotErr, wantErr)
			}
		})
	}
}
