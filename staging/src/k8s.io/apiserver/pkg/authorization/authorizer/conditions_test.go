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

package authorizer_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestConditionsAwareDecision(t *testing.T) {
	unexpectedErr := fmt.Errorf("unexpected things happened")
	otherErr := fmt.Errorf("other error")

	ctx := t.Context()
	sampleAttrs := authorizer.AttributesRecord{}

	tests := []struct {
		name                                 string
		testDecisions                        []authorizer.ConditionsAwareDecision
		wantIsAllow                          bool
		wantIsNoOpinion                      bool
		wantIsDeny                           bool
		wantContainsUnconditionalAllowOrDeny bool
		wantPossibleDecisions                sets.Set[authorizer.Decision]
		wantReason                           string
		wantAnyError                         bool
		wantErrorIs                          error
		wantString                           string
	}{
		{
			name: "zero value",
			testDecisions: []authorizer.ConditionsAwareDecision{
				{},
				authorizer.ConditionsAwareDecisionFromParts(0, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (named1 authorizer.Decision, named2 string, named3 error) {
					return
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDeny:  true,
			wantReason:  "",
			wantErrorIs: nil,
			wantString:  `Deny`,
		},
		{
			name: "deny constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionDeny("foo", unexpectedErr),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionDeny, "foo", unexpectedErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "foo", unexpectedErr
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDeny:  true,
			wantReason:  "foo",
			wantErrorIs: unexpectedErr,
			wantString:  `Deny(reason="foo", err="unexpected things happened")`,
		},
		{
			name: "allow constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionAllow("ok", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionAllow, "ok", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "ok", nil
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsAllow: true,
			wantReason:  "ok",
			wantErrorIs: nil,
			wantString:  `Allow(reason="ok")`,
		},
		{
			name: "noopinion constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionNoOpinion, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionNoOpinion, "", nil
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsNoOpinion: true,
			wantReason:      "",
			wantErrorIs:     nil,
			wantString:      `NoOpinion`,
		},
		{
			name: "from parts: unsupported mode",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "", nil
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDeny:   true,
			wantReason:   "",
			wantAnyError: true,
			wantString:   `Deny(err="unknown unconditional decision type: 42")`,
		},
		{
			name: "from parts: unsupported mode with other error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "foo", otherErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "foo", otherErr
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDeny:  true,
			wantReason:  "foo",
			wantErrorIs: otherErr,
			wantString:  `Deny(reason="foo", err="[other error, unknown unconditional decision type: 42]")`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, d := range tt.testDecisions {
				t.Run(fmt.Sprint(i), func(t *testing.T) {
					isAllowed := d.IsAllow()
					if isAllowed != tt.wantIsAllow {
						t.Errorf("IsAllowed() = %v, want %v", isAllowed, tt.wantIsAllow)
					}
					isNoOpinion := d.IsNoOpinion()
					if isNoOpinion != tt.wantIsNoOpinion {
						t.Errorf("IsNoOpinion() = %v, want %v", isNoOpinion, tt.wantIsNoOpinion)
					}
					isDenied := d.IsDeny()
					if isDenied != tt.wantIsDeny {
						t.Errorf("IsDenied() = %v, want %v", isDenied, tt.wantIsDeny)
					}
					isUnconditional := d.IsUnconditional()
					wantIsUnconditional := tt.wantIsAllow || tt.wantIsDeny || tt.wantIsNoOpinion
					if isUnconditional != wantIsUnconditional {
						t.Errorf("IsUnconditional() = %v, want %v", isUnconditional, wantIsUnconditional)
					}
					containsUnconditionalAllowOrDeny := d.ContainsUnconditionalAllowOrDeny()
					// default assertion value for the plain Allow/Deny cases to avoid stutter
					if tt.wantIsDeny || tt.wantIsAllow {
						tt.wantContainsUnconditionalAllowOrDeny = true
					}
					if containsUnconditionalAllowOrDeny != tt.wantContainsUnconditionalAllowOrDeny {
						t.Errorf("ContainsUnconditionalAllowOrDeny() = %v, want %v", containsUnconditionalAllowOrDeny, tt.wantContainsUnconditionalAllowOrDeny)
					}
					gotPossibleDecisions := d.PossibleDecisions()
					// default assertion value for the plain Allow/Deny/NoOpinion cases to avoid stutter
					if tt.wantPossibleDecisions == nil {
						tt.wantPossibleDecisions = make(sets.Set[authorizer.Decision])
					}
					if tt.wantIsAllow {
						tt.wantPossibleDecisions.Insert(authorizer.DecisionAllow)
					}
					if tt.wantIsDeny {
						tt.wantPossibleDecisions.Insert(authorizer.DecisionDeny)
					}
					if tt.wantIsNoOpinion {
						tt.wantPossibleDecisions.Insert(authorizer.DecisionNoOpinion)
					}
					if !gotPossibleDecisions.Equal(tt.wantPossibleDecisions) {
						t.Errorf("PossibleDecisions() = %v, want %v",
							slices.Sorted(maps.Keys(gotPossibleDecisions)),
							slices.Sorted(maps.Keys(tt.wantPossibleDecisions)))
					}
					// dynamic property-based assertion, ordered after dynamic defaults to tt.wantPossibleDecisions
					wantFailureDecision := authorizer.DecisionNoOpinion
					if tt.wantPossibleDecisions.Has(authorizer.DecisionDeny) {
						wantFailureDecision = authorizer.DecisionDeny
					}
					failureDecision := d.FailureDecision()
					if failureDecision != wantFailureDecision {
						t.Errorf("FailureDecision() = %v, want %v", failureDecision, wantFailureDecision)
					}
					gotReason := d.Reason()
					if gotReason != tt.wantReason {
						t.Errorf("Reason() = %v, want %v", gotReason, tt.wantReason)
					}
					gotError := d.Error()
					if tt.wantAnyError {
						if gotError == nil {
							t.Errorf("Error() = %v, want some error", nil)
						}
					} else {
						if !errors.Is(gotError, tt.wantErrorIs) {
							t.Errorf("Error() = %v, want %v", gotError, tt.wantErrorIs)
						}
					}

					gotString := d.String()
					if gotString != tt.wantString {
						t.Errorf("String() = %v, want %v", gotString, tt.wantString)
					}
				})
			}
		})
	}
}
