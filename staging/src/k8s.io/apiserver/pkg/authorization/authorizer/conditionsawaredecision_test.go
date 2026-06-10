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

func typedNil() authorizer.Condition {
	var c *authorizer.GenericCondition = nil
	return c
}

func TestConditionsAwareDecision(t *testing.T) {
	unexpectedErr := fmt.Errorf("unexpected things happened")
	otherErr := fmt.Errorf("other error")

	ctx := t.Context()
	sampleAttrs := authorizer.AttributesRecord{}

	makeConditionsSlice := func(conditionCount int) []authorizer.Condition {
		allowConditionList := make([]authorizer.Condition, conditionCount)
		for i := range conditionCount {
			allowConditionList[i] = authorizer.GenericCondition{ID: fmt.Sprintf("cond-%d", i)}
		}
		return allowConditionList
	}

	condMapDenyAndAllow := authorizer.ConditionsAwareDecisionConditionsMap(
		[]authorizer.Condition{authorizer.GenericCondition{ID: "deny-1"}},
		nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "allow-1"}},
	)

	tests := []struct {
		name                    string
		testDecisions           []authorizer.ConditionsAwareDecision
		wantIsAllow             bool
		wantIsNoOpinion         bool
		wantIsDeny              bool
		wantIsConditionsMap     bool
		wantContainsAllowOrDeny bool
		wantPossibleDecisions   sets.Set[authorizer.Decision]
		wantReason              string
		wantAnyError            bool
		wantErrorIs             error
		wantString              string
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
		{
			name: "construct valid allow/noopinion conditionsmap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(nil, nil, makeConditionsSlice(authorizer.MaxConditionsPerMap)),
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(allows=128)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "construct valid allow/noopinion/deny conditionsmap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				condMapDenyAndAllow,
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(denies=1, allows=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionDeny, authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "too many Allow conditions",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(nil, nil, makeConditionsSlice(authorizer.MaxConditionsPerMap+1)),
			},
			wantIsNoOpinion: true,
			wantReason:      "failed closed",
			wantAnyError:    true,
			wantString:      `NoOpinion(reason="failed closed", err="too many conditions: 129 exceeds maximum of 128")`,
		},
		{
			name: "too many conditions, with one Deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap([]authorizer.Condition{authorizer.GenericCondition{ID: "deny-cond"}}, nil, makeConditionsSlice(authorizer.MaxConditionsPerMap)),
			},
			wantIsDeny:   true,
			wantReason:   "failed closed",
			wantAnyError: true,
			wantString:   `Deny(reason="failed closed", err="too many conditions: 129 exceeds maximum of 128")`,
		},
		{
			name: "nil condition is a validation error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{ID: "foo"},
						nil,
					},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil, nil,
					[]authorizer.Condition{
						authorizer.GenericCondition{ID: "foo"},
						typedNil(),
					},
				),
			},
			wantIsNoOpinion: true,
			wantReason:      "failed closed",
			wantAnyError:    true,
			wantString:      `NoOpinion(reason="failed closed", err="encountered nil condition")`,
		},
		{
			name: "duplicate IDs",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{authorizer.GenericCondition{ID: "foo"}},
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "foo"}},
				),
			},
			wantIsDeny:   true,
			wantReason:   "failed closed",
			wantAnyError: true,
			wantString:   `Deny(reason="failed closed", err="duplicate condition ID \"foo\"")`,
		},
		{
			name: "condition ID must be a Kubernetes label, one condition error enough to fail closed (in Deny)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{authorizer.GenericCondition{ID: "not a kubernetes label"}},
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "foo"}},
				),
			},
			wantIsDeny:   true,
			wantReason:   "failed closed",
			wantAnyError: true,
			wantString:   `Deny(reason="failed closed", err="invalid condition ID \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "condition ID must be a Kubernetes label, one condition error enough to fail closed (in NoOpinion)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "not a kubernetes label"}},
					[]authorizer.Condition{authorizer.GenericCondition{ID: "foo"}},
				),
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "not a kubernetes label"}},
				),
			},
			wantIsNoOpinion: true,
			wantReason:      "failed closed",
			wantAnyError:    true,
			wantString:      `NoOpinion(reason="failed closed", err="invalid condition ID \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "condition type must be a Kubernetes label",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "bar", Type: "not a kubernetes label"}},
					[]authorizer.Condition{authorizer.GenericCondition{ID: "foo"}},
				),
			},
			wantIsNoOpinion: true,
			wantReason:      "failed closed",
			wantAnyError:    true,
			wantString:      `NoOpinion(reason="failed closed", err="invalid condition type \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "empty ConditionsMap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(nil, nil, nil),
			},
			wantIsNoOpinion: true,
			wantReason:      "no conditions",
			wantAnyError:    true,
			wantString:      `NoOpinion(reason="no conditions", err="at least one condition must be passed to ConditionsAwareDecisionConditionsMap(), got none")`,
		},
		{
			// Short-circuit: only NoOpinion conditions => the constructor folds the result to NoOpinion
			// directly, without ever returning a ConditionsMap (which would then evaluate to NoOpinion anyway).
			name: "noopinion-only conditions short-circuit to NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{authorizer.GenericCondition{ID: "nop-1"}},
					nil,
				),
			},
			wantIsNoOpinion: true,
			wantReason:      "",
			wantString:      `NoOpinion`,
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
					containsAllowOrDeny := d.ContainsAllowOrDeny()
					// default assertion value for the plain Allow/Deny cases to avoid stutter
					if tt.wantIsDeny || tt.wantIsAllow {
						tt.wantContainsAllowOrDeny = true
					}
					if containsAllowOrDeny != tt.wantContainsAllowOrDeny {
						t.Errorf("ContainsAllowOrDeny() = %v, want %v", containsAllowOrDeny, tt.wantContainsAllowOrDeny)
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
					if tt.wantIsConditionsMap {
						// also ensure the ConditionsMap's FailureDecision is aligned
						failureDecision := d.ConditionsMap().FailureDecision()
						if failureDecision != wantFailureDecision {
							t.Errorf("ConditionsMap.FailureDecision() = %v, want %v", failureDecision, wantFailureDecision)
						}
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
