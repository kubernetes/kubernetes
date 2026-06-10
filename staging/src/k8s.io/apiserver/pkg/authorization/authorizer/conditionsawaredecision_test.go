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
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func typedNil() authorizer.Condition {
	var c *authorizer.GenericCondition = nil
	return c
}

// unionDecision builds a ConditionsAwareDecisionUnion from the given decisions, assigning each
// a synthetic authorizerName ("0", "1", ...), and returns the equivalent ConditionsAwareDecision.
// It is a thin shim over the public Add + ToDecision API to keep the existing test cases readable.
func unionDecision(decisions ...authorizer.ConditionsAwareDecision) authorizer.ConditionsAwareDecision {
	var u authorizer.ConditionsAwareDecisionUnion
	for i, d := range decisions {
		u.Add(strconv.Itoa(i), d)
	}
	return u.ToDecision()
}

func TestConditionsAwareDecision(t *testing.T) {
	unexpectedErr := fmt.Errorf("unexpected things happened")
	otherErr := fmt.Errorf("other error")

	genericCond := func(id string) authorizer.Condition {
		return authorizer.GenericCondition{ID: id, Condition: "x", Type: "test"}
	}

	ctx := t.Context()
	sampleAttrs := authorizer.AttributesRecord{}

	makeConditionsSlice := func(conditionCount int) []authorizer.Condition {
		allowConditionList := make([]authorizer.Condition, conditionCount)
		for i := range conditionCount {
			allowConditionList[i] = authorizer.GenericCondition{ID: fmt.Sprintf("cond-%d", i)}
		}
		return allowConditionList
	}

	condMapAllow := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "allow-cond"}},
	)
	condMapDeny := authorizer.ConditionsAwareDecisionConditionsMap(
		[]authorizer.Condition{authorizer.GenericCondition{ID: "deny-cond"}},
		nil, nil,
	)
	condMapDenyAndAllow := authorizer.ConditionsAwareDecisionConditionsMap(
		[]authorizer.Condition{authorizer.GenericCondition{ID: "deny-1"}},
		nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "allow-1"}},
	)

	allow := authorizer.ConditionsAwareDecisionAllow("", nil)
	deny := authorizer.ConditionsAwareDecisionDeny("", nil)
	noOp := authorizer.ConditionsAwareDecisionNoOpinion("", nil)

	tests := []struct {
		name                    string
		testDecisions           []authorizer.ConditionsAwareDecision
		wantIsAllow             bool
		wantIsNoOpinion         bool
		wantIsDeny              bool
		wantIsConditionsMap     bool
		wantIsUnion             bool
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
		// Union constructor simplification cases
		{
			name: "union: empty yields NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(),
			},
			wantIsNoOpinion:         true,
			wantContainsAllowOrDeny: false,
			wantReason:              "",
			wantString:              `NoOpinion`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// A single unconditional decision is simplified to that decision; the reason gets
			// an "%d: %s" index prefix (the index in the union's inner slice) per ToDecision.
			name: "union: single Allow simplifies to Allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionAllow("ok", nil)),
			},
			wantIsAllow:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              "0: ok",
			wantString:              `Allow(reason="0: ok")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: single Deny simplifies to Deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionDeny("denied", nil)),
			},
			wantIsDeny:              true,
			wantContainsAllowOrDeny: true,
			wantReason:              "0: denied",
			wantString:              `Deny(reason="0: denied")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionDeny),
		},
		{
			name: "union: single NoOpinion simplifies to NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsNoOpinion:         true,
			wantContainsAllowOrDeny: false,
			wantReason:              "0: noop",
			wantString:              `NoOpinion(reason="0: noop")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion),
		},
		{
			name: "union: single ConditionsMap wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantReason:              `[""]`,
			wantString:              `Union[ConditionsMap(allows=1)]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "union: single Union wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionAllow("", nil))),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              `[["", ""]]`,
			wantString:              `Union[Union[ConditionsMap(denies=1), Allow]]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: all NoOpinion yields merged NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("a", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("", unexpectedErr),
					authorizer.ConditionsAwareDecisionNoOpinion("c", otherErr),
				),
			},
			wantIsNoOpinion:         true,
			wantContainsAllowOrDeny: false,
			wantReason:              "0: a, 2: c",
			wantErrorIs:             unexpectedErr,
			wantString:              `NoOpinion(reason="0: a, 2: c", err="[1: unexpected things happened, 2: other error]")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// Add short-circuits after the first Allow/Deny leaf, so the trailing Deny("second")
			// is dropped. The remaining inner slice is [NoOpinion, NoOpinion, Allow], so the
			// simplified reason references the Allow at index 2.
			name: "union: Allow before Deny returns Allow, NoOpinions are ignored",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionAllow("first", nil),
					authorizer.ConditionsAwareDecisionDeny("second", nil),
				),
			},
			wantIsAllow:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              "2: first",
			wantString:              `Allow(reason="2: first")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: Deny before Allow returns Deny, NoOpinions are ignored",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionDeny("first", nil),
					authorizer.ConditionsAwareDecisionAllow("second", nil),
				),
			},
			wantIsDeny:              true,
			wantContainsAllowOrDeny: true,
			wantReason:              "2: first",
			wantString:              `Deny(reason="2: first")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionDeny),
		},
		// Actual union decisions (not simplified)
		{
			name: "union: noopinion + conditionsmap(allow) + noopinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("no-op1", nil),
					condMapAllow,
					authorizer.ConditionsAwareDecisionNoOpinion("no-op2", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantReason:              `[no-op1, "", no-op2]`,
			wantString:              `Union[NoOpinion(reason="no-op1"), ConditionsMap(allows=1), NoOpinion(reason="no-op2")]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			// ConditionsMap(allow-only) followed by Allow has PossibleDecisions={Allow}: if the
			// ConditionsMap evaluates to Allow, the answer is Allow; if it evaluates to NoOpinion,
			// the trailing Allow takes over. Either way, the union eagerly simplifies to Allow.
			name: "union: conditionsmap(allow) + allow simplifies to Allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionAllow("allowed", nil)),
			},
			wantIsAllow:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              "1: allowed",
			wantString:              `Allow(reason="1: allowed")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: conditionsmap(allow) + deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionDeny("no", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Deny
			wantReason:              `["", no]`,
			wantString:              `Union[ConditionsMap(allows=1), Deny(reason="no")]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(deny) + noopinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantReason:              `["", noop]`,
			wantString:              `Union[ConditionsMap(denies=1), NoOpinion(reason="noop")]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(deny) + allow with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionAllow("allowed", unexpectedErr)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Allow
			wantReason:              `["", allowed]`,
			wantErrorIs:             unexpectedErr,
			wantString:              `Union[ConditionsMap(denies=1), Allow(reason="allowed", err="unexpected things happened")]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(allow) + conditionsmap(deny)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, condMapDeny),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantReason:              `["", ""]`,
			wantString:              `Union[ConditionsMap(allows=1), ConditionsMap(denies=1)]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			// The inner union [condMapAllow, Allow("ok")] simplifies to Allow(reason="1: ok").
			// The trailing NoOpinion is dropped by the outer Add's short-circuit (an Allow is
			// already present). The remaining outer inner is [condMapAllow, Allow("1: ok")],
			// which again simplifies to Allow with a nested index prefix.
			name: "union: nested with allow simplifies through both levels",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					condMapAllow,
					unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionAllow("ok", nil)),
					authorizer.ConditionsAwareDecisionNoOpinion("don't care", nil),
				),
			},
			wantIsAllow:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              "1: 1: ok",
			wantString:              `Allow(reason="1: 1: ok")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: deep nesting without anything unconditional",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					condMapAllow,
					unionDecision(
						condMapAllow,
						authorizer.ConditionsAwareDecisionNoOpinion("inner", nil),
						unionDecision(
							condMapDeny,
							authorizer.ConditionsAwareDecisionNoOpinion("inner2", nil)),
					),
				),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantReason:              `["", ["", inner, ["", inner2]]]`,
			wantString:              `Union[ConditionsMap(allows=1), Union[ConditionsMap(allows=1), NoOpinion(reason="inner"), Union[ConditionsMap(denies=1), NoOpinion(reason="inner2")]]]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},

		// --- Unconditional leaves carrying a side-channel error ---
		{
			name: "Allow with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionAllow("ok", errors.New("warning")),
			},
			wantIsAllow:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              "ok",
			wantAnyError:            true,
			wantString:              `Allow(reason="ok", err="warning")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow),
		},
		{
			name: "Deny with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionDeny("no", errors.New("warning")),
			},
			wantIsDeny:              true,
			wantContainsAllowOrDeny: true,
			wantReason:              "no",
			wantAnyError:            true,
			wantString:              `Deny(reason="no", err="warning")`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionDeny),
		},
		{
			name: "NoOpinion with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionNoOpinion("meh", errors.New("warning")),
			},
			wantIsNoOpinion:       true,
			wantReason:            "meh",
			wantAnyError:          true,
			wantString:            `NoOpinion(reason="meh", err="warning")`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion),
		},

		// --- Bare ConditionsMap effect-group combinations ---
		{
			name: "conditionsmap: allow-only -> {NoOpinion, Allow}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				condMapAllow,
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(allows=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "conditionsmap: deny-only -> {NoOpinion, Deny}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				condMapDeny,
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(denies=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			name: "conditionsmap: noOpinion + allow -> {NoOpinion, Allow}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					nil,
					[]authorizer.Condition{genericCond("nop-1")},
					[]authorizer.Condition{genericCond("allow-1")},
				),
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(noopinions=1, allows=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "conditionsmap: deny + noOpinion -> {NoOpinion, Deny}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{genericCond("deny-1")},
					[]authorizer.Condition{genericCond("nop-1")},
					nil,
				),
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(denies=1, noopinions=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			name: "conditionsmap: deny + allow -> {NoOpinion, Allow, Deny}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				condMapDenyAndAllow,
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(denies=1, allows=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "conditionsmap: deny + noOpinion + allow -> {NoOpinion, Allow, Deny}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					[]authorizer.Condition{genericCond("deny-1")},
					[]authorizer.Condition{genericCond("nop-1")},
					[]authorizer.Condition{genericCond("allow-1")},
				),
			},
			wantIsConditionsMap:   true,
			wantString:            `ConditionsMap(denies=1, noopinions=1, allows=1)`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},

		// --- Union sequences not already covered above ---
		{
			name: "union: single ConditionsMap(deny) -> {NoOpinion, Deny}",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny),
			},
			wantIsUnion:           true,
			wantReason:            `[""]`,
			wantString:            `Union[ConditionsMap(denies=1)]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			// CM(deny) → Deny short-circuits there; CM(deny) → NoOpinion falls through to Allow.
			// Both outcomes are reachable, so the union stays a Union.
			name: "union: conditionsmap(deny) + allow stays Union",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, allow),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // trailing Allow leaf
			wantReason:              `["", ""]`,
			wantString:              `Union[ConditionsMap(denies=1), Allow]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			// CM(deny) either yields Deny or NoOpinion; the trailing Deny then catches the NoOpinion
			// branch — so the union eagerly simplifies to Deny.
			name: "union: conditionsmap(deny) + deny simplifies to Deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, deny),
			},
			wantIsDeny:              true,
			wantContainsAllowOrDeny: true,
			wantReason:              "",
			wantString:              `Deny`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionDeny),
		},
		{
			// No downstream Allow/Deny to short-circuit, so NoOpinion remains a possible outcome.
			name: "union: conditionsmap(allow) + noopinion (no leaf) stays Union",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, noOp),
			},
			wantIsUnion:           true,
			wantReason:            `["", ""]`,
			wantString:            `Union[ConditionsMap(allows=1), NoOpinion]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "union: single ConditionsMap(deny+allow) wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDenyAndAllow),
			},
			wantIsUnion:           true,
			wantReason:            `[""]`,
			wantString:            `Union[ConditionsMap(denies=1, allows=1)]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: noopinion + conditionsmap(allow) + deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(noOp, condMapAllow, deny),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // trailing Deny leaf
			wantReason:              `["", "", ""]`,
			wantString:              `Union[NoOpinion, ConditionsMap(allows=1), Deny]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(allow) + conditionsmap(deny) + allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, condMapDeny, allow),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true,
			wantReason:              `["", "", ""]`,
			wantString:              `Union[ConditionsMap(allows=1), ConditionsMap(denies=1), Allow]`,
			wantPossibleDecisions:   sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
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
					isConditionsMap := d.IsConditionsMap()
					if isConditionsMap != tt.wantIsConditionsMap {
						t.Errorf("IsConditionsMap() = %v, want %v", isConditionsMap, tt.wantIsConditionsMap)
					}
					isUnion := d.IsUnion()
					if isUnion != tt.wantIsUnion {
						t.Errorf("IsUnion() = %v, want %v", isUnion, tt.wantIsUnion)
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
