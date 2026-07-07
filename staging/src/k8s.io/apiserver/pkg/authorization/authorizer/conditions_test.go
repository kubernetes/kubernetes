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
	"iter"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"testing"

	"github.com/google/cel-go/cel"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// unionDecision builds a ConditionsAwareDecisionUnion from the given decisions, assigning each
// a synthetic authorizerName ("0.example.com", "1.example.com", ...), and returns the equivalent ConditionsAwareDecision.
// It is a thin shim over the public Add + ToDecision API to keep the existing test cases readable.
func unionDecision(decisions ...authorizer.ConditionsAwareDecision) authorizer.ConditionsAwareDecision {
	var u authorizer.ConditionsAwareDecisionUnion
	for i, d := range decisions {
		u.Add(strconv.Itoa(i)+".example.com", d)
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
		name                                 string
		testDecisions                        []authorizer.ConditionsAwareDecision
		wantIsAllow                          bool
		wantIsNoOpinion                      bool
		wantIsDeny                           bool
		wantIsConditionsMap                  bool
		wantIsUnion                          bool
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
			wantReason:      "only NoOpinion conditions always evaluate to NoOpinion",
			wantString:      `NoOpinion(reason="only NoOpinion conditions always evaluate to NoOpinion")`,
		},
		// Union constructor simplification cases
		{
			name: "union: empty yields NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(),
			},
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `NoOpinion`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// A single unconditional decision is simplified to that decision; the reason gets
			// an "%s: %s" authorizer-name prefix (the sub-decision's authorizerName) per ToDecision.
			name: "union: single Allow simplifies to Allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionAllow("ok", nil)),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {ok}`,
			wantString:                           `Allow(reason="0.example.com: {ok}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: single Deny simplifies to Deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionDeny("denied", nil)),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {denied}`,
			wantString:                           `Deny(reason="0.example.com: {denied}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
		},
		{
			name: "union: single NoOpinion simplifies to NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           `0.example.com: {noop}`,
			wantString:                           `NoOpinion(reason="0.example.com: {noop}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			name: "union: single ConditionsMap wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(allows=1)]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "union: single Union wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionAllow("", nil))),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: Union[0.example.com: ConditionsMap(denies=1), 1.example.com: Allow]]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
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
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           `0.example.com: {a}, 2.example.com: {c}`,
			wantErrorIs:                          unexpectedErr,
			wantString:                           `NoOpinion(reason="0.example.com: {a}, 2.example.com: {c}", err="[1.example.com: unexpected things happened, 2.example.com: other error]")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// Inner union folds to NoOpinion with an already-aggregated reason. When the
			// outer union folds, collectReasonsAndErrors matches the inner as an unconditional
			// NoOpinion leaf and prepends only the outer sub-name; the inner's aggregated
			// string appears verbatim after the colon.
			name: "union: nested NoOpinion fold embeds inner aggregate under outer prefix",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(
						authorizer.ConditionsAwareDecisionNoOpinion("a", nil),
						authorizer.ConditionsAwareDecisionNoOpinion("b", nil),
					),
				),
			},
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           `0.example.com: {0.example.com: {a}, 1.example.com: {b}}`,
			wantString:                           `NoOpinion(reason="0.example.com: {0.example.com: {a}, 1.example.com: {b}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// Both inner unions fold to NoOpinion with their own aggregated reasons. The
			// outer union then folds and produces a flattened, prefix-embedded string that
			// contains both inner aggregates in order.
			name: "union: nested NoOpinion fold with siblings at each level",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(
						authorizer.ConditionsAwareDecisionNoOpinion("a1", nil),
						authorizer.ConditionsAwareDecisionNoOpinion("a2", nil),
					),
					unionDecision(
						authorizer.ConditionsAwareDecisionNoOpinion("b1", nil),
						authorizer.ConditionsAwareDecisionNoOpinion("b2", nil),
					),
				),
			},
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           `0.example.com: {0.example.com: {a1}, 1.example.com: {a2}}, 1.example.com: {0.example.com: {b1}, 1.example.com: {b2}}`,
			wantString:                           `NoOpinion(reason="0.example.com: {0.example.com: {a1}, 1.example.com: {a2}}, 1.example.com: {0.example.com: {b1}, 1.example.com: {b2}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// Errors at multiple depths: the inner union's fold wraps its own child's error
			// with the inner sub-name; the outer fold wraps the (already-wrapped) inner
			// aggregate again with the outer sub-name, and additionally aggregates a sibling
			// error at the outer level. Result: two prefix-wrapped errors in a length-2
			// aggregate.
			name: "union: nested NoOpinion fold with errors at multiple depths",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("outer-a", otherErr),
					unionDecision(
						authorizer.ConditionsAwareDecisionNoOpinion("inner-b", unexpectedErr),
						authorizer.ConditionsAwareDecisionNoOpinion("inner-c", nil),
					),
				),
			},
			wantIsNoOpinion:                      true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           `0.example.com: {outer-a}, 1.example.com: {0.example.com: {inner-b}, 1.example.com: {inner-c}}`,
			wantErrorIs:                          unexpectedErr,
			wantString:                           `NoOpinion(reason="0.example.com: {outer-a}, 1.example.com: {0.example.com: {inner-b}, 1.example.com: {inner-c}}", err="[0.example.com: other error, 1.example.com: 0.example.com: unexpected things happened]")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion),
		},
		{
			// Add short-circuits after the first Allow/Deny leaf, so the trailing Deny("second")
			// is dropped. The remaining inner slice is [NoOpinion, NoOpinion, Allow], so the
			// simplified reason references the Allow at authorizer name "2.example.com".
			name: "union: Allow before Deny returns Allow, NoOpinions are ignored",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionAllow("first", nil),
					authorizer.ConditionsAwareDecisionDeny("second", nil),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `2.example.com: {first}`,
			wantString:                           `Allow(reason="2.example.com: {first}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
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
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `2.example.com: {first}`,
			wantString:                           `Deny(reason="2.example.com: {first}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
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
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: NoOpinion(reason="no-op1"), 1.example.com: ConditionsMap(allows=1), 2.example.com: NoOpinion(reason="no-op2")]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			// ConditionsMap(allow-only) followed by Allow has PossibleDecisions={Allow}: if the
			// ConditionsMap evaluates to Allow, the answer is Allow; if it evaluates to NoOpinion,
			// the trailing Allow takes over. Either way, the union eagerly simplifies to Allow.
			name: "union: conditionsmap(allow) + allow simplifies to Allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionAllow("allowed", nil)),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `1.example.com: {allowed}`,
			wantString:                           `Allow(reason="1.example.com: {allowed}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			name: "union: conditionsmap(allow) + deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionDeny("no", nil)),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true, // There is an inner Deny
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(allows=1), 1.example.com: Deny(reason="no")]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(deny) + noopinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(denies=1), 1.example.com: NoOpinion(reason="noop")]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(deny) + allow with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionAllow("allowed", unexpectedErr)),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true, // There is an inner Allow
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(denies=1), 1.example.com: Allow(reason="allowed", err="unexpected things happened")]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(allow) + conditionsmap(deny)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, condMapDeny),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(allows=1), 1.example.com: ConditionsMap(denies=1)]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			// The inner union [condMapAllow, Allow("ok")] simplifies to Allow(reason="1.example.com: ok").
			// The trailing NoOpinion is dropped by the outer Add's short-circuit (an Allow is
			// already present). The remaining outer inner is [condMapAllow, Allow("1.example.com: ok")],
			// which again simplifies to Allow with a nested authorizer-name prefix.
			name: "union: nested with allow simplifies through both levels",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					condMapAllow,
					unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionAllow("ok", nil)),
					authorizer.ConditionsAwareDecisionNoOpinion("don't care", nil),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `1.example.com: {1.example.com: {ok}}`,
			wantString:                           `Allow(reason="1.example.com: {1.example.com: {ok}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			// Two-level nesting where every level folds. Inner fold produces
			// Allow(reason="0.example.com: deep"); outer fold matches that as an
			// unconditional Allow leaf and prefixes with its own sub-name.
			name: "union: two-level nested allow (inner union folds to Allow)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(authorizer.ConditionsAwareDecisionAllow("deep", nil)),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {0.example.com: {deep}}`,
			wantString:                           `Allow(reason="0.example.com: {0.example.com: {deep}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			// Three-level nesting with an error carried through each fold. Each fold wraps
			// the sub-decision's Error() with the sub-name prefix (via fmt.Errorf %w) and
			// puts it back into a single-element aggregate, so the final err prints
			// (without brackets) as three concatenated prefixes.
			name: "union: three-level nested allow with error propagates through every fold",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(
						unionDecision(authorizer.ConditionsAwareDecisionAllow("deep", unexpectedErr)),
					),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {0.example.com: {0.example.com: {deep}}}`,
			wantErrorIs:                          unexpectedErr,
			wantString:                           `Allow(reason="0.example.com: {0.example.com: {0.example.com: {deep}}}", err="0.example.com: 0.example.com: 0.example.com: unexpected things happened")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			// Outer union has a NoOpinion sibling. NoOpinion doesn't match the Allow filter
			// and contributes nothing to the aggregate; only the folded inner Allow's
			// reason surfaces, prefixed with the outer sub-name.
			name: "union: nested allow with sibling NoOpinion in outer",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					authorizer.ConditionsAwareDecisionNoOpinion("outer-noop", nil),
					unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionAllow("inner-allow", nil)),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `1.example.com: {1.example.com: {inner-allow}}`,
			wantString:                           `Allow(reason="1.example.com: {1.example.com: {inner-allow}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			// Contrast with the "two-level nested allow" case: here the nested union does
			// NOT fold (its PossibleDecisions is {NoOpinion, Allow} — two outcomes). During
			// the outer fold, collectReasonsAndErrors recurses INTO the still-Union inner
			// via the IsUnion branch. The inner has no matching Allow leaves (its NoOpinion
			// is filtered out, ConditionsMap is skipped by both branches), so only the
			// outer's direct Allow leaf contributes.
			name: "union: nested allow where inner union stays a Union (recursed into, not folded)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(condMapAllow, authorizer.ConditionsAwareDecisionNoOpinion("inner-noop", nil)),
					authorizer.ConditionsAwareDecisionAllow("outer-allow", nil),
				),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `1.example.com: {outer-allow}`,
			wantString:                           `Allow(reason="1.example.com: {outer-allow}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
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
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: false,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(allows=1), 1.example.com: Union[0.example.com: ConditionsMap(allows=1), 1.example.com: NoOpinion(reason="inner"), 2.example.com: Union[0.example.com: ConditionsMap(denies=1), 1.example.com: NoOpinion(reason="inner2")]]]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},

		// --- Unconditional leaves carrying a side-channel error ---
		{
			name: "Allow with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionAllow("ok", errors.New("warning")),
			},
			wantIsAllow:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           "ok",
			wantAnyError:                         true,
			wantString:                           `Allow(reason="ok", err="warning")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow),
		},
		{
			name: "Deny with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionDeny("no", errors.New("warning")),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           "no",
			wantAnyError:                         true,
			wantString:                           `Deny(reason="no", err="warning")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
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
			wantReason:            "",
			wantString:            `Union[0.example.com: ConditionsMap(denies=1)]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionDeny),
		},
		{
			// CM(deny) → Deny short-circuits there; CM(deny) → NoOpinion falls through to Allow.
			// Both outcomes are reachable, so the union stays a Union.
			name: "union: conditionsmap(deny) + allow stays Union",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, allow),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true, // trailing Allow leaf
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(denies=1), 1.example.com: Allow]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			// CM(deny) either yields Deny or NoOpinion; the trailing Deny then catches the NoOpinion
			// branch — so the union eagerly simplifies to Deny.
			name: "union: conditionsmap(deny) + deny simplifies to Deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDeny, deny),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           "",
			wantString:                           `Deny`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
		},
		{
			// Deny counterpart to the two-level nested Allow case.
			name: "union: two-level nested deny (inner union folds to Deny)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(authorizer.ConditionsAwareDecisionDeny("deep", nil)),
				),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {0.example.com: {deep}}`,
			wantString:                           `Deny(reason="0.example.com: {0.example.com: {deep}}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
		},
		{
			// Three-level Deny fold carrying an error through every level. Same
			// mechanism as the three-level Allow variant.
			name: "union: three-level nested deny with error propagates through every fold",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(
						unionDecision(authorizer.ConditionsAwareDecisionDeny("deep", otherErr)),
					),
				),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `0.example.com: {0.example.com: {0.example.com: {deep}}}`,
			wantErrorIs:                          otherErr,
			wantString:                           `Deny(reason="0.example.com: {0.example.com: {0.example.com: {deep}}}", err="0.example.com: 0.example.com: 0.example.com: other error")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
		},
		{
			// Deny counterpart to "nested allow where inner union stays a Union": the
			// inner union has PossibleDecisions={NoOpinion, Deny} (two outcomes, does not
			// fold). Outer folds to Deny; collectReasonsAndErrors recurses into the still-
			// Union inner but its NoOpinion is filtered out and CM(deny) is skipped, so
			// only the outer Deny leaf contributes.
			name: "union: nested deny where inner union stays a Union (recursed into, not folded)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(
					unionDecision(condMapDeny, authorizer.ConditionsAwareDecisionNoOpinion("inner-noop", nil)),
					authorizer.ConditionsAwareDecisionDeny("outer-deny", nil),
				),
			},
			wantIsDeny:                           true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           `1.example.com: {outer-deny}`,
			wantString:                           `Deny(reason="1.example.com: {outer-deny}")`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionDeny),
		},
		{
			// No downstream Allow/Deny to short-circuit, so NoOpinion remains a possible outcome.
			name: "union: conditionsmap(allow) + noopinion (no leaf) stays Union",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, noOp),
			},
			wantIsUnion:           true,
			wantReason:            "",
			wantString:            `Union[0.example.com: ConditionsMap(allows=1), 1.example.com: NoOpinion]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow),
		},
		{
			name: "union: single ConditionsMap(deny+allow) wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapDenyAndAllow),
			},
			wantIsUnion:           true,
			wantReason:            "",
			wantString:            `Union[0.example.com: ConditionsMap(denies=1, allows=1)]`,
			wantPossibleDecisions: sets.New(authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: noopinion + conditionsmap(allow) + deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(noOp, condMapAllow, deny),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true, // trailing Deny leaf
			wantReason:                           "",
			wantString:                           `Union[0.example.com: NoOpinion, 1.example.com: ConditionsMap(allows=1), 2.example.com: Deny]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
		},
		{
			name: "union: conditionsmap(allow) + conditionsmap(deny) + allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				unionDecision(condMapAllow, condMapDeny, allow),
			},
			wantIsUnion:                          true,
			wantContainsUnconditionalAllowOrDeny: true,
			wantReason:                           "",
			wantString:                           `Union[0.example.com: ConditionsMap(allows=1), 1.example.com: ConditionsMap(denies=1), 2.example.com: Allow]`,
			wantPossibleDecisions:                sets.New(authorizer.DecisionAllow, authorizer.DecisionDeny),
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

var _ authorizer.Authorizer = sampleAuthorizer{}

type sampleAuthorizer struct{}

func (a sampleAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return unconditionalParts(a.ConditionsAwareAuthorize(ctx, attrs))
}

func (a sampleAuthorizer) ConditionsAwareAuthorize(ctx context.Context, attrs authorizer.Attributes) authorizer.ConditionsAwareDecision {
	switch attrs.GetUser().GetName() {
	case "alice":
		return authorizer.ConditionsAwareDecisionAllow("", nil)
	case "bob":
		return authorizer.ConditionsAwareDecisionDeny("", nil)
	case "carol":
		// allow carol to read anything, but require seting the owner=carol label on writes
		switch attrs.GetVerb() {
		case "list":
			return authorizer.ConditionsAwareDecisionAllow("", nil)
		case "update":
			return authorizer.ConditionsAwareDecisionConditionsMap(
				nil, nil,
				[]authorizer.Condition{authorizer.GenericCondition{
					ID: "owner-label-is-set",
					Condition: `
						(oldObject != null ? (has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.owner) && oldObject.metadata.labels.owner == "carol") : true) &&
						(object != null ? (has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.owner) && object.metadata.labels.owner == "carol") : true)
					`,
					Type: "test-cel-conditions-type",
				}},
			)
		default:
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
	case "dave":
		// allow dave to read anything, but never set the classified label on writes
		switch attrs.GetVerb() {
		case "list":
			return authorizer.ConditionsAwareDecisionAllow("", nil)
		case "create", "update", "delete":
			return authorizer.ConditionsAwareDecisionConditionsMap(
				[]authorizer.Condition{
					authorizer.GenericCondition{
						ID:        "deny-supersecret-label-on-oldObject",
						Condition: "oldObject != null && has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.supersecret)",
						Type:      "test-cel-conditions-type",
					},
					authorizer.GenericCondition{
						ID:        "deny-supersecret-label-on-object",
						Condition: "object != null && has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.supersecret)",
						Type:      "test-cel-conditions-type",
					},
				},
				nil, nil,
			)
		default:
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}
}

func (a sampleAuthorizer) EvaluateConditions(ctx context.Context, unevaluated authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if !unevaluated.IsConditionsMap() {
		return unevaluated.FailureDecision(), "failed closed", errors.New("can only evaluate ConditionsMap decisions")
	}

	return celEvaluateConditions(ctx, unevaluated.ConditionsMap(), data)
}

func objWithLabels(lbls map[string]string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{}}
	if len(lbls) > 0 {
		obj.SetLabels(lbls)
	}
	return obj
}

func unconditionalParts(d authorizer.ConditionsAwareDecision) (authorizer.Decision, string, error) {
	switch {
	case d.IsAllow():
		return authorizer.DecisionAllow, d.Reason(), d.Error()
	case d.IsDeny():
		return authorizer.DecisionDeny, d.Reason(), d.Error()
	case d.IsNoOpinion():
		return authorizer.DecisionNoOpinion, d.Reason(), d.Error()
	default:
		// An error is not returned here, as that could yield a HTTP response code of 500 instead of 403.
		// For the use-case described above with regards to calling this function in Authorize, not returning
		// an error is important, as it is valid to always fail closed, as if this happens, no unconditional
		// permissions were given the requestor.
		return d.FailureDecision(), "failed closed: tried to return conditional decision to conditions-unaware authorizer", nil
	}
}

// TestConditionsAwareDecisionConditionsMap_ClonesInputSlices verifies that
// ConditionsAwareDecisionConditionsMap defensively copies its input slices, so that
// callers mutating the slices after construction cannot alter the resulting decision.
func TestConditionsAwareDecisionConditionsMap_ClonesInputSlices(t *testing.T) {
	denyConditions := []authorizer.Condition{authorizer.GenericCondition{ID: "deny-orig"}}
	noOpinionConditions := []authorizer.Condition{authorizer.GenericCondition{ID: "nop-orig"}}
	allowConditions := []authorizer.Condition{authorizer.GenericCondition{ID: "allow-orig"}}

	d := authorizer.ConditionsAwareDecisionConditionsMap(denyConditions, noOpinionConditions, allowConditions)
	if !d.IsConditionsMap() {
		t.Fatalf("expected ConditionsMap decision, got %s", d.String())
	}

	// Mutate every element of every input slice through the caller's backing arrays.
	denyConditions[0] = authorizer.GenericCondition{ID: "deny-mutated"}
	noOpinionConditions[0] = authorizer.GenericCondition{ID: "nop-mutated"}
	allowConditions[0] = authorizer.GenericCondition{ID: "allow-mutated"}

	collect := func(seq iter.Seq[authorizer.Condition]) []string {
		var ids []string
		for c := range seq {
			ids = append(ids, c.GetID())
		}
		return ids
	}

	cm := d.ConditionsMap()
	if got, want := collect(cm.DenyConditions()), []string{"deny-orig"}; !slices.Equal(got, want) {
		t.Errorf("DenyConditions IDs = %v, want %v (caller mutation must not leak)", got, want)
	}
	if got, want := collect(cm.NoOpinionConditions()), []string{"nop-orig"}; !slices.Equal(got, want) {
		t.Errorf("NoOpinionConditions IDs = %v, want %v (caller mutation must not leak)", got, want)
	}
	if got, want := collect(cm.AllowConditions()), []string{"allow-orig"}; !slices.Equal(got, want) {
		t.Errorf("AllowConditions IDs = %v, want %v (caller mutation must not leak)", got, want)
	}
}

func TestSampleAuthorizer(t *testing.T) {
	type evalCase struct {
		name      string
		object    *unstructured.Unstructured
		oldObject *unstructured.Unstructured
		// the first case is with conditions-unaware, the second is conditions-aware.
		authorizeDecision [2]string
		finalDecision     [2]string
	}

	tests := []struct {
		name  string
		attrs authorizer.AttributesRecord
		cases []evalCase
	}{
		// alice: unconditional allow for all verbs
		{
			name: "alice list",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "alice"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{`Allow`, `Allow`}},
			},
		},
		{
			name: "alice create",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "alice"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{`Allow`, `Allow`}},
			},
		},
		// bob: unconditional deny for all verbs
		{
			name: "bob list",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "bob"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "deny", authorizeDecision: [2]string{`Deny`, `Deny`}},
			},
		},
		{
			name: "bob create",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "bob"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "deny", authorizeDecision: [2]string{`Deny`, `Deny`}},
			},
		},
		// carol: allow reads, conditional writes (allow on owner=carol)
		{
			name: "carol list",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{`Allow`, `Allow`}},
			},
		},
		{
			name: "carol update",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "update",
			},
			cases: []evalCase{
				{
					name:      "both objects with owner=carol",
					object:    objWithLabels(map[string]string{"owner": "carol"}),
					oldObject: objWithLabels(map[string]string{"owner": "carol"}),
					authorizeDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(allows=1)`,
					},
					finalDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`Allow(reason="condition \"owner-label-is-set\" allowed the request")`,
					},
				},
				{
					name:      "old with owner=carol, new without",
					object:    objWithLabels(map[string]string{"owner": "carol"}),
					oldObject: objWithLabels(nil),
					authorizeDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(allows=1)`,
					},
					finalDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`NoOpinion(reason="no conditions matched")`,
					},
				},
				{
					name:      "new with owner=carol, old with owner=alice",
					object:    objWithLabels(map[string]string{"owner": "alice"}),
					oldObject: objWithLabels(map[string]string{"owner": "carol"}),
					authorizeDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(allows=1)`,
					},
					finalDecision: [2]string{
						`NoOpinion(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`,
						`NoOpinion(reason="no conditions matched")`,
					},
				},
			},
		},
		{
			name: "carol unsupported verb",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "patch",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion`, `NoOpinion`}},
			},
		},
		// dave: allow reads, conditional writes (deny on supersecret label)
		{
			name: "dave list",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{`Allow`, `Allow`}},
			},
		},

		{
			name: "dave update",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "update",
			},
			cases: []evalCase{
				{
					name:              "both objects with supersecret",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request")`},
				},
				{
					name:              "new with supersecret old without",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(nil),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-object\" denied the request")`},
				},
				{
					name:              "new without old with supersecret",
					object:            objWithLabels(nil),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request")`},
				},
				{
					name:              "both without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion(reason="no conditions matched")`},
				},
			},
		},
		{
			name: "dave create",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "create",
			},
			cases: []evalCase{
				{
					name:              "create with supersecret",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-object\" denied the request")`},
				},
				{
					name:              "create without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion(reason="no conditions matched")`},
				},
			},
		},
		{
			name: "dave delete",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "delete",
			},
			cases: []evalCase{
				{
					name:              "delete with supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request")`},
				},
				{
					name:              "delete without supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(denies=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion(reason="no conditions matched")`},
				},
			},
		},
		{
			name: "dave unsupported verb",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "patch",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion`, `NoOpinion`}},
			},
		},
		// unknown user: no opinion
		{
			name: "unknown user get",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "unknown"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion`, `NoOpinion`}},
			},
		},
	}

	authz := sampleAuthorizer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, tc := range tt.cases {
				// if only the authorization decision is specified, the final one is the same
				if len(tc.finalDecision[0]) == 0 && len(tc.finalDecision[1]) == 0 {
					tc.finalDecision[0] = tc.authorizeDecision[0]
					tc.finalDecision[1] = tc.authorizeDecision[1]
				}
				for i, supportsConditions := range [2]bool{false, true} {
					t.Run(fmt.Sprintf("%s/%t", tc.name, supportsConditions), func(t *testing.T) {
						var decision authorizer.ConditionsAwareDecision
						if supportsConditions {
							decision = authz.ConditionsAwareAuthorize(t.Context(), tt.attrs)
						} else {
							decision = authorizer.ConditionsAwareDecisionFromParts(authz.Authorize(t.Context(), tt.attrs))
						}

						if decision.String() != tc.authorizeDecision[i] {
							t.Errorf("got Authorize() decision %s, want %s", decision.String(), tc.authorizeDecision[i])
						}

						// Only object and oldObject is used in celEvaluateConditions, so let all other values be zero here, as they are anyways unused.
						data := admission.NewAttributesRecord(tc.object, tc.oldObject, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil)

						// Wrap in the ConditionsAwareDecision format just to get an unified string comparison mechanism.
						final := decision
						if !decision.IsUnconditional() {
							final = authorizer.ConditionsAwareDecisionFromParts(authz.EvaluateConditions(t.Context(), decision, data))
						}
						if final.String() != tc.finalDecision[i] {
							t.Errorf("got Evaluate() decision %s, want %s", final.String(), tc.finalDecision[i])
						}
					})
				}
			}
		})
	}
}

func celEvaluateConditions(ctx context.Context, conditionsMap authorizer.ConditionsMap, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	env, err := cel.NewEnv(
		cel.Variable("object", cel.DynType),
		cel.Variable("oldObject", cel.DynType),
	)
	if err != nil {
		return conditionsMap.FailureDecision(), "failed closed", fmt.Errorf("failed to create CEL env: %w", err)
	}

	if data == nil {
		return conditionsMap.FailureDecision(), "failed closed", errors.New("evaluating a CEL condition requires non-nil data.AdmissionControl")
	}

	obj, err := objectToResolveVal(data.GetObject())
	if err != nil {
		return conditionsMap.FailureDecision(), "failed closed", fmt.Errorf("failed to convert object to CEL ref.Val: %w", err)
	}

	oldObj, err := objectToResolveVal(data.GetOldObject())
	if err != nil {
		return conditionsMap.FailureDecision(), "failed closed", fmt.Errorf("failed to convert object to CEL ref.Val: %w", err)
	}

	vars := map[string]any{
		"object":    obj,
		"oldObject": oldObj,
	}

	return conditionsMap.Evaluate(ctx, data, func(ctx context.Context, c authorizer.Condition, _ authorizer.ConditionsData) (bool, error) {
		return evalCEL(env, c.GetCondition(), vars)
	})
}

// evalCEL compiles and evaluates a single CEL expression, returning true/false.
func evalCEL(env *cel.Env, expr string, vars map[string]any) (bool, error) {
	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return false, fmt.Errorf("CEL compile error for %q: %w", expr, issues.Err())
	}
	prg, err := env.Program(ast)
	if err != nil {
		return false, fmt.Errorf("CEL program error for %q: %w", expr, err)
	}
	out, _, err := prg.Eval(vars)
	if err != nil {
		return false, fmt.Errorf("CEL eval error for %q: %w", expr, err)
	}
	result, ok := out.Value().(bool)
	if !ok {
		return false, fmt.Errorf("CEL expression %q did not return bool, got %T", expr, out.Value())
	}
	return result, nil
}

func objectToResolveVal(r runtime.Object) (interface{}, error) {
	if r == nil || reflect.ValueOf(r).IsNil() {
		return nil, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(r)
	if err != nil {
		return nil, err
	}
	return ret, nil
}

func TestConditionsAwareDecisionUnionedDecisions(t *testing.T) {
	condMap := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "test", Condition: "true", Type: "test-type"}},
	)
	noOp := authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)

	t.Run("non-union has empty iterator", func(t *testing.T) {
		noUnionTestcases := []authorizer.ConditionsAwareDecision{
			condMap,
			noOp,
			authorizer.ConditionsAwareDecisionAllow("ok", nil),
			authorizer.ConditionsAwareDecisionDeny("not ok", nil),
		}
		for i, tc := range noUnionTestcases {
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				count := 0
				for range tc.UnionedDecisions() {
					count++
				}
				if count != 0 {
					t.Errorf("expected 0 unioned decisions for %s, got %d", tc, count)
				}
			})
		}
	})

	t.Run("union iterates sub-decisions in order", func(t *testing.T) {
		union := unionDecision(condMap, noOp)
		var got []string
		for _, sub := range union.UnionedDecisions() {
			got = append(got, sub.Reason())
		}
		want := []string{"", "noop"}
		if len(got) != len(want) {
			t.Fatalf("expected %d sub-decisions, got %d", len(want), len(got))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("sub-decision[%d].Reason() = %q, want %q", i, got[i], want[i])
			}
		}
	})

	t.Run("early break in iterator", func(t *testing.T) {
		union := unionDecision(condMap, noOp)
		count := 0
		for range union.UnionedDecisions() {
			count++
			break
		}
		if count != 1 {
			t.Errorf("expected early break after 1 iteration, got %d", count)
		}
	})
}
