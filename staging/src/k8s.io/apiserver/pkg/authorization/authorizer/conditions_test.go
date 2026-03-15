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
	"reflect"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestConditionsAwareDecision(t *testing.T) {
	unexpectedErr := fmt.Errorf("unexpected things happened")
	otherErr := fmt.Errorf("other error")

	ctx := t.Context()
	sampleAttrs := authorizer.AttributesRecord{}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	okAmountOfConditions := make([]authorizer.Condition, authorizer.MaxConditionsPerMap)
	for i := range authorizer.MaxConditionsPerMap {
		okAmountOfConditions[i] = authorizer.GenericCondition{ID: fmt.Sprintf("cond-%d", i), Effect: authorizer.ConditionEffectAllow}
	}

	tooManyConditions := make([]authorizer.Condition, authorizer.MaxConditionsPerMap+1)
	for i := range authorizer.MaxConditionsPerMap + 1 {
		tooManyConditions[i] = authorizer.GenericCondition{ID: fmt.Sprintf("cond-%d", i), Effect: authorizer.ConditionEffectAllow}
	}

	condMapAllow := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "allow-cond", Condition: "something", Effect: authorizer.ConditionEffectAllow, Type: "test-type"},
	)
	condMapDeny := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "deny-cond", Condition: "something", Effect: authorizer.ConditionEffectDeny, Type: "test-type"},
	)

	tests := []struct {
		name                    string
		testDecisions           []authorizer.ConditionsAwareDecision
		wantIsAllowed           bool
		wantIsNoOpinion         bool
		wantIsDenied            bool
		wantIsConditionsMap     bool
		wantIsUnion             bool
		wantIsUnconditional     bool
		wantContainsAllowOrDeny bool
		wantFailClosedIsDeny    bool
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
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "",
			wantErrorIs:             nil,
			wantString:              `Deny`,
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
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "foo",
			wantErrorIs:             unexpectedErr,
			wantString:              `Deny(reason="foo", err="unexpected things happened")`,
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
			wantIsAllowed:           true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantReason:              "ok",
			wantErrorIs:             nil,
			wantString:              `Allow(reason="ok")`,
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
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "",
			wantErrorIs:         nil,
			wantString:          `NoOpinion`,
		},
		{
			name: "from parts: unsupported mode",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "", nil
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "",
			wantAnyError:            true,
			wantString:              `Deny(err="unknown unconditional decision type: 42")`,
		},
		{
			name: "from parts: unsupported mode with other error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "foo", otherErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "foo", otherErr
				}).ConditionsAwareAuthorize(ctx, sampleAttrs),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "foo",
			wantErrorIs:             otherErr,
			wantString:              `Deny(reason="foo", err="[other error, unknown unconditional decision type: 42]")`,
		},
		{
			name: "construct valid conditionsmap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(okAmountOfConditions...),
			},
			wantIsConditionsMap: true,
			wantIsUnconditional: false,
			wantString:          `ConditionsMap(len=128)`,
		},
		{
			name: "too many conditions",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(tooManyConditions...),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "failed closed",
			wantAnyError:            true,
			wantString:              `Deny(reason="failed closed", err="too many conditions: 129 exceeds maximum of 128")`,
		},
		{
			name: "construct valid conditionsmap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectAllow},
					nil,
					typedNil(), // nil, but has the type word set so the normal nil check fails
					authorizer.GenericCondition{ID: "baz", Effect: authorizer.ConditionEffectAllow},
				),
			},
			wantIsConditionsMap: true,
			wantIsUnconditional: false,
			wantString:          `ConditionsMap(len=2)`,
		},
		{
			name: "duplicate IDs, ignores nil",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectAllow},
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectDeny},
				),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "failed closed",
			wantAnyError:            true,
			wantString:              `Deny(reason="failed closed", err="duplicate condition ID \"foo\"")`,
		},
		{
			name: "invalid effect",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffect("nonexistent")},
				),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "failed closed",
			wantAnyError:            true,
			wantString:              `Deny(reason="failed closed", err="condition effect \"nonexistent\" not supported. Supported effects are: [Deny, NoOpinion, Allow]")`,
		},
		{
			name: "condition ID must be a Kubernetes label, one condition error enough to fail closed",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectAllow},
					authorizer.GenericCondition{ID: "not a kubernetes label", Effect: authorizer.ConditionEffectDeny},
				),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "failed closed",
			wantAnyError:            true,
			wantString:              `Deny(reason="failed closed", err="invalid condition ID \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "condition type must be a Kubernetes label",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(
					authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectAllow},
					authorizer.GenericCondition{ID: "bar", Effect: authorizer.ConditionEffectNoOpinion, Type: "not a kubernetes label"},
				),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `NoOpinion(reason="failed closed", err="invalid condition type \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "empty ConditionsMap is NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionsMap(),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "empty ConditionsMap",
			wantString:          `NoOpinion(reason="empty ConditionsMap")`,
		},
		// Union constructor simplification cases
		{
			name: "union: empty yields NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(),
			},
			wantIsNoOpinion:         true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    false,
			wantReason:              "",
			wantString:              `NoOpinion`,
		},
		{
			name: "union: single Allow returned as-is",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(authorizer.ConditionsAwareDecisionAllow("ok", nil)),
			},
			wantIsAllowed:           true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    false,
			wantReason:              "ok",
			wantString:              `Allow(reason="ok")`,
		},
		{
			name: "union: single Deny returned as-is",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(authorizer.ConditionsAwareDecisionDeny("denied", nil)),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "denied",
			wantString:              `Deny(reason="denied")`,
		},
		{
			name: "union: single NoOpinion returned as-is",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsNoOpinion:         true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    false,
			wantReason:              "noop",
			wantString:              `NoOpinion(reason="noop")`,
		},
		{
			name: "union: single ConditionsMap wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapAllow),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    false,
			wantReason:              `[""]`,
			wantString:              `Union[ConditionsMap(len=1)]`,
		},
		{
			name: "union: single Union wrapped",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(authorizer.ConditionsAwareDecisionUnion(condMapDeny, authorizer.ConditionsAwareDecisionAllow("", nil))),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              `[["", ""]]`,
			wantString:              `Union[Union[ConditionsMap(len=1), Allow]]`,
		},
		{
			name: "union: all NoOpinion yields merged NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					authorizer.ConditionsAwareDecisionNoOpinion("a", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("", unexpectedErr),
					authorizer.ConditionsAwareDecisionNoOpinion("c", otherErr),
				),
			},
			wantIsNoOpinion:         true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    false,
			wantReason:              "0: a, 2: c",
			wantErrorIs:             unexpectedErr,
			wantString:              `NoOpinion(reason="0: a, 2: c", err="[1: unexpected things happened, 2: other error]")`,
		},
		{
			name: "union: Allow before Deny returns Allow, NoOpinions are ignored",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionAllow("first", nil),
					authorizer.ConditionsAwareDecisionDeny("second", nil),
				),
			},
			wantIsAllowed:           true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    false,
			wantReason:              "first",
			wantString:              `Allow(reason="first")`,
		},
		{
			name: "union: Deny before Allow returns Deny, NoOpinions are ignored",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionNoOpinion("skip", nil),
					authorizer.ConditionsAwareDecisionDeny("first", nil),
					authorizer.ConditionsAwareDecisionAllow("second", nil),
				),
			},
			wantIsDenied:            true,
			wantIsUnconditional:     true,
			wantContainsAllowOrDeny: true,
			wantFailClosedIsDeny:    true,
			wantReason:              "first",
			wantString:              `Deny(reason="first")`,
		},
		// Actual union decisions (not simplified)
		{
			name: "union: noopinion + conditionsmap(allow) + noopinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					authorizer.ConditionsAwareDecisionNoOpinion("no-op1", nil),
					condMapAllow,
					authorizer.ConditionsAwareDecisionNoOpinion("no-op2", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    false,
			wantReason:              `[no-op1, "", no-op2]`,
			wantString:              `Union[NoOpinion(reason="no-op1"), ConditionsMap(len=1), NoOpinion(reason="no-op2")]`,
		},
		{
			name: "union: conditionsmap(allow) + allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapAllow, authorizer.ConditionsAwareDecisionAllow("allowed", nil)),
			},
			// TODO: This could be simplified to an Allow in the future, if we stick with this "eager" evaluation mode (as opposed to the "lazy" one, as described in the Conditional Authorization KEP)
			// That is, a ConditionsMap of _only_ Allow/NoOpinion effects, followed by an unconditional Allow => Allow, no matter what the data is, and likewise
			// A ConditionsMap of _only_ Deny/NoOpinion effects, followed by an unconditional Deny => Deny, no matter what the data is
			// However, to avoid complicating things too much in the beginning, this is not yet implemented. However, if we choose the Lazy evaluation mode, this optimization cannot be done.
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Allow
			wantFailClosedIsDeny:    false,
			wantReason:              `["", allowed]`,
			wantString:              `Union[ConditionsMap(len=1), Allow(reason="allowed")]`,
		},
		{
			name: "union: conditionsmap(allow) + deny",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapAllow, authorizer.ConditionsAwareDecisionDeny("no", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Deny
			wantFailClosedIsDeny:    true,
			wantReason:              `["", no]`,
			wantString:              `Union[ConditionsMap(len=1), Deny(reason="no")]`,
		},
		{
			name: "union: conditionsmap(deny) + noopinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapDeny, authorizer.ConditionsAwareDecisionNoOpinion("noop", nil)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    true, // There are Deny conditions
			wantReason:              `["", noop]`,
			wantString:              `Union[ConditionsMap(len=1), NoOpinion(reason="noop")]`,
		},
		{
			name: "union: conditionsmap(deny) + allow with error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapDeny, authorizer.ConditionsAwareDecisionAllow("allowed", unexpectedErr)),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Allow
			wantFailClosedIsDeny:    true, // There are Deny conditions
			wantReason:              `["", allowed]`,
			wantErrorIs:             unexpectedErr,
			wantString:              `Union[ConditionsMap(len=1), Allow(reason="allowed", err="unexpected things happened")]`,
		},
		{
			name: "union: conditionsmap(allow) + conditionsmap(deny)",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(condMapAllow, condMapDeny),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    true, // There are Deny conditions
			wantReason:              `["", ""]`,
			wantString:              `Union[ConditionsMap(len=1), ConditionsMap(len=1)]`,
		},
		{
			name: "union: nested with allow",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					condMapAllow,
					authorizer.ConditionsAwareDecisionUnion(condMapAllow, authorizer.ConditionsAwareDecisionAllow("ok", nil)),
					// Note: NoOpinions after a decision for which ContainsAllowOrDeny() == true won't ever affect the decision outcome, and could thus be trimmed in the union constructor.
					// This is implicitly done in the union authorizer, as it short-circuits after it found some decision for which ContainsAllowOrDeny() == true, so this input would not
					// be yielded by Kubernetes own components, but someone else using this library could invoke the union constructor on this input. However, this optimization is skipped
					// as the impact is minor, and the code would get quite a bit more complicated.
					authorizer.ConditionsAwareDecisionNoOpinion("don't care", nil),
				),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: true, // There is an inner Allow
			wantFailClosedIsDeny:    false,
			wantReason:              `["", ["", ok], don't care]`,
			wantString:              `Union[ConditionsMap(len=1), Union[ConditionsMap(len=1), Allow(reason="ok")], NoOpinion(reason="don't care")]`,
		},
		{
			name: "union: deep nesting without anything unconditional",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionUnion(
					condMapAllow,
					authorizer.ConditionsAwareDecisionUnion(
						condMapAllow,
						authorizer.ConditionsAwareDecisionNoOpinion("inner", nil),
						authorizer.ConditionsAwareDecisionUnion(
							condMapDeny,
							authorizer.ConditionsAwareDecisionNoOpinion("inner2", nil)),
					),
				),
			},
			wantIsUnion:             true,
			wantContainsAllowOrDeny: false,
			wantFailClosedIsDeny:    true,
			wantReason:              `["", ["", inner, ["", inner2]]]`,
			wantString:              `Union[ConditionsMap(len=1), Union[ConditionsMap(len=1), NoOpinion(reason="inner"), Union[ConditionsMap(len=1), NoOpinion(reason="inner2")]]]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, d := range tt.testDecisions {
				t.Run(fmt.Sprint(i), func(t *testing.T) {
					isAllowed := d.IsAllowed()
					if isAllowed != tt.wantIsAllowed {
						t.Errorf("IsAllowed() = %v, want %v", isAllowed, tt.wantIsAllowed)
					}
					isNoOpinion := d.IsNoOpinion()
					if isNoOpinion != tt.wantIsNoOpinion {
						t.Errorf("IsNoOpinion() = %v, want %v", isNoOpinion, tt.wantIsNoOpinion)
					}
					isDenied := d.IsDenied()
					if isDenied != tt.wantIsDenied {
						t.Errorf("IsDenied() = %v, want %v", isDenied, tt.wantIsDenied)
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
					if isUnconditional != tt.wantIsUnconditional {
						t.Errorf("IsUnconditional() = %v, want %v", isUnconditional, tt.wantIsUnconditional)
					}
					containsAllowOrDeny := d.ContainsAllowOrDeny()
					if containsAllowOrDeny != tt.wantContainsAllowOrDeny {
						t.Errorf("ContainsAllowOrDeny() = %v, want %v", containsAllowOrDeny, tt.wantContainsAllowOrDeny)
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
					failClosed := d.FailClosedDecision()
					if tt.wantFailClosedIsDeny {
						if failClosed != authorizer.DecisionDeny {
							t.Errorf("want FailClosedDecision() == Deny; got %s", failClosed)
						}
					} else {
						if failClosed != authorizer.DecisionNoOpinion {
							t.Errorf("want FailClosedDecision() == NoOpinion; got %s", failClosed)
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

func typedNil() authorizer.Condition {
	var c *authorizer.GenericCondition = nil
	return c
}

func TestCreateConditionsMapFeatureDisabled(t *testing.T) {
	// Feature gate is disabled (which is the default) in this test
	// Fail closed to NoOpinion, as there are no denies
	d := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectAllow},
	)
	if !d.IsNoOpinion() {
		t.Error("Expected creating a ConditionsMap decision to yield NoOpinion when the feature gate is disabled")
	}
	if !strings.Contains(d.Reason(), "ConditionalAuthorization feature gate is disabled") {
		t.Errorf("Expected reason to tell about feature gate being disabled, got %q", d.Reason())
	}
	// Fail closed to Deny, as there is at least one Deny condition
	d = authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "foo", Effect: authorizer.ConditionEffectDeny},
	)
	if !d.IsDenied() {
		t.Error("Expected creating a ConditionsMap decision to yield Deny when the feature gate is disabled")
	}
	if !strings.Contains(d.Reason(), "ConditionalAuthorization feature gate is disabled") {
		t.Errorf("Expected reason to tell about feature gate being disabled, got %q", d.Reason())
	}
}

func TestConditionsMapEvaluate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	evalErr := errors.New("eval error")

	trueResult := authorizer.ConditionEvaluationResultBoolean(true)
	falseResult := authorizer.ConditionEvaluationResultBoolean(false)
	errResult := authorizer.ConditionEvaluationResultError(evalErr)

	cond := func(id string, effect authorizer.ConditionEffect, result authorizer.ConditionEvaluationResult) authorizer.GenericCondition {
		return authorizer.GenericCondition{
			ID:     id,
			Effect: effect,
			EvaluateFunc: func(context.Context, authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return result
			},
		}
	}
	condDesc := func(id string, effect authorizer.ConditionEffect, desc string, result authorizer.ConditionEvaluationResult) authorizer.GenericCondition {
		c := cond(id, effect, result)
		c.Description = desc
		return c
	}
	unevalCond := func(id string, effect authorizer.ConditionEffect) authorizer.GenericCondition {
		return authorizer.GenericCondition{ID: id, Effect: effect} // nil EvaluateFunc → unevaluatable
	}

	type subCase struct {
		name         string
		conditions   []authorizer.Condition
		evaluateFunc func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult
	}

	tests := []struct {
		name     string
		subCases []subCase
		// All sub-cases must produce a decision whose String() equals wantString.
		wantString string
		// For ConditionsMap results, additionally verify structure:
		wantIsConditionsMap bool
		wantDenyCount       int
		wantNoOpinionCount  int
		wantAllowCount      int
	}{
		// ============================================================
		// Deny: at least one deny condition matched
		// ============================================================
		{
			name:       "deny: at least one deny condition matched",
			wantString: `Deny(reason="condition \"deny-1\" denied the request")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("deny-1", authorizer.ConditionEffectDeny, trueResult)},
				},
				{
					name: "matching deny trumps any other case",
					conditions: []authorizer.Condition{
						cond("nop-yes", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("nop-err", authorizer.ConditionEffectNoOpinion, errResult),
						cond("nop-no", authorizer.ConditionEffectNoOpinion, falseResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
						cond("allow-yes", authorizer.ConditionEffectAllow, trueResult),
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						unevalCond("allow-uneval", authorizer.ConditionEffectAllow),
						cond("deny-no", authorizer.ConditionEffectDeny, falseResult),
						unevalCond("deny-uneval", authorizer.ConditionEffectDeny),
						cond("deny-err", authorizer.ConditionEffectDeny, errResult),
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
					},
				},
				{
					name: "with erroring deny (error ignored due to match)",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						cond("deny-err", authorizer.ConditionEffectDeny, errResult),
					},
				},
				{
					name: "with unevaluatable deny (ignored due to match)",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						unevalCond("deny-uneval", authorizer.ConditionEffectDeny),
					},
				},
				{
					name: "with false+error+unevaluatable deny (all ignored due to match)",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						cond("deny-2", authorizer.ConditionEffectDeny, falseResult),
						cond("deny-err", authorizer.ConditionEffectDeny, errResult),
						unevalCond("deny-uneval", authorizer.ConditionEffectDeny),
					},
				},
				{
					name: "deny match takes precedence over matching nop and allow; only fast conditions-evaluation",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
					evaluateFunc: func(ctx context.Context, cd authorizer.ConditionsData, c authorizer.Condition) authorizer.ConditionEvaluationResult {
						panic("should never be called, as all conditions could readily be evaluated")
					},
				},
				{
					name: "deny match with false nop and allow",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-1", authorizer.ConditionEffectAllow, falseResult),
					},
				},
				{
					name: "deny match with unevaluatable nop and allow",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						unevalCond("nop-1", authorizer.ConditionEffectNoOpinion),
						unevalCond("allow-1", authorizer.ConditionEffectAllow),
					},
				},
				{
					name: "deny match with erroring nop and allow",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, trueResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, errResult),
						cond("allow-1", authorizer.ConditionEffectAllow, errResult),
					},
				},
				{
					name: "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns true)",
					conditions: []authorizer.Condition{
						unevalCond("deny-1", authorizer.ConditionEffectDeny),
					},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						return authorizer.ConditionEvaluationResultBoolean(true)
					},
				},
			},
		},
		{
			name:       "deny: at least one deny condition matched with description",
			wantString: `Deny(reason="condition \"deny-1\" denied the request with description \"access denied\"")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{condDesc("deny-1", authorizer.ConditionEffectDeny, "access denied", trueResult)},
				},
				{
					name: "with false nop and allow",
					conditions: []authorizer.Condition{
						condDesc("deny-1", authorizer.ConditionEffectDeny, "access denied", trueResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-1", authorizer.ConditionEffectAllow, falseResult),
					},
				},
			},
		},

		// ============================================================
		// Deny: error, fail closed
		// ============================================================
		{
			name:       "deny: error fail closed",
			wantString: `Deny(reason="one or more conditional evaluation errors occurred", err="condition \"deny-1\" with effect=Deny produced error: eval error")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("deny-1", authorizer.ConditionEffectDeny, errResult)},
				},
				{
					name: "with false deny",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, errResult),
						cond("deny-2", authorizer.ConditionEffectDeny, falseResult),
					},
				},
				{
					name: "error takes precedence over unevaluatable deny",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, errResult),
						unevalCond("deny-uneval", authorizer.ConditionEffectDeny),
					},
				},
				{
					name: "deny error trumps noopinion and allow of any form",
					conditions: []authorizer.Condition{
						cond("nop-yes", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("nop-err", authorizer.ConditionEffectNoOpinion, errResult),
						cond("nop-no", authorizer.ConditionEffectNoOpinion, falseResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
						cond("allow-yes", authorizer.ConditionEffectAllow, trueResult),
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						unevalCond("allow-uneval", authorizer.ConditionEffectAllow),
						cond("deny-no", authorizer.ConditionEffectDeny, falseResult),
						unevalCond("deny-uneval", authorizer.ConditionEffectDeny),
						cond("deny-1", authorizer.ConditionEffectDeny, errResult),
					},
				},
			},
		},
		// TODO: Showcase a test with two deny errors erroring at the same time.

		// ============================================================
		// NoOpinion: at least one noopinion condition matched
		// ============================================================
		{
			name:       "noopinion: at least one noopinion condition matched",
			wantString: `NoOpinion(reason="condition \"nop-1\" evaluated to NoOpinion")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult)},
				},
				{
					name: "noopinion match trumps any noopinion or allow form",
					conditions: []authorizer.Condition{
						cond("nop-err", authorizer.ConditionEffectNoOpinion, errResult),
						cond("nop-no", authorizer.ConditionEffectNoOpinion, falseResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
						cond("allow-yes", authorizer.ConditionEffectAllow, trueResult),
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						unevalCond("allow-uneval", authorizer.ConditionEffectAllow),
						cond("deny-no", authorizer.ConditionEffectDeny, falseResult),

						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
					},
				},
				{
					name: "with erroring nop (error ignored due to match)",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("nop-err", authorizer.ConditionEffectNoOpinion, errResult),
					},
				},
				{
					name: "with unevaluatable nop (ignored due to match)",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
					},
				},
				{
					name: "with false+error+unevaluatable nop (all ignored due to match)",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("nop-2", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("nop-err", authorizer.ConditionEffectNoOpinion, errResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
					},
				},
				{
					name: "nop match takes precedence over matching allow",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "with false deny, nop matches",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "nop match with unevaluatable allow",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						unevalCond("allow-1", authorizer.ConditionEffectAllow),
					},
				},
				{
					name: "nop match with erroring allow",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, errResult),
					},
				},
			},
		},
		{
			name:       "noopinion: at least one noopinion condition matched with description",
			wantString: `NoOpinion(reason="condition \"nop-1\" evaluated to NoOpinion with description \"not relevant\"")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{condDesc("nop-1", authorizer.ConditionEffectNoOpinion, "not relevant", trueResult)},
				},
				{
					name: "with false deny and allow",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						condDesc("nop-1", authorizer.ConditionEffectNoOpinion, "not relevant", trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, falseResult),
					},
				},
			},
		},

		// ============================================================
		// NoOpinion: error, fail closed (from nop)
		// ============================================================
		{
			name:       "noopinion: nop error fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="condition \"nop-1\" with effect=NoOpinion produced error: eval error")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("nop-1", authorizer.ConditionEffectNoOpinion, errResult)},
				},
				{
					name: "noopinion error trumps noopinion unevaluated and any other allow",
					conditions: []authorizer.Condition{
						cond("nop-no", authorizer.ConditionEffectNoOpinion, falseResult),
						unevalCond("nop-uneval", authorizer.ConditionEffectNoOpinion),
						cond("allow-yes", authorizer.ConditionEffectAllow, trueResult),
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						unevalCond("allow-uneval", authorizer.ConditionEffectAllow),
						cond("deny-no", authorizer.ConditionEffectDeny, falseResult),

						cond("nop-1", authorizer.ConditionEffectNoOpinion, errResult),
					},
				},
				{
					name: "nop error trumps matching allow",
					conditions: []authorizer.Condition{
						cond("nop-1", authorizer.ConditionEffectNoOpinion, errResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "with false deny, nop error, matching allow",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, errResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
			},
		},

		// ============================================================
		// NoOpinion: error, fail closed (from allow)
		// ============================================================
		{
			name:       "noopinion: single allow error fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="condition \"allow-1\" with effect=Allow produced error: eval error")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("allow-1", authorizer.ConditionEffectAllow, errResult)},
				},
				{
					name: "with false deny and nop",
					conditions: []authorizer.Condition{
						cond("deny-no", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-no", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						unevalCond("allow-uneval", authorizer.ConditionEffectAllow),

						cond("allow-1", authorizer.ConditionEffectAllow, errResult),
					},
				},
				{
					name:       "via evaluateFunc fallback (condition unevaluatable, evaluateFunc errors)",
					conditions: []authorizer.Condition{unevalCond("allow-1", authorizer.ConditionEffectAllow)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						return authorizer.ConditionEvaluationResultError(evalErr)
					},
				},
				{
					name:       "condition errors, evaluateFunc panics (not called)",
					conditions: []authorizer.Condition{cond("allow-1", authorizer.ConditionEffectAllow, errResult)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						panic("should not be called")
					},
				},
			},
		},
		{
			name:       "noopinion: multiple allow errors fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="[condition \"allow-1\" with effect=Allow produced error: eval error, condition \"allow-2\" with effect=Allow produced error: eval error]")`,
			subCases: []subCase{
				{
					name: "minimal",
					conditions: []authorizer.Condition{
						cond("allow-1", authorizer.ConditionEffectAllow, errResult),
						cond("allow-2", authorizer.ConditionEffectAllow, errResult),
					},
				},
			},
		},

		// ============================================================
		// NoOpinion: no conditions matched
		// ============================================================
		{
			name:       "noopinion: no conditions matched",
			wantString: `NoOpinion(reason="no conditions matched")`,
			subCases: []subCase{
				{
					name:       "single deny false",
					conditions: []authorizer.Condition{cond("deny-1", authorizer.ConditionEffectDeny, falseResult)},
				},
				{
					name:       "single nop false",
					conditions: []authorizer.Condition{cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult)},
				},
				{
					name:       "single allow false",
					conditions: []authorizer.Condition{cond("allow-1", authorizer.ConditionEffectAllow, falseResult)},
				},
				{
					name: "all effects false",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-1", authorizer.ConditionEffectAllow, falseResult),
					},
				},
				{
					name:       "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns false)",
					conditions: []authorizer.Condition{unevalCond("allow-1", authorizer.ConditionEffectAllow)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						return authorizer.ConditionEvaluationResultBoolean(false)
					},
				},
			},
		},

		// ============================================================
		// NoOpinion: unevaluatable nop with no allow conditions
		// ============================================================
		{
			name:       "noopinion: unevaluatable nop, no allow -> NoOpinion",
			wantString: `NoOpinion(reason="at least one NoOpinion condition matched, or no conditions matched")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{unevalCond("nop-1", authorizer.ConditionEffectNoOpinion)},
				},
				{
					name: "with false deny",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						unevalCond("nop-1", authorizer.ConditionEffectNoOpinion),
					},
				},
			},
		},

		// ============================================================
		// Allow: at least one allow condition matched
		// ============================================================
		{
			name:       "allow: at least one allow condition matched",
			wantString: `Allow(reason="condition \"allow-1\" allowed the request")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{cond("allow-1", authorizer.ConditionEffectAllow, trueResult)},
				},
				{
					name: "with false allow",
					conditions: []authorizer.Condition{
						cond("allow-no", authorizer.ConditionEffectAllow, falseResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "with false deny and nop",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name:       "evaluateFunc panics (not called, condition self-evaluates)",
					conditions: []authorizer.Condition{cond("allow-1", authorizer.ConditionEffectAllow, trueResult)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						panic("should not be called")
					},
				},
				{
					name:       "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns true)",
					conditions: []authorizer.Condition{unevalCond("allow-1", authorizer.ConditionEffectAllow)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						return authorizer.ConditionEvaluationResultBoolean(true)
					},
				},
			},
		},
		{
			name:       "allow: at least one allow condition matched with description",
			wantString: `Allow(reason="condition \"allow-1\" allowed the request with description \"access granted\"")`,
			subCases: []subCase{
				{
					name:       "minimal",
					conditions: []authorizer.Condition{condDesc("allow-1", authorizer.ConditionEffectAllow, "access granted", trueResult)},
				},
				{
					name: "with false deny and nop",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						condDesc("allow-1", authorizer.ConditionEffectAllow, "access granted", trueResult),
					},
				},
			},
		},

		// ============================================================
		// Allow: condition matched with error warning
		// ============================================================
		{
			name:       "allow: condition matched with error warning from other allow",
			wantString: `Allow(reason="condition \"allow-1\" allowed the request", err="condition \"allow-err\" with effect=Allow produced error: eval error")`,
			subCases: []subCase{
				{
					name: "minimal",
					conditions: []authorizer.Condition{
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "with false deny and nop",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						cond("allow-err", authorizer.ConditionEffectAllow, errResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
			},
		},

		// ============================================================
		// ConditionsMap: refined map with unevaluatable conditions
		// ============================================================
		{
			name:                "conditionsmap: deny unevaluatable, nop and allow present",
			wantString:          `ConditionsMap(len=3)`,
			wantIsConditionsMap: true,
			wantDenyCount:       1,
			wantNoOpinionCount:  1,
			wantAllowCount:      1,
			subCases: []subCase{
				{
					name: "minimal",
					conditions: []authorizer.Condition{
						unevalCond("deny-1", authorizer.ConditionEffectDeny),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "one deny false, one deny unevaluatable",
					conditions: []authorizer.Condition{
						cond("deny-false", authorizer.ConditionEffectDeny, falseResult),
						unevalCond("deny-1", authorizer.ConditionEffectDeny),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, trueResult),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
			},
		},
		{
			name:                "conditionsmap: nop unevaluatable, allow present",
			wantString:          `ConditionsMap(len=2)`,
			wantIsConditionsMap: true,
			wantNoOpinionCount:  1,
			wantAllowCount:      1,
			subCases: []subCase{
				{
					name: "minimal",
					conditions: []authorizer.Condition{
						unevalCond("nop-1", authorizer.ConditionEffectNoOpinion),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
				{
					name: "with false deny",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						unevalCond("nop-1", authorizer.ConditionEffectNoOpinion),
						cond("allow-1", authorizer.ConditionEffectAllow, trueResult),
					},
				},
			},
		},
		{
			name:                "conditionsmap: allow unevaluatable",
			wantString:          `ConditionsMap(len=1)`,
			wantIsConditionsMap: true,
			wantAllowCount:      1,
			subCases: []subCase{
				{
					name:       "minimal (nil evaluateFunc)",
					conditions: []authorizer.Condition{unevalCond("allow-1", authorizer.ConditionEffectAllow)},
				},
				{
					name:       "evaluateFunc also returns unevaluatable",
					conditions: []authorizer.Condition{unevalCond("allow-1", authorizer.ConditionEffectAllow)},
					evaluateFunc: func(context.Context, authorizer.ConditionsData, authorizer.Condition) authorizer.ConditionEvaluationResult {
						return authorizer.ConditionsEvaluationResultUnevaluatable()
					},
				},
				{
					name: "with false deny and nop",
					conditions: []authorizer.Condition{
						cond("deny-1", authorizer.ConditionEffectDeny, falseResult),
						cond("nop-1", authorizer.ConditionEffectNoOpinion, falseResult),
						unevalCond("allow-1", authorizer.ConditionEffectAllow),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, sc := range tt.subCases {
				t.Run(sc.name, func(t *testing.T) {
					// Construct the ConditionsMap via the constructor to exercise validation.
					decision := authorizer.ConditionsAwareDecisionConditionsMap(sc.conditions...)
					if !decision.IsConditionsMap() {
						t.Fatalf("expected ConditionsMap from constructor, got %s", decision.String())
					}
					cm := decision.ConditionsMap()

					result := cm.Evaluate(t.Context(), authorizer.ConditionsData{}, sc.evaluateFunc)
					if got := result.String(); got != tt.wantString {
						t.Errorf("got decision %s, want %s", got, tt.wantString)
					}
					if tt.wantIsConditionsMap {
						if !result.IsConditionsMap() {
							t.Fatalf("expected ConditionsMap decision, got %s", result.String())
						}
						rcm := result.ConditionsMap()
						gotDeny, gotNoOpinion, gotAllow := countConditions(rcm)
						if gotDeny != tt.wantDenyCount {
							t.Errorf("deny count = %d, want %d", gotDeny, tt.wantDenyCount)
						}
						if gotNoOpinion != tt.wantNoOpinionCount {
							t.Errorf("noopinion count = %d, want %d", gotNoOpinion, tt.wantNoOpinionCount)
						}
						if gotAllow != tt.wantAllowCount {
							t.Errorf("allow count = %d, want %d", gotAllow, tt.wantAllowCount)
						}
					}
				})
			}
		})
	}
}

func countConditions(cm authorizer.ConditionsMap) (deny, noopinion, allow int) {
	for range cm.DenyConditions() {
		deny++
	}
	for range cm.NoOpinionConditions() {
		noopinion++
	}
	for range cm.AllowConditions() {
		allow++
	}
	return
}

// TestConditionsMapEvaluateDeepCopy verifies that when a refined ConditionsMap is returned
// because some conditions are unevaluatable, the non-evaluated conditions from lower-priority
// effect groups are deep-copied and independent from the original.
func TestConditionsMapEvaluateDeepCopy(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	marker := "original"

	// Unevaluatable deny → triggers refined ConditionsMap with deep-copied nop and allow.
	decision := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "deny-uneval", Effect: authorizer.ConditionEffectDeny},
		&deepCopyTracker{id: "nop-1", effect: authorizer.ConditionEffectNoOpinion, marker: &marker},
		&deepCopyTracker{id: "allow-1", effect: authorizer.ConditionEffectAllow, marker: &marker},
	)
	if !decision.IsConditionsMap() {
		t.Fatalf("expected ConditionsMap from constructor, got %s", decision.String())
	}
	cm := decision.ConditionsMap()

	result := cm.Evaluate(t.Context(), authorizer.ConditionsData{}, nil)
	if !result.IsConditionsMap() {
		t.Fatalf("expected refined ConditionsMap, got %s", result.String())
	}

	refined := result.ConditionsMap()
	if refined.Length() != 3 {
		t.Fatalf("expected 3 conditions in refined map, got %d", refined.Length())
	}

	// Mutate the original marker.
	marker = "mutated"

	// Verify the deep-copied conditions in the refined ConditionsMap still have "original".
	for c := range refined.NoOpinionConditions() {
		tracker := c.(*deepCopyTracker)
		if *tracker.marker != "original" {
			t.Errorf("deep copy failed for noopinion condition: marker = %q, want %q", *tracker.marker, "original")
		}
	}
	for c := range refined.AllowConditions() {
		tracker := c.(*deepCopyTracker)
		if *tracker.marker != "original" {
			t.Errorf("deep copy failed for allow condition: marker = %q, want %q", *tracker.marker, "original")
		}
	}
}

// deepCopyTracker is a Condition implementation with a pointer field to verify deep copy behavior.
type deepCopyTracker struct {
	id     string
	effect authorizer.ConditionEffect
	marker *string
}

func (c *deepCopyTracker) GetID() string                         { return c.id }
func (c *deepCopyTracker) GetEffect() authorizer.ConditionEffect { return c.effect }
func (c *deepCopyTracker) GetType() string                       { return "" }
func (c *deepCopyTracker) GetCondition() string                  { return "" }
func (c *deepCopyTracker) GetDescription() string                { return "" }
func (c *deepCopyTracker) Evaluate(context.Context, authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
	return authorizer.ConditionsEvaluationResultUnevaluatable()
}
func (c *deepCopyTracker) DeepCopy() authorizer.Condition {
	cp := *c
	if c.marker != nil {
		m := *c.marker
		cp.marker = &m
	}
	return &cp
}

var _ authorizer.Authorizer = sampleAuthorizer{}

type sampleAuthorizer struct{}

func (a sampleAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return a.ConditionsAwareAuthorize(ctx, attrs).UnconditionalParts()
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
				authorizer.GenericCondition{
					ID: "owner-label-is-set",
					Condition: `
						(oldObject != null ? (has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.owner) && oldObject.metadata.labels.owner == "carol") : true) &&
						(object != null ? (has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.owner) && object.metadata.labels.owner == "carol") : true)
					`,
					Effect: authorizer.ConditionEffectAllow,
					Type:   "test-cel-conditions-type",
				},
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
				authorizer.GenericCondition{
					ID:        "deny-supersecret-label-on-oldObject",
					Condition: "oldObject != null && has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.supersecret)",
					Effect:    authorizer.ConditionEffectDeny,
					Type:      "test-cel-conditions-type",
				},
				authorizer.GenericCondition{
					ID:        "deny-supersecret-label-on-object",
					Condition: "object != null && has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.supersecret)",
					Effect:    authorizer.ConditionEffectDeny,
					Type:      "test-cel-conditions-type",
				})
		default:
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}
}

func (a sampleAuthorizer) EvaluateConditions(ctx context.Context, unevaluated authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if unevaluated.IsUnconditional() {
		return unevaluated.UnconditionalParts()
	}
	if !unevaluated.IsConditionsMap() {
		return unevaluated.FailClosedDecision(), "failed closed", errors.New("can only evaluate unconditional or ConditionsMap decisions")
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

func TestSampleAuthorizer(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
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
		// carol: allow reads, conditional writes (EffectAllow on owner=carol)
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
						`ConditionsMap(len=1)`,
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
						`ConditionsMap(len=1)`,
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
						`ConditionsMap(len=1)`,
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
		// dave: allow reads, conditional writes (EffectDeny on supersecret label)
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
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request, condition \"deny-supersecret-label-on-object\" denied the request")`},
				},
				{
					name:              "new with supersecret old without",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(nil),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-object\" denied the request")`},
				},
				{
					name:              "new without old with supersecret",
					object:            objWithLabels(nil),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request")`},
				},
				{
					name:              "both without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
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
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-object\" denied the request")`},
				},
				{
					name:              "create without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
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
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request")`},
				},
				{
					name:              "delete without supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(len=2)`},
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
						data := authorizer.ConditionsData{
							AdmissionControl: admission.NewAttributesRecord(tc.object, tc.oldObject, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil),
						}

						// Wrap in the ConditionsAwareDecision format just to get an unified string comparison mechanism.
						final := authorizer.ConditionsAwareDecisionFromParts(authz.EvaluateConditions(t.Context(), decision, data))
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
		return conditionsMap.FailClosedDecision(), "failed closed", fmt.Errorf("failed to create CEL env: %w", err)
	}

	if data.AdmissionControl == nil {
		return conditionsMap.FailClosedDecision(), "failed closed", errors.New("evaluating a CEL condition requires non-nil data.AdmissionControl")
	}

	obj, err := objectToResolveVal(data.AdmissionControl.GetObject())
	if err != nil {
		return conditionsMap.FailClosedDecision(), "failed closed", fmt.Errorf("failed to convert object to CEL ref.Val: %w", err)
	}

	oldObj, err := objectToResolveVal(data.AdmissionControl.GetOldObject())
	if err != nil {
		return conditionsMap.FailClosedDecision(), "failed closed", fmt.Errorf("failed to convert object to CEL ref.Val: %w", err)
	}

	vars := map[string]any{
		"object":    obj,
		"oldObject": oldObj,
	}

	return conditionsMap.Evaluate(ctx, data, func(ctx context.Context, _ authorizer.ConditionsData, c authorizer.Condition) authorizer.ConditionEvaluationResult {
		return evalCEL(env, c.GetCondition(), vars)
	}).UnconditionalParts()
}

// evalCEL compiles and evaluates a single CEL expression, returning true/false.
func evalCEL(env *cel.Env, expr string, vars map[string]any) authorizer.ConditionEvaluationResult {
	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return authorizer.ConditionEvaluationResultError(fmt.Errorf("CEL compile error for %q: %w", expr, issues.Err()))
	}
	prg, err := env.Program(ast)
	if err != nil {
		return authorizer.ConditionEvaluationResultError(fmt.Errorf("CEL program error for %q: %w", expr, err))
	}
	out, _, err := prg.Eval(vars)
	if err != nil {
		return authorizer.ConditionEvaluationResultError(fmt.Errorf("CEL eval error for %q: %w", expr, err))
	}
	result, ok := out.Value().(bool)
	if !ok {
		return authorizer.ConditionEvaluationResultError(fmt.Errorf("CEL expression %q did not return bool, got %T", expr, out.Value()))
	}
	return authorizer.ConditionEvaluationResultBoolean(result)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	condMap := authorizer.ConditionsAwareDecisionConditionsMap(
		authorizer.GenericCondition{ID: "test", Condition: "true", Effect: authorizer.ConditionEffectAllow, Type: "test-type"},
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
		union := authorizer.ConditionsAwareDecisionUnion(condMap, noOp)
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
		union := authorizer.ConditionsAwareDecisionUnion(condMap, noOp)
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
