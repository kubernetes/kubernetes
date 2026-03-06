/*
Copyright 2026 The Kubernetes Authors.

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
	"strings"
	"testing"

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
	samplePref := authorizer.ConditionsEncodingPreference{}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	tests := []struct {
		name                string
		testDecisions       []authorizer.ConditionsAwareDecision
		wantIsAllowed       bool
		wantIsNoOpinion     bool
		wantIsDenied        bool
		wantIsConditionsMap bool
		wantIsUnconditional bool
		wantReason          string
		wantAnyError        bool
		wantErrorIs         error
		wantString          string
	}{
		{
			name: "zero value",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecision{},
				authorizer.ConditionsAwareDecisionFromParts(0, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (named1 authorizer.Decision, named2 string, named3 error) {
					return
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "",
			wantErrorIs:         nil,
			wantString:          `Deny("", <nil>)`,
		},
		{
			name: "deny constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionDeny("foo", unexpectedErr),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionDeny, "foo", unexpectedErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "foo", unexpectedErr
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "foo",
			wantErrorIs:         unexpectedErr,
			wantString:          `Deny("foo", "unexpected things happened")`,
		},
		{
			name: "allow constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionAllow("ok", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionAllow, "ok", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "ok", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsAllowed:       true,
			wantIsUnconditional: true,
			wantReason:          "ok",
			wantErrorIs:         nil,
			wantString:          `Allow("ok", <nil>)`,
		},
		{
			name: "noopinion constructor",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				authorizer.ConditionsAwareDecisionFromParts(authorizer.DecisionNoOpinion, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return authorizer.DecisionNoOpinion, "", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "",
			wantErrorIs:         nil,
			wantString:          `NoOpinion("", <nil>)`,
		},
		{
			name: "from parts: unsupported mode",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "", nil),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "", nil
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "",
			wantAnyError:        true,
			wantString:          `Deny("", "unknown unconditional decision type: 42")`,
		},
		{
			name: "from parts: unsupported mode with other error",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionFromParts(42, "foo", otherErr),
				authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
					return 42, "foo", otherErr
				}).AuthorizeConditionsAware(ctx, sampleAttrs, samplePref),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "foo",
			wantErrorIs:         otherErr,
			wantString:          `Deny("foo", "[other error, unknown unconditional decision type: 42]")`,
		},
		{
			name: "construct valid conditionsmap",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					maps.All(map[string]authorizer.Condition{
						"foo": {
							Condition:   "ok",
							Effect:      authorizer.ConditionEffectAllow,
							Description: "foo",
						},
					}),
					"",
					nil,
				),
			},
			wantIsConditionsMap: true,
			wantIsUnconditional: false,
			wantReason:          "",
			wantString:          `ConditionsMap(target="AdmissionControl", type="foo-type", len=1, reason="", err=<nil>)`,
		},
		{
			name: "duplicate IDs",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					func(yield func(string, authorizer.Condition) bool) {
						cond1 := authorizer.Condition{
							Condition: "foo",
							Effect:    authorizer.ConditionEffectAllow,
						}
						cond2 := authorizer.Condition{
							Condition: "bar",
							Effect:    authorizer.ConditionEffectDeny,
						}
						if !yield("foo", cond1) {
							return
						}
						if !yield("foo", cond2) {
							return
						}
					},
					"",
					nil,
				),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `Deny("failed closed", "duplicate condition ID \"foo\"")`,
		},
		{
			name: "invalid effect",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					maps.All(map[string]authorizer.Condition{
						"foo": {
							Condition: "ok",
							Effect:    authorizer.ConditionEffect("nonexistent"),
						},
					}),
					"",
					nil,
				),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `Deny("failed closed", "condition effect \"nonexistent\" not supported. Supported effects are: [Allow Deny NoOpinion]")`,
		},
		{
			name: "empty condition invalid, one condition error is enough to fail closed",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					maps.All(map[string]authorizer.Condition{
						"foo": {
							Effect: authorizer.ConditionEffectAllow,
						},
						"deny": {
							Condition: "ok",
							Effect:    authorizer.ConditionEffectDeny,
						},
					}),
					"",
					nil,
				),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `Deny("failed closed", "condition \"foo\" has empty Condition string")`,
		},
		{
			name: "condition ID must be a Kubernetes label",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					maps.All(map[string]authorizer.Condition{
						"not a kubernetes label": {
							Condition: "ok",
							Effect:    authorizer.ConditionEffectDeny,
						},
					}),
					"",
					nil,
				),
			},
			wantIsDenied:        true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `Deny("failed closed", "invalid condition ID \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "condition type must be a Kubernetes label",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("not a kubernetes label"),
					maps.All(map[string]authorizer.Condition{
						"ok": {
							Condition: "ok",
							Effect:    authorizer.ConditionEffectNoOpinion,
						},
					}),
					"",
					nil,
				),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `NoOpinion("failed closed", "invalid condition type \"not a kubernetes label\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")`,
		},
		{
			name: "condition target must be supported",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTarget("not supported"),
					authorizer.ConditionType("foo"),
					maps.All(map[string]authorizer.Condition{
						"ok": {
							Condition: "ok",
							Effect:    authorizer.ConditionEffectNoOpinion,
						},
					}),
					"",
					nil,
				),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "failed closed",
			wantAnyError:        true,
			wantString:          `NoOpinion("failed closed", "conditions target \"not supported\" not supported. Supported targets are: [AdmissionControl]")`,
		},
		{
			name: "empty ConditionsMap is NoOpinion",
			testDecisions: []authorizer.ConditionsAwareDecision{
				authorizer.ConditionsAwareDecisionConditionMap(
					authorizer.ConditionsTargetAdmissionControl,
					authorizer.ConditionType("foo-type"),
					maps.All(map[string]authorizer.Condition{}),
					"ignored",
					otherErr,
				),
			},
			wantIsNoOpinion:     true,
			wantIsUnconditional: true,
			wantReason:          "empty ConditionsMap",
			wantErrorIs:         otherErr,
			wantString:          `NoOpinion("empty ConditionsMap", "other error")`,
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
					isUnconditional := d.IsUnconditional()
					if isUnconditional != tt.wantIsUnconditional {
						t.Errorf("IsUnconditional() = %v, want %v", isUnconditional, tt.wantIsUnconditional)
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

func TestCreateConditionsMapFeatureDisabled(t *testing.T) {
	// Feature gate is disabled (which is the default) in this test
	// Fail closed to NoOpinion, as there are no denies
	d := authorizer.ConditionsAwareDecisionConditionMap(
		authorizer.ConditionsTargetAdmissionControl,
		authorizer.ConditionType("foo-type"),
		maps.All(map[string]authorizer.Condition{
			"foo": {
				Condition:   "ok",
				Effect:      authorizer.ConditionEffectAllow,
				Description: "foo",
			},
		}),
		"",
		nil,
	)
	if !d.IsNoOpinion() {
		t.Error("Expected creating a ConditionsMap decision to yield NoOpinion when the feature gate is disabled")
	}
	if !strings.Contains(d.Error().Error(), "ConditionalAuthorization feature gate is disabled") {
		t.Error("Expected error to tell about feature gate being disabled")
	}
	// Fail closed to Deny, as there is at least one Deny condition
	d = authorizer.ConditionsAwareDecisionConditionMap(
		authorizer.ConditionsTargetAdmissionControl,
		authorizer.ConditionType("foo-type"),
		maps.All(map[string]authorizer.Condition{
			"foo": {
				Condition:   "ok",
				Effect:      authorizer.ConditionEffectDeny,
				Description: "foo",
			},
		}),
		"",
		nil,
	)
	if !d.IsDenied() {
		t.Error("Expected creating a ConditionsMap decision to yield NoOpinion when the feature gate is disabled")
	}
	if !strings.Contains(d.Error().Error(), "ConditionalAuthorization feature gate is disabled") {
		t.Error("Expected error to tell about feature gate being disabled")
	}
}
