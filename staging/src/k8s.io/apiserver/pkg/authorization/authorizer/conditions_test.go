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

var _ authorizer.Authorizer = sampleAuthorizer{}

type sampleAuthorizer struct{}

func (a sampleAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionPartsFromConditionsAware(a.AuthorizeConditionsAware(ctx, attrs, authorizer.ConditionsEncodingPreferenceOptimized()))
}

func (a sampleAuthorizer) AuthorizeConditionsAware(ctx context.Context, attrs authorizer.Attributes, _ authorizer.ConditionsEncodingPreference) authorizer.ConditionsAwareDecision {
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
			return authorizer.ConditionsAwareDecisionConditionMap(authorizer.ConditionsTargetAdmissionControl, "test-cel-conditions-type", maps.All(map[string]authorizer.Condition{
				"owner-label-is-set": {
					Condition: `
						(oldObject != null ? (has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.owner) && oldObject.metadata.labels.owner == "carol") : true) &&
						(object != null ? (has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.owner) && object.metadata.labels.owner == "carol") : true)
					`,
					Effect: authorizer.ConditionEffectAllow,
				},
			}), "", nil)
		default:
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
	case "dave":
		// allow dave to read anything, but never set the classified label on writes
		switch attrs.GetVerb() {
		case "list":
			return authorizer.ConditionsAwareDecisionAllow("", nil)
		case "create", "update", "delete":
			return authorizer.ConditionsAwareDecisionConditionMap(authorizer.ConditionsTargetAdmissionControl, "test-cel-conditions-type", maps.All(map[string]authorizer.Condition{
				"deny-supersecret-label-on-oldObject": {
					Condition: "oldObject != null && has(oldObject.metadata) && has(oldObject.metadata.labels) && has(oldObject.metadata.labels.supersecret)",
					Effect:    authorizer.ConditionEffectDeny,
				},
				"deny-supersecret-label-on-object": {
					Condition: "object != null && has(object.metadata) && has(object.metadata.labels) && has(object.metadata.labels.supersecret)",
					Effect:    authorizer.ConditionEffectDeny,
				},
			}), "", nil)
		default:
			return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		}
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}
}

func (a sampleAuthorizer) EvaluateConditions(ctx context.Context, unevaluated authorizer.ConditionsAwareDecision, data authorizer.ConditionsData, builtin authorizer.BuiltinConditionsMapEvaluators) authorizer.ConditionsAwareDecision {
	if unevaluated.IsUnconditional() {
		return unevaluated
	}
	/* TODO
	if unevaluated.IsConditionalChain() {
		return unevaluated.FailClosedDecision(), errors.New("conditionschain unsupported")
	}*/
	if data.AdmissionControl() == nil {
		return unevaluated.FailClosedDecision(errors.New("only supports conditions for write requests"))
	}

	return celEvaluateConditions(unevaluated.ConditionsMap(), data.AdmissionControl())
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
		// the first case is with ConditionsModeNone, the second with ConditionsModeHumanReadable
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
				{name: "allow", authorizeDecision: [2]string{`Allow("", <nil>)`, `Allow("", <nil>)`}},
			},
		},
		{
			name: "alice create",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "alice"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{`Allow("", <nil>)`, `Allow("", <nil>)`}},
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
				{name: "deny", authorizeDecision: [2]string{`Deny("", <nil>)`, `Deny("", <nil>)`}},
			},
		},
		{
			name: "bob create",
			attrs: authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "bob"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "deny", authorizeDecision: [2]string{`Deny("", <nil>)`, `Deny("", <nil>)`}},
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
				{name: "allow", authorizeDecision: [2]string{`Allow("", <nil>)`, `Allow("", <nil>)`}},
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
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=1, reason="", err=<nil>)`,
					},
					finalDecision: [2]string{
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`Allow("[condition \"owner-label-is-set\" allowed the request]", <nil>)`,
					},
				},
				{
					name:      "old with owner=carol, new without",
					object:    objWithLabels(map[string]string{"owner": "carol"}),
					oldObject: objWithLabels(nil),
					authorizeDecision: [2]string{
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=1, reason="", err=<nil>)`,
					},
					finalDecision: [2]string{
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`NoOpinion("no conditions matched", <nil>)`,
					},
				},
				{
					name:      "new with owner=carol, old with owner=alice",
					object:    objWithLabels(map[string]string{"owner": "alice"}),
					oldObject: objWithLabels(map[string]string{"owner": "carol"}),
					authorizeDecision: [2]string{
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=1, reason="", err=<nil>)`,
					},
					finalDecision: [2]string{
						`NoOpinion("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`,
						`NoOpinion("no conditions matched", <nil>)`,
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
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion("", <nil>)`, `NoOpinion("", <nil>)`}},
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
				{name: "allow", authorizeDecision: [2]string{`Allow("", <nil>)`, `Allow("", <nil>)`}},
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
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `Deny("[condition \"deny-supersecret-label-on-object\" denied the request condition \"deny-supersecret-label-on-oldObject\" denied the request]", <nil>)`},
				},
				{
					name:              "new with supersecret old without",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(nil),
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `Deny("[condition \"deny-supersecret-label-on-object\" denied the request]", <nil>)`},
				},
				{
					name:              "new without old with supersecret",
					object:            objWithLabels(nil),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `Deny("[condition \"deny-supersecret-label-on-oldObject\" denied the request]", <nil>)`},
				},
				{
					name:              "both without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion("no conditions matched", <nil>)`},
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
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `Deny("[condition \"deny-supersecret-label-on-object\" denied the request]", <nil>)`},
				},
				{
					name:              "create without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion("no conditions matched", <nil>)`},
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
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `Deny("[condition \"deny-supersecret-label-on-oldObject\" denied the request]", <nil>)`},
				},
				{
					name:              "delete without supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `ConditionsMap(target="AdmissionControl", type="test-cel-conditions-type", len=2, reason="", err=<nil>)`},
					finalDecision:     [2]string{`Deny("failed closed", "tried to return conditional decision to conditions-unaware authorizer")`, `NoOpinion("no conditions matched", <nil>)`},
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
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion("", <nil>)`, `NoOpinion("", <nil>)`}},
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
				{name: "no opinion", authorizeDecision: [2]string{`NoOpinion("", <nil>)`, `NoOpinion("", <nil>)`}},
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
							decision = authz.AuthorizeConditionsAware(t.Context(), tt.attrs, authorizer.ConditionsEncodingPreferenceOptimized())
						} else {
							decision = authorizer.ConditionsAwareDecisionFromParts(authz.Authorize(t.Context(), tt.attrs))
						}

						if decision.String() != tc.authorizeDecision[i] {
							t.Errorf("got Authorize() decision %s, want %s", decision.String(), tc.authorizeDecision[i])
						}

						// Only object and oldObject is used in celEvaluateConditions, so let all other values be zero here, as they are anyways unused.
						data := authorizer.MakeConditionsDataAdmissionControl(admission.NewAttributesRecord(tc.object, tc.oldObject, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil))

						final := authz.EvaluateConditions(t.Context(), decision, data, nil)
						if final.String() != tc.finalDecision[i] {
							t.Errorf("got Evaluate() decision %s, want %s", final.String(), tc.finalDecision[i])
						}
					})
				}
			}
		})
	}
}

func celEvaluateConditions(conditionsMap authorizer.ConditionsMap, admissionData authorizer.ConditionsDataAdmissionControl) authorizer.ConditionsAwareDecision {
	env, err := cel.NewEnv(
		cel.Variable("object", cel.DynType),
		cel.Variable("oldObject", cel.DynType),
	)
	if err != nil {
		return conditionsMap.FailClosedDecision(fmt.Errorf("failed to create CEL env: %v", err))
	}

	obj, err := objectToResolveVal(admissionData.GetObject())
	if err != nil {
		return conditionsMap.FailClosedDecision(fmt.Errorf("failed to convert object to CEL ref.Val: %v", err))
	}

	oldObj, err := objectToResolveVal(admissionData.GetOldObject())
	if err != nil {
		return conditionsMap.FailClosedDecision(fmt.Errorf("failed to convert object to CEL ref.Val: %v", err))
	}

	vars := map[string]any{
		"object":    obj,
		"oldObject": oldObj,
	}

	evaluated, _ := authorizer.EvaluateConditionsMap(conditionsMap, authorizer.ConditionsTargetAdmissionControl, "test-cel-conditions-type", func(expr string) (bool, error) {
		return evalCEL(env, expr, vars)
	})
	return evaluated
}

// evalCEL compiles and evaluates a single CEL expression, returning true/false.
func evalCEL(env *cel.Env, expr string, vars map[string]any) (bool, error) {
	ast, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return false, fmt.Errorf("CEL compile error for %q: %v", expr, issues.Err())
	}
	prg, err := env.Program(ast)
	if err != nil {
		return false, fmt.Errorf("CEL program error for %q: %v", expr, err)
	}
	out, _, err := prg.Eval(vars)
	if err != nil {
		return false, fmt.Errorf("CEL eval error for %q: %v", expr, err)
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
