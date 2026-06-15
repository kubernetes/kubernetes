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
	"reflect"
	"slices"
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

func (a sampleAuthorizer) ConditionalAuthorizerName() string {
	return "sampleauthorizer.example.com"
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
					finalDecision:     [2]string{`Deny(reason="failed closed: tried to return conditional decision to conditions-unaware authorizer")`, `Deny(reason="condition \"deny-supersecret-label-on-oldObject\" denied the request, condition \"deny-supersecret-label-on-object\" denied the request")`},
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
						data := authorizer.ConditionsData{
							AdmissionControl: admission.NewAttributesRecord(tc.object, tc.oldObject, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil),
						}

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

	if data.AdmissionControl == nil {
		return conditionsMap.FailureDecision(), "failed closed", errors.New("evaluating a CEL condition requires non-nil data.AdmissionControl")
	}

	obj, err := objectToResolveVal(data.AdmissionControl.GetObject())
	if err != nil {
		return conditionsMap.FailureDecision(), "failed closed", fmt.Errorf("failed to convert object to CEL ref.Val: %w", err)
	}

	oldObj, err := objectToResolveVal(data.AdmissionControl.GetOldObject())
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
