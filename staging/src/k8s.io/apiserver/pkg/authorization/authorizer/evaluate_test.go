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
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestPartiallyEvaluateConditionsAwareDecision(t *testing.T) {
	// mkCM builds a ConditionsMap decision from individually-tagged (effect, condition) pairs,
	// preserving the inline readability of the original test cases (which carried .Effect on
	// each GenericCondition).
	mkCM := func(items ...effectCondition) authorizer.ConditionsAwareDecision {
		var deny, nop, allow []authorizer.Condition
		for _, it := range items {
			switch it.effect {
			case effectAllow:
				allow = append(allow, it.cond)
			case effectDeny:
				deny = append(deny, it.cond)
			case effectNoOpinion:
				nop = append(nop, it.cond)
			}
		}
		return authorizer.ConditionsAwareDecisionConditionsMap(deny, nop, allow)
	}

	// genericCond is a shorthand for an authorizer.GenericCondition. Description is optional.
	cnd := func(effect conditionEffect, id, condition, typ, description string) effectCondition {
		return effectCondition{
			effect: effect,
			cond: authorizer.GenericCondition{
				ID: id, Condition: condition, Type: typ, Description: description,
			},
		}
	}

	type testCase struct {
		name string

		// decision is the input passed to PartiallyEvaluateConditionsAwareDecision.
		decision authorizer.ConditionsAwareDecision

		// noACRReviewer: in the original test suite this flag meant "no webhook required for
		// this case because the partial evaluator simplifies fully to an unconditional decision".
		// Here it means: the partial result must be Unconditional (Allow/Deny/NoOpinion), and
		// only wantDecision / wantReason are checked.
		noACRReviewer bool

		// builtinConditionsEvaluator is the PartialEvaluateConditionFunc supplied to the partial
		// evaluator. Returning Unevaluatable leaves the condition in a refined ConditionsMap.
		builtinConditionsEvaluator authorizer.PartialEvaluateConditionFunc

		wantDecision authorizer.Decision
		wantReason   string

		// verifyPartial is the replacement for verifyACR. When the partial result is still
		// conditional (noACRReviewer == false), it asserts the shape of the returned
		// ConditionsAwareDecision tree.
		verifyPartial func(t *testing.T, partial authorizer.ConditionsAwareDecision)
	}

	tests := []testCase{
		{
			name: "full builtin evaluation of one ConditionsMap => Deny",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			noACRReviewer: true,
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			},
			wantDecision: authorizer.DecisionDeny,
			wantReason:   `condition "d" denied the request with description "very bad"`,
		},
		{
			name: "full builtin evaluation of one ConditionsMap => NoOpinion",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			noACRReviewer: true,
			builtinConditionsEvaluator: func(_ context.Context, _ authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(false)
			},
			wantDecision: authorizer.DecisionNoOpinion,
			wantReason:   `no conditions matched`,
		},
		{
			name: "full builtin evaluation of one ConditionsMap => Allow",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			noACRReviewer: true,
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			},
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `condition "c" allowed the request with description "all ok"`,
		},
		{
			// The opaque allow condition cannot be evaluated in-process, so the partial result
			// is a refined ConditionsMap containing only that condition. (Previously a webhook
			// was consulted to finish the evaluation; here we just verify the partial tree.)
			name: "partial builtin evaluation of one ConditionsMap => refined ConditionsMap",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "opaque", "all ok"),       // needs a webhook due to opaque type
				cnd(effectDeny, "d", "d", "transparent", "very bad"), // simplified in-process
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			verifyPartial: assertDecisionTree(snapDecision{
				Kind: "ConditionsMap",
				CM: &snapCM{
					Allow: []snapCondition{
						{ID: "c", Condition: "c", Type: "opaque", Description: "all ok"},
					},
				},
			}),
		},
		{
			name: "builtin evaluation of union succeeds => Allow",
			decision: unionDecision(
				mkCM(
					cnd(effectAllow, "a", "a", "transparent", ""),
					cnd(effectDeny, "b", "b", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "transparent", ""),
				),
			),
			noACRReviewer: true,
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			},
			wantDecision: authorizer.DecisionAllow,
			wantReason:   `condition "c" allowed the request`,
		},
		{
			name: "builtin evaluation of union succeeds => Deny",
			decision: unionDecision(
				mkCM(
					cnd(effectAllow, "a", "a", "transparent", ""),
					cnd(effectDeny, "b", "b", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "transparent", ""),
				),
			),
			noACRReviewer: true,
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			},
			wantDecision: authorizer.DecisionDeny,
			wantReason:   `condition "d" denied the request`,
		},
		{
			// First CM has an opaque allow condition that cannot be simplified, so the union
			// short-circuits to "collect remaining sub-decisions as-is" after that point. The
			// second CM is preserved unchanged (it never gets a chance to be evaluated).
			name: "first conditionsmap cannot be simplified fully",
			decision: unionDecision(
				mkCM(
					cnd(effectAllow, "a", "a", "opaque", ""),
					cnd(effectDeny, "b", "b", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "transparent", ""),
				),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			verifyPartial: assertDecisionTree(snapDecision{
				Kind: "Union",
				Union: []snapDecision{
					{
						Kind: "ConditionsMap",
						CM: &snapCM{
							Allow: []snapCondition{{ID: "a", Condition: "a", Type: "opaque"}},
						},
					},
					{
						Kind: "ConditionsMap",
						CM: &snapCM{
							Deny:  []snapCondition{{ID: "d", Condition: "d", Type: "transparent"}},
							Allow: []snapCondition{{ID: "c", Condition: "c", Type: "transparent"}},
						},
					},
				},
			}),
		},
		{
			// First CM simplifies fully to NoOpinion (none of its transparent conditions match).
			// Second CM has an opaque deny condition, so it cannot be simplified and stays a
			// (refined) ConditionsMap. Third entry is an unconditional Deny that survives as-is.
			name: "first conditionsmap can be simplified fully, but not second",
			decision: unionDecision(
				mkCM(
					cnd(effectAllow, "a", "a", "transparent", ""),
					cnd(effectDeny, "b", "b", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "opaque", ""),
				),
				authorizer.ConditionsAwareDecisionDeny("something later denies", nil),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.PartialConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			verifyPartial: assertDecisionTree(snapDecision{
				Kind: "Union",
				Union: []snapDecision{
					{Kind: "NoOpinion", Reason: "no conditions matched"},
					{
						Kind: "ConditionsMap",
						CM: &snapCM{
							Deny:  []snapCondition{{ID: "d", Condition: "d", Type: "opaque"}},
							Allow: []snapCondition{{ID: "c", Condition: "c", Type: "transparent"}},
						},
					},
					{Kind: "Deny", Reason: "something later denies"},
				},
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := authorizer.PartiallyEvaluateConditionsAwareDecision(
				t.Context(),
				tt.decision,
				authorizer.ConditionsData{},
				tt.builtinConditionsEvaluator,
			)

			if tt.noACRReviewer {
				if !got.IsUnconditional() {
					t.Fatalf("expected unconditional decision, got %s", got.String())
				}
				gotDecision, gotReason, _ := unconditionalParts(got)
				if gotDecision != tt.wantDecision {
					t.Errorf("decision = %v, want %v", gotDecision, tt.wantDecision)
				}
				if gotReason != tt.wantReason {
					t.Errorf("reason = %q, want %q", gotReason, tt.wantReason)
				}
				return
			}
			if tt.verifyPartial == nil {
				t.Fatalf("test case %q must set either noACRReviewer or verifyPartial", tt.name)
			}
			tt.verifyPartial(t, got)
		})
	}
}

// conditionEffect mirrors the deny/noOpinion/allow categorization the old
// GenericCondition.Effect field encoded, but lives entirely in the test rather than the API.
type conditionEffect int

const (
	effectAllow conditionEffect = iota
	effectDeny
	effectNoOpinion
)

// effectCondition pairs a Condition with the effect slice it should be placed into.
type effectCondition struct {
	effect conditionEffect
	cond   authorizer.Condition
}

// snapDecision is a deep-comparable snapshot of an authorizer.ConditionsAwareDecision tree,
// used in lieu of comparing against the (now-removed) authorizationv1alpha1 wire types.
type snapDecision struct {
	Kind   string // "Allow" | "Deny" | "NoOpinion" | "ConditionsMap" | "Union"
	Reason string
	CM     *snapCM
	Union  []snapDecision
}

type snapCM struct {
	Deny      []snapCondition
	NoOpinion []snapCondition
	Allow     []snapCondition
}

type snapCondition struct {
	ID          string
	Condition   string
	Type        string
	Description string
}

func snapshotDecision(d authorizer.ConditionsAwareDecision) snapDecision {
	switch {
	case d.IsAllow():
		return snapDecision{Kind: "Allow", Reason: d.Reason()}
	case d.IsDeny():
		return snapDecision{Kind: "Deny", Reason: d.Reason()}
	case d.IsNoOpinion():
		return snapDecision{Kind: "NoOpinion", Reason: d.Reason()}
	case d.IsConditionsMap():
		return snapDecision{Kind: "ConditionsMap", CM: snapshotConditionsMap(d.ConditionsMap())}
	case d.IsUnion():
		var subs []snapDecision
		for _, sub := range d.UnionedDecisions() {
			subs = append(subs, snapshotDecision(sub))
		}
		return snapDecision{Kind: "Union", Union: subs}
	}
	return snapDecision{Kind: "Unknown"}
}

func snapshotConditionsMap(cm authorizer.ConditionsMap) *snapCM {
	out := &snapCM{}
	for c := range cm.DenyConditions() {
		out.Deny = append(out.Deny, snapshotCondition(c))
	}
	for c := range cm.NoOpinionConditions() {
		out.NoOpinion = append(out.NoOpinion, snapshotCondition(c))
	}
	for c := range cm.AllowConditions() {
		out.Allow = append(out.Allow, snapshotCondition(c))
	}
	return out
}

func snapshotCondition(c authorizer.Condition) snapCondition {
	return snapCondition{
		ID:          c.GetID(),
		Condition:   c.GetCondition(),
		Type:        c.GetType(),
		Description: c.GetDescription(),
	}
}

// assertDecisionTree returns a verifyPartial that snapshots the actual partial decision and
// compares it against want using cmp.Diff. This is the modern equivalent of the verifyACR
// helper that used cmp.Diff on the (now-removed) authorizationv1alpha1 wire decision tree.
func assertDecisionTree(want snapDecision) func(t *testing.T, got authorizer.ConditionsAwareDecision) {
	return func(t *testing.T, got authorizer.ConditionsAwareDecision) {
		t.Helper()
		if diff := cmp.Diff(want, snapshotDecision(got)); diff != "" {
			t.Errorf("partial decision mismatch (-want +got):\n%s", diff)
		}
	}
}
