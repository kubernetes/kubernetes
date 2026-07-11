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
	// mkCM builds a ConditionsMap decision from individually-tagged (effect, condition) pairs.
	mkCM := func(items ...effectCondition) authorizer.ConditionsAwareDecision {
		var deny, nop, allow []authorizer.Condition
		for _, it := range items {
			switch it.effect {
			case effectDeny:
				deny = append(deny, it.cond)
			case effectNoOpinion:
				nop = append(nop, it.cond)
			case effectAllow:
				allow = append(allow, it.cond)
			}
		}
		return authorizer.ConditionsAwareDecisionConditionsMap(deny, nop, allow)
	}

	// cnd is a shorthand for a labeled authorizer.GenericCondition. Description is optional.
	cnd := func(effect conditionEffect, id, condition, typ, description string) effectCondition {
		return effectCondition{
			effect: effect,
			cond: authorizer.GenericCondition{
				ID: id, Condition: condition, Type: typ, Description: description,
			},
		}
	}

	tests := []struct {
		name string

		// decision is the input passed to PartiallyEvaluateConditionsAwareDecision.
		decision authorizer.ConditionsAwareDecision

		// builtinConditionsEvaluator is the MaybeEvaluateConditionFunc supplied to the partial
		// evaluator. Returning Unevaluatable leaves the condition in a refined ConditionsMap.
		builtinConditionsEvaluator authorizer.MaybeEvaluateConditionFunc

		// want is the expected snapshot of the returned ConditionsAwareDecision.
		want snapDecision
	}{
		{
			name: "full builtin evaluation of one ConditionsMap => Deny",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			},
			want: snapDecision{Kind: "Deny", Reason: `condition "d" denied the request with description "very bad"`},
		},
		{
			name: "full builtin evaluation of one ConditionsMap => NoOpinion",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			builtinConditionsEvaluator: func(_ context.Context, _ authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(false)
			},
			want: snapDecision{Kind: "NoOpinion", Reason: "no conditions matched"},
		},
		{
			name: "full builtin evaluation of one ConditionsMap => Allow",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "transparent", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			},
			want: snapDecision{Kind: "Allow", Reason: `condition "c" allowed the request with description "all ok"`},
		},
		{
			// The opaque allow condition cannot be evaluated in-process, so the partial result
			// is a refined ConditionsMap containing only that condition.
			name: "partial builtin evaluation of one ConditionsMap => refined ConditionsMap",
			decision: mkCM(
				cnd(effectAllow, "c", "c", "opaque", "all ok"),
				cnd(effectDeny, "d", "d", "transparent", "very bad"),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			want: snapDecision{
				Kind: "ConditionsMap",
				CM: &snapCM{
					Allow: []snapCondition{
						{ID: "c", Condition: "c", Type: "opaque", Description: "all ok"},
					},
				},
			},
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
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
			},
			want: snapDecision{Kind: "Allow", Reason: `1.example.com: {condition "c" allowed the request}`},
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
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "d")
			},
			want: snapDecision{Kind: "Deny", Reason: `1.example.com: {condition "d" denied the request}`},
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
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			want: snapDecision{
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
			},
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
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "c")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			want: snapDecision{
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
			},
		},
		{
			name: "first allow conditionsmap => NoOpinion, noopinion, third conditional deny, finally Deny => Deny",
			decision: unionDecision(
				mkCM(
					cnd(effectAllow, "allow-false", "a", "transparent", ""),
				),
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				mkCM(
					cnd(effectDeny, "b", "b", "opaque", ""),
				),
				authorizer.ConditionsAwareDecisionDeny("something later denies", nil),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "allow-true")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			want: snapDecision{
				Kind:   "Deny",
				Reason: `3.example.com: {something later denies}`,
			},
		},
		{
			name: "first deny conditionsmap => NoOpinion, second conditional allow, noopinion, finally Allow => Allow",
			decision: unionDecision(
				mkCM(
					cnd(effectDeny, "deny-false", "a", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "b", "b", "opaque", ""),
				),
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				authorizer.ConditionsAwareDecisionAllow("something later allows", nil),
			),
			builtinConditionsEvaluator: func(_ context.Context, condition authorizer.Condition, _ authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
				if condition.GetType() == "transparent" {
					return authorizer.ConditionEvaluationResultBoolean(condition.GetCondition() == "deny-true")
				}
				return authorizer.ConditionsEvaluationResultUnevaluatable()
			},
			want: snapDecision{
				Kind:   "Allow",
				Reason: `3.example.com: {something later allows}`,
			},
		},
		{
			name: "evaluateConditionFn can be nil, and evaluation to concrete can still succeed",
			decision: unionDecision(
				mkCM(
					// This should take precedence and make the outcome Deny
					effectCondition{
						effect: effectDeny,
						cond: authorizer.GenericCondition{
							ID: "foo",
							EvaluateFunc: func(ctx context.Context, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
								return authorizer.ConditionEvaluationResultBoolean(true)
							},
						},
					},
					cnd(effectAllow, "c", "c", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "opaque", ""),
				),
				authorizer.ConditionsAwareDecisionDeny("something later denies", nil),
			),
			want: snapDecision{Kind: "Deny", Reason: `0.example.com: {condition "foo" denied the request}`},
		},
		{
			name: "evaluateConditionFn can be nil, and refinement can still happen",
			decision: unionDecision(
				mkCM(
					// This condition should be removed from the refined ConditionsAwareDecision
					effectCondition{
						effect: effectDeny,
						cond: authorizer.GenericCondition{
							ID: "foo",
							EvaluateFunc: func(ctx context.Context, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
								return authorizer.ConditionEvaluationResultBoolean(false)
							},
						},
					},
					cnd(effectAllow, "c", "c", "transparent", ""),
				),
				mkCM(
					cnd(effectAllow, "c", "c", "transparent", ""),
					cnd(effectDeny, "d", "d", "opaque", ""),
				),
				authorizer.ConditionsAwareDecisionDeny("something later denies", nil),
			),
			want: snapDecision{
				Kind: "Union",
				Union: []snapDecision{
					{
						Kind: "ConditionsMap",
						CM: &snapCM{
							Allow: []snapCondition{{ID: "c", Condition: "c", Type: "transparent"}},
						},
					},
					{
						Kind: "ConditionsMap",
						CM: &snapCM{
							Deny:  []snapCondition{{ID: "d", Condition: "d", Type: "opaque"}},
							Allow: []snapCondition{{ID: "c", Condition: "c", Type: "transparent"}},
						},
					},
					{Kind: "Deny", Reason: "something later denies"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := authorizer.PartiallyEvaluateConditionsAwareDecision(
				t.Context(),
				tt.decision,
				nil,
				tt.builtinConditionsEvaluator,
			)
			if diff := cmp.Diff(tt.want, snapshotDecision(got)); diff != "" {
				t.Errorf("decision mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// conditionEffect categorizes a Condition into the deny/noOpinion/allow slice of a ConditionsMap.
// The ordering matches the argument order of ConditionsAwareDecisionConditionsMap.
type conditionEffect int

const (
	effectDeny conditionEffect = iota
	effectNoOpinion
	effectAllow
)

// effectCondition pairs a Condition with the effect slice it should be placed into.
type effectCondition struct {
	effect conditionEffect
	cond   authorizer.Condition
}

// snapDecision is a deep-comparable snapshot of an authorizer.ConditionsAwareDecision tree.
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
