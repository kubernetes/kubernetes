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

package authorizer

import (
	"context"
	"errors"
	"testing"
)

func TestConditionsMapPartiallyEvaluate(t *testing.T) {
	evalErr := errors.New("eval error")

	trueResult := ConditionEvaluationResultBoolean(true)
	falseResult := ConditionEvaluationResultBoolean(false)
	errResult := ConditionEvaluationResultError(evalErr)

	cond := func(id string, result ConditionEvaluationResult) GenericCondition {
		return GenericCondition{
			ID: id,
			EvaluateFunc: func(context.Context, ConditionsData) ConditionEvaluationResult {
				return result
			},
		}
	}
	condDesc := func(id string, desc string, result ConditionEvaluationResult) GenericCondition {
		c := cond(id, result)
		c.Description = desc
		return c
	}
	unevalCond := func(id string) GenericCondition {
		return GenericCondition{ID: id} // nil EvaluateFunc → unevaluatable
	}

	type subCase struct {
		name                         string
		denyConditions               []Condition
		noOpinionConditions          []Condition
		allowConditions              []Condition
		evaluateFunc                 func(context.Context, Condition, ConditionsData) ConditionEvaluationResult
		disableConditionsMapEvaluate bool
	}

	tests := []struct {
		name     string
		subCases []subCase
		// All sub-cases must produce a decision whose String() equals wantString.
		wantString string
	}{
		// ============================================================
		// Deny: at least one deny condition matched
		// ============================================================
		{
			name:       "deny: at least one deny condition matched",
			wantString: `Deny(reason="condition \"deny-1\" denied the request")`,
			subCases: []subCase{
				{
					name:           "minimal",
					denyConditions: []Condition{cond("deny-1", trueResult)},
				},
				{
					name:                         "matching deny trumps any other case",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-no", falseResult),
						unevalCond("deny-uneval"),
						cond("deny-err", errResult),
						cond("deny-1", trueResult),
					},
					noOpinionConditions: []Condition{
						cond("nop-yes", trueResult),
						cond("nop-err", errResult),
						cond("nop-no", falseResult),
						unevalCond("nop-uneval"),
					},
					allowConditions: []Condition{
						cond("allow-yes", trueResult),
						cond("allow-no", falseResult),
						cond("allow-err", errResult),
						unevalCond("allow-uneval"),
					},
				},
				{
					name: "with erroring deny (error ignored due to match)",
					denyConditions: []Condition{
						cond("deny-1", trueResult),
						cond("deny-err", errResult),
					},
				},
				{
					name:                         "with unevaluatable deny (ignored due to match)",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-1", trueResult),
						unevalCond("deny-uneval"),
					},
				},
				{
					name:                         "with false+error+unevaluatable deny (all ignored due to match)",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-1", trueResult),
						cond("deny-2", falseResult),
						cond("deny-err", errResult),
						unevalCond("deny-uneval"),
					},
				},
				{
					name:                "deny match takes precedence over matching nop and allow; only fast conditions-evaluation",
					denyConditions:      []Condition{cond("deny-1", trueResult)},
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						panic("should never be called, as all conditions could readily be evaluated")
					},
				},
				{
					name:                "deny match with false nop and allow",
					denyConditions:      []Condition{cond("deny-1", trueResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions:     []Condition{cond("allow-1", falseResult)},
				},
				{
					name:                "deny match with unevaluatable nop and allow",
					denyConditions:      []Condition{cond("deny-1", trueResult)},
					noOpinionConditions: []Condition{unevalCond("nop-1")},
					allowConditions:     []Condition{unevalCond("allow-1")},
				},
				{
					name:                "deny match with erroring nop and allow",
					denyConditions:      []Condition{cond("deny-1", trueResult)},
					noOpinionConditions: []Condition{cond("nop-1", errResult)},
					allowConditions:     []Condition{cond("allow-1", errResult)},
				},
				{
					name:           "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns true)",
					denyConditions: []Condition{unevalCond("deny-1")},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						return ConditionEvaluationResultBoolean(true)
					},
				},
			},
		},
		{
			name:       "deny: at least one deny condition matched with description",
			wantString: `Deny(reason="condition \"deny-1\" denied the request with description \"access denied\"")`,
			subCases: []subCase{
				{
					name:           "minimal",
					denyConditions: []Condition{condDesc("deny-1", "access denied", trueResult)},
				},
				{
					name:                "with false nop and allow",
					denyConditions:      []Condition{condDesc("deny-1", "access denied", trueResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions:     []Condition{cond("allow-1", falseResult)},
				},
			},
		},

		// ============================================================
		// Deny: error, fail closed
		// ============================================================
		{
			name:       "deny: error fail closed",
			wantString: `Deny(reason="one or more conditional evaluation errors occurred", err="condition \"deny-1\" with effect=Deny evaluated to an error: eval error")`,
			subCases: []subCase{
				{
					name:           "minimal",
					denyConditions: []Condition{cond("deny-1", errResult)},
				},
				{
					name: "with false deny",
					denyConditions: []Condition{
						cond("deny-1", errResult),
						cond("deny-2", falseResult),
					},
				},
				{
					name:                         "error takes precedence over unevaluatable deny",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-1", errResult),
						unevalCond("deny-uneval"),
					},
				},
				{
					name:                         "deny error trumps noopinion and allow of any form",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-no", falseResult),
						unevalCond("deny-uneval"),
						cond("deny-1", errResult),
					},
					noOpinionConditions: []Condition{
						cond("nop-yes", trueResult),
						cond("nop-err", errResult),
						cond("nop-no", falseResult),
						unevalCond("nop-uneval"),
					},
					allowConditions: []Condition{
						cond("allow-yes", trueResult),
						cond("allow-no", falseResult),
						cond("allow-err", errResult),
						unevalCond("allow-uneval"),
					},
				},
			},
		},
		{
			name:       "deny: error fail closed",
			wantString: `Deny(reason="one or more conditional evaluation errors occurred", err="[condition \"deny-1\" with effect=Deny evaluated to an error: eval error, condition \"deny-2\" with effect=Deny evaluated to an error: eval error]")`,
			subCases: []subCase{
				{
					name:           "minimal",
					denyConditions: []Condition{cond("deny-1", errResult), cond("deny-2", errResult)},
				},
			},
		},

		// ============================================================
		// NoOpinion: at least one noopinion condition matched
		// ============================================================
		{
			name:       "noopinion: at least one noopinion condition matched",
			wantString: `NoOpinion(reason="condition \"nop-1\" evaluated to NoOpinion")`,
			subCases: []subCase{
				{
					name:                "simple",
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
				},
				{
					name:                         "noopinion match trumps any noopinion or allow form",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-no", falseResult),
					},
					noOpinionConditions: []Condition{
						cond("nop-err", errResult),
						cond("nop-no", falseResult),
						unevalCond("nop-uneval"),
						cond("nop-1", trueResult),
					},
					allowConditions: []Condition{
						cond("allow-yes", trueResult),
						cond("allow-no", falseResult),
						cond("allow-err", errResult),
						unevalCond("allow-uneval"),
					},
				},
				{
					name: "erroring nop (error ignored due to match)",
					noOpinionConditions: []Condition{
						cond("nop-1", trueResult),
						cond("nop-err", errResult),
					},
				},
				{
					name:                         "unevaluatable nop (ignored due to match)",
					disableConditionsMapEvaluate: true,
					noOpinionConditions: []Condition{
						cond("nop-1", trueResult),
						unevalCond("nop-uneval"),
					},
				},
				{
					name:                         "false+error+unevaluatable nop (all ignored due to match)",
					disableConditionsMapEvaluate: true,
					noOpinionConditions: []Condition{
						cond("nop-1", trueResult),
						cond("nop-2", falseResult),
						cond("nop-err", errResult),
						unevalCond("nop-uneval"),
					},
				},
				{
					name:                "nop match takes precedence over matching allow",
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
				{
					name:                "with false deny, nop matches",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
				{
					name:                "nop match with unevaluatable allow",
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{unevalCond("allow-1")},
				},
				{
					name:                "nop match with erroring allow",
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{cond("allow-1", errResult)},
				},
			},
		},
		{
			name:       "noopinion: at least one noopinion condition matched with description",
			wantString: `NoOpinion(reason="condition \"nop-1\" evaluated to NoOpinion with description \"not relevant\"")`,
			subCases: []subCase{
				{
					name:                "simple",
					noOpinionConditions: []Condition{condDesc("nop-1", "not relevant", trueResult)},
				},
				{
					name:                "with false deny and allow",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{condDesc("nop-1", "not relevant", trueResult)},
					allowConditions:     []Condition{cond("allow-1", falseResult)},
				},
			},
		},

		// ============================================================
		// NoOpinion: error, fail closed (from nop)
		// ============================================================
		{
			name:       "noopinion: nop error fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="condition \"nop-1\" with effect=NoOpinion evaluated to an error: eval error")`,
			subCases: []subCase{
				{
					name:                "simple",
					noOpinionConditions: []Condition{cond("nop-1", errResult)},
				},
				{
					name:                         "noopinion error trumps noopinion unevaluated and any other allow",
					disableConditionsMapEvaluate: true,
					noOpinionConditions: []Condition{
						cond("nop-no", falseResult),
						unevalCond("nop-uneval"),
						cond("nop-1", errResult),
					},
					allowConditions: []Condition{
						cond("allow-yes", trueResult),
						cond("allow-no", falseResult),
						cond("allow-err", errResult),
						unevalCond("allow-uneval"),
					},
					denyConditions: []Condition{cond("deny-no", falseResult)},
				},
				{
					name:                "nop error trumps matching allow",
					noOpinionConditions: []Condition{cond("nop-1", errResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
				{
					name:                "with false deny, nop error, matching allow",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", errResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
			},
		},

		// ============================================================
		// NoOpinion: error, fail closed (from allow)
		// ============================================================
		{
			name:       "noopinion: single allow error fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="condition \"allow-1\" with effect=Allow evaluated to an error: eval error")`,
			subCases: []subCase{
				{
					name:            "minimal",
					allowConditions: []Condition{cond("allow-1", errResult)},
				},
				{
					name:                         "with false deny and nop",
					disableConditionsMapEvaluate: true,
					denyConditions:               []Condition{cond("deny-no", falseResult)},
					noOpinionConditions:          []Condition{cond("nop-no", falseResult)},
					allowConditions: []Condition{
						cond("allow-no", falseResult),
						unevalCond("allow-uneval"),
						cond("allow-1", errResult),
					},
				},
				{
					name:            "via evaluateFunc fallback (condition unevaluatable, evaluateFunc errors)",
					allowConditions: []Condition{unevalCond("allow-1")},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						return ConditionEvaluationResultError(evalErr)
					},
				},
				{
					name:            "condition errors, evaluateFunc panics (not called)",
					allowConditions: []Condition{cond("allow-1", errResult)},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						panic("should not be called")
					},
				},
			},
		},
		{
			name:       "noopinion: multiple allow errors fail closed",
			wantString: `NoOpinion(reason="one or more conditional evaluation errors occurred", err="[condition \"allow-1\" with effect=Allow evaluated to an error: eval error, condition \"allow-2\" with effect=Allow evaluated to an error: eval error]")`,
			subCases: []subCase{
				{
					name: "minimal",
					allowConditions: []Condition{
						cond("allow-1", errResult),
						cond("allow-2", errResult),
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
					name:           "single deny false",
					denyConditions: []Condition{cond("deny-1", falseResult)},
				},
				{
					name:                "single nop false",
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
				},
				{
					name:            "single allow false",
					allowConditions: []Condition{cond("allow-1", falseResult)},
				},
				{
					name:                "all effects false",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions:     []Condition{cond("allow-1", falseResult)},
				},
				{
					name:            "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns false)",
					allowConditions: []Condition{unevalCond("allow-1")},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						return ConditionEvaluationResultBoolean(false)
					},
				},
			},
		},

		// ============================================================
		// NoOpinion: unevaluatable nop with no allow conditions
		// ============================================================
		{
			name:       "noopinion: unevaluatable nop, no allow -> NoOpinion",
			wantString: `NoOpinion(reason="only NoOpinion conditions always evaluate to NoOpinion")`,
			subCases: []subCase{
				{
					name:                         "false deny and unevaluatable noopinion folds to noopinion",
					disableConditionsMapEvaluate: true,
					denyConditions:               []Condition{cond("deny-1", falseResult)},
					noOpinionConditions:          []Condition{unevalCond("nop-1")},
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
					name:            "minimal",
					allowConditions: []Condition{cond("allow-1", trueResult)},
				},
				{
					name: "with false allow",
					allowConditions: []Condition{
						cond("allow-no", falseResult),
						cond("allow-1", trueResult),
					},
				},
				{
					name:                "with false deny and nop",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
				{
					name:            "evaluateFunc panics (not called, condition self-evaluates)",
					allowConditions: []Condition{cond("allow-1", trueResult)},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						panic("should not be called")
					},
				},
				{
					name:            "via evaluateFunc fallback (condition unevaluatable, evaluateFunc returns true)",
					allowConditions: []Condition{unevalCond("allow-1")},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						return ConditionEvaluationResultBoolean(true)
					},
				},
			},
		},
		{
			name:       "allow: at least one allow condition matched with description",
			wantString: `Allow(reason="condition \"allow-1\" allowed the request with description \"access granted\"")`,
			subCases: []subCase{
				{
					name:            "minimal",
					allowConditions: []Condition{condDesc("allow-1", "access granted", trueResult)},
				},
				{
					name:                "with false deny and nop",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions:     []Condition{condDesc("allow-1", "access granted", trueResult)},
				},
			},
		},

		// ============================================================
		// Allow: matching allow short-circuits and drops any prior errors
		// ============================================================
		{
			// Iteration is: allow-err (accumulated as an error), then allow-1 (true) which
			// short-circuits. The short-circuit path returns the reason with nil errors,
			// so the accumulated allow-err warning is dropped and not surfaced on Allow.
			name:       "allow: matching allow short-circuits and drops prior error",
			wantString: `Allow(reason="condition \"allow-1\" allowed the request")`,
			subCases: []subCase{
				{
					name: "minimal",
					allowConditions: []Condition{
						cond("allow-err", errResult),
						cond("allow-1", trueResult),
					},
				},
				{
					name:                "with false deny and nop",
					denyConditions:      []Condition{cond("deny-1", falseResult)},
					noOpinionConditions: []Condition{cond("nop-1", falseResult)},
					allowConditions: []Condition{
						cond("allow-err", errResult),
						cond("allow-1", trueResult),
					},
				},
			},
		},

		// ============================================================
		// ConditionsMap: refined map with unevaluatable conditions
		// ============================================================
		{
			name:       "conditionsmap: deny unevaluatable, nop and allow present",
			wantString: `ConditionsMap(denies=1, noopinions=1, allows=1)`,
			subCases: []subCase{
				{
					name:                         "minimal",
					disableConditionsMapEvaluate: true,
					denyConditions:               []Condition{unevalCond("deny-1")},
					noOpinionConditions:          []Condition{cond("nop-1", trueResult)},
					allowConditions:              []Condition{cond("allow-1", trueResult)},
				},
				{
					name:                         "one deny false, one deny unevaluatable",
					disableConditionsMapEvaluate: true,
					denyConditions: []Condition{
						cond("deny-false", falseResult),
						unevalCond("deny-1"),
					},
					noOpinionConditions: []Condition{cond("nop-1", trueResult)},
					allowConditions:     []Condition{cond("allow-1", trueResult)},
				},
			},
		},
		{
			name:       "conditionsmap: nop unevaluatable, allow present",
			wantString: `ConditionsMap(noopinions=1, allows=1)`,
			subCases: []subCase{
				{
					name:                         "minimal",
					disableConditionsMapEvaluate: true,
					noOpinionConditions:          []Condition{unevalCond("nop-1")},
					allowConditions:              []Condition{cond("allow-1", trueResult)},
				},
				{
					name:                         "with false deny",
					disableConditionsMapEvaluate: true,
					denyConditions:               []Condition{cond("deny-1", falseResult)},
					noOpinionConditions:          []Condition{unevalCond("nop-1")},
					allowConditions:              []Condition{cond("allow-1", trueResult)},
				},
			},
		},
		{
			name:       "conditionsmap: allow unevaluatable",
			wantString: `ConditionsMap(allows=1)`,
			subCases: []subCase{
				{
					name:                         "minimal (nil evaluateFunc)",
					disableConditionsMapEvaluate: true,
					allowConditions:              []Condition{unevalCond("allow-1")},
				},
				{
					name:                         "evaluateFunc also returns unevaluatable",
					disableConditionsMapEvaluate: true,
					allowConditions:              []Condition{unevalCond("allow-1")},
					evaluateFunc: func(context.Context, Condition, ConditionsData) ConditionEvaluationResult {
						return ConditionsEvaluationResultUnevaluatable()
					},
				},
				{
					name:                         "with false deny and nop",
					disableConditionsMapEvaluate: true,
					denyConditions:               []Condition{cond("deny-1", falseResult)},
					noOpinionConditions:          []Condition{cond("nop-1", falseResult)},
					allowConditions:              []Condition{unevalCond("allow-1")},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, sc := range tt.subCases {
				t.Run(sc.name, func(t *testing.T) {
					// when testing a ConditionsMap of only NoOpinion conditions, add a "filler" Deny condition
					// that evaluates to false, it doesn't anyways influence the outcome, but to avoid the constructor
					// to always statically fold the ConditionsAwareDecision to NoOpinion.
					if len(sc.denyConditions) == 0 && len(sc.allowConditions) == 0 {
						sc.denyConditions = append(sc.denyConditions, cond("filler-deny-false", falseResult))
					}

					// Construct the ConditionsMap via the constructor to exercise validation.
					decision := ConditionsAwareDecisionConditionsMap(sc.denyConditions, sc.noOpinionConditions, sc.allowConditions)
					if !decision.IsConditionsMap() {
						t.Fatalf("expected ConditionsMap from constructor, got %s", decision.String())
					}
					cm := decision.ConditionsMap()

					// Always assert the test case against the more general partiallyEvaluateConditionsMapInternal function
					result := partiallyEvaluateConditionsMapInternal(t.Context(), cm, nil, sc.evaluateFunc)
					if got := result.String(); got != tt.wantString {
						t.Errorf("partiallyEvaluateConditionsMapInternal: got decision %s, want %s", got, tt.wantString)
					}
					// However, when possible (no unevaluatable conditions), also call ConditionsMap.Evaluate
					if !sc.disableConditionsMapEvaluate {
						// wrap the three parts returned from ConditionsMap.Evaluate in a ConditionsAwareDecision just to get unified string assertions
						result = ConditionsAwareDecisionFromParts(cm.Evaluate(t.Context(), nil, func(ctx context.Context, condition Condition, data ConditionsData) (bool, error) {
							if sc.evaluateFunc == nil {
								t.Fatalf("ConditionsMap.Evaluate doesn't support unevaluatable conditions, set sc.testConditionsMapEvaluate=false")
								panic("unreachable, ensure no return")
							}

							r := sc.evaluateFunc(ctx, condition, data)
							switch {
							case r.IsTrue():
								return true, nil
							case r.IsFalse():
								return false, nil
							case r.IsError():
								return false, r.Error()
							default:
								t.Fatalf("ConditionsMap.Evaluate doesn't support unevaluatable conditions, set sc.testConditionsMapEvaluate=false")
								panic("unreachable, ensure no return")
							}
						}))
						if got := result.String(); got != tt.wantString {
							t.Errorf("ConditionsMap.Evaluate: got decision %s, want %s", got, tt.wantString)
						}
					}
				})
			}
		})
	}
}
