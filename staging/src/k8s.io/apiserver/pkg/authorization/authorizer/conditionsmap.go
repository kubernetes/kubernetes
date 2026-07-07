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
	"fmt"
	"iter"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ConditionsMap is a map of conditions, where each condition has a map-unique
// ID, is a function from ConditionsData -> bool and has Allow/Deny/NoOpinion effect.
// It must be constructed through ConditionsAwareDecisionConditionsMap.
// During construction, all conditions are validated.
type ConditionsMap struct {
	// invariant: len(denyConditions) != 0 || len(allowConditions) != 0
	//
	// slices are used here instead of actual maps, as the ConditionsMap does
	// not need to lookup single elements. It's called a "map" as uniqueness of
	// the IDs (keys) across all conditions (values) in the map is enforced.
	denyConditions      []Condition
	noOpinionConditions []Condition
	allowConditions     []Condition
}

// FailureDecision returns either a Deny or NoOpinion decision that the caller can
// use if the caller encounters an unrecoverable error while processing a
// ConditionsAwareDecision. If this decision is or could evaluate to Deny
// this function returns DecisionDeny to the caller, in order to fail closed
// as a conservative approximation. Otherwise, NoOpinion is returned.
func (c ConditionsMap) FailureDecision() Decision {
	if c.PossibleDecisions().Has(DecisionDeny) {
		// Note: this only happens if len(denyConditions) > 0
		return DecisionDeny
	}
	return DecisionNoOpinion
}

// Length returns the number of elements in the map.
func (c ConditionsMap) Length() int {
	return len(c.denyConditions) + len(c.noOpinionConditions) + len(c.allowConditions)
}

// DenyConditions returns the Deny conditions in this map.
// The conditions are returned in the same order as they were supplied to ConditionsAwareDecisionConditionsMap.
func (c ConditionsMap) DenyConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.denyConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// NoOpinionConditions returns the NoOpinion conditions in this map.
// The conditions are returned in the same order as they were supplied to ConditionsAwareDecisionConditionsMap.
func (c ConditionsMap) NoOpinionConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.noOpinionConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// AllowConditions returns the Allow conditions in this map.
// The conditions are returned in the same order as they were supplied to ConditionsAwareDecisionConditionsMap.
func (c ConditionsMap) AllowConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.allowConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// PossibleDecisions details what are the possible decision outcomes of this
// ConditionsAwareDecision. The return value is a subset of {Allow, Deny, NoOpinion},
// but never the empty set. If the set only contains a single value, it means the
// decision is unconditional.
func (c ConditionsMap) PossibleDecisions() sets.Set[Decision] {
	possibleDecisions := sets.New(DecisionNoOpinion)
	if len(c.allowConditions) > 0 {
		possibleDecisions.Insert(DecisionAllow)
	}
	if len(c.denyConditions) > 0 {
		possibleDecisions.Insert(DecisionDeny)
	}
	return possibleDecisions
}

// MaxConditionsPerMap is the maximum number of conditions allowed in a single ConditionsMap.
const MaxConditionsPerMap = 128

// ConditionsAwareDecisionConditionsMap creates a ConditionsMap decision.
// The conditions are grouped by their effects: Deny, NoOpinion and Allow, that function as follows:
//   - Deny: If a Deny condition evaluates to true, the ConditionsMap necessarily evaluates to Deny.
//     In this case, no further authorizers are consulted.
//   - NoOpinion: If a NoOpinion condition evaluates to true, the given authorizer's ConditionsMap cannot
//     evaluate to Allow anymore, but necessarily Deny or NoOpinion, depending on whether there are any true
//     Deny conditions. However, later authorizers in the chain can still Allow or Deny.
//     It is effectively a softer deny that just overrides the authorizer's own allow policies.
//   - Allow: If any Allow condition evaluates to true, the ConditionsMap evaluates to Allow,
//     unless any Deny/NoOpinion condition also evaluates to true (in which case the Deny/NoOpinion conditions
//     have precedence).
//
// All conditions must be non-nil, this function panics if any condition is nil.
func ConditionsAwareDecisionConditionsMap(denyConditions []Condition, noOpinionConditions []Condition, allowConditions []Condition) ConditionsAwareDecision {
	// if there are any Deny conditions, Deny is a possible decision, and thus should we fail closed with Deny in that case
	hasDenyEffect := len(denyConditions) > 0
	makeFailClosedError := func(err error) ConditionsAwareDecision {
		if hasDenyEffect {
			return ConditionsAwareDecisionDeny("failed closed", err)
		}
		return ConditionsAwareDecisionNoOpinion("failed closed", err)
	}

	// enforce minimum 1 and maximum amount of conditions per map
	conditionsAmount := len(denyConditions) + len(noOpinionConditions) + len(allowConditions)
	if conditionsAmount == 0 {
		// Does not use makeFailClosedError, but NoOpinion directly, as in this branch there are no deny conditions, so NoOpinion is safe
		return ConditionsAwareDecisionNoOpinion("no conditions", fmt.Errorf("at least one condition must be passed to ConditionsAwareDecisionConditionsMap(), got none"))
	}
	if conditionsAmount > MaxConditionsPerMap {
		return makeFailClosedError(fmt.Errorf("too many conditions: %d exceeds maximum of %d", conditionsAmount, MaxConditionsPerMap))
	}
	// short-circuit case: if no Allow or Deny conditions exist, then evaluating this ConditionsMap can never evaluate to Allow or Deny,
	// but only NoOpinion, regardless of the data. Thus just fold to NoOpinion directly.
	// this upholds the len(denyConditions) != 0 || len(allowConditions) != 0 invariant on the ConditionsMap.
	if len(denyConditions) == 0 && len(allowConditions) == 0 {
		return ConditionsAwareDecisionNoOpinion("only NoOpinion conditions always evaluate to NoOpinion", nil)
	}

	seenIDs := sets.New[string]()

	if err := validateConditions(seenIDs, denyConditions); err != nil {
		return makeFailClosedError(err)
	}
	if err := validateConditions(seenIDs, noOpinionConditions); err != nil {
		return makeFailClosedError(err)
	}
	if err := validateConditions(seenIDs, allowConditions); err != nil {
		return makeFailClosedError(err)
	}

	return ConditionsAwareDecision{
		decisionType: conditionsAwareDecisionTypeConditionsMap,
		conditionsMap: ConditionsMap{
			// ensure immutability of the ConditionsMap after construction by not sharing caller-owned references
			denyConditions:      slices.Clone(denyConditions),
			noOpinionConditions: slices.Clone(noOpinionConditions),
			allowConditions:     slices.Clone(allowConditions),
		},
	}
}

func validateConditions(seenIDs sets.Set[string], conditions []Condition) error {
	for _, condition := range conditions {

		id := condition.GetID()
		if seenIDs.Has(id) {
			return fmt.Errorf("duplicate condition ID %q", id)
		}
		seenIDs.Insert(id)

		// Validate ID as a label key.
		if errs := content.IsLabelKey(id); len(errs) > 0 {
			return fmt.Errorf("invalid condition ID %q: %s", id, strings.Join(errs, "; "))
		}

		// Validate type as a label key, if set.
		if conditionType := condition.GetType(); len(conditionType) != 0 {
			if errs := content.IsLabelKey(conditionType); len(errs) > 0 {
				return fmt.Errorf("invalid condition type %q: %s", conditionType, strings.Join(errs, "; "))
			}
		}
	}
	return nil
}

// GenericCondition is a generic implementation of the Condition interface,
// with optional support for fast in-process conditions evaluation, by
// setting EvaluateFunc non-nil.
type GenericCondition struct {
	ID           string
	Condition    string
	Type         string
	Description  string
	EvaluateFunc func(ctx context.Context, data ConditionsData) ConditionEvaluationResult
}

var _ Condition = GenericCondition{}

func (c GenericCondition) GetID() string {
	return c.ID
}

func (c GenericCondition) GetCondition() string {
	return c.Condition
}

func (c GenericCondition) GetType() string {
	return c.Type
}

func (c GenericCondition) GetDescription() string {
	return c.Description
}
func (c GenericCondition) Evaluate(ctx context.Context, data ConditionsData) ConditionEvaluationResult {
	if c.EvaluateFunc == nil {
		return ConditionsEvaluationResultUnevaluatable()
	}
	return c.EvaluateFunc(ctx, data)
}
