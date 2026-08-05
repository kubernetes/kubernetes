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
	"errors"
	"fmt"
	"slices"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

type namedConditionsAwareDecision struct {
	conditionalAuthorizerName string
	d                         ConditionsAwareDecision
}

// ConditionsAwareDecisionUnion is a builder struct for a Union-typed ConditionsAwareDecision.
// ConditionsAwareDecisionUnion is not thread-safe.
type ConditionsAwareDecisionUnion struct {
	// Note: Remember to register any fields added here to the deepcopy in ToDecision
	inner []namedConditionsAwareDecision
	errs  []error
	// containsUnconditionalAllowOrDeny is a memorized value of whether any of
	// the conditions in inner contain an unconditional Allow or Deny leaf.
	containsUnconditionalAllowOrDeny bool
	// subDecisionsPossibleDecisions is a memorized union of all added decisions'
	// PossibleDecisions. Important: This is NOT the same as
	// ConditionsAwareDecisionUnion.PossibleDecisions, as that also
	// requires extra NoOpinion handling, adding NoOpinion in the empty
	// union case, and removing it if containsUnconditionalAllowOrDeny == true.
	subDecisionsPossibleDecisions sets.Set[Decision]
}

// Add adds a named sub-decision to the current Union ConditionsAwareDecision builder.
// Add is a no-op if ContainsUnconditionalAllowOrDeny() is true, as adding decisions after an Allow or
// Deny will never change the final evaluated outcome.
// An error might be returned if the name is empty or a duplicate, the bool signifies whether
// the Add was a no-op.
// Add is not thread-safe.
func (unionMap *ConditionsAwareDecisionUnion) Add(conditionalAuthorizerName string, d ConditionsAwareDecision) {
	if unionMap.ContainsUnconditionalAllowOrDeny() {
		return // all items after the first concrete Allow or Deny aren't anyways used in evaluation, so they are not added to inner
	}

	if len(conditionalAuthorizerName) == 0 {
		unionMap.errs = append(unionMap.errs, fmt.Errorf("conditionalAuthorizerName must be non-empty"))
	}
	if unionMap.hasConditionalAuthorizerName(conditionalAuthorizerName) {
		// Note: We don't short-circuit here, as we want to see all decisions "until the end", such that we can fail closed stronger if needed.
		unionMap.errs = append(unionMap.errs, fmt.Errorf("duplicate conditionalAuthorizerName %q", conditionalAuthorizerName))
	}
	// Once we've seen an unconditional Allow or Deny somewhere in the chain, we can stop accepting
	// other decisions, as they won't ever apply.
	if d.ContainsUnconditionalAllowOrDeny() {
		unionMap.containsUnconditionalAllowOrDeny = true
	}
	unionMap.inner = append(unionMap.inner, namedConditionsAwareDecision{conditionalAuthorizerName: conditionalAuthorizerName, d: d})
	// Memorize the possible decisions. This is sound, as ConditionsAwareDecisions are immutable after construction.
	if unionMap.subDecisionsPossibleDecisions == nil {
		unionMap.subDecisionsPossibleDecisions = sets.New[Decision]()
	}
	unionMap.subDecisionsPossibleDecisions.Insert(d.PossibleDecisions().UnsortedList()...)
}

func (unionMap *ConditionsAwareDecisionUnion) hasConditionalAuthorizerName(conditionalAuthorizerName string) bool {
	return slices.ContainsFunc(unionMap.inner, func(nd namedConditionsAwareDecision) bool {
		return nd.conditionalAuthorizerName == conditionalAuthorizerName
	})
}

// FailureDecision returns either a Deny or NoOpinion decision that the caller can
// use if the caller encounters an unrecoverable error while processing a
// ConditionsAwareDecision. If this decision is or could evaluate to Deny
// this function returns DecisionDeny to the caller, in order to fail closed
// as a conservative approximation. Otherwise, NoOpinion is returned.
// FailureDecision is not thread-safe.
func (unionMap *ConditionsAwareDecisionUnion) FailureDecision() Decision {
	if unionMap.PossibleDecisions().Has(DecisionDeny) {
		return DecisionDeny
	}
	return DecisionNoOpinion
}

// ContainsUnconditionalAllowOrDeny returns true whether there union contains at least one
// Allow or Deny decision within the unioned decisions.
// ContainsUnconditionalAllowOrDeny is not thread-safe.
func (unionMap *ConditionsAwareDecisionUnion) ContainsUnconditionalAllowOrDeny() bool {
	return unionMap.containsUnconditionalAllowOrDeny
}

// PossibleDecisions details what are the possible decision outcomes of this
// ConditionsAwareDecisionUnion. The return value is a subset of {Allow, Deny, NoOpinion},
// but never the empty set.
// PossibleDecisions is not thread-safe.
func (unionMap *ConditionsAwareDecisionUnion) PossibleDecisions() sets.Set[Decision] {
	// Default response is NoOpinion.
	// Always return a fresh set, don't allow the caller to mutate any internal state.
	possibleDecisions := sets.New(DecisionNoOpinion)
	// Add the memorized possible decisions after adding them.
	if unionMap.subDecisionsPossibleDecisions != nil {
		possibleDecisions.Insert(unionMap.subDecisionsPossibleDecisions.UnsortedList()...)
	}
	// When there is an Allow or Deny leaf somewhere, the default response NoOpinion won't ever be returned
	if unionMap.ContainsUnconditionalAllowOrDeny() {
		possibleDecisions.Delete(DecisionNoOpinion)
	}
	return possibleDecisions
}

// ConditionsAwareDecisionUnion unions some amount of decisions together into a tree structure,
// where Allow/Deny/NoOpinion/ConditionsMap decisions are leafs, and Union decisions are internal
// tree nodes.
// ToDecision is not thread-safe.
func (unionMap *ConditionsAwareDecisionUnion) ToDecision() ConditionsAwareDecision {
	// If we encountered any errors (e.g. duplicate authorizernames) while building the slice,
	// fail closed.
	if len(unionMap.errs) != 0 {
		err := utilerrors.NewAggregate(unionMap.errs)
		return ConditionsAwareDecisionFromParts(unionMap.FailureDecision(), "failed closed", err)
	}

	// If we only have one possible outcome, consolidate to an unconditional decision without evaluation.
	if possibleDecisions := unionMap.PossibleDecisions(); possibleDecisions.Len() == 1 {
		onlyPossibleDecision := possibleDecisions.UnsortedList()[0]
		// Collect the certainly deciding decisions' reasons and errors. When the only possible
		// decision is Allow or Deny, we in principle know it's the last decision added, but
		// keep the logic generic and aligned with NoOpinion, for which we aggregate together
		// the reasons and errors for all of the same-type decisions.
		var aggregateReasons []string
		var aggregateErrors []error
		for _, namedDecision := range unionMap.inner {
			reasons, errs := collectReasonsAndErrors([]string{namedDecision.conditionalAuthorizerName}, namedDecision.d, onlyPossibleDecision)
			aggregateReasons = append(aggregateReasons, reasons...)
			aggregateErrors = append(aggregateErrors, errs...)
		}

		switch onlyPossibleDecision {
		case DecisionDeny:
			// For example, a union of decisions with possible outcomes "[Deny, NoOpinion], [NoOpinion], [Deny], [Allow]" yields possible outcome [Deny] always,
			// regardless of how the ConditionsMap in the beginning evaluates.
			return ConditionsAwareDecisionDeny(strings.Join(aggregateReasons, ", "), utilerrors.NewAggregate(aggregateErrors))
		case DecisionNoOpinion:
			// This happens for instance when called on the empty slice, then the only possible mode is NoOpinion
			// This can only happen if there were only NoOpinions in the chain, so we can gather them here. TODO: (formally) verify this
			return ConditionsAwareDecisionNoOpinion(strings.Join(aggregateReasons, ", "), utilerrors.NewAggregate(aggregateErrors))
		case DecisionAllow:
			// For example, a union of decisions with possible outcomes "[Allow, NoOpinion], [NoOpinion], [Allow], [Deny]" yields possible outcome [Allow] always,
			// regardless of how the ConditionsMap in the beginning evaluates.
			return ConditionsAwareDecisionAllow(strings.Join(aggregateReasons, ", "), utilerrors.NewAggregate(aggregateErrors))
		default:
			return ConditionsAwareDecisionDeny("failed closed", errors.New("should be unreachable: ConditionsAwareDecision should only contain Allow/Deny/NoOpinion"))
		}
	}

	return ConditionsAwareDecision{
		decisionType: conditionsAwareDecisionTypeUnion,
		union: ConditionsAwareDecisionUnion{
			// ensure ConditionsAwareDecision immutability by not sharing any references between the builder and the result
			inner:                            slices.Clone(unionMap.inner),
			containsUnconditionalAllowOrDeny: unionMap.containsUnconditionalAllowOrDeny,
			subDecisionsPossibleDecisions:    unionMap.subDecisionsPossibleDecisions.Clone(),
		},
	}
}

func collectReasonsAndErrors(authorizerNamePrefix []string, condAwareDecision ConditionsAwareDecision, unconditionalDecisionToCollect Decision) (aggregateReasons []string, aggregateErrors []error) {
	if (unconditionalDecisionToCollect == DecisionAllow && condAwareDecision.IsAllow()) ||
		(unconditionalDecisionToCollect == DecisionNoOpinion && condAwareDecision.IsNoOpinion()) ||
		(unconditionalDecisionToCollect == DecisionDeny && condAwareDecision.IsDeny()) {
		if reason := condAwareDecision.Reason(); len(reason) != 0 {
			aggregateReasons = append(aggregateReasons, fmt.Sprintf("%s: {%s}", strings.Join(authorizerNamePrefix, ": "), reason))
		}
		if err := condAwareDecision.Error(); err != nil {
			aggregateErrors = append(aggregateErrors, fmt.Errorf("%s: %w", strings.Join(authorizerNamePrefix, ": "), err))
		}
		return
	}
	if condAwareDecision.IsUnion() {
		for authorizerName, subDecision := range condAwareDecision.UnionedDecisions() {
			reasons, errs := collectReasonsAndErrors(slices.Concat(authorizerNamePrefix, []string{authorizerName}), subDecision, unconditionalDecisionToCollect)
			aggregateReasons = append(aggregateReasons, reasons...)
			aggregateErrors = append(aggregateErrors, errs...)
		}
		return
	}
	return
}
