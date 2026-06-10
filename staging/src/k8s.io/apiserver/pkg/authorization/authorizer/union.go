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
	authorizerName string
	d              ConditionsAwareDecision
}

// ConditionsAwareDecisionUnion is an unioned conditions-aware decision type, keyed by authorizer name.
type ConditionsAwareDecisionUnion struct {
	inner []namedConditionsAwareDecision
	errs  []error
}

func (unionMap *ConditionsAwareDecisionUnion) Add(authorizerName string, d ConditionsAwareDecision) {
	// Ignore authorizerName validation when the authorizer gave an unconditional decision, as that won't get evaluated
	if !d.IsUnconditional() && slices.ContainsFunc(unionMap.inner, func(nd namedConditionsAwareDecision) bool { return nd.authorizerName == authorizerName }) {
		unionMap.errs = append(unionMap.errs, fmt.Errorf("duplicate authorizerName %q", authorizerName))
		// add decision anyways to inner, so we can take into account its content when failing closed
		unionMap.inner = append(unionMap.inner, namedConditionsAwareDecision{authorizerName: authorizerName, d: d})
		return
	}
	if unionMap.ContainsAllowOrDeny() {
		return // all items after the first concrete Allow or Deny aren't anyways used in evaluation, so they are not added to inner
	}
	unionMap.inner = append(unionMap.inner, namedConditionsAwareDecision{authorizerName: authorizerName, d: d})
}

// FailureDecision returns either a Deny or NoOpinion decision to fail closed
// whenever processing a decision fails. If the decision contains one or
// more Deny decisions or conditions, one must fail closed with Deny, as that could or would
// have been the if the condition evaluation did not error. Otherwise, NoOpinion is returned.
// TODO: Use PossibleDecisions instead
func (unionMap ConditionsAwareDecisionUnion) FailureDecision() Decision {
	for _, subDecision := range unionMap.inner {
		if subDecision.d.FailureDecision() == DecisionDeny {
			return DecisionDeny
		}
	}
	return DecisionNoOpinion
}

// ContainsAllowOrDeny returns true whether there union contains at least one
// Allow or Deny decision within the unioned decisions.
func (unionMap ConditionsAwareDecisionUnion) ContainsAllowOrDeny() bool {
	for _, subDecision := range unionMap.inner {
		if subDecision.d.ContainsAllowOrDeny() {
			return true
		}
	}
	return false
}

func (unionMap ConditionsAwareDecisionUnion) PossibleDecisions() sets.Set[Decision] {
	union := sets.New(DecisionNoOpinion) // Default response is NoOpinion
	for _, subDecision := range unionMap.inner {
		union.Insert(subDecision.d.PossibleDecisions().UnsortedList()...)
		// Short-circuit on the first Allow or Deny, after that, decisions don't matter.
		if subDecision.d.ContainsAllowOrDeny() {
			// When there is an Allow or Deny leaf somewhere, the default response NoOpinion won't ever be returned
			union.Delete(DecisionNoOpinion)
			return union
		}
	}
	return union
}

// ConditionsAwareDecisionUnion unions some amount of decisions together into a tree structure,
// where Allow/Deny/NoOpinion/ConditionsMap decisions are leafs, and Union decisions are internal
// tree nodes.
func (unionMap ConditionsAwareDecisionUnion) ToDecision() ConditionsAwareDecision {
	// If we encountered any errors (e.g. duplicate authorizernames) while building the slice,
	// fail closed.
	if len(unionMap.errs) != 0 {
		err := utilerrors.NewAggregate(unionMap.errs)
		return ConditionsAwareDecisionFromParts(unionMap.FailureDecision(), "failed closed", err)
	}

	// If we only have one possible decision, it can readily be evaluated without evaluation.
	if possibleDecisions := unionMap.PossibleDecisions(); possibleDecisions.Len() == 1 {
		onlyPossibleDecision := possibleDecisions.UnsortedList()[0]
		// Collect at least the certainly deciding decisions' reasons and errors. TODO: could we expand this?
		reasonlist := make([]string, 0, len(unionMap.inner))
		errlist := make([]error, 0, len(unionMap.inner))
		for i, subDecision := range unionMap.inner {
			if (onlyPossibleDecision == DecisionAllow && subDecision.d.IsAllow()) ||
				(onlyPossibleDecision == DecisionNoOpinion && subDecision.d.IsNoOpinion()) ||
				(onlyPossibleDecision == DecisionDeny && subDecision.d.IsDeny()) {
				if reason := subDecision.d.Reason(); len(reason) != 0 {
					reasonlist = append(reasonlist, fmt.Sprintf("%d: %s", i, reason))
				}
				if err := subDecision.d.Error(); err != nil {
					errlist = append(errlist, fmt.Errorf("%d: %w", i, err))
				}
			}
		}

		switch onlyPossibleDecision {
		case DecisionAllow:
			// For example, a union of decisions with possible outcomes "[Allow, NoOpinion], [NoOpinion], [Allow], [Deny]" yields possible outcome [Allow] always,
			// regardless of how the ConditionsMap in the beginning evaluates.
			return ConditionsAwareDecisionAllow(strings.Join(reasonlist, ", "), utilerrors.NewAggregate(errlist))
		case DecisionNoOpinion:
			// This happens for instance when called on the empty slice, then the only possible mode is NoOpinion
			// This can only happen if there were only NoOpinions in the chain, so we can gather them here. TODO: (formally) verify this
			return ConditionsAwareDecisionNoOpinion(strings.Join(reasonlist, ", "), utilerrors.NewAggregate(errlist))
		case DecisionDeny:
			// For example, a union of decisions with possible outcomes "[Deny, NoOpinion], [NoOpinion], [Deny], [Allow]" yields possible outcome [Deny] always,
			// regardless of how the ConditionsMap in the beginning evaluates.
			return ConditionsAwareDecisionDeny(strings.Join(reasonlist, ", "), utilerrors.NewAggregate(errlist))
		default:
			return ConditionsAwareDecisionDeny("failed closed", errors.New("should be unreachable: ConditionsAwareDecision should only contain Allow/Deny/NoOpinion"))
		}
	}

	return ConditionsAwareDecision{
		decisionType: conditionsAwareDecisionTypeUnion,
		union: ConditionsAwareDecisionUnion{
			// avoid assigning unionMap here, as then unionMap.Add could change the returned decision
			inner: slices.Clone(unionMap.inner),
		},
	}
}
