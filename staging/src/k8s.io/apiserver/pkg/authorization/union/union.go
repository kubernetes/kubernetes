/*
Copyright 2014 The Kubernetes Authors.

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

// Package union implements an authorizer that combines multiple subauthorizer.
// The union authorizer iterates over each subauthorizer and returns the first
// decision that is either an Allow decision or a Deny decision. If a
// subauthorizer returns a NoOpinion, then the union authorizer moves onto the
// next authorizer or, if the subauthorizer was the last authorizer, returns
// NoOpinion as the aggregate decision. I.e. union authorizer creates an
// aggregate decision and supports short-circuit allows and denies from
// subauthorizers.
package union

import (
	"context"
	"errors"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// unionAuthzHandler authorizer against a chain of authorizer.Authorizer
type unionAuthzHandler []authorizer.Authorizer

// New returns an authorizer that authorizes against a chain of authorizer.Authorizer objects
func New(authorizationHandlers ...authorizer.Authorizer) authorizer.Authorizer {
	return unionAuthzHandler(authorizationHandlers)
}

// Authorizes against a chain of authorizer.Authorizer objects and returns nil if successful and returns error if unsuccessful
func (authzHandler unionAuthzHandler) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	var (
		errlist    []error
		reasonlist []string
	)

	for _, currAuthzHandler := range authzHandler {
		decision, reason, err := currAuthzHandler.Authorize(ctx, a)

		if err != nil {
			errlist = append(errlist, err)
		}
		if len(reason) != 0 {
			reasonlist = append(reasonlist, reason)
		}
		switch decision {
		case authorizer.DecisionAllow, authorizer.DecisionDeny:
			return decision, reason, err
		case authorizer.DecisionNoOpinion:
			// continue to the next authorizer
		}
	}

	return authorizer.DecisionNoOpinion, strings.Join(reasonlist, "\n"), utilerrors.NewAggregate(errlist)
}

// ConditionsAwareAuthorize is not conditions-aware, converts the Authorize decision.
func (authzHandler unionAuthzHandler) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	var decisions []authorizer.ConditionsAwareDecision

	for _, currAuthzHandler := range authzHandler {
		// Precondition: All previously seen leaf decisions were either of NoOpinion or ConditionsMap type.

		// Call the authorizer on its conditions-aware method, and add the decision to the slice,
		// regardless of type. This due to that later in EvaluateConditions, the decision index
		// in the slice is what correlates a decision with the authorizer that should be used
		// for evaluating it (if needed).
		decision := currAuthzHandler.ConditionsAwareAuthorize(ctx, a)
		decisions = append(decisions, decision)

		// If there is any Allow/Deny decision leaf, no need to walk the chain further.
		if decision.ContainsAllowOrDeny() {
			return authorizer.ConditionsAwareDecisionUnion(decisions...)
		}
		// => all leaves are NoOpinion or ConditionsMap, continue to the next authorizer
	}

	// If we reached here, all leaf decisions were either of NoOpinion or ConditionsMap type.
	// If all decisions were NoOpinions, the constructor folds into a single NoOpinion decision.
	return authorizer.ConditionsAwareDecisionUnion(decisions...)
}

// EvaluateConditions is not supported by this authorizer.
func (authzHandler unionAuthzHandler) EvaluateConditions(ctx context.Context, unevaluatedDecision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	// Stopping condition for the recursion: Nothing to evaluate here.
	if unevaluatedDecision.IsUnconditional() {
		return unevaluatedDecision.UnconditionalParts()
	}
	// This should never happen, an authorizer shall only be called back on an unevaluatedDecision that was returned from
	// AuthorizeConditionsAware(). However, unionAuthzHandler.AuthorizeConditionsAware never returns a "bare" ConditionsMap,
	// but either Allow/Deny/NoOpinion (the case above), or Union[...], even if the union only contains one element.
	if unevaluatedDecision.IsConditionsMap() {
		return unevaluatedDecision.FailClosedDecision(), "failed closed", errors.New("union authorizer never returns a bare ConditionsMap, cannot evaluate")
	}

	// This logic directly maps 1:1 with Authorize(), now that we get unconditional responses from the evaluation process.
	var (
		errlist    []error
		reasonlist []string
	)

	for i, unevaluatedSubDecision := range unevaluatedDecision.UnionedDecisions() {
		// Precondition: All previously seen leaf decisions were or evaluated to NoOpinion, or some unrecognized mode.

		// If we get to an Allow or Deny in the union chain, we have our answer.
		if unevaluatedSubDecision.IsAllowed() || unevaluatedSubDecision.IsDenied() {
			return unevaluatedSubDecision.UnconditionalParts()
		}

		var decision authorizer.Decision
		var reason string
		var err error
		if unevaluatedSubDecision.IsNoOpinion() {
			// NoOpinions cannot be evaluated, but we should make sure to save the reason and error.
			decision, reason, err = authorizer.DecisionNoOpinion, unevaluatedSubDecision.Reason(), unevaluatedSubDecision.Error()
		} else {
			// ConditionsMap or Union types are evaluated by their authorizer
			decision, reason, err = authzHandler[i].EvaluateConditions(ctx, unevaluatedSubDecision, data)
		}

		if err != nil {
			errlist = append(errlist, err)
		}
		if len(reason) != 0 {
			reasonlist = append(reasonlist, reason)
		}

		switch decision {
		case authorizer.DecisionAllow, authorizer.DecisionDeny:
			return decision, reason, err
		case authorizer.DecisionNoOpinion:
			// continue to the next authorizer
		}
	}

	return authorizer.DecisionNoOpinion, strings.Join(reasonlist, "\n"), utilerrors.NewAggregate(errlist)
}

// unionAuthzRulesHandler authorizer against a chain of authorizer.RuleResolver
type unionAuthzRulesHandler []authorizer.RuleResolver

// NewRuleResolvers returns an authorizer that authorizes against a chain of authorizer.Authorizer objects
func NewRuleResolvers(authorizationHandlers ...authorizer.RuleResolver) authorizer.RuleResolver {
	return unionAuthzRulesHandler(authorizationHandlers)
}

// RulesFor against a chain of authorizer.RuleResolver objects and returns nil if successful and returns error if unsuccessful
func (authzHandler unionAuthzRulesHandler) RulesFor(ctx context.Context, user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	var (
		errList              []error
		resourceRulesList    []authorizer.ResourceRuleInfo
		nonResourceRulesList []authorizer.NonResourceRuleInfo
	)
	incompleteStatus := false

	for _, currAuthzHandler := range authzHandler {
		resourceRules, nonResourceRules, incomplete, err := currAuthzHandler.RulesFor(ctx, user, namespace)

		if incomplete {
			incompleteStatus = true
		}
		if err != nil {
			errList = append(errList, err)
		}
		if len(resourceRules) > 0 {
			resourceRulesList = append(resourceRulesList, resourceRules...)
		}
		if len(nonResourceRules) > 0 {
			nonResourceRulesList = append(nonResourceRulesList, nonResourceRules...)
		}
	}

	return resourceRulesList, nonResourceRulesList, incompleteStatus, utilerrors.NewAggregate(errList)
}
