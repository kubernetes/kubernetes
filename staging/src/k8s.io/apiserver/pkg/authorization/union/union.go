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
	"fmt"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

var _ = authorizer.Authorizer(unionAuthzHandler{})

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
	var decisions authorizer.ConditionsAwareDecisionUnion

	for _, currAuthzHandler := range authzHandler {
		// Precondition: All previously seen leaf decisions were either of NoOpinion or ConditionsMap type.

		// Call the authorizer on its conditions-aware method, and add the decision to the slice,
		// regardless of type. This due to that later in EvaluateConditions, the decision index
		// in the slice is what correlates a decision with the authorizer that should be used
		// for evaluating it (if needed).
		decision := currAuthzHandler.ConditionsAwareAuthorize(ctx, a)
		decisions.Add(currAuthzHandler.AuthorizerName(), decision)

		// If there is any Allow/Deny decision leaf, no need to walk the chain further.
		if decisions.ContainsAllowOrDeny() {
			return decisions.ToDecision()
		}
		// => all leaves are NoOpinion or ConditionsMap, continue to the next authorizer
	}

	// If we reached here, all leaf decisions were either of NoOpinion or ConditionsMap type.
	// If all decisions were NoOpinions, the constructor folds into a single NoOpinion decision.
	return decisions.ToDecision()
}

// EvaluateConditions is not supported by this authorizer.
func (authzHandler unionAuthzHandler) EvaluateConditions(ctx context.Context, unevaluatedDecision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	// This should never happen, an authorizer shall only be called back on the union unevaluatedDecision that was returned from
	// AuthorizeConditionsAware(). The caller should never call EvaluateConditions on an Allow/Deny/NoOpinion
	if !unevaluatedDecision.IsUnion() {
		return unevaluatedDecision.FailureDecision(), "failed closed", errors.New("the union authorizer can only evaluate union ConditionsAwareDecisions")
	}

	// This logic directly maps 1:1 with Authorize(), now that we get unconditional responses from the evaluation process.
	var (
		errlist    []error
		reasonlist []string
	)

	for currentAuthorizerName, unevaluatedSubDecision := range unevaluatedDecision.UnionedDecisions() {
		// Precondition: All previously seen leaf decisions were or evaluated to NoOpinion, or some unrecognized mode.

		// If we get to an Allow or Deny in the union chain, we have our answer.
		if unevaluatedSubDecision.IsAllow() {
			return authorizer.DecisionAllow, unevaluatedSubDecision.Reason(), unevaluatedSubDecision.Error()
		}
		if unevaluatedSubDecision.IsDeny() {
			return authorizer.DecisionDeny, unevaluatedSubDecision.Reason(), unevaluatedSubDecision.Error()
		}

		var decision authorizer.Decision
		var reason string
		var err error
		if unevaluatedSubDecision.IsNoOpinion() {
			// NoOpinions cannot be evaluated, but we should make sure to save the reason and error.
			decision, reason, err = authorizer.DecisionNoOpinion, unevaluatedSubDecision.Reason(), unevaluatedSubDecision.Error()
		} else {
			// ConditionsMap or Union types are evaluated by their authorizer
			decision, reason, err = authzHandler.evaluateConditions(ctx, currentAuthorizerName, unevaluatedSubDecision, data)
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

func (authzHandler unionAuthzHandler) evaluateConditions(ctx context.Context, authorizerName string, unevaluatedSubDecision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	authorizer, err := authzHandler.getAuthorizerWithName(authorizerName)
	if err != nil {
		return unevaluatedSubDecision.FailureDecision(), "failed closed", err
	}
	return authorizer.EvaluateConditions(ctx, unevaluatedSubDecision, data)
}

func (authzHandler unionAuthzHandler) getAuthorizerWithName(authorizerName string) (authorizer.Authorizer, error) {
	authorizers := make([]authorizer.Authorizer, 0, 1)
	for _, a := range authzHandler {
		if authorizerName == a.AuthorizerName() {
			authorizers = append(authorizers, a)
		}
	}
	switch len(authorizers) {
	case 0:
		return nil, fmt.Errorf("no authorizer with name %q found", authorizerName)
	case 1:
		return authorizers[0], nil
	default:
		return nil, fmt.Errorf("ambiguous match: found %d authorizers with name %q, don't know which one to pick", len(authorizers), authorizerName)
	}
}

// AuthorizerName returns a name for the union authorizer itself. Sub-authorizers retain
// their own names; this is just a label for the wrapping union.
func (authzHandler unionAuthzHandler) AuthorizerName() string {
	delegateNames := make([]string, 0, len(authzHandler))
	for _, a := range authzHandler {
		delegateNames = append(delegateNames, a.AuthorizerName())
	}
	return fmt.Sprintf("authorizer.k8s.io/Union[%s]", strings.Join(delegateNames, ", "))
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
