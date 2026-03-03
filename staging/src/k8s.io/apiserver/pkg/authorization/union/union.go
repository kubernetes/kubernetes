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

// This means that we got a concrete Allow or Deny or a conditional Allow
// Note that there may be conditional Denies before a concrete Allow, and
// a conditional Allow before a concrete Deny.

// Authorizes against a chain of authorizer.Authorizer objects and returns nil if successful and returns error if unsuccessful
func (authzHandler unionAuthzHandler) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, error) {
	var (
		errlist       []error
		decisionChain authorizer.ConditionalDecisionChain
	)

	for _, currAuthzHandler := range authzHandler {
		// Precondition: All previously seen decisions were either NoOpinion or Conditional.
		decision, err := currAuthzHandler.Authorize(ctx, a)

		if err != nil {
			// TODO: Wrap the errors to be of the form "authenticator 'foo' returned error: %w"
			errlist = append(errlist, err)
		}
		decisionChain = append(decisionChain, decision)

		// If we got a concrete Allow/Deny decision, no need to walk the chain further.
		if decisionChain.HasConcreteResponse() {
			// TODO: should we capture the reasons and errors from earlier conditional decisions?
			return authorizer.DecisionConditionalChain(decisionChain...), utilerrors.NewAggregate(errlist)
		}
	}

	return authorizer.DecisionConditionalChain(decisionChain...), utilerrors.NewAggregate(errlist)
}

func (authzHandler unionAuthzHandler) EvaluateConditions(ctx context.Context, unevaluatedDecision authorizer.Decision, data authorizer.ConditionData) (authorizer.Decision, error) {
	if unevaluatedDecision.IsAllowed() || unevaluatedDecision.IsDenied() || unevaluatedDecision.IsNoOpinion() {
		return unevaluatedDecision, nil
	}
	// TODO: better separation between IsConditional and IsConditionalChain
	if unevaluatedDecision.IsConditional() {
		return unevaluatedDecision.FailClosedDecision(), errors.New("plain ConditionSet unsupported")
	}

	errlist := []error{}
	for i, unevaluatedSubDecision := range unevaluatedDecision.ConditionalChain() {
		// Whenever we reach a concrete Allow/Deny in the list, that is our answer
		if unevaluatedSubDecision.IsAllowed() || unevaluatedSubDecision.IsDenied() {
			return unevaluatedSubDecision, nil
		}
		// No point in trying to evaluate conditions for a NoOpinion decision, skip
		if unevaluatedSubDecision.IsNoOpinion() {
			continue
		}

		conditionsAuthorizer := authzHandler[i]
		evalResult, err := conditionsAuthorizer.EvaluateConditions(ctx, unevaluatedSubDecision, data)
		if evalResult.IsAllowed() || evalResult.IsDenied() {
			return evalResult, err
		}

		if err != nil {
			errlist = append(errlist, err)
		}

		if evalResult.IsNoOpinion() {
			continue
		}

		// We do not yet support evaluating conditional to conditional
		err = errors.New("unsupported to evaluate conditional to conditional")
		if err != nil {
			errlist = append(errlist, err)
		}
		failClosedDecision := unevaluatedSubDecision.FailClosedDecision()
		if failClosedDecision.IsDenied() {
			return failClosedDecision, err
		}
	}
	// Everything evaluated to NoOpinion
	// TODO: Aggregate the reasons here too?
	return authorizer.DecisionNoOpinion(), utilerrors.NewAggregate(errlist)
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
