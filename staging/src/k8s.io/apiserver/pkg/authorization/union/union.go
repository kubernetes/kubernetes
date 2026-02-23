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
func (authzHandler unionAuthzHandler) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, error) {
	var (
		errlist    []error
		reasonlist []string
	)

	for _, currAuthzHandler := range authzHandler {
		decision, err := currAuthzHandler.Authorize(ctx, a)
		// Ignore previous errors/reasons from NoOpinion responses
		if decision.IsAllowed() || decision.IsDenied() {
			return decision, err
		}
		// If NoOpinion, save the reasons and errors, if any
		if err != nil {
			errlist = append(errlist, err)
		}
		reason := decision.Reason()
		if len(reason) != 0 {
			reasonlist = append(reasonlist, reason)
		}
	}

	return authorizer.DecisionNoOpinion(reasonlist...), utilerrors.NewAggregate(errlist)
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
