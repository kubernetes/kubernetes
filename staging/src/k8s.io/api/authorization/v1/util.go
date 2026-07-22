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

package v1

import (
	"k8s.io/apimachinery/pkg/util/sets"
)

var conditionalAuthorizationDecisionTypes = sets.New(
	// unconditional
	ConditionsAwareDecisionTypeAllow,
	ConditionsAwareDecisionTypeDeny,
	ConditionsAwareDecisionTypeNoOpinion,
	// conditional
	ConditionsAwareDecisionTypeConditionsMap,
	ConditionsAwareDecisionTypeUnion,
)

var unconditionalAuthorizationDecisionTypes = sets.New(
	// unconditional
	ConditionsAwareDecisionTypeAllow,
	ConditionsAwareDecisionTypeDeny,
	ConditionsAwareDecisionTypeNoOpinion,
)

// SupportsConditionalAuthorization returns whether the AuthorizationOptions support conditional authorization,
// that is, a superset of the decision types {Allow, Deny, NoOpinion, ConditionMap, Union}.
func (ao *AuthorizationOptions) SupportsConditionalAuthorization() bool {
	return ao.GetHandledDecisionTypes().IsSuperset(conditionalAuthorizationDecisionTypes)
}

// SupportsUnconditionalAuthorization returns whether the AuthorizationOptions support unconditional authorization,
// that is, a superset of the decision types {Allow, Deny, NoOpinion}.
// All SubjectAccessReview callers are expected to support these modes.
// If ao is nil, this function returns true.
// Note that more specific capabilities (e.g. conditional authorization) shall be checked for first, as
// HandledDecisionTypes={Allow, Deny, NoOpinion, ConditionMap, Union} yields both
// SupportsConditionalAuthorization() == true and SupportsUnconditionalAuthorization() == true.
func (ao *AuthorizationOptions) SupportsUnconditionalAuthorization() bool {
	return ao.GetHandledDecisionTypes().IsSuperset(unconditionalAuthorizationDecisionTypes)
}

// GetHandledDecisionTypes returns a set of client-handled decision types.
// If ao is nil, the default set of {Allow, Deny, NoOpinion} is returned.
func (ao *AuthorizationOptions) GetHandledDecisionTypes() sets.Set[ConditionsAwareDecisionType] {
	if ao == nil {
		return UnconditionalAuthorizationDecisionTypes()
	}
	return sets.New(ao.HandledDecisionTypes...)
}

// ConditionalAuthorizationDecisionTypes returns the decision types that a client
// need to support to handle conditional authorization: {Allow, Deny, NoOpinion, ConditionMap, Union}.
func ConditionalAuthorizationDecisionTypes() sets.Set[ConditionsAwareDecisionType] {
	return conditionalAuthorizationDecisionTypes.Clone() // always return fresh copies, never expose the original data
}

// UnconditionalAuthorizationDecisionTypes returns the decision types that a client
// need to support to handle conditional authorization: {Allow, Deny, NoOpinion}.
func UnconditionalAuthorizationDecisionTypes() sets.Set[ConditionsAwareDecisionType] {
	return unconditionalAuthorizationDecisionTypes.Clone() // always return fresh copies, never expose the original data
}
