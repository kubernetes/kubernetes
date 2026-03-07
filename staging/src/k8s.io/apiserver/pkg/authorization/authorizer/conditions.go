/*
Copyright 2026 The Kubernetes Authors.

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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// ErrorConditionEvaluationNotSupported is returned by authorizer implementations
// that do not support condition evaluation.
var ErrorConditionEvaluationNotSupported = errors.New("condition evaluation not supported")

// ConditionsAwareDecision models an authorization decision that is conditions-aware.
// It is an enum type of the following five variants:
// - Allow: unconditional Allow.
// - Deny: unconditional Deny.
// - NoOpinion: unconditional NoOpinion.
// - Conditional: conditional on some previously-unseen data.
// - Union: an ordered list of sub-decisions, which forms a tree of decisions.
//
// The zero value (ConditionsAwareDecision{}) is equivalent to ConditionsAwareDecisionDeny().
// A Decision is passed by value.
// Decision equality must be checked with Decision.Equal, not reflect.DeepEqual.
type ConditionsAwareDecision struct {
	unconditionalDecision Decision

	reason string
	err    error
}

// ConditionsAwareDecisionDeny constructs a Deny decision with the given reason and error.
func ConditionsAwareDecisionDeny(reason string, err error) ConditionsAwareDecision {
	return ConditionsAwareDecision{
		// DecisionDeny == int(0) == zero value
		// => ConditionsAwareDecision{} == ConditionsAwareDecisionDeny()
		unconditionalDecision: DecisionDeny,
		reason:                reason,
		err:                   err,
	}
}

// ConditionsAwareDecisionAllow constructs an Allow decision with the given reason.
func ConditionsAwareDecisionAllow(reason string, err error) ConditionsAwareDecision {
	return ConditionsAwareDecision{
		unconditionalDecision: DecisionAllow,
		reason:                reason,
		err:                   err,
	}
}

// ConditionsAwareDecisionNoOpinion constructs a NoOpinion decision with the given reason.
func ConditionsAwareDecisionNoOpinion(reason string, err error) ConditionsAwareDecision {
	return ConditionsAwareDecision{
		unconditionalDecision: DecisionNoOpinion,
		reason:                reason,
		err:                   err,
	}
}

// DecisionFromParts is meant to be used by conditions-unaware Authorizer implementations
// in order to implement Authorizer.AuthorizeConditionsAware as:
// "return authorizer.DecisionFromParts(self.Authorize(ctx, a))"
func ConditionsAwareDecisionFromParts(unconditional Decision, reason string, err error) ConditionsAwareDecision {
	switch unconditional {
	case DecisionAllow:
		return ConditionsAwareDecisionAllow(reason, err)
	case DecisionNoOpinion:
		return ConditionsAwareDecisionNoOpinion(reason, err)
	case DecisionDeny:
		return ConditionsAwareDecisionDeny(reason, err)
	default:
		return ConditionsAwareDecisionDeny(reason, utilerrors.NewAggregate(
			[]error{
				err,
				fmt.Errorf("unknown unconditional decision type: %d", unconditional),
			},
		))
	}
}

// INVARIANT: Exactly one of Is* must return true at all times.

// IsAllowed returns true if the decision is an unconditional Allow.
func (d ConditionsAwareDecision) IsAllowed() bool {
	return d.unconditionalDecision == DecisionAllow
}

// IsNoOpinion returns true if the decision is an unconditional NoOpinion.
func (d ConditionsAwareDecision) IsNoOpinion() bool {
	return d.unconditionalDecision == DecisionNoOpinion
}

// IsDenied returns true if the decision is an unconditional Deny.
func (d ConditionsAwareDecision) IsDenied() bool {
	// The decision is a Deny whenever none of the other modes apply
	// NOTE: A Conditional decision is encoded as
	// d.unconditionalDecision == 0 == decisionDeny && d.conditionSet != nil, so it
	// is not enough to check d.unconditionalDecision == decisionDeny
	// This is because the zero value of the struct must be a Deny
	return !d.IsAllowed() && !d.IsNoOpinion()
}

func (d ConditionsAwareDecision) Reason() string {
	return d.reason
}

func (d ConditionsAwareDecision) Error() error {
	return d.err
}

// String returns a human-readable representation of the decision.
func (d ConditionsAwareDecision) String() string {
	errStr := "<nil>"
	if d.err != nil {
		errStr = fmt.Sprintf("%q", d.err.Error())
	}
	if d.IsAllowed() {
		return fmt.Sprintf("Allow(%q, %s)", d.reason, errStr)
	}
	if d.IsNoOpinion() {
		return fmt.Sprintf("NoOpinion(%q, %s)", d.reason, errStr)
	}
	// Deny is written such that if none of the other modes apply,
	// IsDenied() is true.
	return fmt.Sprintf("Deny(%q, %s)", d.reason, errStr)
}

// ConditionsMap is a map of conditions of a given type.
type ConditionsMap struct {
}

// BuiltinConditionsMapEvaluators represents a list of builtin
// conditions evaluators, that may be used to evaluate a ConditionMap
// before performing e.g. expensive webhook calls.
type BuiltinConditionsMapEvaluators []BuiltinConditionsMapEvaluator

// ConditionsEncodingPreference specifies how the client wants conditions to be
// returned by the authorizer. The authorizer can freely choose to respect or
// ignore this hint.
//
// The zero value is equal to ConditionsEncodingPreferenceOptimized(), which means
// "encode conditions in the most efficient/optimized form possible, without descriptions".
type ConditionsEncodingPreference struct {
	humanReadableConditions bool
	includeDescription      bool
}

// ConditionsEncodingPreferenceOptimized is the default, which asks the authorizer to
// encode conditions in the most efficient/optimized form possible, without descriptions.
//
// This mode is used in the hot path, in the WithAuthorization HTTP filter.
func ConditionsEncodingPreferenceOptimized() ConditionsEncodingPreference {
	return ConditionsEncodingPreference{
		humanReadableConditions: false,
		includeDescription:      false,
	}
}

// ConditionsEncodingPreferenceHumanReadable asks the authorizer to encode conditions
// in a form that is as human-readable as possible, and include desciptions if possible.
//
// This mode is recommended to be used by clients to e.g. SelfSubjectAccessReview.
func ConditionsEncodingPreferenceHumanReadable() ConditionsEncodingPreference {
	return ConditionsEncodingPreference{
		humanReadableConditions: true,
		includeDescription:      true,
	}
}

// HumanReadableConditions returns true if the caller would prefer human-readable conditions.
func (pref ConditionsEncodingPreference) HumanReadableConditions() bool {
	return pref.humanReadableConditions
}

// IncludeDescription returns true if the caller would prefer descriptions with the condition.
func (pref ConditionsEncodingPreference) IncludeDescription() bool {
	return pref.includeDescription
}

// ConditionsData is an enum type for various evaluation targets conditions
// can be written against.
type ConditionsData struct {
}
