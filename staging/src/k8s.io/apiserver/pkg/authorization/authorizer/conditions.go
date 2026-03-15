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
	"strings"

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
// A ConditionsAwareDecision is passed by value.
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

// ConditionsAwareDecisionAllow constructs an Allow decision with the given reason and error.
func ConditionsAwareDecisionAllow(reason string, err error) ConditionsAwareDecision {
	return ConditionsAwareDecision{
		unconditionalDecision: DecisionAllow,
		reason:                reason,
		err:                   err,
	}
}

// ConditionsAwareDecisionNoOpinion constructs a NoOpinion decision with the given reason and error.
func ConditionsAwareDecisionNoOpinion(reason string, err error) ConditionsAwareDecision {
	return ConditionsAwareDecision{
		unconditionalDecision: DecisionNoOpinion,
		reason:                reason,
		err:                   err,
	}
}

// ConditionsAwareDecisionFromParts is meant to be used by conditions-unaware Authorizer implementations
// in order to implement Authorizer.ConditionsAwareAuthorize as:
// "return authorizer.ConditionsAwareDecisionFromParts(self.Authorize(ctx, a))"
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
	return d.unconditionalDecision == DecisionDeny // == 0 == zero value
}

// IsUnconditional is true if d is Allowed, Denied or NoOpinion.
func (d ConditionsAwareDecision) IsUnconditional() bool {
	return d.IsAllowed() || d.IsDenied() || d.IsNoOpinion()
}

// UnconditionalParts turns a ConditionsAwareDecision into the
// triple that Authorizer.Authorize expects. If the decision is
// conditional, the returned condition is Deny if there were at least
// some Deny condition, otherwise NoOpinion.
// This function is meant to be called when IsUnconditional() == true.
//
// If the authorizer is conditions-aware, it can choose to only implement
// real business logic in the ConditionsAwareAuthorize method, and implement
// Authorize() as "return self.ConditionsAwareAuthorize(ctx, attrs).UnconditionalParts()"
func (d ConditionsAwareDecision) UnconditionalParts() (Decision, string, error) {
	switch {
	case d.IsAllowed():
		return DecisionAllow, d.Reason(), d.Error()
	case d.IsDenied():
		return DecisionDeny, d.Reason(), d.Error()
	case d.IsNoOpinion():
		return DecisionNoOpinion, d.Reason(), d.Error()
	default:
		// An error is not returned here, as that could yield a HTTP response code of 500 instead of 403.
		// For the use-case described above with regards to calling this function in Authorize, not returning
		// an error is important, as it is valid to always fail closed, as if this happens, no unconditional
		// permissions were given the requestor.
		return d.FailClosedDecision(), "failed closed: tried to return conditional decision to conditions-unaware authorizer", nil
	}
}

// FailClosedDecision returns either a Deny or NoOpinion decision to fail closed
// whenever processing a decision fails. If the decision contains one or
// more Deny decisions or conditions, one must fail closed with Deny, as that could or would
// have been the if the condition evaluation did not error. Otherwise, NoOpinion is returned.
func (d ConditionsAwareDecision) FailClosedDecision() Decision {
	if d.IsAllowed() || d.IsNoOpinion() {
		return DecisionNoOpinion
	}
	// TODO(luxas): In the follow-up PR, add the logic for ConditionsMap and Union here, which
	// makes this function useful.
	// => d.IsDenied() == true
	return DecisionDeny
}

// Reason returns the reason associated with the decision.
func (d ConditionsAwareDecision) Reason() string {
	return d.reason
}

// Error returns the error associated with the decision.
func (d ConditionsAwareDecision) Error() error {
	return d.err
}

// String returns a human-readable representation of the decision.
func (d ConditionsAwareDecision) String() string {
	params := []string{}
	if len(d.reason) != 0 {
		params = append(params, fmt.Sprintf("reason=%q", d.reason))
	}
	if d.err != nil {
		params = append(params, fmt.Sprintf("err=%q", d.err.Error()))
	}
	paramsStr := func() string {
		if len(params) == 0 {
			return ""
		}
		return fmt.Sprintf("(%s)", strings.Join(params, ", "))
	}
	if d.IsAllowed() {
		return fmt.Sprintf("Allow%s", paramsStr())
	}
	if d.IsNoOpinion() {
		return fmt.Sprintf("NoOpinion%s", paramsStr())
	}
	// Deny is written such that if none of the other modes apply,
	// IsDenied() is true.
	return fmt.Sprintf("Deny%s", paramsStr())
}

// ConditionsData is an enum type for various evaluation targets conditions
// can be written against.
// TODO(luxas): Implement this in the follow-up PR.
type ConditionsData struct {
}
