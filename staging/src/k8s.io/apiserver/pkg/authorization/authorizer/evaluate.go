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
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// partialConditionEvaluationResultType is a small enum for the type of ConditionEvaluationResult
type partialConditionEvaluationResultType int

const (
	partialConditionEvaluationResultTypeUnevaluatable partialConditionEvaluationResultType = iota
	partialConditionEvaluationResultTypeTrue
	partialConditionEvaluationResultTypeFalse
	partialConditionEvaluationResultTypeError
)

// ConditionEvaluationResult is an enum type with four variants:
// - true and false: Evaluation was successful, and evaluated to this value
// - error: The condition could be evaluated, but errored during eval.
// - unevaluatable: The condition cannot readily be evaluated. This is the struct zero value.
type ConditionEvaluationResult struct {
	resultType partialConditionEvaluationResultType
	err        error
}

// ConditionEvaluationResultBoolean constructs an evaluation result with a boolean value.
func ConditionEvaluationResultBoolean(evalResult bool) ConditionEvaluationResult {
	if evalResult {
		return ConditionEvaluationResult{resultType: partialConditionEvaluationResultTypeTrue}
	}
	return ConditionEvaluationResult{resultType: partialConditionEvaluationResultTypeFalse}
}

// ConditionEvaluationResultError indicates that the condition could be evaluated, but failed.
func ConditionEvaluationResultError(err error) ConditionEvaluationResult {
	if err == nil {
		return ConditionEvaluationResult{
			resultType: partialConditionEvaluationResultTypeError,
			err:        errors.New("unspecified evaluation error"),
		}
	}
	return ConditionEvaluationResult{
		resultType: partialConditionEvaluationResultTypeError,
		err:        err,
	}
}

// ConditionsEvaluationResultUnevaluatable indicates direct conditions evaluation is not possible.
func ConditionsEvaluationResultUnevaluatable() ConditionEvaluationResult {
	return ConditionEvaluationResult{
		resultType: partialConditionEvaluationResultTypeUnevaluatable, // == 0 (which matches the zero value of the struct)
	}
}

// IsTrue indicates that the conditions evaluation was successful, and evaluated to true, which means it influences the ConditionsMap decision.
func (r ConditionEvaluationResult) IsTrue() bool {
	return r.resultType == partialConditionEvaluationResultTypeTrue
}

// IsFalse indicates that the conditions evaluation was successful, but evaluated to false, and it not thus taken into account.
func (r ConditionEvaluationResult) IsFalse() bool {
	return r.resultType == partialConditionEvaluationResultTypeFalse
}

// IsError indicates whether conditions evaluation failed.
func (r ConditionEvaluationResult) IsError() bool {
	return r.resultType == partialConditionEvaluationResultTypeError
}

// Error returns the evaluation error, if any.
func (r ConditionEvaluationResult) Error() error { return r.err }

// IsUnevaluatable is true whenever none of the other variants is, that is, the zero value.
func (r ConditionEvaluationResult) IsUnevaluatable() bool {
	return r.resultType == partialConditionEvaluationResultTypeUnevaluatable
}

// Evaluate evaluates the ConditionsMap primarily using the Conditions' own Evaluate() function,
// and secondarily using evaluateConditionFn, if set.
func (c ConditionsMap) Evaluate(ctx context.Context, data ConditionsData, evaluateConditionFn EvaluateConditionFunc) (Decision, string, error) {
	// This is a translation between the generic, private function, and the interface we want to expose to callers.
	return partiallyEvaluateConditionsMapInternal(ctx, c, data, func(ctx context.Context, cond Condition, condData ConditionsData) ConditionEvaluationResult {
		// Because we never return "unevaluatable", the returned ConditionsAwareDecision is always one of Allow/Deny/NoOpinion, and thus can we split it into unconditionalParts
		applied, err := evaluateConditionFn(ctx, cond, condData)
		if err != nil {
			return ConditionEvaluationResultError(err)
		}
		return ConditionEvaluationResultBoolean(applied)
	}).unconditionalParts()
}

// partiallyEvaluateConditionsMapInternal evaluates the ConditionsMap primarily using the Conditions' own Evaluate() function,
// and secondarily using evaluateFunc, if set. If evaluateFunc is non-nil and never returns
// ConditionsEvaluationResultUnevaluatable, the returned decision is guaranteed to be Allow/Deny/NoOpinion.
// However, this method can also be used to evaluate a subset of the conditions (e.g. for builtin
// conditions evaluators that support a certain conditions type), returning ConditionsEvaluationResultUnevaluatable
// for conditions that the evaluator does not recognize. In the latter case, a partially evaluated, deep copied
// ConditionsMap might be returned.
func partiallyEvaluateConditionsMapInternal(ctx context.Context, c ConditionsMap, data ConditionsData, evaluateConditionFn PartialEvaluateConditionFunc) ConditionsAwareDecision {
	evalCond := func(cond Condition) ConditionEvaluationResult {
		// First, try to use the condition's own evaluate function.
		// Fallback to evaluateConditionFn if set and unevaluatable
		result := cond.Evaluate(ctx, data)
		if result.IsUnevaluatable() && evaluateConditionFn != nil {
			return evaluateConditionFn(ctx, cond, data)
		}
		return result
	}

	if len(c.denyConditions) != 0 {
		appliedDenyReasons, denyErrors, unevaluatedDenyConditions := conditionsToAppliedErroredUnevaluated(c.DenyConditions(), evalCond, "Deny", "denied the request")
		// If any deny conditions evaluated to true, return Deny
		// Deny conditions that apply take precedence over deny conditions that error, as even if the erroring
		// deny conditions wouldn't have errored, the applied deny conditions would have produced the same Deny decision.
		if len(appliedDenyReasons) != 0 {
			// A nil error must be returned here, in order for the WithAuthorization handler to return 403 and not 500.
			return ConditionsAwareDecisionDeny(strings.Join(appliedDenyReasons, ", "), nil)
		}
		// If any deny errors were encountered, fail closed
		if len(denyErrors) != 0 {
			return ConditionsAwareDecisionDeny("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(denyErrors))
		}

		// When len(unevaluatedDenyConditions) != 0, the possible outcomes are [Deny, NoOpinion] or [Deny, Allow] (depending on whether)
		// there is some matching NoOpinion/Allow condition or not. This means that we need to return another, possibly refined ConditionsMap
		if len(unevaluatedDenyConditions) != 0 {
			return ConditionsAwareDecisionConditionsMap(
				unevaluatedDenyConditions,
				slices.Clone(c.noOpinionConditions),
				slices.Clone(c.allowConditions))
		}
	}
	// If we got here, all Deny conditions could be evaluated, and evaluated to false, nil
	if len(c.noOpinionConditions) != 0 {
		appliedNoOpinionReasons, noOpinionErrors, unevaluatedNoOpinionConditions := conditionsToAppliedErroredUnevaluated(c.NoOpinionConditions(), evalCond, "NoOpinion", "evaluated to NoOpinion")
		// If any NoOpinion conditions evaluated to true, return NoOpinion
		if len(appliedNoOpinionReasons) != 0 {
			return ConditionsAwareDecisionNoOpinion(strings.Join(appliedNoOpinionReasons, ", "), nil)
		}
		// If any NoOpinion errors were encountered, fail closed to NoOpinion as if the conditions would have matched
		if len(noOpinionErrors) != 0 {
			return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(noOpinionErrors))
		}
		// When len(unevaluatedNoOpinionConditions) != 0, the possible outcomes are [NoOpinion] or [NoOpinion, Allow]. (depending on whether)
		// there is some matching Allow condition or not. This means that we need to return another, possibly refined ConditionsMap, unless
		// there are no Allow conditions, in which the decision is always NoOpinion.
		if len(unevaluatedNoOpinionConditions) != 0 {
			// If there are no allow conditions, then either some unevaluated NoOpinion applies, in which the decision is NoOpinion, or all unevaluated
			// NoOpinion conditions evaluate to false, no allow condition applies (as there are none), so the default NoOpinion is returned. In either
			// case under that assumption, the return value is NoOpinion.
			if len(c.allowConditions) == 0 {
				return ConditionsAwareDecisionNoOpinion("at least one NoOpinion condition matched, or no conditions matched", nil)
			}

			// Otherwise, the possible outcomes are [NoOpinion, Allow]. Return a possibly refined ConditionsMap.
			return ConditionsAwareDecisionConditionsMap(
				nil,
				unevaluatedNoOpinionConditions,
				slices.Clone(c.allowConditions))
		}
	}
	// If we got here, all Deny and NoOpinion conditions could be evaluated, and evaluated to false, nil
	if len(c.allowConditions) != 0 {
		appliedAllowReasons, allowErrors, unevaluatedAllowConditions := conditionsToAppliedErroredUnevaluated(c.AllowConditions(), evalCond, "Allow", "allowed the request")
		// If there were at least one Allow condition that applied, then evaluation is successful, even if there
		// were some errors that happened. Those are in this case considered warnings.
		if len(appliedAllowReasons) != 0 {
			return ConditionsAwareDecisionAllow(strings.Join(appliedAllowReasons, ", "), utilerrors.NewAggregate(allowErrors))
		}
		// However, if no Allow condition evaluated to true, but at least one errored, return that as an error to the caller
		if len(allowErrors) != 0 {
			return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(allowErrors))
		}
		// When len(unevaluatedAllowConditions) != 0, the possible outcomes are [NoOpinion, Allow].
		// Return a possibly refined ConditionsMap with the Allow conditions that could not be evaluated.
		if len(unevaluatedAllowConditions) != 0 {
			return ConditionsAwareDecisionConditionsMap(nil, nil, unevaluatedAllowConditions)
		}
	}

	// All conditions evaluated to false. This means a simple default NoOpinion.
	return ConditionsAwareDecisionNoOpinion("no conditions matched", nil)
}

func conditionsToAppliedErroredUnevaluated(conditions iter.Seq[Condition], evalCond func(cond Condition) ConditionEvaluationResult, effect, appliedDescription string) ([]string, []error, []Condition) {
	errs := []error{}
	appliedCondReasons := []string{}
	unevaluatedConditions := []Condition{}
	for cond := range conditions {
		id := cond.GetID()
		evalResult := evalCond(cond)
		switch {
		case evalResult.IsUnevaluatable():
			unevaluatedConditions = append(unevaluatedConditions, cond)
			continue
		case evalResult.IsError():
			errs = append(errs, fmt.Errorf("condition %q with effect=%s produced error: %w", id, effect, evalResult.Error()))
			continue
		case evalResult.IsTrue():
			reason := fmt.Sprintf("condition %q %s", id, appliedDescription)
			if desc := cond.GetDescription(); len(desc) != 0 {
				reason += fmt.Sprintf(" with description %q", desc)
			}
			appliedCondReasons = append(appliedCondReasons, reason)
			continue
		default: // => evalResult.IsFalse() == true
			continue
		}
	}
	// Arguments are returned in the order that they should be considered.
	return appliedCondReasons, errs, unevaluatedConditions
}

// PartiallyEvaluateConditionsAwareDecision evaluates the ConditionsAwareDecision primarily using any conditions' own Evaluate() function,
// and secondarily/optionally using evaluateConditionFn, if set. If evaluateConditionFn is non-nil and never returns
// ConditionsEvaluationResultUnevaluatable, the returned decision is guaranteed to be Allow/Deny/NoOpinion.
// However, this method can also be used to evaluate a subset of the conditions (e.g. for builtin
// conditions evaluators that only support a certain conditions type), returning ConditionsEvaluationResultUnevaluatable
// for conditions that the evaluator does not recognize. In the latter case, a partially evaluated ConditionsAwareDecision is returned.
func PartiallyEvaluateConditionsAwareDecision(ctx context.Context, unevaluatedDecision ConditionsAwareDecision, data ConditionsData, evaluateConditionFn PartialEvaluateConditionFunc) ConditionsAwareDecision {
	switch {
	case unevaluatedDecision.IsConditionsMap():
		// Try to evaluate or refine the leaf ConditionsMap using the builtin evaluator.
		return partiallyEvaluateConditionsMapInternal(ctx, unevaluatedDecision.ConditionsMap(), data, evaluateConditionFn)
	case unevaluatedDecision.IsUnion():
		var newDecisionChain ConditionsAwareDecisionUnion
		// Recursively walk through the decision DAG in a depth-first manner.

		collectAndShortcircuitOnly := false
		for authorizerName, unevaluatedSubDecision := range unevaluatedDecision.UnionedDecisions() {
			// If collectAndShortcircuitOnly == true, a conditional decision that couldn't
			// be evaluated to Allow/Deny/NoOpinion was encountered during a previous
			// loop iteration. Then all latter decisions stay unevaluated.
			if collectAndShortcircuitOnly {
				newDecisionChain.Add(authorizerName, unevaluatedSubDecision)
				continue
			}

			// When !collectAndShortcircuitOnly: All decisions so far in newDecisionChain are NoOpinions.

			// Try evaluating or refining the leaf ConditionsMaps in this tree of decisions.
			possiblyEvaluatedSubDecision := PartiallyEvaluateConditionsAwareDecision(ctx, unevaluatedSubDecision, data, evaluateConditionFn)

			// Always preserve the ordering of the decisions, even for NoOpinions, as we might use their reasons/errors
			newDecisionChain.Add(authorizerName, possiblyEvaluatedSubDecision)

			switch {
			case possiblyEvaluatedSubDecision.IsDeny(), possiblyEvaluatedSubDecision.IsAllow():
				// We successfully evaluated to something, and because all previously-seen
				// decisions were NoOpinions, we can simplify to Allow/Deny here.
				return possiblyEvaluatedSubDecision
			case possiblyEvaluatedSubDecision.IsNoOpinion():
				continue
			case possiblyEvaluatedSubDecision.IsConditionsMap(), possiblyEvaluatedSubDecision.IsUnion():
				// This means that there is no chance of evaluating to an unconditional decision using builtinConditionsEvaluator.
				// Thus, instead of continuing to try to evaluate later ConditionsMaps in-process,
				// whose computation might be wasted if previous authorizer's ConditionsMaps indeed
				// turn out to be Allow/Deny (and not NoOpinion), just short-circuit and delegate to the authorizer to evaluate.
				//
				// collectAndShortcircuitOnly is used to preserve the tail of the union, without
				// evaluating the suffix.
				collectAndShortcircuitOnly = true
				continue
			default:
				// Fail closed with the FailureDecision of the whole unevaluatedDecision
				return ConditionsAwareDecisionFromParts(unevaluatedDecision.FailureDecision(), "failed closed", fmt.Errorf("unknown ConditionsAwareDecision variant: %s", unevaluatedSubDecision))
			}
		}
		// If we got here, the first not-NoOpinion decision was Union or ConditionsMap, which means
		// we cannot simplify it. Return a possibly refined decision chain to delegate to the authorizer.
		return newDecisionChain.ToDecision()
	default:
		// No simplification possible
		return unevaluatedDecision
	}
}
