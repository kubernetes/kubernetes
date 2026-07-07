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
// and secondarily using evaluateConditionFn, if set. The first matching (true) condition
// short-circuits the process, in order for the evaluation to be as efficient as possible.
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
// for conditions that the evaluator does not recognize. In the latter case, a partially evaluated,
// ConditionsMap might be returned.
func partiallyEvaluateConditionsMapInternal(ctx context.Context, c ConditionsMap, data ConditionsData, evaluateConditionFn MaybeEvaluateConditionFunc) ConditionsAwareDecision {
	evalCond := func(cond Condition) ConditionEvaluationResult {
		// First, try to use the condition's own evaluate function.
		// Fallback to evaluateConditionFn if set and unevaluatable
		result := cond.Evaluate(ctx, data)
		if result.IsUnevaluatable() && evaluateConditionFn != nil {
			return evaluateConditionFn(ctx, cond, data)
		}
		return result
	}

	// General logic: Deny > NoOpinion > Allow. Within a set of same-effect conditions true > error > unevaluatable > false.

	if len(c.denyConditions) != 0 {
		appliedReason, errored, unevaluated := conditionsToAppliedErroredUnevaluated(
			c.DenyConditions(), evalCond, "Deny", "denied the request")

		if len(appliedReason) != 0 {
			// No errors are returned here, as that would turn this into a 500 instead of 403 in the WithAuthorization HTTP filter
			// TODO(luxas): We might want to change the WithAuthorization logic such that it becomes possible for an authorizer to
			// surface non-critical errors also in the precence of denies.
			return ConditionsAwareDecisionDeny(appliedReason, nil)
		}
		if len(errored) != 0 {
			return ConditionsAwareDecisionDeny("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(errored))
		}
		if len(unevaluated) != 0 {
			return ConditionsAwareDecisionConditionsMap(unevaluated, c.noOpinionConditions, c.allowConditions)
		}
	}

	// If we got here, all Deny conditions could be evaluated, and evaluated to false, nil

	if len(c.noOpinionConditions) != 0 {
		appliedReason, errored, unevaluated := conditionsToAppliedErroredUnevaluated(
			c.NoOpinionConditions(), evalCond, "NoOpinion", "evaluated to NoOpinion")

		if len(appliedReason) != 0 {
			// No errors are returned here, as that would turn this into a 500 instead of 403 in the WithAuthorization HTTP filter
			return ConditionsAwareDecisionNoOpinion(appliedReason, nil)
		}
		if len(errored) != 0 {
			return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(errored))
		}
		if len(unevaluated) != 0 {
			// Note: If len(c.allowConditions) == 0, ConditionsAwareDecisionConditionsMap will fold this into an unconditional NoOpinion.
			return ConditionsAwareDecisionConditionsMap(nil, unevaluated, c.allowConditions)
		}
	}

	// If we got here, all Deny and NoOpinion conditions could be evaluated, and evaluated to false, nil

	if len(c.allowConditions) != 0 {
		appliedReason, errored, unevaluated := conditionsToAppliedErroredUnevaluated(
			c.AllowConditions(), evalCond, "Allow", "allowed the request")

		if len(appliedReason) != 0 {
			// Errors could technically be returned with an Allow (and be logged), but no errors are returned for
			// the short-circuit true case, as the error list might be incomplete.
			return ConditionsAwareDecisionAllow(appliedReason, nil)
		}
		if len(errored) != 0 {
			return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(errored))
		}
		if len(unevaluated) != 0 {
			return ConditionsAwareDecisionConditionsMap(nil, nil, unevaluated)
		}
	}

	// All conditions evaluated to false. This means a simple default NoOpinion.
	return ConditionsAwareDecisionNoOpinion("no conditions matched", nil)
}

// Note: the signature is string, error, []Condition, as that is the order in which the return values should be processed
// The first true short-circuits the evaluation with a nil error, even though some potentially happened before the first
// true evaluation, as the errors list in that case wouldn't necessarily be comprehensive.
func conditionsToAppliedErroredUnevaluated(conditions iter.Seq[Condition], evalCond func(cond Condition) ConditionEvaluationResult, effect, appliedDescription string) (string, []error, []Condition) {
	var errs []error
	var unevaluatedConditions []Condition
	for cond := range conditions {
		id := cond.GetID()
		evalResult := evalCond(cond)
		switch {
		case evalResult.IsTrue():
			reason := fmt.Sprintf("condition %q %s", id, appliedDescription)
			if desc := cond.GetDescription(); len(desc) != 0 {
				reason += fmt.Sprintf(" with description %q", desc)
			}
			// Note: nil is returned for errors here as the list is not comprehensive in the short-circuit case.
			return reason, nil, nil
		case evalResult.IsFalse():
			continue
		case evalResult.IsUnevaluatable():
			unevaluatedConditions = append(unevaluatedConditions, cond)
			continue
		case evalResult.IsError():
			errs = append(errs, fmt.Errorf("condition %q with effect=%s evaluated to an error: %w", id, effect, evalResult.Error()))
			continue
		default:
			errs = append(errs, fmt.Errorf("condition %q with effect=%s evaluated to an error: unknown evaluation result %v", id, effect, evalResult))
			continue
		}
	}
	return "", errs, unevaluatedConditions
}

// PartiallyEvaluateConditionsAwareDecision evaluates the ConditionsAwareDecision primarily using any conditions' own Evaluate() function,
// and secondarily/optionally using evaluateConditionFn, if set. If evaluateConditionFn is non-nil and never returns
// ConditionsEvaluationResultUnevaluatable, the returned decision is guaranteed to be Allow/Deny/NoOpinion.
// However, this method can also be used to evaluate a subset of the conditions (e.g. for builtin
// conditions evaluators that only support a certain conditions type), returning ConditionsEvaluationResultUnevaluatable
// for conditions that the evaluator does not recognize. In the latter case, a partially evaluated ConditionsAwareDecision is returned.
// When evaluating a ConditionsMap, only the first matching (true) condition short-circuits the process,
// in order for the evaluation to be as efficient as possible.
func PartiallyEvaluateConditionsAwareDecision(ctx context.Context, unevaluatedDecision ConditionsAwareDecision, data ConditionsData, evaluateConditionFn MaybeEvaluateConditionFunc) ConditionsAwareDecision {
	switch {
	case unevaluatedDecision.IsConditionsMap():
		// Try to evaluate or refine the leaf ConditionsMap using the builtin evaluator.
		return partiallyEvaluateConditionsMapInternal(ctx, unevaluatedDecision.ConditionsMap(), data, evaluateConditionFn)
	case unevaluatedDecision.IsUnion():
		var newDecisionChain ConditionsAwareDecisionUnion
		// Recursively walk through the decision DAG in a depth-first manner.
		// The logic in this function should be the same as in the union authorizer's ConditionsAwareAuthorize.

		collectTailAfterConditionalDecision := false
		for authorizerName, unevaluatedSubDecision := range unevaluatedDecision.UnionedDecisions() {
			// If collectAndShortcircuitOnly == true, a conditional decision that couldn't
			// be evaluated to Allow/Deny/NoOpinion was encountered during a previous
			// loop iteration. Then all latter decisions stay unevaluated.
			if collectTailAfterConditionalDecision {
				newDecisionChain.Add(authorizerName, unevaluatedSubDecision)
				continue
			}

			// When !collectTailAfterConditionalDecision: All decisions so far in newDecisionChain are NoOpinions,
			// as if there existed a previous decision of type ConditionsMap or Union, collectTailAfterConditionalDecision == true,
			// and if there existed an Allow or Deny earlier, the function would have returned.

			// Try evaluating or refining the leaf ConditionsMaps in this tree of decisions.
			possiblyEvaluatedSubDecision := PartiallyEvaluateConditionsAwareDecision(ctx, unevaluatedSubDecision, data, evaluateConditionFn)
			newDecisionChain.Add(authorizerName, possiblyEvaluatedSubDecision)

			// If there is any Allow/Deny decision leaf, no need to walk the chain further.
			if possiblyEvaluatedSubDecision.ContainsUnconditionalAllowOrDeny() {
				return newDecisionChain.ToDecision()
			}

			// Stop partially evaluating after encountering a conditional decision
			if !possiblyEvaluatedSubDecision.IsUnconditional() {
				collectTailAfterConditionalDecision = true
			}
		}
		// If we reached here, all leaf decisions were either of NoOpinion or ConditionsMap type.
		// If all decisions were NoOpinions, the constructor folds into a single NoOpinion decision.
		return newDecisionChain.ToDecision()
	default:
		// No simplification possible
		return unevaluatedDecision
	}
}
