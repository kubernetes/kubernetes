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
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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

	conditionsMap ConditionsMap
	union         conditionsAwareDecisionUnionSlice

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

// IsConditionsMap returns true if the decision is a conditional response
// with a map of conditions to evaluate.
func (d ConditionsAwareDecision) IsConditionsMap() bool {
	return d.conditionsMap.Length() != 0
}

// IsUnion returns true if the decision consists of other sub-decisions
// unioned together in a tree-like structure.
func (d ConditionsAwareDecision) IsUnion() bool {
	return len(d.union) != 0
}

// IsDenied returns true if the decision is an unconditional Deny.
func (d ConditionsAwareDecision) IsDenied() bool {
	// The decision is a Deny whenever none of the other modes apply
	// All other Is* checks require some property of the struct to be
	// distinct from its zero value, which then implies that IsDenied
	// will be true for the zero value.
	return !d.IsAllowed() && !d.IsNoOpinion() && !d.IsConditionsMap() && !d.IsUnion()
}

// ConditionsMap returns the ConditionsMap, which is non-empty
// if and only if IsConditionsMap is true.
func (d ConditionsAwareDecision) ConditionsMap() ConditionsMap {
	return d.conditionsMap
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
	if d.IsConditionsMap() {
		return d.conditionsMap.FailClosedDecision()
	}
	if d.IsUnion() {
		return d.union.FailClosedDecision()
	}
	// => d.IsDenied() == true
	return DecisionDeny
}

// ContainsAllowOrDeny returns true whether there union contains at least one
// Allow or Deny decision within the tree of decisions.
func (d ConditionsAwareDecision) ContainsAllowOrDeny() bool {
	if d.IsAllowed() || d.IsDenied() {
		return true
	}
	if d.IsNoOpinion() || d.IsConditionsMap() {
		return false
	}
	return d.union.ContainsAllowOrDeny()
}

// CanBecomeAllowed returns true if there exists some ConditionsData for which
// the ConditionsAwareDecision would evaluate to Allow for.
func (d ConditionsAwareDecision) CanBecomeAllowed() bool {
	if d.IsAllowed() {
		return true
	}
	if d.IsConditionsMap() {
		return d.conditionsMap.CanBecomeAllowed()
	}
	if d.IsUnion() {
		return d.union.CanBecomeAllowed()
	}
	return false // if Denied or NoOpinion
}

// UnionedDecisions returns an iterator for unioned sub-decisions.
// This iterator is non-empty if and only if IsUnion() == true.
// The sub-decisions are iterated in their priority order.
func (d ConditionsAwareDecision) UnionedDecisions() iter.Seq2[int, ConditionsAwareDecision] {
	return func(yield func(int, ConditionsAwareDecision) bool) {
		for i, subDecision := range d.union {
			if !yield(i, subDecision) {
				return
			}
		}
	}
}

// Reason returns the reason supplied when constructing the decision
// (if Allow/Deny/NoOpinion/ConditionsMap), or an aggregated reason (if Union).
func (d ConditionsAwareDecision) Reason() string {
	if d.IsUnion() {
		b := strings.Builder{}
		b.WriteByte('[')
		for i, sub := range d.union {
			if i != 0 {
				b.WriteString(", ")
			}
			reason := sub.Reason()
			if len(reason) != 0 {
				b.WriteString(sub.Reason())
			} else {
				b.WriteString(`""`)
			}
		}
		b.WriteByte(']')
		return b.String()
	}
	return d.reason
}

// Error returns the error supplied when constructing the decision
// (if Allow/Deny/NoOpinion/ConditionsMap), or an aggregated error (if Union).
func (d ConditionsAwareDecision) Error() error {
	if d.IsUnion() {
		errlist := make([]error, len(d.union))
		for i, sub := range d.union {
			errlist[i] = sub.Error()
		}
		return utilerrors.NewAggregate(errlist)
	}
	return d.err
}

// String returns a human-readable representation of the decision.
func (d ConditionsAwareDecision) String() string {
	if d.IsUnion() {
		// No need to take d.reason or d.err into account, as they are always zero for the union.
		b := strings.Builder{}
		b.WriteString("Union[")
		for i, sub := range d.union {
			if i != 0 {
				b.WriteString(", ")
			}
			b.WriteString(sub.String())
		}
		b.WriteByte(']')
		return b.String()
	}

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
	if d.IsConditionsMap() {
		params = append(params, fmt.Sprintf("len=%d", d.conditionsMap.Length()))
		return fmt.Sprintf("ConditionsMap%s", paramsStr())
	}
	// Deny is written such that if none of the other modes apply,
	// IsDenied() is true.
	return fmt.Sprintf("Deny%s", paramsStr())
}

// ConditionEffect specifies how a condition evaluating to true should be handled.
type ConditionEffect string

const (
	// ConditionEffectDeny means that if this condition evaluates to true,
	// the ConditionsMap necessarily evaluates to Deny. No further authorizers
	// are consulted.
	ConditionEffectDeny ConditionEffect = "Deny"

	// ConditionEffectNoOpinion means that if this condition evaluates to true,
	// the given authorizer's ConditionsMap cannot evaluate to Allow anymore, but
	// necessarily Deny or NoOpinion, depending on whether there are any true
	// EffectDeny conditions.
	// However, later authorizers in the chain can still Allow or Deny.
	// It is effectively a softer deny that just overrides the authorizer's own
	// allow policies.
	ConditionEffectNoOpinion ConditionEffect = "NoOpinion"

	// ConditionEffectAllow means that if this condition evaluates to true,
	// the ConditionsMap evaluates to Allow, unless any Deny/NoOpinion condition
	// also evaluates to true (in which case the Deny/NoOpinion conditions have
	// precedence).
	ConditionEffectAllow ConditionEffect = "Allow"
)

// ConditionsMap is a map of conditions of a given type, and represents
// the conditional decision from the authorizer.
// It must be constructed through ConditionsAwareDecisionConditionsMap.
// During construction, all Conditions are validated and ensured to be non-nil.
type ConditionsMap struct {
	// invariant: when the decision is of type ConditionsMap, Length() != 0,
	// which means that at least one of these slices has an element in it.
	//
	// slices are used here instead of actual maps, as the ConditionsMap does
	// not need to lookup single elements. It's called a "map" as uniqueness of
	// the IDs (keys) across all conditions (values) in the map is enforced.
	denyConditions      []Condition
	noOpinionConditions []Condition
	allowConditions     []Condition
}

// FailClosedDecision returns either a Deny or NoOpinion decision to fail closed
// whenever processing a decision fails. If the decision contains one or
// more Deny decisions or conditions, one must fail closed with Deny, as that could or would
// have been the if the condition evaluation did not error. Otherwise, NoOpinion is returned.
func (c ConditionsMap) FailClosedDecision() Decision {
	for cond := range c.Conditions() {
		if cond.GetEffect() == ConditionEffectDeny {
			return DecisionDeny
		}
	}
	return DecisionNoOpinion
}

// ConditionEvaluationResult is an enum type with four variants:
// - true and false: Evaluation was successful, and evaluated to this value
// - error: The condition could be evaluated, but errored during eval.
// - unevaluatable: The condition cannot readily be evaluated. This is the struct zero value.
type ConditionEvaluationResult struct {
	isTrue  bool
	isFalse bool
	err     error
}

// ConditionEvaluationResultBoolean constructs an evaluation result with a boolean value.
func ConditionEvaluationResultBoolean(evalResult bool) ConditionEvaluationResult {
	if evalResult {
		return ConditionEvaluationResult{isTrue: true}
	}
	return ConditionEvaluationResult{isFalse: true}
}

// ConditionEvaluationResultError indicates that the condition could be evaluated, but failed.
func ConditionEvaluationResultError(err error) ConditionEvaluationResult {
	return ConditionEvaluationResult{err: err}
}

// ConditionsEvaluationResultUnevaluatable indicates direct conditions evaluation is not possible.
func ConditionsEvaluationResultUnevaluatable() ConditionEvaluationResult {
	return ConditionEvaluationResult{}
}

// IsTrue indicates that the conditions evaluation was successful, and evaluated to true, which means it influences the ConditionsMap decision.
func (r ConditionEvaluationResult) IsTrue() bool { return r.isTrue }

// IsFalse indicates that the conditions evaluation was successful, but evaluated to false, and it not thus taken into account.
func (r ConditionEvaluationResult) IsFalse() bool { return r.isFalse }

// IsError indicates whether conditions evaluation failed.
func (r ConditionEvaluationResult) IsError() bool { return r.err != nil }

// Error returns the evaluation error, if any.
func (r ConditionEvaluationResult) Error() error { return r.err }

// IsUnevaluatable is true whenever none of the other variants is, that is, the zero value.
func (r ConditionEvaluationResult) IsUnevaluatable() bool {
	return !r.IsTrue() && !r.IsFalse() && !r.IsError()
}

// Condition represents one authorization condition that is part of a ConditionsMap.
type Condition interface {
	// GetID uniquely identifies this condition within the scope of the authorizer
	// that authored it. Validated as a Kubernetes label key.
	// Required.
	GetID() string

	// GetEffect specifies how the condition evaluating to "true" should be treated.
	// Required.
	GetEffect() ConditionEffect

	// GetType describes the type of the condition, if there are multiple possibilities.
	// Should be formatted as a Kubernetes label key.
	// Any domain suffix of *.k8s.io or *.kubernetes.io is reserved for Kubernetes use.
	// Optional. Can be omitted if the authorizer already knows how to evaluate the condition.
	GetType() string

	// GetCondition returns a string encoding of the condition to be evaluated.
	// It is a pure, deterministic function from ConditionsData to a boolean (or error).
	// Might or might not be human-readable.
	// Optional, if the ID alone is enough for the authorizer to know how to evaluate the condition.
	GetCondition() string

	// GetDescription is an optional human-friendly description that can be shown
	// as an error message or for debugging. Optional.
	GetDescription() string

	// DeepCopy returns a deep copy of the Condition.
	DeepCopy() Condition

	// Evaluate evaluates the condition to a boolean, returns an error, or returns "unevaluatable".
	// If an authorizer already has a pre-compiled condition, this avoids one serialization roundtrip,
	// with potentially expensive deserialization/parsing. However, if the condition underwent a
	// serialize/deserialize roundtrip (e.g. when the caller is an aggregated API server), the authorizer
	// might have to evaluate the condition from its serialized form using evaluateFunc in
	// ConditionsMap.Evaluate.
	Evaluate(ctx context.Context, data ConditionsData) ConditionEvaluationResult
}

// Length returns the number of elements in the map.
func (c ConditionsMap) Length() int {
	return len(c.denyConditions) + len(c.noOpinionConditions) + len(c.allowConditions)
}

// CanBecomeAllowed returns true if this ConditionsMap has at least one
// effect=Allow condition, which means that there exists some ConditionsData
// for which the ConditionsMap could evaluate to Allow.
func (c ConditionsMap) CanBecomeAllowed() bool {
	return len(c.allowConditions) != 0
}

// Conditions returns all conditions in this map.
// The order in which elements are returned is deterministic but undefined.
func (c ConditionsMap) Conditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.denyConditions {
			if !yield(cond) {
				return
			}
		}
		for _, cond := range c.noOpinionConditions {
			if !yield(cond) {
				return
			}
		}
		for _, cond := range c.allowConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// DenyConditions returns the Deny conditions in this map.
// The order in which elements are returned is deterministic but undefined.
func (c ConditionsMap) DenyConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.denyConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// NoOpinionConditions returns the NoOpinion conditions in this map.
// The order in which elements are returned is deterministic but undefined.
func (c ConditionsMap) NoOpinionConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.noOpinionConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

// AllowConditions returns the Allow conditions in this map.
// The order in which elements are returned is deterministic but undefined.
func (c ConditionsMap) AllowConditions() iter.Seq[Condition] {
	return func(yield func(Condition) bool) {
		for _, cond := range c.allowConditions {
			if !yield(cond) {
				return
			}
		}
	}
}

const (
	// MaxConditionsPerMap is the maximum number of conditions allowed in a single ConditionsMap.
	MaxConditionsPerMap = 128
)

// ConditionsAwareDecisionConditionsMap creates a ConditionsMap decision.
func ConditionsAwareDecisionConditionsMap(conditions ...Condition) ConditionsAwareDecision {

	// enforce maximum amount of conditions per map
	if len(conditions) > MaxConditionsPerMap {
		return ConditionsAwareDecisionDeny("failed closed", fmt.Errorf("too many conditions: %d exceeds maximum of %d", len(conditions), MaxConditionsPerMap))
	}

	denyConditions := []Condition{}
	noOpinionConditions := []Condition{}
	allowConditions := []Condition{}
	seenIDs := sets.New[string]()
	errlist := []error{}
	hasDenyEffect := false
	makeFailClosedError := func(err error) ConditionsAwareDecision {
		if hasDenyEffect {
			return ConditionsAwareDecisionDeny("failed closed", err)
		}
		return ConditionsAwareDecisionNoOpinion("failed closed", err)
	}
	for _, condition := range conditions {
		// ignore nil conditions.
		if isNilValue(condition) {
			continue
		}

		// Fail closed using Deny if there was at least one Deny condition in the map.
		effect := condition.GetEffect()
		if effect == ConditionEffectDeny {
			hasDenyEffect = true
		}

		id := condition.GetID()
		if seenIDs.Has(id) {
			errlist = append(errlist, fmt.Errorf("duplicate condition ID %q", id))
			continue
		}
		seenIDs.Insert(id)

		// Validate ID as a label key.
		if errs := content.IsLabelKey(id); len(errs) > 0 {
			errlist = append(errlist, fmt.Errorf("invalid condition ID %q: %s", id, strings.Join(errs, "; ")))
			continue
		}

		// Validate type as a label key, if set.
		if conditionType := condition.GetType(); len(conditionType) != 0 {
			if errs := content.IsLabelKey(conditionType); len(errs) > 0 {
				errlist = append(errlist, fmt.Errorf("invalid condition type %q: %s", conditionType, strings.Join(errs, "; ")))
				continue
			}
		}

		// TODO(luxas): Add condition and description byte limits here or in authorizationapivalidation?

		switch effect {
		case ConditionEffectDeny:
			denyConditions = append(denyConditions, condition)
		case ConditionEffectNoOpinion:
			noOpinionConditions = append(noOpinionConditions, condition)
		case ConditionEffectAllow:
			allowConditions = append(allowConditions, condition)
		default:
			// Fail closed if there are unknown effects
			return ConditionsAwareDecisionDeny("failed closed", fmt.Errorf("condition effect %q not supported. Supported effects are: [Deny, NoOpinion, Allow]", effect))
		}
	}

	// check errors before len(ConditionsMap) == 0, as some errors might have made the map be empty
	// although there were items in the iterator
	if err := utilerrors.NewAggregate(errlist); err != nil {
		// the error is returned first here, not in the loop, to make sure we saw all conditions,
		// and fail closed with deny if there were any deny conditions
		return makeFailClosedError(err)
	}

	// an empty ConditionsMap always evaluates to NoOpinion
	// ignore conditionType being invalid or the feature gate not being set in this case, as it does not matter
	// This must be done as the invariant of the decision's IsConditionsMap is whether the map has non-zero length.
	totalLen := len(denyConditions) + len(noOpinionConditions) + len(allowConditions)
	if totalLen == 0 {
		return ConditionsAwareDecisionNoOpinion("empty ConditionsMap", nil)
	}

	// Do not allow constructing Conditional decisions when the feature gate is off
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
		// Fail closed "softer" than makeFailClosedError, as in this case the authorizer isn't malfunctioning, but the _caller_ of the
		// authorizer just called ConditionsAwareAuthorize even though the feature is off. The caller _shouldn't_ do this, but there is
		// no way of us preventing it. However, instead of returning an error, which could lead to a response code 500, just tell the caller
		// through the reason that as the feature gate is off, the returned decision is "rounded down" (which most likely yields a 403).
		if hasDenyEffect {
			return ConditionsAwareDecisionDeny("authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled", nil)
		}
		return ConditionsAwareDecisionNoOpinion("authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled", nil)
	}

	return ConditionsAwareDecision{
		conditionsMap: ConditionsMap{
			denyConditions:      denyConditions,
			noOpinionConditions: noOpinionConditions,
			allowConditions:     allowConditions,
		},
	}
}

func isNilValue(i interface{}) bool {
	if i == nil {
		return true // both type and data nil
	}
	v := reflect.ValueOf(i)
	switch v.Kind() {
	// v.IsNil() panics if the kind is anything else than these,
	// the list is taken from the IsNil source code
	case reflect.Chan, reflect.Func, reflect.Map,
		reflect.Pointer, reflect.UnsafePointer,
		reflect.Interface, reflect.Slice:
		return v.IsNil() // type non-nil, but data nil
	}
	return false // data non-nil
}

// GenericCondition is a generic implementation of the Condition interface,
// with optional support for fast in-process conditions evaluation, by
// setting EvaluateFunc non-nil.
type GenericCondition struct {
	ID           string
	Effect       ConditionEffect
	Condition    string
	Type         string
	Description  string
	EvaluateFunc func(ctx context.Context, data ConditionsData) ConditionEvaluationResult
}

var _ Condition = GenericCondition{}

func (c GenericCondition) GetID() string {
	return c.ID
}
func (c GenericCondition) GetEffect() ConditionEffect {
	return c.Effect
}
func (c GenericCondition) GetCondition() string {
	return c.Condition
}
func (c GenericCondition) GetType() string {
	return c.Type
}
func (c GenericCondition) GetDescription() string {
	return c.Description
}
func (c GenericCondition) Evaluate(ctx context.Context, data ConditionsData) ConditionEvaluationResult {
	if c.EvaluateFunc == nil {
		return ConditionsEvaluationResultUnevaluatable()
	}
	return c.EvaluateFunc(ctx, data)
}

func (c GenericCondition) DeepCopy() Condition {
	return c // no values passed by reference
}

// Evaluate evaluates the ConditionsMap primarily using the Conditions' own Evaluate() function,
// and secondarily using evaluateFunc, if set. If evaluateFunc is non-nil and never returns
// ConditionsEvaluationResultUnevaluatable, the returned decision is guaranteed to be Allow/Deny/NoOpinion.
// However, this method can also be used to evaluate a subset of the conditions (e.g. for builtin
// conditions evaluators that support a certain conditions type), returning ConditionsEvaluationResultUnevaluatable
// for conditions that the evaluator does not recognize. In the latter case, a partially evaluated, deep copied
// ConditionsMap might be returned.
func (c ConditionsMap) Evaluate(ctx context.Context, data ConditionsData, evaluateFunc func(context.Context, ConditionsData, Condition) ConditionEvaluationResult) ConditionsAwareDecision {
	evalCond := func(cond Condition) ConditionEvaluationResult {
		return cond.Evaluate(ctx, data)
	}
	if evaluateFunc != nil {
		evalCond = func(cond Condition) ConditionEvaluationResult {
			result := cond.Evaluate(ctx, data)
			if !result.IsUnevaluatable() {
				return result
			}
			return evaluateFunc(ctx, data, cond)
		}
	}

	if len(c.denyConditions) != 0 {
		denyErrors := []error{}
		appliedDenyReasons := []string{}
		unevaluatedDenyConditions := []Condition{}
		for cond := range c.DenyConditions() {
			id := cond.GetID()
			evalResult := evalCond(cond)
			switch {
			case evalResult.IsUnevaluatable():
				unevaluatedDenyConditions = append(unevaluatedDenyConditions, cond)
				continue
			case evalResult.IsError():
				denyErrors = append(denyErrors, fmt.Errorf("condition %q with effect=Deny produced error: %w", id, evalResult.Error()))
				continue
			case evalResult.IsTrue():
				reason := fmt.Sprintf("condition %q denied the request", id)
				if desc := cond.GetDescription(); len(desc) != 0 {
					reason += fmt.Sprintf(" with description %q", desc)
				}
				appliedDenyReasons = append(appliedDenyReasons, reason)
				continue
			default: // => evalResult.IsFalse() == true
				continue
			}
		}
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
			return ConditionsAwareDecision{
				conditionsMap: ConditionsMap{
					denyConditions:      unevaluatedDenyConditions,
					noOpinionConditions: deepCopyConditions(c.noOpinionConditions),
					allowConditions:     deepCopyConditions(c.allowConditions),
				},
			}
		}
	}
	// If we got here, all Deny conditions could be evaluated, and evaluated to false, nil
	if len(c.noOpinionConditions) != 0 {
		noOpinionErrors := []error{}
		appliedNoOpinionReasons := []string{}
		unevaluatedNoOpinionConditions := []Condition{}
		for cond := range c.NoOpinionConditions() {
			id := cond.GetID()
			evalResult := evalCond(cond)
			switch {
			case evalResult.IsUnevaluatable():
				unevaluatedNoOpinionConditions = append(unevaluatedNoOpinionConditions, cond)
				continue
			case evalResult.IsError():
				noOpinionErrors = append(noOpinionErrors, fmt.Errorf("condition %q with effect=NoOpinion produced error: %w", id, evalResult.Error()))
				continue
			case evalResult.IsTrue():
				reason := fmt.Sprintf("condition %q evaluated to NoOpinion", id)
				if desc := cond.GetDescription(); len(desc) != 0 {
					reason += fmt.Sprintf(" with description %q", desc)
				}
				appliedNoOpinionReasons = append(appliedNoOpinionReasons, reason)
				continue
			default: // => evalResult.IsFalse() == true
				continue
			}
		}
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
			return ConditionsAwareDecision{
				conditionsMap: ConditionsMap{
					denyConditions:      nil,
					noOpinionConditions: unevaluatedNoOpinionConditions,
					// Technically, one could greedily try evaluating the Allow conditions and whether none of them evaluate to true,
					// directly fold to NoOpinion, even though there are unevaluated NoOpinion conditions.
					allowConditions: deepCopyConditions(c.allowConditions),
				},
			}
		}
	}
	// If we got here, all Deny and NoOpinion conditions could be evaluated, and evaluated to false, nil
	if len(c.allowConditions) != 0 {
		allowErrors := []error{}
		appliedAllowReasons := []string{}
		unevaluatedAllowConditions := []Condition{}
		for cond := range c.AllowConditions() {
			id := cond.GetID()
			evalResult := evalCond(cond)
			switch {
			case evalResult.IsUnevaluatable():
				unevaluatedAllowConditions = append(unevaluatedAllowConditions, cond)
				continue
			case evalResult.IsError():
				allowErrors = append(allowErrors, fmt.Errorf("condition %q with effect=Allow produced error: %w", id, evalResult.Error()))
				continue
			case evalResult.IsTrue():
				reason := fmt.Sprintf("condition %q allowed the request", id)
				if desc := cond.GetDescription(); len(desc) != 0 {
					reason += fmt.Sprintf(" with description %q", desc)
				}
				appliedAllowReasons = append(appliedAllowReasons, reason)
				continue
			default: // => evalResult.IsFalse() == true
				continue
			}
		}
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
			return ConditionsAwareDecision{
				conditionsMap: ConditionsMap{
					denyConditions:      nil,
					noOpinionConditions: nil,
					allowConditions:     unevaluatedAllowConditions,
				},
			}
		}
	}

	// All conditions evaluated to false. This means a simple default NoOpinion.
	return ConditionsAwareDecisionNoOpinion("no conditions matched", nil)
}

func deepCopyConditions(originals []Condition) []Condition {
	copied := make([]Condition, len(originals))
	for i, original := range originals {
		copied[i] = original.DeepCopy()
	}
	return copied
}

// conditionsAwareDecisionUnionSlice is an unioned conditions-aware decision type.
// Order of the decisions matter.
type conditionsAwareDecisionUnionSlice []ConditionsAwareDecision

// FailClosedDecision returns either a Deny or NoOpinion decision to fail closed
// whenever processing a decision fails. If the decision contains one or
// more Deny decisions or conditions, one must fail closed with Deny, as that could or would
// have been the if the condition evaluation did not error. Otherwise, NoOpinion is returned.
func (unionSlice conditionsAwareDecisionUnionSlice) FailClosedDecision() Decision {
	for _, subDecision := range unionSlice {
		if subDecision.FailClosedDecision() == DecisionDeny {
			return DecisionDeny
		}
	}
	return DecisionNoOpinion
}

// ContainsAllowOrDeny returns true whether there union contains at least one
// Allow or Deny decision within the unioned decisions.
func (unionSlice conditionsAwareDecisionUnionSlice) ContainsAllowOrDeny() bool {
	for _, subDecision := range unionSlice {
		if subDecision.ContainsAllowOrDeny() {
			return true
		}
	}
	return false
}

// CanBecomeAllowed returns true if there exists some ConditionsData for which
// the unionSlice would evaluate to Allow for.
func (unionSlice conditionsAwareDecisionUnionSlice) CanBecomeAllowed() bool {
	for _, subDecision := range unionSlice {
		if subDecision.IsDenied() {
			return false
		}
		if subDecision.IsAllowed() {
			return true
		}
		if subDecision.IsConditionsMap() && subDecision.CanBecomeAllowed() {
			return true
		}
		if subDecision.IsUnion() && subDecision.CanBecomeAllowed() {
			return true
		}
	}
	return false
}

// ConditionsAwareDecisionUnion unions some amount of decisions together into a tree structure,
// where Allow/Deny/NoOpinion/ConditionsMap decisions are leafs, and Union decisions are internal
// tree nodes.
func ConditionsAwareDecisionUnion(decisions ...ConditionsAwareDecision) ConditionsAwareDecision {
	// If there are no decisions, no authorizer had any opinion about the request
	// This also ensures the invariant that a Union decision always has len(d.union) != 0.
	if len(decisions) == 0 {
		return ConditionsAwareDecisionNoOpinion("", nil)
	}

	// No need to wrap only one element
	if len(decisions) == 1 {
		// No need to wrap one Allow/Deny/NoOpinion in a union
		if decisions[0].IsUnconditional() {
			return decisions[0]
		}

		// However, ConditionsMap and Union sub-decisions must always be wrapped, such that
		// the DAG structure is preserved (the union type is an internal node, which is used
		// to route evaluation of the ConditionsMap to the right authorizer).
		return ConditionsAwareDecision{
			// Note that unconditionalDecision == 0 => Deny only if d.conditionsMap and d.union are both zero-valued
			unconditionalDecision: 0,
			union:                 decisions,
		}
	}

	// Search for the first decision that is not a NoOpinion
	onlyNoOpinion := true
	reasonlist := make([]string, 0, len(decisions))
	errlist := make([]error, 0, len(decisions))
	for i, d := range decisions {
		if d.IsNoOpinion() {
			if reason := d.Reason(); len(reason) != 0 {
				reasonlist = append(reasonlist, fmt.Sprintf("%d: %s", i, d.Reason()))
			}
			if err := d.Error(); err != nil {
				errlist = append(errlist, fmt.Errorf("%d: %w", i, err))
			}
			continue
		}
		onlyNoOpinion = false

		// If we see an Allow or Deny, and previously only saw NoOpinions, return Allow/Deny
		if d.IsAllowed() || d.IsDenied() {
			return d
		}
		// If a ConditionsMap or Union decision is the first not-NoOpinion response,
		// we cannot simplify it in any way.
		break
	}

	// If we got through this loop without setting onlyNoOpinion => false, all elements were NoOpinions
	if onlyNoOpinion {
		return ConditionsAwareDecisionNoOpinion(strings.Join(reasonlist, ", "), utilerrors.NewAggregate(errlist))
	}

	// By this we know that:
	// - There are at least two elements
	// - The first not-NoOpinion decision in the list is either Conditional or Union => at least one not-NoOpinion
	return ConditionsAwareDecision{
		union: decisions,
	}
}

// ConditionsData is an enum type for various evaluation targets conditions
// can be written against.
type ConditionsData struct {
	// AdmissionControl holds the data available during admission control.
	// Callers must verify that this is non-nil before using.
	AdmissionControl ConditionsDataAdmissionControl
}

// AdmissionOperation represents the admission operation,
// for example CREATE, UPDATE, DELETE. The constants are
// defined in k8s.io/apiserver/pkg/admission, but the
// type is defined here, because this package is more generic
// than the admission package (thus avoiding import cycles)
type AdmissionOperation string

// ConditionsDataAdmissionControl represents the data available during admission control, for conditions
// to evaluate against. This is by design a subset of admission.Attributes.
type ConditionsDataAdmissionControl interface {
	// GetName returns the name of the object as presented in the request. On a CREATE operation, the client
	// may omit name and rely on the server to generate the name. If that is the case, this method will return
	// the empty string
	GetName() string
	// GetNamespace is the namespace associated with the request (if any)
	GetNamespace() string
	// GetResource is the name of the resource being requested. This is not the kind. For example: pods
	GetResource() schema.GroupVersionResource
	// GetSubresource is the name of the subresource being requested. This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	GetSubresource() string
	// GetOperation is the operation being performed
	GetOperation() AdmissionOperation
	// GetOperationOptions is the options for the operation being performed
	GetOperationOptions() runtime.Object
	// IsDryRun indicates that modifications will definitely not be persisted for this request. This is to prevent
	// admission controllers with side effects and a method of reconciliation from being overwhelmed.
	// However, a value of false for this does not mean that the modification will be persisted, because it
	// could still be rejected by a subsequent validation step.
	IsDryRun() bool
	// GetObject is the object from the incoming request prior to default values being applied
	GetObject() runtime.Object
	// GetOldObject is the existing object. Only populated for UPDATE and DELETE requests.
	GetOldObject() runtime.Object
	// GetKind is the type of object being manipulated.  For example: Pod
	GetKind() schema.GroupVersionKind
	// GetUserInfo is information about the requesting user
	GetUserInfo() user.Info
}
