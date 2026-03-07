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
	"iter"
	"maps"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
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
// A Decision is passed by value.
// Decision equality must be checked with Decision.Equal, not reflect.DeepEqual.
type ConditionsAwareDecision struct {
	unconditionalDecision Decision

	conditionsMap ConditionsMap

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

// IsConditionsMap returns true if the decision is a conditional response
// with a map of conditions to evaluate.
func (d ConditionsAwareDecision) IsConditionsMap() bool {
	return len(d.conditionsMap.conditions) != 0
}

// IsDenied returns true if the decision is an unconditional Deny.
func (d ConditionsAwareDecision) IsDenied() bool {
	// The decision is a Deny whenever none of the other modes apply
	// NOTE: A conditional decision is encoded as
	// d.unconditionalDecision == 0 == decisionDeny && d.conditionMap.conditions != nil, so it
	// is not enough to check d.unconditionalDecision == decisionDeny
	// This is because the zero value of the struct must be a Deny
	return !d.IsAllowed() && !d.IsNoOpinion() && !d.IsConditionsMap()
}

// IsUnconditional is true if d is Allowed, Denied or NoOpinion.
func (d ConditionsAwareDecision) IsUnconditional() bool {
	return d.IsAllowed() || d.IsDenied() || d.IsNoOpinion()
}

// ConditionsMap returns the ConditionsMap, which is non-empty
// if and only if IsConditionsMap is true.
func (d ConditionsAwareDecision) ConditionsMap() ConditionsMap {
	return d.conditionsMap
}

func (d ConditionsAwareDecision) Reason() string {
	return d.reason
}

func (d ConditionsAwareDecision) Error() error {
	return d.err
}

// FailClosedDecision returns either a Deny or NoOpinion Decision to fail closed
// whenever processing a decision fails. If the decision contains one or
// more Deny conditions, the Decision must be Deny, as that could have been the
// answer if the evaluation had been successful. Otherwise, NoOpinion is returned.
func (d ConditionsAwareDecision) FailClosedDecision(err error) ConditionsAwareDecision {
	if d.IsAllowed() || d.IsNoOpinion() {
		return ConditionsAwareDecisionNoOpinion("failed closed", err)
	}
	if d.IsConditionsMap() {
		return d.conditionsMap.FailClosedDecision(err)
	}
	// => d.IsDenied() == true
	return ConditionsAwareDecisionDeny("failed closed", err)
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
	if d.IsConditionsMap() {
		return fmt.Sprintf("ConditionsMap(target=%q, type=%q, len=%d, reason=%q, err=%s)", d.conditionsMap.conditionTarget, d.conditionsMap.conditionType, len(d.conditionsMap.conditions), d.reason, errStr)
	}
	// Deny is written such that if none of the other modes apply,
	// IsDenied() is true.
	return fmt.Sprintf("Deny(%q, %s)", d.reason, errStr)
}

// Maximum limits for conditions and condition sets.
const (
	// MaxConditionsPerSet is the maximum number of conditions allowed in a single ConditionsMap.
	MaxConditionsPerSet = 128
	// MaxConditionBytes is the maximum size in bytes for a single Condition.Condition and Condition.Description string.
	MaxConditionBytes = 10240
)

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

// Validate validates that the given ConditionEffect is known to the system.
func (e ConditionEffect) Validate() error {
	if !supportedConditionEffects.Has(e) {
		return fmt.Errorf("condition effect %q not supported. Supported effects are: %v", e, slices.Sorted(maps.Keys(supportedConditionEffects)))
	}
	return nil
}

var supportedConditionEffects = sets.New(ConditionEffectDeny, ConditionEffectNoOpinion, ConditionEffectAllow)

// ConditionType represents a type of authorization conditions.
// Should be formatted as a Kubernetes label key.
// Any domain suffix of *.k8s.io or *.kubernetes.io is reserved.
type ConditionType string

func (ct ConditionType) Validate() error {
	if errs := content.IsLabelKey(string(ct)); len(errs) > 0 {
		return fmt.Errorf("invalid condition type %q: %s", ct, strings.Join(errs, "; "))
	}
	return nil
}

// Condition represents a single condition to be evaluated against ConditionsData.
// A condition is a pure, deterministic function from ConditionsData to a boolean.
type Condition struct {
	// Condition is an opaque string that represents the condition to be evaluated.
	// It is a pure, deterministic function from ConditionsData to a boolean.
	// Might or might not be human-readable. Maximum MaxConditionBytes bytes.
	Condition string

	// Effect specifies how the condition evaluating to "true" should be treated.
	Effect ConditionEffect

	// Description is an optional human-friendly description that can be shown
	// as an error message or for debugging.
	Description string
}

// validateCondition validates a single Condition.
func (cond Condition) Validate(id string) error {
	// Validate ID as a label key.
	if errs := content.IsLabelKey(id); len(errs) > 0 {
		return fmt.Errorf("invalid condition ID %q: %s", id, strings.Join(errs, "; "))
	}

	// Validate Condition strings length.
	if len(cond.Condition) == 0 {
		return fmt.Errorf("condition %q has empty Condition string", id)
	}
	if len(cond.Condition) > MaxConditionBytes {
		return fmt.Errorf("condition %q exceeds maximum length of %d bytes (saw %d bytes)", id, MaxConditionBytes, len(cond.Condition))
	}
	if len(cond.Description) > MaxConditionBytes {
		return fmt.Errorf("condition description %q exceeds maximum length of %d bytes (saw %d bytes)", id, MaxConditionBytes, len(cond.Condition))
	}

	return cond.Effect.Validate()
}

// ConditionsMap is a map of conditions of a given type, and represents
// the conditional decision from the authorizer.
// It must be constructed through DecisionConditional.
type ConditionsMap struct {
	// conditionTarget represents what data the conditions are written against.
	conditionTarget ConditionsTarget

	// conditionType is the format/encoding/language of the conditions in this set.
	// Any type starting with `k8s.io/` is reserved for Kubernetes condition types.
	// Validated as a label key.
	conditionType ConditionType

	// conditions is the set of conditions to evaluate.
	// The string ID uniquely identifies the condition within the scope of the authorizer
	// that authored the condition. Validated as a Kubernetes label key, i.e.
	// (<DNS1123 subdomain>/)[-A-Za-z0-9_.]{1,63}.
	// IDs with the 'k8s.io/' prefix are reserved for Kubernetes.
	conditions map[string]Condition
}

// Type returns the condition type (format/encoding/language) of the conditions
// in this map.
func (c ConditionsMap) Type() ConditionType {
	return c.conditionType
}

// Target returns the condition target of the conditions in this map.
func (c ConditionsMap) Target() ConditionsTarget {
	return c.conditionTarget
}

// Conditions returns all conditions in this map, sorted by ID.
func (c ConditionsMap) Conditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for _, id := range slices.Sorted(maps.Keys(c.conditions)) {
			cond := c.conditions[id]
			if !yield(id, cond) {
				return
			}
		}
	}
}

// DenyConditions returns the Deny conditions in this map, sorted by ID.
func (c ConditionsMap) DenyConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for _, id := range slices.Sorted(maps.Keys(c.conditions)) {
			cond := c.conditions[id]
			if cond.Effect != ConditionEffectDeny {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

// NoOpinionConditions returns the NoOpinion conditions in this map, sorted by ID.
func (c ConditionsMap) NoOpinionConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for _, id := range slices.Sorted(maps.Keys(c.conditions)) {
			cond := c.conditions[id]
			if cond.Effect != ConditionEffectNoOpinion {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

// AllowConditions returns the Allow conditions in this map, sorted by ID.
func (c ConditionsMap) AllowConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for _, id := range slices.Sorted(maps.Keys(c.conditions)) {
			cond := c.conditions[id]
			if cond.Effect != ConditionEffectAllow {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

// FailClosedDecision returns either a Deny or NoOpinion Decision to fail closed
// whenever evaluating a ConditionSet fails. If the ConditionSet has one or
// more Deny conditions, the Decision must be Deny, as that could have been the
// answer if the evaluation had been successful. Otherwise, NoOpinion is returned.
func (c ConditionsMap) FailClosedDecision(err error) ConditionsAwareDecision {
	for _, cond := range c.conditions {
		if cond.Effect == ConditionEffectDeny {
			return ConditionsAwareDecisionDeny("failed closed", err)
		}
	}
	return ConditionsAwareDecisionNoOpinion("failed closed", err)
}

// ConditionsAwareDecisionConditionMap creates a ConditionsMap decision. One can use maps.All to create an iterator from a map[string]Condition.
func ConditionsAwareDecisionConditionMap(conditionTarget ConditionsTarget, conditionType ConditionType, conditionsIter iter.Seq2[string, Condition], reason string, err error) ConditionsAwareDecision {
	conditionMap := map[string]Condition{}
	seenIDs := sets.New[string]()
	errlist := []error{}
	hasDenyEffect := false
	makeFailClosedError := func(err error) ConditionsAwareDecision {
		if hasDenyEffect {
			return ConditionsAwareDecisionDeny("failed closed", err)
		}
		return ConditionsAwareDecisionNoOpinion("failed closed", err)
	}
	for id, condition := range conditionsIter {
		// Fail closed if there are unknown effects
		if err := condition.Effect.Validate(); err != nil {
			return ConditionsAwareDecisionDeny("failed closed", err)
		}
		if condition.Effect == ConditionEffectDeny {
			hasDenyEffect = true
		}
		if seenIDs.Has(id) {
			errlist = append(errlist, fmt.Errorf("duplicate condition ID %q", id))
			continue
		}
		seenIDs.Insert(id)

		if err := condition.Validate(id); err != nil {
			errlist = append(errlist, err)
			continue
		}
		conditionMap[id] = condition
		// defensively stop directly when having seen too many conditions
		if len(conditionMap) > MaxConditionsPerSet {
			return ConditionsAwareDecisionDeny("failed closed", fmt.Errorf("too many conditions: %d exceeds maximum of %d", len(conditionMap), MaxConditionsPerSet))
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
	if len(conditionMap) == 0 {
		return ConditionsAwareDecisionNoOpinion("empty ConditionsMap", err)
	}

	// Do not allow constructing Conditional decisions when the feature gate is off
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
		return makeFailClosedError(fmt.Errorf("cannot construct conditional decision: the ConditionalAuthorization feature gate is disabled"))
	}

	if err := conditionTarget.Validate(); err != nil {
		return makeFailClosedError(err)
	}

	if err := conditionType.Validate(); err != nil {
		return makeFailClosedError(err)
	}

	return ConditionsAwareDecision{
		unconditionalDecision: 0,
		conditionsMap: ConditionsMap{
			conditionTarget: conditionTarget,
			conditionType:   conditionType,
			conditions:      conditionMap,
		},
		reason: reason,
		err:    err,
	}
}

// EvaluateConditionsMap evaluates the conditions in the map into a concrete Allow/Deny/NoOpinion Decision, given an
// evaluation function with a given supported condition type.
// This is a reference implementation that other conditional authorizers can use if convenient.
// The returned boolean quantifies whether the evaluation succeeded, that is, did _not_ have to fail closed
// due to a critical error. This allows the caller to take different actions depending of if evaluation was successful or not.
func EvaluateConditionsMap(conditionsMap ConditionsMap, supportedConditionTarget ConditionsTarget, supportedConditionType ConditionType, eval func(string) (bool, error)) (ConditionsAwareDecision, bool) {
	if conditionsMap.Target() != supportedConditionTarget {
		return conditionsMap.FailClosedDecision(fmt.Errorf("unsupported condition target: %q, expected: %q", conditionsMap.Target(), supportedConditionTarget)), false
	}
	if conditionsMap.Type() != supportedConditionType {
		return conditionsMap.FailClosedDecision(fmt.Errorf("unsupported condition type: %q, expected: %q", conditionsMap.Type(), supportedConditionType)), false
	}

	denyErrors := []error{}
	appliedDenyReasons := []string{}
	for id, cond := range conditionsMap.DenyConditions() {
		applies, err := eval(cond.Condition)
		if err != nil {
			denyErrors = append(denyErrors, fmt.Errorf("Deny condition %q produced error: %w", id, err))
			continue
		}
		if applies {
			reason := fmt.Sprintf("condition %q denied the request", id)
			if len(cond.Description) != 0 {
				reason += fmt.Sprintf(" with description %q", cond.Description)
			}
			appliedDenyReasons = append(appliedDenyReasons, reason)
			continue
		}
	}
	// If any deny conditions evaluated to true, return Deny
	if len(appliedDenyReasons) != 0 {
		return ConditionsAwareDecisionDeny(fmt.Sprintf("%v", appliedDenyReasons), nil), true
	}
	// If any deny errors were encountered, fail closed
	if len(denyErrors) != 0 {
		return ConditionsAwareDecisionDeny("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(denyErrors)), false
	}

	noOpinionErrors := []error{}
	appliedNoOpinionReasons := []string{}
	for id, cond := range conditionsMap.NoOpinionConditions() {
		applies, err := eval(cond.Condition)
		if err != nil {
			noOpinionErrors = append(noOpinionErrors, fmt.Errorf("NoOpinion condition %q produced error: %w", id, err))
			continue
		}
		if applies {
			reason := fmt.Sprintf("condition %q evaluated to NoOpinion", id)
			if len(cond.Description) != 0 {
				reason += fmt.Sprintf(" with description %q", cond.Description)
			}
			appliedNoOpinionReasons = append(appliedNoOpinionReasons, reason)
			continue
		}
	}
	// If any NoOpinion conditions evaluated to true, return NoOpinion
	if len(appliedNoOpinionReasons) != 0 {
		return ConditionsAwareDecisionNoOpinion(fmt.Sprintf("%v", appliedNoOpinionReasons), nil), true
	}
	// If any NoOpinion errors were encountered, fail closed to NoOpinion as if the conditions would have matched
	if len(noOpinionErrors) != 0 {
		return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(noOpinionErrors)), false
	}

	allowErrors := []error{}
	appliedAllowReasons := []string{}
	for id, cond := range conditionsMap.AllowConditions() {
		applies, err := eval(cond.Condition)
		if err != nil {
			allowErrors = append(allowErrors, fmt.Errorf("Allow condition %q produced error: %w", id, err))
			continue
		}
		if applies {
			reason := fmt.Sprintf("condition %q allowed the request", id)
			if len(cond.Description) != 0 {
				reason += fmt.Sprintf(" with description %q", cond.Description)
			}
			appliedAllowReasons = append(appliedAllowReasons, reason)
			continue
		}
	}
	// If there were at least one Allow condition that applied, then evaluation is successful, even if there
	// were some errors that happened. Those are in this case considered warnings.
	if len(appliedAllowReasons) != 0 {
		return ConditionsAwareDecisionAllow(fmt.Sprintf("%v", appliedAllowReasons), utilerrors.NewAggregate(allowErrors)), true
	}
	// However, if no Allow condition evaluated to true, but at least one errored, return that as an error to the caller
	if len(allowErrors) != 0 {
		return ConditionsAwareDecisionNoOpinion("one or more conditional evaluation errors occurred", utilerrors.NewAggregate(allowErrors)), false
	}
	// Otherwise, no condition evaluated to true, and no condition errored. This means a simple NoOpinion.
	return ConditionsAwareDecisionNoOpinion("no conditions matched", nil), true
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

// ConditionsTarget represents a target data set a condition is set to evaluate against.
type ConditionsTarget string

const (
	// ConditionsTargetAdmissionControl represents that a condition can be written against
	// the data available in admission, for example, Object and OldObject.
	ConditionsTargetAdmissionControl ConditionsTarget = "AdmissionControl"
)

// Validate validates that the given ConditionsTarget is known to the system.
func (t ConditionsTarget) Validate() error {
	if !supportedTargets.Has(t) {
		return fmt.Errorf("conditions target %q not supported. Supported targets are: %v", t, slices.Sorted(maps.Keys(supportedTargets)))
	}
	return nil
}

var supportedTargets = sets.New(ConditionsTargetAdmissionControl)

// MakeConditionsDataAdmissionControl constructs a ConditionsData struct with
// conditions data available during admission.
func MakeConditionsDataAdmissionControl(data ConditionsDataAdmissionControl) ConditionsData {
	return ConditionsData{
		target:    ConditionsTargetAdmissionControl,
		admission: data,
	}
}

// ConditionsData is an enum type for various evaluation targets conditions
// can be written against.
// This struct upholds the invariants that:
// a) Target() is always non-empty
// b) At most one of the target getters return something non-nil
type ConditionsData struct {
	target    ConditionsTarget
	admission ConditionsDataAdmissionControl
}

// Target returns the target of this data set
func (d ConditionsData) Target() ConditionsTarget {
	if len(d.target) == 0 {
		return ConditionsTargetAdmissionControl
	}
	return d.target
}

// AdmissionControl returns the admission control-related data, if set.
// Callers must verify that the return value is non-nil before using.
func (d ConditionsData) AdmissionControl() ConditionsDataAdmissionControl {
	return d.admission
}

// AdmissionOperation represents the admission operation,
// for example CREATE, UPDATE, DELETE. The constants are
// defined in k8s.io/apiserver/pkg/admission, but the
// type is defined here, because this package is more generic
// than the admission package (thus avoiding import cycles)
type AdmissionOperation string

// ConditionsDataAdmissionControl is a subset of the admission.Attributes,
// against which authorization conditions may be written.
type ConditionsDataAdmissionControl interface {
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
}
