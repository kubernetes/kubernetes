/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// ErrorConditionEvaluationNotSupported is returned by authorizer implementations
// that do not support condition evaluation.
var ErrorConditionEvaluationNotSupported = errors.New("condition evaluation not supported")

// Maximum limits for conditions and condition sets.
const (
	// MaxConditionsPerSet is the maximum number of conditions allowed in a single ConditionSet.
	MaxConditionsPerSet = 128
	// MaxConditionBytes is the maximum size in bytes for a single Condition.Condition and Condition.Description string.
	MaxConditionBytes = 10240
)

// ConditionEffect specifies how a condition evaluating to true should be handled.
type ConditionEffect string

const (
	// ConditionEffectDeny means that if this condition evaluates to true,
	// the ConditionSet necessarily evaluates to Deny. No further authorizers
	// are consulted.
	ConditionEffectDeny ConditionEffect = "Deny"

	// ConditionEffectNoOpinion means that if this condition evaluates to true,
	// the given authorizer's ConditionSet cannot evaluate to Allow anymore, but
	// necessarily Deny or NoOpinion, depending on whether there are any true
	// EffectDeny conditions.
	// However, later authorizers in the chain can still Allow or Deny.
	// It is effectively a softer deny that just overrides the authorizer's own
	// allow policies.
	ConditionEffectNoOpinion ConditionEffect = "NoOpinion"

	// ConditionEffectAllow means that if this condition evaluates to true,
	// the ConditionSet evaluates to Allow, unless any Deny/NoOpinion condition
	// also evaluates to true (in which case the Deny/NoOpinion conditions have
	// precedence).
	ConditionEffectAllow ConditionEffect = "Allow"
)

// Condition represents a single condition to be evaluated against ConditionData.
// A condition is a pure, deterministic function from ConditionData to a Boolean.
type Condition struct {
	// Condition is an opaque string that represents the condition to be evaluated.
	// It is a pure, deterministic function from ConditionData to a Boolean.
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

	// Validate Effect.
	switch cond.Effect {
	case ConditionEffectAllow, ConditionEffectDeny, ConditionEffectNoOpinion:
		// valid
	default:
		return fmt.Errorf("condition %q has invalid effect %q", id, cond.Effect)
	}

	return nil
}

// ConditionSet represents a conditional response from an authorizer.
// It must be constructed through DecisionConditional.
type ConditionSet struct {
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
// in this set.
func (c *ConditionSet) Type() ConditionType {
	return c.conditionType
}

// Conditions returns the conditions in this set. The returned slice must not be
// modified.
func (c *ConditionSet) Conditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for id, cond := range c.conditions {
			if !yield(id, cond) {
				return
			}
		}
	}
}

func (c *ConditionSet) DenyConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for id, cond := range c.conditions {
			if cond.Effect != ConditionEffectDeny {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

func (c *ConditionSet) NoOpinionConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for id, cond := range c.conditions {
			if cond.Effect != ConditionEffectNoOpinion {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

func (c *ConditionSet) AllowConditions() iter.Seq2[string, Condition] {
	return func(yield func(string, Condition) bool) {
		for id, cond := range c.conditions {
			if cond.Effect != ConditionEffectAllow {
				continue
			}
			if !yield(id, cond) {
				return
			}
		}
	}
}

func (c *ConditionSet) Equal(other *ConditionSet) bool {
	if c == nil || other == nil {
		// if both are nil => true
		// if c nil, but other non-nil => false
		// if c non-nil, but other nil => false
		return (c == nil) == (other == nil)
	}
	// both non-nil
	return false // TODO implement semantic equivalence
}

// CanBecomeAllowed returns true if this ConditionSet has at least one
// effect=Allow condition, or wraps an unconditional Allow decision.
func (c *ConditionSet) CanBecomeAllowed() bool {
	for _, cond := range c.conditions {
		if cond.Effect == ConditionEffectAllow {
			return true
		}
	}
	return false
}

// FailClosedDecision returns either a Deny or NoOpinion Decision to fail closed
// whenever evaluating a ConditionSet fails. If the ConditionSet has one or
// more Deny conditions, the Decision must be Deny, as that could have been the
// answer if the evaluation had been successful. Otherwise, NoOpinion is returned.
func (c *ConditionSet) FailClosedDecision() Decision {
	for _, cond := range c.conditions {
		if cond.Effect == ConditionEffectDeny {
			return DecisionDeny()
		}
	}
	return DecisionNoOpinion()
}

// EvaluateConditionSet evaluates the conditions in the set into a concrete Allow/Deny/NoOpinion Decision, given an
// evaluation function with a given supported condition type.
// This is a reference implementation that other conditional authorizers can use if convenient.
func EvaluateConditionSet(conditionSet *ConditionSet, supportedConditionType ConditionType, eval func(string) (bool, error)) (Decision, []error, error) {
	if conditionSet == nil {
		return DecisionNoOpinion(), nil, nil
	}

	if conditionSet.Type() != supportedConditionType {
		return conditionSet.FailClosedDecision(), nil, fmt.Errorf("unsupported condition type: %q", conditionSet.Type())
	}

	denyErrors := []error{}
	appliedDenyReasons := []string{}
	for id, cond := range conditionSet.DenyConditions() {
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
	// If any deny errors were encountered, fail closed
	if len(denyErrors) != 0 {
		return DecisionDeny("one or more conditional evaluation errors occurred"), nil, utilerrors.NewAggregate(denyErrors)
	}
	// If any deny conditions evaluated to true, return Deny
	if len(appliedDenyReasons) != 0 {
		return DecisionDeny(appliedDenyReasons...), nil, nil
	}

	noOpinionErrors := []error{}
	appliedNoOpinionReasons := []string{}
	for id, cond := range conditionSet.NoOpinionConditions() {
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
	// If any NoOpinion errors were encountered, fail closed to NoOpinion as if the conditions would have matched
	if len(noOpinionErrors) != 0 {
		return DecisionNoOpinion("one or more conditional evaluation errors occurred"), nil, utilerrors.NewAggregate(noOpinionErrors)
	}
	// If any NoOpinion conditions evaluated to true, return NoOpinion
	if len(appliedNoOpinionReasons) != 0 {
		return DecisionNoOpinion(appliedNoOpinionReasons...), nil, nil
	}

	allowErrors := []error{}
	appliedAllowReasons := []string{}
	for id, cond := range conditionSet.AllowConditions() {
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
	// If there were at least one Allow condition that applied, ignore any evaluation errors, return those as
	// non-critical warnings.
	if len(appliedAllowReasons) != 0 {
		return DecisionAllow(appliedAllowReasons...), allowErrors, nil
	}
	// However, if no Allow condition evaluated to true, but at least one errored, return that as an error to the caller
	if len(allowErrors) != 0 {
		return DecisionNoOpinion("one or more conditional evaluation errors occurred"), nil, utilerrors.NewAggregate(allowErrors)
	}
	// Otherwise, no condition evaluated to true, and no condition errored. This means a simple NoOpinion.
	return DecisionNoOpinion("no conditions matched"), nil, nil
}

// ConditionType represents a type of authorization conditions
// Should be formatted as a Kubernetes label.
// Any domain suffix of *.k8s.io or *.kubernetes.io is reserved
// TODO: Should this be a private struct that must be instantiated through the constructor
type ConditionType string

func (ct ConditionType) Validate() error {
	if errs := content.IsLabelKey(string(ct)); len(errs) > 0 {
		return fmt.Errorf("invalid condition type %q: %s", ct, strings.Join(errs, "; "))
	}
	return nil
}

// BuiltinConditionSetEvaluator is a ConditionSetEvaluator that can evaluate
// conditions of a specific type in-process, without requiring a webhook call.
/*type BuiltinConditionSetEvaluator interface {
	// BuiltinEvaluateConditions evaluates a decision given more information in ConditionData.
	// If the evaluator does not know how to evaluate the given Decision, it MUST return the
	// same decision as return decision, nil. TODO: add a boolean?
	// The resulting Decision may be concrete (Allow/Deny/NoOpinion), or again conditional, if the
	// data in ConditionData is partial.
	BuiltinEvaluateConditions(ctx context.Context, decision Decision, data ConditionData) (Decision, error)

	// SupportedConditionTypes returns the condition type that this evaluator
	// supports (e.g. "k8s.io/cel").
	// SupportedConditionTypes() sets.Set[ConditionType]
}*/

// TODO: Provide a recursive decision (de)serialization function, to avoid duplication?

// ConditionData provides the data that was unknown at authorization time but
// is now available for condition evaluation. This includes the request object,
// the old object (for updates/deletes), the operation, and options.
// All top-level getters are mutually exclusive with each other.
type ConditionData interface {
	// WriteRequest provides data for a condition that is targeting a normal write request
	// (verbs=create, update, patch, delete, deletecollection or a connect request).
	// Evaluating a ConditionSet against WriteRequest must return a concrete decision (Allow/Deny/NoOpinion).
	WriteRequest() WriteRequestConditionData

	// ImpersonationRequest provides data known at the time of impersonation. Evaluating a condition
	// against the data of ImpersonationRequest can result in a concrete decision (Allow/Deny/NoOpinion)
	// or another conditional decision, with conditions written against WriteRequest.
	ImpersonationRequest() ImpersonationRequestConditionData
}

type WriteRequestConditionData interface {
	// GetOperation returns the operation being performed (e.g. "CREATE", "UPDATE",
	// "DELETE", "CONNECT").
	GetOperation() string

	// GetOperationOptions returns the options for the operation being performed.
	GetOperationOptions() runtime.Object

	// GetObject returns the object from the incoming request prior to default
	// values being applied. Only populated for CREATE and UPDATE requests.
	// If the object of a code type (e.g. a Pod), it should be given in its internal
	// form, that is, *api.Pod. Otherwise, the object should be *unstructured.Unstructured.
	// TODO(luxas): Verify that CRs are *unstructured.Unstructured in admission.
	GetObject() runtime.Object

	// GetOldObject returns the existing object. Only populated for UPDATE and
	// DELETE requests.
	// If the object of a code type (e.g. a Pod), it should be given in its internal
	// form, that is, *api.Pod. Otherwise, the object should be *unstructured.Unstructured.
	GetOldObject() runtime.Object
}

type ImpersonationRequestConditionData interface {
	// GetVerb returns the kube verb associated with API requests (this includes get, list, watch, create, update, patch, delete, deletecollection, and proxy),
	// or the lowercased HTTP verb associated with non-API requests (this includes get, put, post, patch, and delete)
	GetVerb() string

	// When IsReadOnly() == true, the request has no side effects, other than
	// caching, logging, and other incidentals.
	IsReadOnly() bool

	// The namespace of the object, if a request is for a REST object.
	GetNamespace() string

	// The kind of object, if a request is for a REST object.
	GetResource() string

	// GetSubresource returns the subresource being requested, if present
	GetSubresource() string

	// GetName returns the name of the object as parsed off the request.  This will not be present for all request types, but
	// will be present for: get, update, delete
	GetName() string

	// The group of the resource, if a request is for a REST object.
	GetAPIGroup() string

	// GetAPIVersion returns the version of the group requested, if a request is for a REST object.
	GetAPIVersion() string

	// IsResourceRequest returns true for requests to API resources, like /api/v1/nodes,
	// and false for non-resource endpoints like /api, /healthz
	IsResourceRequest() bool

	// GetPath returns the path of the request
	GetPath() string

	// ParseFieldSelector is lazy, thread-safe, and stores the parsed result and error.
	// It returns an error if the field selector cannot be parsed.
	// The returned requirements must be treated as readonly and not modified.
	GetFieldSelector() (fields.Requirements, error)

	// ParseLabelSelector is lazy, thread-safe, and stores the parsed result and error.
	// It returns an error if the label selector cannot be parsed.
	// The returned requirements must be treated as readonly and not modified.
	GetLabelSelector() (labels.Requirements, error)
}
