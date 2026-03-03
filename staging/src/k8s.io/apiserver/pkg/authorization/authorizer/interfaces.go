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

package authorizer

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// Attributes is an interface used by an Authorizer to get information about a request
// that is used to make an authorization decision.
type Attributes interface {
	// GetUser returns the user.Info object to authorize
	GetUser() user.Info

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

	// GetConditionsMode returns the conditions mode that the caller has requested.
	// If the caller does not support conditions, this returns ConditionsModeNone (empty string).
	GetConditionsMode() ConditionsMode
}

type BuiltinConditionSetEvaluator interface {
	// BuiltinEvaluateConditions evaluates a condition set given more information in ConditionData.
	// The resulting Decision may be concrete (Allow/Deny/NoOpinion), or again conditional, if the
	// data in ConditionData is partial.
	// If the builtin evaluator does not know how to evaluate the given decision, it should just
	// return unevaluated, false, nil.
	// A builtin evaluator might also evaluate the decision DAG partially, in which it can return
	// evaluated != unevaluated, but fullyEvaluated == false.
	// TODO: Change all no-op implementations to just return decision, false, nil.
	BuiltinEvaluateConditions(ctx context.Context, conditionSet *ConditionSet, data ConditionData) (fullyEvaluatedDecision *Decision, err error)
}

// Authorizer makes an authorization decision based on information gained by making
// zero or more calls to methods of the Attributes interface. It might return
// an error together with any decision. It is then up to the caller to decide
// whether that error is critical or not.
type Authorizer interface {
	Authorize(ctx context.Context, a Attributes) (Decision, error)

	// TODO: Should this also return a boolean, or I guess the assumption is that fullyEvaluated == true always?
	EvaluateConditions(ctx context.Context, decision Decision, data ConditionData) (Decision, error)
}

// AuthorizerFunc implements Authorizer using a function.
// It does not support conditional authorization.
type AuthorizerFunc func(ctx context.Context, a Attributes) (Decision, error)

func (f AuthorizerFunc) Authorize(ctx context.Context, a Attributes) (Decision, error) {
	return f(ctx, a)
}

func (f AuthorizerFunc) EvaluateConditions(ctx context.Context, decision Decision, data ConditionData) (Decision, error) {
	return DecisionDeny(), ErrorConditionEvaluationNotSupported
}

// RuleResolver provides a mechanism for resolving the list of rules that apply to a given user within a namespace.
type RuleResolver interface {
	// RulesFor get the list of cluster wide rules, the list of rules in the specific namespace, incomplete status and errors.
	RulesFor(ctx context.Context, user user.Info, namespace string) ([]ResourceRuleInfo, []NonResourceRuleInfo, bool, error)
}

// RequestAttributesGetter provides a function that extracts Attributes from an http.Request
type RequestAttributesGetter interface {
	GetRequestAttributes(user.Info, *http.Request) Attributes
}

// ConditionsMode specifies how, if at all, the client wants conditions to be
// returned by the authorizer. The default (empty string) means conditions are
// not supported by the caller.
type ConditionsMode string

const (
	// ConditionsModeNone indicates that the client does not support conditions.
	ConditionsModeNone ConditionsMode = ""

	// ConditionsModeHumanReadable indicates that the client wants a
	// human-readable condition and description, if possible.
	ConditionsModeHumanReadable ConditionsMode = "HumanReadable"

	// ConditionsModeOptimized indicates that the client wants an
	// optimized conditions encoding without description, if possible.
	ConditionsModeOptimized ConditionsMode = "Optimized"
)

// AttributesRecord implements Attributes interface.
type AttributesRecord struct {
	User            user.Info
	Verb            string
	Namespace       string
	APIGroup        string
	APIVersion      string
	Resource        string
	Subresource     string
	Name            string
	ResourceRequest bool
	Path            string

	FieldSelectorRequirements fields.Requirements
	FieldSelectorParsingErr   error
	LabelSelectorRequirements labels.Requirements
	LabelSelectorParsingErr   error

	// ConditionsMode indicates how conditions should be returned. Defaults to ConditionsModeNone.
	ConditionsMode ConditionsMode
}

func (a AttributesRecord) GetUser() user.Info {
	return a.User
}

func (a AttributesRecord) GetVerb() string {
	return a.Verb
}

func (a AttributesRecord) IsReadOnly() bool {
	return a.Verb == "get" || a.Verb == "list" || a.Verb == "watch"
}

func (a AttributesRecord) GetNamespace() string {
	return a.Namespace
}

func (a AttributesRecord) GetResource() string {
	return a.Resource
}

func (a AttributesRecord) GetSubresource() string {
	return a.Subresource
}

func (a AttributesRecord) GetName() string {
	return a.Name
}

func (a AttributesRecord) GetAPIGroup() string {
	return a.APIGroup
}

func (a AttributesRecord) GetAPIVersion() string {
	return a.APIVersion
}

func (a AttributesRecord) IsResourceRequest() bool {
	return a.ResourceRequest
}

func (a AttributesRecord) GetPath() string {
	return a.Path
}

func (a AttributesRecord) GetFieldSelector() (fields.Requirements, error) {
	return a.FieldSelectorRequirements, a.FieldSelectorParsingErr
}

func (a AttributesRecord) GetLabelSelector() (labels.Requirements, error) {
	return a.LabelSelectorRequirements, a.LabelSelectorParsingErr
}

func (a AttributesRecord) GetConditionsMode() ConditionsMode {
	return a.ConditionsMode
}

// unconditionalDecision is the internal enum type backing Decision.
type unconditionalDecision int

const (
	decisionDeny unconditionalDecision = iota
	decisionAllow
	decisionNoOpinion
)

// Decision models an authorization decision. It can be Allow, Deny, or NoOpinion.
// The zero value (Decision{}) is equivalent to DecisionDeny("").
// A Decision is passed by value.
// Decision equality must be checked with Decision.Equal, not reflect.DeepEqual.
type Decision struct {
	unconditionalDecision unconditionalDecision

	conditionSet *ConditionSet

	decisionChain ConditionalDecisionChain

	// each reasons element should be a non-empty string
	reasons []string
}

// DecisionAllow constructs an Allow decision with the given reason.
func DecisionAllow(reasons ...string) Decision {
	return Decision{
		unconditionalDecision: decisionAllow,
		// on purpose nil
		conditionSet:  nil,
		decisionChain: nil,
		reasons:       nil,
	}.WithAdditionalReasons(reasons...)
}

// DecisionDeny constructs a Deny decision with the given reason.
func DecisionDeny(reasons ...string) Decision {
	return Decision{
		unconditionalDecision: decisionDeny,
		// on purpose nil
		conditionSet:  nil,
		decisionChain: nil,
		reasons:       nil,
	}.WithAdditionalReasons(reasons...)
}

// DecisionNoOpinion constructs a NoOpinion decision with the given reason.
func DecisionNoOpinion(reasons ...string) Decision {
	return Decision{
		unconditionalDecision: decisionNoOpinion,
		// on purpose nil
		conditionSet:  nil,
		decisionChain: nil,
		reasons:       nil,
	}.WithAdditionalReasons(reasons...)
}

// TODO: Should reason be encoded on the Decision struct in the SAR API?

// TODO: How to build the Decision type from the serialized SAR when one needs to provide the authorizer?
func DecisionConditional(attrs Attributes, conditionType ConditionType, conditionsIter iter.Seq2[string, Condition], reasons ...string) (Decision, error) {
	conditionSet := map[string]Condition{}
	seenIDs := sets.New[string]()
	errlist := []error{}
	failClosedError := DecisionNoOpinion()
	for id, condition := range conditionsIter {
		if condition.Effect == ConditionEffectDeny {
			failClosedError = DecisionDeny()
		}
		if seenIDs.Has(id) {
			errlist = append(errlist, fmt.Errorf("duplicate condition ID %q", id))
			continue
		}
		if err := condition.Validate(id); err != nil {
			errlist = append(errlist, err)
			continue
		}
		conditionSet[id] = condition
		// defensively stop directly when having seen too many conditions
		if len(conditionSet) > MaxConditionsPerSet {
			return DecisionDeny(), fmt.Errorf("too many conditions: %d exceeds maximum of %d", len(conditionSet), MaxConditionsPerSet)
		}
	}

	// Do not allow constructing Conditional decisions when the feature gate is off
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
		return failClosedError, fmt.Errorf("cannot construct conditional decision: the ConditionalAuthorization feature gate is disabled")
	}

	// check errors before len(conditionSet) == 0, as some errors might have made the map be empty
	// although there were items in the iterator
	if err := utilerrors.NewAggregate(errlist); err != nil {
		// the error is returned first here, not in the loop, to make sure we saw all conditions,
		// and fail closed with deny if there were any deny conditions
		return failClosedError, err
	}

	// an empty conditionset always evaluates to NoOpinion
	// ignore conditionType being invalid in this case, as it does not matter
	if len(conditionSet) == 0 {
		return DecisionNoOpinion("empty ConditionSet"), nil
	}

	if err := conditionType.Validate(); err != nil {
		return failClosedError, err
	}

	// Protect against authorizers that forget to fail closed for clients that aren't conditions-aware
	if attrs == nil || attrs.GetConditionsMode() == ConditionsModeNone {
		return failClosedError.
			WithAdditionalReasons("client does not support conditions, but authorizer tried to return a conditional response"), nil
	}

	return Decision{
		unconditionalDecision: 0,
		conditionSet: &ConditionSet{
			conditionType: conditionType,
			conditions:    conditionSet,
		},
		decisionChain: nil,
		reasons:       nil,
	}.WithAdditionalReasons(reasons...), nil
}

// ConditionalDecisionChain is an aggregate Decision type. Order of decisions matter.
type ConditionalDecisionChain []Decision

func (chain ConditionalDecisionChain) CanBecomeAllowed() bool {
	for _, subDecision := range chain {
		if subDecision.IsDenied() {
			return false
		}
		if subDecision.IsAllowed() {
			return true
		}
		if subDecision.IsConditional() && subDecision.CanBecomeAllowed() {
			return true
		}
		if subDecision.IsConditionalChain() && subDecision.CanBecomeAllowed() {
			return true
		}
	}
	return false
}

func (chain ConditionalDecisionChain) FailClosedDecision() Decision {
	for _, subDecision := range chain {
		if subDecision.FailClosedDecision().IsDenied() {
			return DecisionDeny()
		}
	}
	return DecisionNoOpinion()
}

func (chain ConditionalDecisionChain) HasConcreteResponse() bool {
	for _, subDecision := range chain {
		if subDecision.HasConcreteResponse() {
			return true
		}
	}
	return false
}

// TODO: Make sure one cannot build a cyclic graph here.
func DecisionConditionalChain(decisions ...Decision) Decision {
	if len(decisions) == 0 {
		return DecisionNoOpinion()
	}
	if len(decisions) == 1 {
		d := decisions[0]
		if d.IsAllowed() || d.IsDenied() || d.IsNoOpinion() {
			return d
		}
		// else, wrap the conditional response in a chain, so that the caller knows which
		// authorizer authored the condition
	}
	// Everything is NoOpinion => NoOpinion
	if allItems(decisions, func(d Decision) bool { return d.IsNoOpinion() }) {
		// TODO: Gather errors here
		return DecisionNoOpinion()
	}
	// There is at least one decision that is not NoOpinion. If this is the last one, return it
	if allItems(decisions[:len(decisions)-1], func(d Decision) bool { return d.IsNoOpinion() }) {
		d := decisions[len(decisions)-1] // last item of the slice is the only non-NoOpinion
		if d.IsAllowed() || d.IsDenied() || d.IsNoOpinion() {
			return d
		}
	}

	return Decision{
		unconditionalDecision: 0,
		conditionSet:          nil,
		decisionChain:         decisions,
		reasons:               nil,
	}
}

func allItems(decisions []Decision, pred func(d Decision) bool) bool {
	for _, d := range decisions {
		if !pred(d) {
			return false
		}
	}
	return true
}

// INVARIANT: Exactly one of IsAllowed, IsNoOpinion, IsConditional and IsDenied must
// always be true.

// IsAllowed returns true if the decision is Allow.
func (d Decision) IsAllowed() bool {
	return d.unconditionalDecision == decisionAllow
}

// IsNoOpinion returns true if the decision is NoOpinion.
func (d Decision) IsNoOpinion() bool {
	return d.unconditionalDecision == decisionNoOpinion
}

func (d Decision) IsConditional() bool {
	return d.conditionSet != nil
}

func (d Decision) IsConditionalChain() bool {
	return d.decisionChain != nil
}

func (d Decision) CanBecomeAllowed() bool {
	if d.IsAllowed() {
		return true
	}
	if d.IsDenied() || d.IsNoOpinion() {
		return false
	}
	if d.IsConditional() {
		return d.conditionSet.CanBecomeAllowed()
	}
	if d.IsConditionalChain() {
		return d.decisionChain.CanBecomeAllowed()
	}
	return false
}

func (d Decision) HasConcreteResponse() bool {
	if d.IsAllowed() || d.IsDenied() {
		return true
	}
	if d.IsNoOpinion() || d.IsConditional() {
		return false
	}
	return d.decisionChain.HasConcreteResponse()
}

// IsDenied returns true if the decision is Deny.
func (d Decision) IsDenied() bool {
	// The decision is a Deny whenever none of the other modes apply
	// NOTE: A Conditional decision is encoded as
	// d.unconditionalDecision == 0 == decisionDeny && d.conditionSet != nil, so it
	// is not enough to check d.unconditionalDecision == decisionDeny
	// This is because the zero value of the struct must be a Deny
	return !d.IsAllowed() && !d.IsNoOpinion() && !d.IsConditional() && !d.IsConditionalChain()
}

// IsConcrete is true if d is Allowed, Denied or NoOpinion.
func (d Decision) IsConcrete() bool {
	return d.IsAllowed() || d.IsDenied() || d.IsNoOpinion()
}

func (d Decision) FailClosedDecision() Decision {
	if d.IsConditional() {
		return d.conditionSet.FailClosedDecision()
	}
	if d.IsConditionalChain() {
		return d.decisionChain.FailClosedDecision()
	}
	return DecisionNoOpinion()
}

// Reason returns the reason string associated with this decision.
func (d Decision) Reason() string {
	if len(d.reasons) == 0 {
		return ""
	}
	if len(d.reasons) == 1 {
		return d.reasons[0]
	}
	return fmt.Sprintf("%v", d.reasons)
}

// Equal returns whether d is equal to other.
// Decision equality is defined as:
// "do the two decisions yield the same final outcome (request allowed/denied), for any input?"
// Note that this equality notion does not take reason into account.
func (d Decision) Equal(other Decision) bool {
	if d.IsAllowed() && other.IsAllowed() {
		return true
	}
	if d.IsDenied() && other.IsDenied() {
		return true
	}
	if d.IsNoOpinion() && other.IsNoOpinion() {
		return true
	}
	if d.IsConditional() && other.IsConditional() {
		return d.conditionSet.Equal(other.conditionSet)
	}
	if d.IsConditionalChain() && other.IsConditionalChain() {
		if len(d.decisionChain) != len(other.decisionChain) {
			return false
		}
		for i := range d.decisionChain {
			if !d.decisionChain[i].Equal(other.decisionChain[i]) {
				return false
			}
		}
	}
	return false
}

func (d Decision) ConditionSet() *ConditionSet {
	return d.conditionSet
}

func (d Decision) ConditionalChain() ConditionalDecisionChain {
	return d.decisionChain
}

// WithAdditionalReasons creates a new Decision with additional reasons
func (d Decision) WithAdditionalReasons(additionalReasons ...string) Decision {
	if len(additionalReasons) == 0 {
		return d
	}

	nonEmptyReasons := make([]string, 0, len(additionalReasons))
	for _, additionalReason := range additionalReasons {
		if len(additionalReason) != 0 {
			nonEmptyReasons = append(nonEmptyReasons, additionalReason)
		}
	}
	return Decision{
		unconditionalDecision: d.unconditionalDecision,
		reasons:               append(d.reasons, nonEmptyReasons...),
	}
}

// String returns a human-readable representation of the decision.
func (d Decision) String() string {
	if d.IsAllowed() {
		return "Allow"
	}
	if d.IsDenied() {
		return "Deny"
	}
	if d.IsNoOpinion() {
		return "NoOpinion"
	}
	if d.IsConditional() {
		return fmt.Sprintf("Conditional(type=%q, len=%d)", d.conditionSet.conditionType, len(d.conditionSet.conditions))
	}
	if d.IsConditionalChain() {
		subdecisionStrings := make([]string, 0, len(d.decisionChain))
		for _, subDecision := range d.decisionChain {
			subdecisionStrings = append(subdecisionStrings, subDecision.String())
		}
		return fmt.Sprintf("ConditionalChain[%s]", strings.Join(subdecisionStrings, ", "))
	}
	return "Unknown" // should never happen, according to our invariant
}
