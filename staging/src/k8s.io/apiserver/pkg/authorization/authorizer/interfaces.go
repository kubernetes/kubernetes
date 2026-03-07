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
	"net/http"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/user"
)

// Attributes is an interface used by an Authorizer to get information about a
// request's metadata, that is used to compute an uncondititional or conditional
// authorization decision.
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
}

// Authorizer makes an authorization decision based on information gained by making
// zero or more calls to methods of the Attributes interface. It might return
// an error together with any decision. It is up to the caller to decide whether
// that error is critical or not.
type Authorizer interface {
	// Authorize returns an unconditional decision based on the Attributes alone.
	Authorize(ctx context.Context, a Attributes) (authorized Decision, reason string, err error)

	// AuthorizeConditionsAware returns an unconditional, conditional, or unioned
	// decision, where the error and reason is part of the Decision struct.
	//
	// encodingPreference allows the caller to tell the authorizer how it prefers the conditions to be
	// encoded, e.g. in optimized or human-readable form.
	// The authorizer is free to respect or ignore this preference.
	//
	// An authorizer who is not conditions-aware MUST implement this function as
	// "return authorizer.ConditionsAwareDecisionFromParts(self.Authorize(ctx, a))",
	// such that conditions-aware callers to this authorizer get the same output
	// as if they called Authorize. Callers are only expected to call one of
	// Authorize or AuthorizeConditionsAware, not both.
	AuthorizeConditionsAware(ctx context.Context, a Attributes, encodingPreference ConditionsEncodingPreference) ConditionsAwareDecision

	// EvaluateConditions evaluates a conditional or unioned ConditionsAwareDecision against previously-unknown data,
	// and returns another ConditionsAwareDecision (with reason and error as part of the struct).
	//
	// A conditional decision may only be returned if the conditions depend on information
	// not supplied in data.
	//
	// The authorizer might make use of the builtin evaluators, in case evaluation otherwise would be expensive.
	//
	// An authorizer who does not support conditions should fail closed and return
	// ConditionsAwareDecisionDeny("", ErrorConditionEvaluationNotSupported)
	EvaluateConditions(ctx context.Context, decision ConditionsAwareDecision, data ConditionsData, builtinEvaluators BuiltinConditionsMapEvaluators) ConditionsAwareDecision
}

// BuiltinConditionsMapEvaluator provides conditions evaluation capabilities for generic
// condition types, for example, for conditions expressed using Kubernetes CEL syntax.
type BuiltinConditionsMapEvaluator interface {
	// BuiltinEvaluateConditions evaluates a conditions map given more information in ConditionData.
	//
	// The resulting Decision may be concrete (Allow/Deny/NoOpinion), or again conditional, if the
	// data in ConditionData is partial. The returned decision must not be of the union variant.
	//
	// If the builtin evaluator does not know how to evaluate the given decision, it should just
	// return nil, nil.
	BuiltinEvaluateConditions(ctx context.Context, conditionsMap ConditionsMap, data ConditionsData) (fullyEvaluatedDecision *ConditionsAwareDecision, err error)
}

var _ Authorizer = AuthorizerFunc(nil)

type AuthorizerFunc func(ctx context.Context, a Attributes) (Decision, string, error)

func (f AuthorizerFunc) Authorize(ctx context.Context, a Attributes) (Decision, string, error) {
	return f(ctx, a)
}

func (f AuthorizerFunc) AuthorizeConditionsAware(ctx context.Context, a Attributes, _ ConditionsEncodingPreference) ConditionsAwareDecision {
	return ConditionsAwareDecisionFromParts(f.Authorize(ctx, a))
}

func (f AuthorizerFunc) EvaluateConditions(_ context.Context, _ ConditionsAwareDecision, _ ConditionsData, _ BuiltinConditionsMapEvaluators) ConditionsAwareDecision {
	return ConditionsAwareDecisionDeny("", ErrorConditionEvaluationNotSupported)
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

// Decision represents an final, unconditional authorization decision.
// The zero value (0) of Decision is DecisionDeny.
type Decision int

const (
	// DecisionDeny means that an authorizer decided to deny the action.
	DecisionDeny Decision = iota
	// DecisionAllow means that an authorizer decided to allow the action.
	DecisionAllow
	// DecisionNoOpinion means that an authorizer has no opinion on whether
	// to allow or deny an action. If there are multiple unioned authorizers,
	// this means that the request can thus get allowed by some later authorizer.
	DecisionNoOpinion
)

func (d Decision) String() string {
	switch d {
	case DecisionDeny:
		return "Deny"
	case DecisionAllow:
		return "Allow"
	case DecisionNoOpinion:
		return "NoOpinion"
	default:
		return fmt.Sprintf("Unknown (%d)", int(d))
	}
}

// DecisionPartsFromConditionsAware turns a ConditionsAwareDecision into the
// triple that Authorizer.Authorize expects.
func DecisionPartsFromConditionsAware(d ConditionsAwareDecision) (Decision, string, error) {
	switch {
	case d.IsAllowed():
		return DecisionAllow, d.Reason(), d.Error()
	case d.IsDenied():
		return DecisionDeny, d.Reason(), d.Error()
	case d.IsNoOpinion():
		return DecisionNoOpinion, d.Reason(), d.Error()
	default:
		return DecisionDeny, "failed closed", fmt.Errorf("tried to return conditional decision to conditions-unaware authorizer")
	}
}
