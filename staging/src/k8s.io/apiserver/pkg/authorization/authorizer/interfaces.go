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
}

// Authorizer makes an authorization decision based on information gained by making
// zero or more calls to methods of the Attributes interface. It might return
// an error together with any decision. It is then up to the caller to decide
// whether that error is critical or not.
type Authorizer interface {
	Authorize(ctx context.Context, a Attributes) (Decision, error)
}

// AuthorizerFunc implements Authorizer using a function.
type AuthorizerFunc func(ctx context.Context, a Attributes) (Decision, error)

func (f AuthorizerFunc) Authorize(ctx context.Context, a Attributes) (Decision, error) {
	return f(ctx, a)
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

// decision is the internal enum type backing Decision.
type decision int

const (
	decisionDeny decision = iota
	decisionAllow
	decisionNoOpinion
)

// Decision models an authorization decision. It can be Allow, Deny, or NoOpinion.
// The zero value is equivalent to DecisionDeny("").
// A Decision is passed by value.
// Decision equality must be checked with Decision.Equal, not reflect.DeepEqual.
type Decision struct {
	decision decision
	// each reasons element should be a non-empty string
	reasons []string
}

// DecisionAllow constructs an Allow decision with the given reason.
func DecisionAllow(reasons ...string) Decision {
	return Decision{decision: decisionAllow, reasons: nil}.WithAdditionalReasons(reasons...)
}

// DecisionDeny constructs a Deny decision with the given reason.
func DecisionDeny(reasons ...string) Decision {
	return Decision{decision: decisionDeny, reasons: nil}.WithAdditionalReasons(reasons...)
}

// DecisionNoOpinion constructs a NoOpinion decision with the given reason.
func DecisionNoOpinion(reasons ...string) Decision {
	return Decision{decision: decisionNoOpinion, reasons: nil}.WithAdditionalReasons(reasons...)
}

// IsAllowed returns true if the decision is Allow.
func (d Decision) IsAllowed() bool { return d.decision == decisionAllow }

// IsDenied returns true if the decision is Deny.
func (d Decision) IsDenied() bool { return d.decision == decisionDeny }

// IsNoOpinion returns true if the decision is NoOpinion.
func (d Decision) IsNoOpinion() bool { return d.decision == decisionNoOpinion }

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
	return d.decision == other.decision
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
		decision: d.decision,
		reasons:  append(d.reasons, nonEmptyReasons...),
	}
}

// String returns a human-readable representation of the decision.
func (d Decision) String() string {
	switch d.decision {
	case decisionDeny:
		return "Deny"
	case decisionAllow:
		return "Allow"
	case decisionNoOpinion:
		return "NoOpinion"
	default:
		return fmt.Sprintf("Unknown (%d)", int(d.decision))
	}
}
