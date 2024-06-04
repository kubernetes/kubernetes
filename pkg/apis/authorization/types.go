/*
Copyright 2015 The Kubernetes Authors.

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

package authorization

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SubjectAccessReview checks whether or not a user or group can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".
type SubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated
	Spec SubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SelfSubjectAccessReview checks whether or the current user can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".  Self is a special case, because users should always be able
// to check whether they can perform an action
type SelfSubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated.
	Spec SelfSubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LocalSubjectAccessReview checks whether or not a user or group can perform an action in a given namespace.
// Having a namespace scoped resource makes it much easier to grant namespace scoped policy that includes permissions
// checking.
type LocalSubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated.  spec.namespace must be equal to the namespace
	// you made the request against.  If empty, it is defaulted.
	Spec SubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// ResourceAttributes includes the authorization attributes available for resource requests to the Authorizer interface
type ResourceAttributes struct {
	// Namespace is the namespace of the action being requested.  Currently, there is no distinction between no namespace and all namespaces
	// "" (empty) is defaulted for LocalSubjectAccessReviews
	// "" (empty) is empty for cluster-scoped resources
	// "" (empty) means "all" for namespace scoped resources from a SubjectAccessReview or SelfSubjectAccessReview
	Namespace string
	// Verb is a kubernetes resource API verb, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	Verb string
	// Group is the API Group of the Resource.  "*" means all.
	Group string
	// Version is the API Version of the Resource.  "*" means all.
	Version string
	// Resource is one of the existing resource types.  "*" means all.
	Resource string
	// Subresource is one of the existing resource types.  "" means none.
	Subresource string
	// Name is the name of the resource being requested for a "get" or deleted for a "delete". "" (empty) means all.
	Name string
	// fieldSelector describes the limitation on access based on field.  It can only limit access, not broaden it.
	//
	// This field  is alpha-level. To use this field, you must enable the
	// `AuthorizeWithSelectors` feature gate (disabled by default).
	// +optional
	FieldSelector *FieldSelectorAttributes
	// labelSelector describes the limitation on access based on labels.  It can only limit access, not broaden it.
	//
	// This field  is alpha-level. To use this field, you must enable the
	// `AuthorizeWithSelectors` feature gate (disabled by default).
	// +optional
	LabelSelector *LabelSelectorAttributes
}

// LabelSelectorAttributes indicates a label limited access.
// Webhook authors are encouraged to
// * ensure rawSelector and requirements are not both set
// * consider the requirements field if set
// * not try to parse or consider the rawSelector field if set. This is to avoid another CVE-2022-2880 (i.e. getting different systems to agree on how exactly to parse a query is not something we want), see https://www.oxeye.io/resources/golang-parameter-smuggling-attack for more details.
// For the *SubjectAccessReview endpoints of the kube-apiserver:
// * If rawSelector is empty and requirements are empty, the request is not limited.
// * If rawSelector is present and requirements are empty, the rawSelector will be parsed and limited if the parsing succeeds.
// * If rawSelector is empty and requirements are present, the requirements should be honored
// * If rawSelector is present and requirements are present, the request is invalid.
type LabelSelectorAttributes struct {
	// rawSelector is the serialization of a field selector that would be included in a query parameter.
	// Webhook implementations are encouraged to ignore rawSelector.
	// The kube-apiserver's *SubjectAccessReview will parse the rawSelector as long as the requirements are not present.
	// +optional
	RawSelector string

	// requirements is the parsed interpretation of a label selector.
	// All requirements must be met for a resource instance to match the selector.
	// Webhook implementations should handle requirements, but how to handle them is up to the webhook.
	// Since requirements can only limit the request, it is safe to authorize as unlimited request if the requirements
	// are not understood.
	// +optional
	// +listType=atomic
	Requirements []metav1.LabelSelectorRequirement
}

// FieldSelectorAttributes indicates a field limited access.
// Webhook authors are encouraged to
// * ensure rawSelector and requirements are not both set
// * consider the requirements field if set
// * not try to parse or consider the rawSelector field if set. This is to avoid another CVE-2022-2880 (i.e. getting different systems to agree on how exactly to parse a query is not something we want), see https://www.oxeye.io/resources/golang-parameter-smuggling-attack for more details.
// For the *SubjectAccessReview endpoints of the kube-apiserver:
// * If rawSelector is empty and requirements are empty, the request is not limited.
// * If rawSelector is present and requirements are empty, the rawSelector will be parsed and limited if the parsing succeeds.
// * If rawSelector is empty and requirements are present, the requirements should be honored
// * If rawSelector is present and requirements are present, the request is invalid.
type FieldSelectorAttributes struct {
	// rawSelector is the serialization of a field selector that would be included in a query parameter.
	// Webhook implementations are encouraged to ignore rawSelector.
	// The kube-apiserver's *SubjectAccessReview will parse the rawSelector as long as the requirements are not present.
	// +optional
	RawSelector string

	// requirements is the parsed interpretation of a field selector.
	// All requirements must be met for a resource instance to match the selector.
	// Webhook implementations should handle requirements, but how to handle them is up to the webhook.
	// Since requirements can only limit the request, it is safe to authorize as unlimited request if the requirements
	// are not understood.
	// +optional
	// +listType=atomic
	Requirements []FieldSelectorRequirement
}

type FieldSelectorRequirement struct {
	// key is the field selector key that the requirement applies to.
	Key string
	// operator represents a key's relationship to a set of values.
	// Valid operators are In, NotIn, Exists, DoesNotExist
	// The list of operators may grow in the future.
	// Webhook authors are encouraged to ignore unrecognized operators and assume they don't limit the request.
	// The semantics of "all requirements are AND'd" will not change, so other requirements can continue to be enforced.
	Operator metav1.LabelSelectorOperator
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty.
	// +optional
	// +listType=atomic
	Values []string
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// Path is the URL path of the request
	Path string
	// Verb is the standard HTTP verb
	Verb string
}

// SubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAttributes
// and NonResourceAttributes must be set
type SubjectAccessReviewSpec struct {
	// ResourceAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes

	// User is the user you're testing for.
	// If you specify "User" but not "Group", then is it interpreted as "What if User were not a member of any groups
	User string
	// Groups is the groups you're testing for.
	Groups []string
	// Extra corresponds to the user.Info.GetExtra() method from the authenticator.  Since that is input to the authorizer
	// it needs a reflection here.
	Extra map[string]ExtraValue
	// UID information about the requesting user.
	UID string
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
type ExtraValue []string

// SelfSubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAttributes
// and NonResourceAttributes must be set
type SelfSubjectAccessReviewSpec struct {
	// ResourceAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes
}

// SubjectAccessReviewStatus represents the current state of a SubjectAccessReview.
type SubjectAccessReviewStatus struct {
	// Allowed is required. True if the action would be allowed, false otherwise.
	Allowed bool
	// Denied is optional. True if the action would be denied, otherwise
	// false. If both allowed is false and denied is false, then the
	// authorizer has no opinion on whether to authorize the action. Denied
	// may not be true if Allowed is true.
	Denied bool
	// Reason is optional.  It indicates why a request was allowed or denied.
	Reason string
	// EvaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	EvaluationError string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SelfSubjectRulesReview enumerates the set of actions the current user can perform within a namespace.
// The returned list of actions may be incomplete depending on the server's authorization mode,
// and any errors experienced during the evaluation. SelfSubjectRulesReview should be used by UIs to show/hide actions,
// or to quickly let an end user reason about their permissions. It should NOT Be used by external systems to
// drive authorization decisions as this raises confused deputy, cache lifetime/revocation, and correctness concerns.
// SubjectAccessReview, and LocalAccessReview are the correct way to defer authorization decisions to the API server.
type SelfSubjectRulesReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated.
	Spec SelfSubjectRulesReviewSpec

	// Status is filled in by the server and indicates the set of actions a user can perform.
	Status SubjectRulesReviewStatus
}

// SelfSubjectRulesReviewSpec defines the specification for SelfSubjectRulesReview.
type SelfSubjectRulesReviewSpec struct {
	// Namespace to evaluate rules for. Required.
	Namespace string
}

// SubjectRulesReviewStatus contains the result of a rules check. This check can be incomplete depending on
// the set of authorizers the server is configured with and any errors experienced during evaluation.
// Because authorization rules are additive, if a rule appears in a list it's safe to assume the subject has that permission,
// even if that list is incomplete.
type SubjectRulesReviewStatus struct {
	// ResourceRules is the list of actions the subject is allowed to perform on resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	ResourceRules []ResourceRule
	// NonResourceRules is the list of actions the subject is allowed to perform on non-resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	NonResourceRules []NonResourceRule
	// Incomplete is true when the rules returned by this call are incomplete. This is most commonly
	// encountered when an authorizer, such as an external authorizer, doesn't support rules evaluation.
	Incomplete bool
	// EvaluationError can appear in combination with Rules. It indicates an error occurred during
	// rule evaluation, such as an authorizer that doesn't support rule evaluation, and that
	// ResourceRules and/or NonResourceRules may be incomplete.
	EvaluationError string
}

// ResourceRule is the list of actions the subject is allowed to perform on resources. The list ordering isn't significant,
// may contain duplicates, and possibly be incomplete.
type ResourceRule struct {
	// Verb is a list of kubernetes resource API verbs, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	Verbs []string
	// APIGroups is the name of the APIGroup that contains the resources.  If multiple API groups are specified, any action requested against one of
	// the enumerated resources in any API group will be allowed.  "*" means all.
	APIGroups []string
	// Resources is a list of resources this rule applies to.  "*" means all in the specified apiGroups.
	//  "*/foo" represents the subresource 'foo' for all resources in the specified apiGroups.
	Resources []string
	// ResourceNames is an optional white list of names that the rule applies to.  An empty set means that everything is allowed.  "*" means all.
	ResourceNames []string
}

// NonResourceRule holds information that describes a rule for the non-resource
type NonResourceRule struct {
	// Verb is a list of kubernetes non-resource API verbs, like: get, post, put, delete, patch, head, options.  "*" means all.
	Verbs []string

	// NonResourceURLs is a set of partial urls that a user should have access to.  *s are allowed, but only as the full,
	// final step in the path.  "*" means all.
	NonResourceURLs []string
}
