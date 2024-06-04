/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.6

// SubjectAccessReview checks whether or not a user or group can perform an action.
type SubjectAccessReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated
	Spec SubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	// +optional
	Status SubjectAccessReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.19

// SelfSubjectAccessReview checks whether or the current user can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".  Self is a special case, because users should always be able
// to check whether they can perform an action
type SelfSubjectAccessReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated.  user and groups must be empty
	Spec SelfSubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	// +optional
	Status SubjectAccessReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +genclient
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.19

// LocalSubjectAccessReview checks whether or not a user or group can perform an action in a given namespace.
// Having a namespace scoped resource makes it much easier to grant namespace scoped policy that includes permissions
// checking.
type LocalSubjectAccessReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated.  spec.namespace must be equal to the namespace
	// you made the request against.  If empty, it is defaulted.
	Spec SubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	// +optional
	Status SubjectAccessReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourceAttributes includes the authorization attributes available for resource requests to the Authorizer interface
type ResourceAttributes struct {
	// Namespace is the namespace of the action being requested.  Currently, there is no distinction between no namespace and all namespaces
	// "" (empty) is defaulted for LocalSubjectAccessReviews
	// "" (empty) is empty for cluster-scoped resources
	// "" (empty) means "all" for namespace scoped resources from a SubjectAccessReview or SelfSubjectAccessReview
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`
	// Verb is a kubernetes resource API verb, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	// +optional
	Verb string `json:"verb,omitempty" protobuf:"bytes,2,opt,name=verb"`
	// Group is the API Group of the Resource.  "*" means all.
	// +optional
	Group string `json:"group,omitempty" protobuf:"bytes,3,opt,name=group"`
	// Version is the API Version of the Resource.  "*" means all.
	// +optional
	Version string `json:"version,omitempty" protobuf:"bytes,4,opt,name=version"`
	// Resource is one of the existing resource types.  "*" means all.
	// +optional
	Resource string `json:"resource,omitempty" protobuf:"bytes,5,opt,name=resource"`
	// Subresource is one of the existing resource types.  "" means none.
	// +optional
	Subresource string `json:"subresource,omitempty" protobuf:"bytes,6,opt,name=subresource"`
	// Name is the name of the resource being requested for a "get" or deleted for a "delete". "" (empty) means all.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,7,opt,name=name"`
	// fieldSelector describes the limitation on access based on field.  It can only limit access, not broaden it.
	//
	// This field  is alpha-level. To use this field, you must enable the
	// `AuthorizeWithSelectors` feature gate (disabled by default).
	// +optional
	FieldSelector *FieldSelectorAttributes `json:"fieldSelector,omitempty" protobuf:"bytes,8,opt,name=fieldSelector"`
	// labelSelector describes the limitation on access based on labels.  It can only limit access, not broaden it.
	//
	// This field  is alpha-level. To use this field, you must enable the
	// `AuthorizeWithSelectors` feature gate (disabled by default).
	// +optional
	LabelSelector *LabelSelectorAttributes `json:"labelSelector,omitempty" protobuf:"bytes,9,opt,name=labelSelector"`
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
	RawSelector string `json:"rawSelector,omitempty" protobuf:"bytes,1,opt,name=rawSelector"`

	// requirements is the parsed interpretation of a label selector.
	// All requirements must be met for a resource instance to match the selector.
	// Webhook implementations should handle requirements, but how to handle them is up to the webhook.
	// Since requirements can only limit the request, it is safe to authorize as unlimited request if the requirements
	// are not understood.
	// +optional
	// +listType=atomic
	Requirements []metav1.LabelSelectorRequirement `json:"requirements,omitempty" protobuf:"bytes,2,rep,name=requirements"`
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
	RawSelector string `json:"rawSelector,omitempty" protobuf:"bytes,1,opt,name=rawSelector"`

	// requirements is the parsed interpretation of a field selector.
	// All requirements must be met for a resource instance to match the selector.
	// Webhook implementations should handle requirements, but how to handle them is up to the webhook.
	// Since requirements can only limit the request, it is safe to authorize as unlimited request if the requirements
	// are not understood.
	// +optional
	// +listType=atomic
	Requirements []FieldSelectorRequirement `json:"requirements,omitempty" protobuf:"bytes,2,rep,name=requirements"`
}

type FieldSelectorRequirement struct {
	// key is the field selector key that the requirement applies to.
	Key string `json:"key" protobuf:"bytes,1,opt,name=key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators are In, NotIn, Exists, DoesNotExist
	// The list of operators may grow in the future.
	// Webhook authors are encouraged to ignore unrecognized operators and assume they don't limit the request.
	// The semantics of "all requirements are AND'd" will not change, so other requirements can continue to be enforced.
	Operator metav1.LabelSelectorOperator `json:"operator" protobuf:"bytes,2,opt,name=operator,casttype=LabelSelectorOperator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty.
	// +optional
	// +listType=atomic
	Values []string `json:"values,omitempty" protobuf:"bytes,3,rep,name=values"`
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// Path is the URL path of the request
	// +optional
	Path string `json:"path,omitempty" protobuf:"bytes,1,opt,name=path"`
	// Verb is the standard HTTP verb
	// +optional
	Verb string `json:"verb,omitempty" protobuf:"bytes,2,opt,name=verb"`
}

// SubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAuthorizationAttributes
// and NonResourceAuthorizationAttributes must be set
type SubjectAccessReviewSpec struct {
	// ResourceAuthorizationAttributes describes information for a resource access request
	// +optional
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty" protobuf:"bytes,1,opt,name=resourceAttributes"`
	// NonResourceAttributes describes information for a non-resource access request
	// +optional
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty" protobuf:"bytes,2,opt,name=nonResourceAttributes"`

	// User is the user you're testing for.
	// If you specify "User" but not "Groups", then is it interpreted as "What if User were not a member of any groups
	// +optional
	User string `json:"user,omitempty" protobuf:"bytes,3,opt,name=user"`
	// Groups is the groups you're testing for.
	// +optional
	// +listType=atomic
	Groups []string `json:"groups,omitempty" protobuf:"bytes,4,rep,name=groups"`
	// Extra corresponds to the user.Info.GetExtra() method from the authenticator.  Since that is input to the authorizer
	// it needs a reflection here.
	// +optional
	Extra map[string]ExtraValue `json:"extra,omitempty" protobuf:"bytes,5,rep,name=extra"`
	// UID information about the requesting user.
	// +optional
	UID string `json:"uid,omitempty" protobuf:"bytes,6,opt,name=uid"`
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type ExtraValue []string

func (t ExtraValue) String() string {
	return fmt.Sprintf("%v", []string(t))
}

// SelfSubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAuthorizationAttributes
// and NonResourceAuthorizationAttributes must be set
type SelfSubjectAccessReviewSpec struct {
	// ResourceAuthorizationAttributes describes information for a resource access request
	// +optional
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty" protobuf:"bytes,1,opt,name=resourceAttributes"`
	// NonResourceAttributes describes information for a non-resource access request
	// +optional
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty" protobuf:"bytes,2,opt,name=nonResourceAttributes"`
}

// SubjectAccessReviewStatus
type SubjectAccessReviewStatus struct {
	// Allowed is required. True if the action would be allowed, false otherwise.
	Allowed bool `json:"allowed" protobuf:"varint,1,opt,name=allowed"`
	// Denied is optional. True if the action would be denied, otherwise
	// false. If both allowed is false and denied is false, then the
	// authorizer has no opinion on whether to authorize the action. Denied
	// may not be true if Allowed is true.
	// +optional
	Denied bool `json:"denied,omitempty" protobuf:"varint,4,opt,name=denied"`
	// Reason is optional.  It indicates why a request was allowed or denied.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,2,opt,name=reason"`
	// EvaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,3,opt,name=evaluationError"`
}

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.19

// SelfSubjectRulesReview enumerates the set of actions the current user can perform within a namespace.
// The returned list of actions may be incomplete depending on the server's authorization mode,
// and any errors experienced during the evaluation. SelfSubjectRulesReview should be used by UIs to show/hide actions,
// or to quickly let an end user reason about their permissions. It should NOT Be used by external systems to
// drive authorization decisions as this raises confused deputy, cache lifetime/revocation, and correctness concerns.
// SubjectAccessReview, and LocalAccessReview are the correct way to defer authorization decisions to the API server.
type SelfSubjectRulesReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated.
	Spec SelfSubjectRulesReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates the set of actions a user can perform.
	// +optional
	Status SubjectRulesReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// SelfSubjectRulesReviewSpec defines the specification for SelfSubjectRulesReview.
type SelfSubjectRulesReviewSpec struct {
	// Namespace to evaluate rules for. Required.
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`
}

// SubjectRulesReviewStatus contains the result of a rules check. This check can be incomplete depending on
// the set of authorizers the server is configured with and any errors experienced during evaluation.
// Because authorization rules are additive, if a rule appears in a list it's safe to assume the subject has that permission,
// even if that list is incomplete.
type SubjectRulesReviewStatus struct {
	// ResourceRules is the list of actions the subject is allowed to perform on resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	// +listType=atomic
	ResourceRules []ResourceRule `json:"resourceRules" protobuf:"bytes,1,rep,name=resourceRules"`
	// NonResourceRules is the list of actions the subject is allowed to perform on non-resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	// +listType=atomic
	NonResourceRules []NonResourceRule `json:"nonResourceRules" protobuf:"bytes,2,rep,name=nonResourceRules"`
	// Incomplete is true when the rules returned by this call are incomplete. This is most commonly
	// encountered when an authorizer, such as an external authorizer, doesn't support rules evaluation.
	Incomplete bool `json:"incomplete" protobuf:"bytes,3,rep,name=incomplete"`
	// EvaluationError can appear in combination with Rules. It indicates an error occurred during
	// rule evaluation, such as an authorizer that doesn't support rule evaluation, and that
	// ResourceRules and/or NonResourceRules may be incomplete.
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,4,opt,name=evaluationError"`
}

// ResourceRule is the list of actions the subject is allowed to perform on resources. The list ordering isn't significant,
// may contain duplicates, and possibly be incomplete.
type ResourceRule struct {
	// Verb is a list of kubernetes resource API verbs, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	// +listType=atomic
	Verbs []string `json:"verbs" protobuf:"bytes,1,rep,name=verbs"`

	// APIGroups is the name of the APIGroup that contains the resources.  If multiple API groups are specified, any action requested against one of
	// the enumerated resources in any API group will be allowed.  "*" means all.
	// +optional
	// +listType=atomic
	APIGroups []string `json:"apiGroups,omitempty" protobuf:"bytes,2,rep,name=apiGroups"`
	// Resources is a list of resources this rule applies to.  "*" means all in the specified apiGroups.
	//  "*/foo" represents the subresource 'foo' for all resources in the specified apiGroups.
	// +optional
	// +listType=atomic
	Resources []string `json:"resources,omitempty" protobuf:"bytes,3,rep,name=resources"`
	// ResourceNames is an optional white list of names that the rule applies to.  An empty set means that everything is allowed.  "*" means all.
	// +optional
	// +listType=atomic
	ResourceNames []string `json:"resourceNames,omitempty" protobuf:"bytes,4,rep,name=resourceNames"`
}

// NonResourceRule holds information that describes a rule for the non-resource
type NonResourceRule struct {
	// Verb is a list of kubernetes non-resource API verbs, like: get, post, put, delete, patch, head, options.  "*" means all.
	// +listType=atomic
	Verbs []string `json:"verbs" protobuf:"bytes,1,rep,name=verbs"`

	// NonResourceURLs is a set of partial urls that a user should have access to.  *s are allowed, but only as the full,
	// final step in the path.  "*" means all.
	// +optional
	// +listType=atomic
	NonResourceURLs []string `json:"nonResourceURLs,omitempty" protobuf:"bytes,2,rep,name=nonResourceURLs"`
}
