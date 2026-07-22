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
// +k8s:supportsSubresource="/status"
type SubjectAccessReview struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	// +k8s:opaqueType
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec holds information about the request being evaluated
	Spec SubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status is filled in by the server and indicates whether the request is allowed or not
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
// +k8s:supportsSubresource="/status"
type SelfSubjectAccessReview struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	// +k8s:opaqueType
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec holds information about the request being evaluated.  user and groups must be empty
	Spec SelfSubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status is filled in by the server and indicates whether the request is allowed or not
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
// +k8s:supportsSubresource="/status"
type LocalSubjectAccessReview struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	// +k8s:opaqueType
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec holds information about the request being evaluated.  spec.namespace must be equal to the namespace
	// you made the request against.  If empty, it is defaulted.
	Spec SubjectAccessReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status is filled in by the server and indicates whether the request is allowed or not
	// +optional
	Status SubjectAccessReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourceAttributes includes the authorization attributes available for resource requests to the Authorizer interface
type ResourceAttributes struct {
	// namespace is the namespace of the action being requested.  Currently, there is no distinction between no namespace and all namespaces
	// "" (empty) is defaulted for LocalSubjectAccessReviews
	// "" (empty) is empty for cluster-scoped resources
	// "" (empty) means "all" for namespace scoped resources from a SubjectAccessReview or SelfSubjectAccessReview
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`
	// verb is a kubernetes resource API verb, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	// +optional
	Verb string `json:"verb,omitempty" protobuf:"bytes,2,opt,name=verb"`
	// group is the API Group of the Resource.  "*" means all.
	// +optional
	Group string `json:"group,omitempty" protobuf:"bytes,3,opt,name=group"`
	// version is the API Version of the Resource.  "*" means all.
	// +optional
	Version string `json:"version,omitempty" protobuf:"bytes,4,opt,name=version"`
	// resource is one of the existing resource types.  "*" means all.
	// +optional
	Resource string `json:"resource,omitempty" protobuf:"bytes,5,opt,name=resource"`
	// subresource is one of the existing resource types.  "" means none.
	// +optional
	Subresource string `json:"subresource,omitempty" protobuf:"bytes,6,opt,name=subresource"`
	// name is the name of the resource being requested for a "get" or deleted for a "delete". "" (empty) means all.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,7,opt,name=name"`
	// fieldSelector describes the limitation on access based on field.  It can only limit access, not broaden it.
	// +optional
	FieldSelector *FieldSelectorAttributes `json:"fieldSelector,omitempty" protobuf:"bytes,8,opt,name=fieldSelector"`
	// labelSelector describes the limitation on access based on labels.  It can only limit access, not broaden it.
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
	Requirements []metav1.FieldSelectorRequirement `json:"requirements,omitempty" protobuf:"bytes,2,rep,name=requirements"`
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// path is the URL path of the request
	// +optional
	Path string `json:"path,omitempty" protobuf:"bytes,1,opt,name=path"`
	// verb is the standard HTTP verb
	// +optional
	Verb string `json:"verb,omitempty" protobuf:"bytes,2,opt,name=verb"`
}

// SubjectAccessReviewSpec is a description of the access request.  Exactly one of resourceAttributes
// and nonResourceAttributes must be set
type SubjectAccessReviewSpec struct {
	// resourceAttributes describes information for a resource access request
	// +optional
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:unionMember
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty" protobuf:"bytes,1,opt,name=resourceAttributes"`
	// nonResourceAttributes describes information for a non-resource access request
	// +optional
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:unionMember
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty" protobuf:"bytes,2,opt,name=nonResourceAttributes"`

	// user is the user you're testing for.
	// If you specify "User" but not "Groups", then is it interpreted as "What if User were not a member of any groups
	// +optional
	User string `json:"user,omitempty" protobuf:"bytes,3,opt,name=user"`
	// groups is the groups you're testing for.
	// +optional
	// +listType=atomic
	Groups []string `json:"groups,omitempty" protobuf:"bytes,4,rep,name=groups"`
	// extra corresponds to the user.Info.GetExtra() method from the authenticator.  Since that is input to the authorizer
	// it needs a reflection here.
	// +optional
	Extra map[string]ExtraValue `json:"extra,omitempty" protobuf:"bytes,5,rep,name=extra"`
	// uid information about the requesting user.
	// +optional
	UID string `json:"uid,omitempty" protobuf:"bytes,6,opt,name=uid"`

	// authorizationOptions contains options for specifying the client's authorization abilities.
	// Requires the ConditionalAuthorization feature to be enabled.
	// +optional
	// +k8s:optional
	// +featureGate=ConditionalAuthorization
	// +k8s:ifDisabled("ConditionalAuthorization")=+k8s:forbidden
	AuthorizationOptions *AuthorizationOptions `json:"authorizationOptions,omitempty" protobuf:"bytes,7,opt,name=authorizationOptions"`
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type ExtraValue []string

func (t ExtraValue) String() string {
	return fmt.Sprintf("%v", []string(t))
}

// SelfSubjectAccessReviewSpec is a description of the access request.  Exactly one of resourceAttributes
// and nonResourceAttributes must be set
type SelfSubjectAccessReviewSpec struct {
	// resourceAttributes describes information for a resource access request
	// +optional
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:unionMember
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty" protobuf:"bytes,1,opt,name=resourceAttributes"`
	// nonResourceAttributes describes information for a non-resource access request
	// +optional
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:unionMember
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty" protobuf:"bytes,2,opt,name=nonResourceAttributes"`

	// authorizationOptions contains options for specifying the client's authorization abilities.
	// Requires the ConditionalAuthorization feature to be enabled.
	// +optional
	// +k8s:optional
	// +featureGate=ConditionalAuthorization
	// +k8s:ifDisabled("ConditionalAuthorization")=+k8s:forbidden
	AuthorizationOptions *AuthorizationOptions `json:"authorizationOptions,omitempty" protobuf:"bytes,3,opt,name=authorizationOptions"`
}

// SubjectAccessReviewStatus
type SubjectAccessReviewStatus struct {
	// allowed is required. True if the action would be allowed, false otherwise.
	// allowed=true is mutually exclusive with denied=true and conditionalDecision != nil.
	Allowed bool `json:"allowed" protobuf:"varint,1,opt,name=allowed"`
	// denied is optional. True if the action would be denied, otherwise false
	// If allowed is false, denied is false, and conditionalDecision is unset,
	// then the authorizer has no opinion on whether to authorize the action.
	// denied=true is mutually exclusive with allowed=true and conditionalDecision != nil.
	// +optional
	Denied bool `json:"denied,omitempty" protobuf:"varint,4,opt,name=denied"`
	// reason is optional.  It indicates why a request was allowed or denied.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,2,opt,name=reason"`
	// evaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,3,opt,name=evaluationError"`

	// conditionalDecision represents a conditional decision returned by the authorizer.
	// Mutually exclusive with allowed=true and denied=true.
	// The top-level decision type should be ConditionsAwareDecisionTypeConditionsMap or
	// ConditionsAwareDecisionTypeUnion, as Allow/Deny/NoOpinion decisions can be represented
	// with SubjectAccessReviewStatus.Allowed and SubjectAccessReviewStatus.Denied alone.
	// May only be set if spec.conditionalAuthorization is non-null.
	// Requires the ConditionalAuthorization feature to be enabled.
	// +optional
	// +k8s:optional
	// +featureGate=ConditionalAuthorization
	// +k8s:ifDisabled("ConditionalAuthorization")=+k8s:forbidden
	ConditionalDecision *ConditionsAwareDecision `json:"conditionalDecision,omitempty" protobuf:"bytes,5,opt,name=conditionalDecision"`
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
// +k8s:supportsSubresource="/status"
type SelfSubjectRulesReview struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	// +k8s:opaqueType
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec holds information about the request being evaluated.
	Spec SelfSubjectRulesReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status is filled in by the server and indicates the set of actions a user can perform.
	// +optional
	Status SubjectRulesReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// SelfSubjectRulesReviewSpec defines the specification for SelfSubjectRulesReview.
type SelfSubjectRulesReviewSpec struct {
	// namespace to evaluate rules for. Required.
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`
}

// SubjectRulesReviewStatus contains the result of a rules check. This check can be incomplete depending on
// the set of authorizers the server is configured with and any errors experienced during evaluation.
// Because authorization rules are additive, if a rule appears in a list it's safe to assume the subject has that permission,
// even if that list is incomplete.
type SubjectRulesReviewStatus struct {
	// resourceRules is the list of actions the subject is allowed to perform on resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	// +listType=atomic
	ResourceRules []ResourceRule `json:"resourceRules" protobuf:"bytes,1,rep,name=resourceRules"`
	// nonResourceRules is the list of actions the subject is allowed to perform on non-resources.
	// The list ordering isn't significant, may contain duplicates, and possibly be incomplete.
	// +listType=atomic
	NonResourceRules []NonResourceRule `json:"nonResourceRules" protobuf:"bytes,2,rep,name=nonResourceRules"`
	// incomplete is true when the rules returned by this call are incomplete. This is most commonly
	// encountered when an authorizer, such as an external authorizer, doesn't support rules evaluation.
	Incomplete bool `json:"incomplete" protobuf:"bytes,3,rep,name=incomplete"`
	// evaluationError can appear in combination with Rules. It indicates an error occurred during
	// rule evaluation, such as an authorizer that doesn't support rule evaluation, and that
	// ResourceRules and/or NonResourceRules may be incomplete.
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,4,opt,name=evaluationError"`
}

// ResourceRule is the list of actions the subject is allowed to perform on resources. The list ordering isn't significant,
// may contain duplicates, and possibly be incomplete.
type ResourceRule struct {
	// verbs is a list of kubernetes resource API verbs, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	// +listType=atomic
	Verbs []string `json:"verbs" protobuf:"bytes,1,rep,name=verbs"`

	// apiGroups is the name of the APIGroup that contains the resources.  If multiple API groups are specified, any action requested against one of
	// the enumerated resources in any API group will be allowed.  "*" means all.
	// +optional
	// +listType=atomic
	APIGroups []string `json:"apiGroups,omitempty" protobuf:"bytes,2,rep,name=apiGroups"`
	// resources is a list of resources this rule applies to.  "*" means all in the specified apiGroups.
	//  "*/foo" represents the subresource 'foo' for all resources in the specified apiGroups.
	// +optional
	// +listType=atomic
	Resources []string `json:"resources,omitempty" protobuf:"bytes,3,rep,name=resources"`
	// resourceNames is an optional white list of names that the rule applies to.  An empty set means that everything is allowed.  "*" means all.
	// +optional
	// +listType=atomic
	ResourceNames []string `json:"resourceNames,omitempty" protobuf:"bytes,4,rep,name=resourceNames"`
}

// NonResourceRule holds information that describes a rule for the non-resource
type NonResourceRule struct {
	// verbs is a list of kubernetes non-resource API verbs, like: get, post, put, delete, patch, head, options.  "*" means all.
	// +listType=atomic
	Verbs []string `json:"verbs" protobuf:"bytes,1,rep,name=verbs"`

	// nonResourceURLs is a set of partial urls that a user should have access to.  *s are allowed, but only as the full,
	// final step in the path.  "*" means all.
	// +optional
	// +listType=atomic
	NonResourceURLs []string `json:"nonResourceURLs,omitempty" protobuf:"bytes,2,rep,name=nonResourceURLs"`
}

// AuthorizationOptions contains options for specifying the client's authorization abilities.
type AuthorizationOptions struct {
	// handledDecisionTypes specifies what decision types the client can handle in the context it is in.
	// Currently valid values are:
	// - [Allow, Deny, NoOpinion] (for conditions-unaware clients) or
	// - [Allow, Deny, NoOpinion, ConditionsMap, Union] (for conditions-aware clients)
	// If the authorizer would like to return conditions, but the client does not opt in to handle those here,
	//   the authorizer must fail closed to a safe unconditional decision using ConditionsAwareDecision.FailureDecision()
	//   (Deny if any Deny conditions were present, otherwise NoOpinion).
	// Order does not matter in this slice; set semantics should be used.
	// The server should not reject unrecognized decision types (hence the k8s:opaqueType), but focus on whether the client
	// supports a mode that the server does. All clients must support "classic", conditions-unaware authorization.
	// +listType=set
	// +k8s:listType=set
	// +k8s:required
	// +required
	// +k8s:eachVal=+k8s:opaqueType
	HandledDecisionTypes []ConditionsAwareDecisionType `json:"handledDecisionTypes" protobuf:"bytes,1,rep,name=handledDecisionTypes"`
}

// Condition represents a single authorization condition to be evaluated against
// data available later in the request chain, e.g. objects available in admission.
type Condition struct {
	// id uniquely identifies this condition within the scope of the authorizer
	// that authored it and ConditionsMap it is part of. Validated as a Kubernetes label key.
	// Any domain of form *.k8s.io or *.kubernetes.io is reserved for Kubernetes use.
	// +k8s:required
	// +k8s:format=k8s-label-key
	// +required
	ID string `json:"id" protobuf:"bytes,1,opt,name=id"`

	// condition returns a string encoding of the condition to be evaluated.
	// It is a pure, deterministic function from ConditionsData to a boolean (or error).
	// Might or might not be human-readable.
	// Optional, if the ID alone is enough for the authorizer to know how to evaluate the condition.
	// +k8s:beta=+k8s:maxBytes=10240
	// +k8s:optional
	// +optional
	Condition string `json:"condition,omitempty" protobuf:"bytes,2,opt,name=condition"`

	// type describes the type of the condition, if there are multiple possibilities.
	// Should be formatted as a Kubernetes label key.
	// Any domain of form *.k8s.io or *.kubernetes.io is reserved for Kubernetes use.
	// Optional. Can be omitted if the authorizer already knows how to evaluate the condition.
	// +k8s:format=k8s-label-key
	// +k8s:optional
	// +optional
	Type string `json:"type,omitempty" protobuf:"bytes,3,opt,name=type"`

	// description is an optional human-friendly description that can be shown
	// as an error message or for debugging. Optional.
	// +k8s:beta=+k8s:maxBytes=1024
	// +k8s:optional
	// +optional
	Description string `json:"description,omitempty" protobuf:"bytes,4,opt,name=description"`
}

// ConditionsMap represents a map of conditions, keyed by ID across all conditions, across
// all effects. The ConditionsMap must have at least one Allow condition or at least one
// Deny condition. It cannot contain more than 128 conditions in total. The conditions are evaluated
// against data available later, to determine whether the authorizer that authored the conditions
// allows or denies the request.
// If all conditions in the map evaluate to false, the final decision must be NoOpinion.
type ConditionsMap struct {
	// denyConditions contains the conditions with Deny effect. If any such condition evaluates to
	// true or error, the ConditionsMap as a whole must evaluate to Deny.
	// +listType=map
	// +listMapKey=id
	// +k8s:listType=map
	// +k8s:listMapKey=id
	// +k8s:maxItems=128
	// +k8s:optional
	// +optional
	DenyConditions []Condition `json:"denyConditions" protobuf:"bytes,1,rep,name=denyConditions"`

	// noOpinionConditions contains the conditions with NoOpinion effect. If any such condition evaluates to
	// true or error, the ConditionsMap as a whole must evaluate to NoOpinion.
	// +listType=map
	// +listMapKey=id
	// +k8s:listType=map
	// +k8s:listMapKey=id
	// +k8s:maxItems=128
	// +k8s:optional
	// +optional
	NoOpinionConditions []Condition `json:"noOpinionConditions" protobuf:"bytes,2,rep,name=noOpinionConditions"`

	// allowConditions contains the conditions with Allow effect. If any such condition evaluates to
	// true, the ConditionsMap as a whole must evaluate to Allow.
	// +listType=map
	// +listMapKey=id
	// +k8s:listType=map
	// +k8s:listMapKey=id
	// +k8s:maxItems=128
	// +k8s:optional
	// +optional
	AllowConditions []Condition `json:"allowConditions" protobuf:"bytes,3,rep,name=allowConditions"`
}

// ConditionsAwareDecisionType is an enum representing what kind of authorization decision
// the ConditionsAwareDecision represents.
// +k8s:enum
type ConditionsAwareDecisionType string

const (
	// ConditionsAwareDecisionTypeDeny represents an unconditional Deny authorizer decision.
	ConditionsAwareDecisionTypeDeny ConditionsAwareDecisionType = "Deny"

	// ConditionsAwareDecisionTypeAllow represents an unconditional Allow authorizer decision.
	ConditionsAwareDecisionTypeAllow ConditionsAwareDecisionType = "Allow"

	// ConditionsAwareDecisionTypeNoOpinion represents an unconditional NoOpinion authorizer decision,
	// which means that the authorizer does not have a specific opinion on whether the request
	// should be allowed or denied, and thus can other authorizers later in the union have their say.
	ConditionsAwareDecisionTypeNoOpinion ConditionsAwareDecisionType = "NoOpinion"

	// ConditionsAwareDecisionTypeConditionsMap represents an authorizer decision that is dependent
	// on request data available later in the request chain, and thus at this stage conditional.
	ConditionsAwareDecisionTypeConditionsMap ConditionsAwareDecisionType = "ConditionsMap"

	// ConditionsAwareDecisionTypeUnion is a decision type whose final decision is computed by
	// an ordered list of sub-authorizers, with their individual decisions. A decision can thus
	// be represented as a tree, with Union decisions being internal nodes, and
	// Deny/Allow/NoOpinion/ConditionsMap decisions being leaf nodes, which are visited in depth-first order.
	ConditionsAwareDecisionTypeUnion ConditionsAwareDecisionType = "Union"
)

// ConditionsAwareDecision represents one authorizer's decision. It is an enum type,
// with variants described in ConditionsAwareDecisionType, plus a reason and error.
type ConditionsAwareDecision struct {
	// type describes the type of the decision, and acts as an enum discriminator.
	// +k8s:beta=+k8s:unionDiscriminator
	// +k8s:required
	// +required
	Type ConditionsAwareDecisionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=ConditionsAwareDecisionType"`

	// deny represents an unconditional Deny decision.
	// Must be non-null when type == "Deny", otherwise this field must be unset.
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	Deny *UnconditionalDecision `json:"deny,omitempty" protobuf:"bytes,2,opt,name=deny"`

	// noOpinion represents an unconditional NoOpinion decision.
	// Must be non-null when type == "NoOpinion", otherwise this field must be unset.
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	NoOpinion *UnconditionalDecision `json:"noOpinion,omitempty" protobuf:"bytes,3,opt,name=noOpinion"`

	// allow represents an unconditional Allow decision.
	// Must be non-null when type == "Allow", otherwise this field must be unset.
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	Allow *UnconditionalDecision `json:"allow,omitempty" protobuf:"bytes,4,opt,name=allow"`

	// conditionsMap represents a conditional decision, modelled as a map of conditions.
	// Must be non-null when type == "ConditionsMap", otherwise this field must be unset.
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	ConditionsMap *ConditionsMap `json:"conditionsMap,omitempty" protobuf:"bytes,5,opt,name=conditionsMap"`

	// union forms an ordered tree of decisions, where the union decision is represented by
	// an internal node, and all other decision types are leaf nodes. During evaluation, the
	// leaf decisions are evaluated in depth-first order, until an Allow or Deny decision is found.
	// The order of the decisions must match exactly the order of the authorizers in the union authorizer.
	// At least one of the leaves must be of type ConditionsMap, as otherwise the union could be trivially
	// reduced to just a single Allow/Deny/NoOpinion.
	//
	// Must have at least one element when type == "Union", otherwise this field must be unset.
	//
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	// +k8s:listType=map
	// +k8s:listMapKey=authorizerName
	// +listType=map
	// +listMapKey=authorizerName
	Union []NamedConditionsAwareDecision `json:"union,omitempty" protobuf:"bytes,6,rep,name=union"`
}

// NamedConditionsAwareDecision is a named ConditionsAwareDecision, returned by a unioned authorizer.
type NamedConditionsAwareDecision struct {
	// authorizerName details the name of the authorizer that authored the condition, such that
	// the right Decision can be paired with the right authorizer when evaluating the conditions,
	// even across API server replicas. The name must be stable over time.
	// This name must be unique within a given union authorizer, not necessarily globally.
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	AuthorizerName string `json:"authorizerName" protobuf:"bytes,1,opt,name=authorizerName"`

	// decision carries the inner decision returned from the authorizer.
	// +required
	Decision ConditionsAwareDecision `json:"decision" protobuf:"bytes,2,rep,name=decision"`
}

// UnconditionalDecision represents the data associated with an unconditional decision.
type UnconditionalDecision struct {
	// reason is optional. It indicates why a request was allowed or denied.
	// +k8s:optional
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,1,opt,name=reason"`

	// evaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	// +k8s:optional
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,2,opt,name=evaluationError"`
}
