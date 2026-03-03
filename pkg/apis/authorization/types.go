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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/apis/admission"
	"k8s.io/kubernetes/pkg/apis/authentication"
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
	// +optional
	FieldSelector *FieldSelectorAttributes
	// labelSelector describes the limitation on access based on labels.  It can only limit access, not broaden it.
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
	Requirements []metav1.FieldSelectorRequirement
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// Path is the URL path of the request
	Path string
	// Verb is the standard HTTP verb
	Verb string
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

// ConditionalAuthorizationOptions contains options for requesting conditional authorization.
type ConditionalAuthorizationOptions struct {
	// ConditionsMode specifies how conditions should be returned.
	ConditionsMode ConditionsMode
}

// SubjectAccessReviewConditionEffect specifies how a condition evaluating to
// true should be treated.
type SubjectAccessReviewConditionEffect string

const (
	// SubjectAccessReviewConditionEffectAllow means that if this condition
	// evaluates to true, the ConditionSet evaluates to Allow, unless any
	// Deny/NoOpinion condition also evaluates to true.
	SubjectAccessReviewConditionEffectAllow SubjectAccessReviewConditionEffect = "Allow"

	// SubjectAccessReviewConditionEffectDeny means that if this condition
	// evaluates to true, the ConditionSet necessarily evaluates to Deny.
	SubjectAccessReviewConditionEffectDeny SubjectAccessReviewConditionEffect = "Deny"

	// SubjectAccessReviewConditionEffectNoOpinion means that if this condition
	// evaluates to true, the given authorizer's ConditionSet cannot evaluate
	// to Allow anymore.
	SubjectAccessReviewConditionEffectNoOpinion SubjectAccessReviewConditionEffect = "NoOpinion"
)

// SubjectAccessReviewCondition represents a single condition to be evaluated
// against admission attributes.
type SubjectAccessReviewCondition struct {
	// ID uniquely identifies this condition within the scope of the authorizer.
	ID string
	// Effect specifies how the condition evaluating to "true" should be treated.
	Effect SubjectAccessReviewConditionEffect
	// Condition is an opaque string that represents the condition to be evaluated.
	Condition string
	// Description is an optional human-friendly description.
	Description string
}

// SubjectAccessReviewAuthorizationDecision represents one authorizer's decision in
// the condition set chain.
type SubjectAccessReviewAuthorizationDecision struct {
	// Allowed specifies whether this element is unconditionally allowed.
	Allowed bool
	// Denied specifies whether this element is unconditionally denied.
	Denied bool
	// ConditionsType describes the type of all conditions in the Conditions slice.
	ConditionsType string
	// Conditions is an unordered set of conditions to evaluate.
	Conditions []SubjectAccessReviewCondition
	// ConditionalDecisionChain is a nested chain for composite authorizers.
	ConditionalDecisionChain []SubjectAccessReviewAuthorizationDecision
	// Reason indicates why a request was allowed or denied by this authorizer.
	Reason string
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
	// ConditionalAuthorization contains options for requesting conditional authorization.
	ConditionalAuthorization *ConditionalAuthorizationOptions
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
	// ConditionalAuthorization contains options for requesting conditional authorization.
	ConditionalAuthorization *ConditionalAuthorizationOptions
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
	// ConditionalDecisionChain is an ordered list of condition sets, where every item
	// of the list represents one authorizer's Decision response.
	// When conditionalDecisionChain is non-null, Allowed and Denied must be false.
	ConditionalDecisionChain []SubjectAccessReviewAuthorizationDecision
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

// AuthorizationConditionsReview types

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36

// AuthorizationConditionsReview describes a request to evaluate authorization conditions.
type AuthorizationConditionsReview struct {
	metav1.TypeMeta
	// Not actually used, only here to get a lister implementation
	metav1.ObjectMeta

	// Request describes the attributes for the authorization conditions request.
	// +optional
	Request *AuthorizationConditionsRequest
	// Response describes the attributes for the authorization conditions response.
	// +optional
	Response *AuthorizationConditionsResponse
}

// AuthorizationConditionsRequest describes the authorization conditions request.
type AuthorizationConditionsRequest struct {
	Decision SubjectAccessReviewAuthorizationDecision

	WriteRequest *AuthorizationConditionsWriteRequest
}

type AuthorizationConditionsWriteRequest struct {
	// UID is an identifier for the individual request/response. It allows us to distinguish instances of requests which are
	// otherwise identical (parallel requests, requests when earlier requests did not modify etc)
	// The UID is meant to track the round trip (request/response) between the KAS and the WebHook, not the user request.
	// It is suitable for correlating log entries between the webhook and apiserver, for either auditing or debugging.
	// TODO: Does this need to be here?
	UID types.UID
	// Kind is the fully-qualified type of object being submitted (for example, v1.Pod or autoscaling.v1.Scale)
	Kind metav1.GroupVersionKind
	// Resource is the fully-qualified resource being requested (for example, v1.pods)
	Resource metav1.GroupVersionResource
	// SubResource is the subresource being requested, if any (for example, "status" or "scale")
	// +optional
	SubResource string

	// RequestKind is the fully-qualified type of the original API request (for example, v1.Pod or autoscaling.v1.Scale).
	// If this is specified and differs from the value in "kind", an equivalent match and conversion was performed.
	//
	// For example, if deployments can be modified via apps/v1 and apps/v1beta1, and a webhook registered a rule of
	// `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]` and `matchPolicy: Equivalent`,
	// an API request to apps/v1beta1 deployments would be converted and sent to the webhook
	// with `kind: {group:"apps", version:"v1", kind:"Deployment"}` (matching the rule the webhook registered for),
	// and `requestKind: {group:"apps", version:"v1beta1", kind:"Deployment"}` (indicating the kind of the original API request).
	//
	// See documentation for the "matchPolicy" field in the webhook configuration type for more details.
	// +optional
	RequestKind *metav1.GroupVersionKind
	// RequestResource is the fully-qualified resource of the original API request (for example, v1.pods).
	// If this is specified and differs from the value in "resource", an equivalent match and conversion was performed.
	//
	// For example, if deployments can be modified via apps/v1 and apps/v1beta1, and a webhook registered a rule of
	// `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]` and `matchPolicy: Equivalent`,
	// an API request to apps/v1beta1 deployments would be converted and sent to the webhook
	// with `resource: {group:"apps", version:"v1", resource:"deployments"}` (matching the resource the webhook registered for),
	// and `requestResource: {group:"apps", version:"v1beta1", resource:"deployments"}` (indicating the resource of the original API request).
	//
	// See documentation for the "matchPolicy" field in the webhook configuration type.
	// +optional
	RequestResource *metav1.GroupVersionResource
	// RequestSubResource is the name of the subresource of the original API request, if any (for example, "status" or "scale")
	// If this is specified and differs from the value in "subResource", an equivalent match and conversion was performed.
	// See documentation for the "matchPolicy" field in the webhook configuration type.
	// +optional
	RequestSubResource string

	// Name is the name of the object as presented in the request.  On a CREATE operation, the client may omit name and
	// rely on the server to generate the name.  If that is the case, this field will contain an empty string.
	// +optional
	Name string
	// Namespace is the namespace associated with the request (if any).
	// +optional
	Namespace string
	// Operation is the operation being performed. This may be different than the operation
	// requested. e.g. a patch can result in either a CREATE or UPDATE Operation.
	Operation         admission.Operation
	AuthorizationVerb string

	// UserInfo is information about the requesting user
	UserInfo authentication.UserInfo
	// Object is the object from the incoming request.
	// +optional
	Object runtime.RawExtension
	// OldObject is the existing object. Only populated for DELETE and UPDATE requests.
	// +optional
	OldObject runtime.RawExtension
	// DryRun indicates that modifications will definitely not be persisted for this request.
	// Defaults to false.
	// +optional
	DryRun *bool
	// Options is the operation option structure of the operation being performed.
	// e.g. `meta.k8s.io/v1.DeleteOptions` or `meta.k8s.io/v1.CreateOptions`. This may be
	// different than the options the caller provided. e.g. for a patch request the performed
	// Operation might be a CREATE, in which case the Options will a
	// `meta.k8s.io/v1.CreateOptions` even though the caller provided `meta.k8s.io/v1.PatchOptions`.
	// +optional
	Options runtime.RawExtension
}

// AuthorizationConditionsResponse describes an authorization conditions response.
type AuthorizationConditionsResponse struct {
	// UID is an identifier for the individual request/response.
	// This must be copied over from the corresponding AuthorizationConditionsRequest.
	// TODO: Does this need to be here?
	UID types.UID

	SubjectAccessReviewAuthorizationDecision

	// Result contains extra details into why an authorization conditions request was denied.
	// This field IS NOT consulted in any way if "Allowed" is "true".
	// +optional
	Result *metav1.Status

	// AuditAnnotations is an unstructured key value map set by remote admission controller (e.g. error=image-blacklisted).
	// MutatingAdmissionWebhook and ValidatingAdmissionWebhook admission controller will prefix the keys with
	// admission webhook name (e.g. imagepolicy.example.com/error=image-blacklisted). AuditAnnotations will be provided by
	// the admission webhook to add additional context to the audit log for this request.
	// TODO: Does this need to be here?
	// +optional
	AuditAnnotations map[string]string

	// warnings is a list of warning messages to return to the requesting API client.
	// Warning messages describe a problem the client making the API request should correct or be aware of.
	// Limit warnings to 120 characters if possible.
	// Warnings over 256 characters and large numbers of warnings may be truncated.
	// TODO: Does this need to be here?
	// +optional
	// +listType=atomic
	Warnings []string
}
