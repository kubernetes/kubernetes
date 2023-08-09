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

package admissionregistration

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Rule is a tuple of APIGroups, APIVersion, and Resources.It is recommended
// to make sure that all the tuple expansions are valid.
type Rule struct {
	// APIGroups is the API groups the resources belong to. '*' is all groups.
	// If '*' is present, the length of the slice must be one.
	// Required.
	APIGroups []string

	// APIVersions is the API versions the resources belong to. '*' is all versions.
	// If '*' is present, the length of the slice must be one.
	// Required.
	APIVersions []string

	// Resources is a list of resources this rule applies to.
	//
	// For example:
	// 'pods' means pods.
	// 'pods/log' means the log subresource of pods.
	// '*' means all resources, but not subresources.
	// 'pods/*' means all subresources of pods.
	// '*/scale' means all scale subresources.
	// '*/*' means all resources and their subresources.
	//
	// If wildcard is present, the validation rule will ensure resources do not
	// overlap with each other.
	//
	// Depending on the enclosing object, subresources might not be allowed.
	// Required.
	Resources []string

	// scope specifies the scope of this rule.
	// Valid values are "Cluster", "Namespaced", and "*"
	// "Cluster" means that only cluster-scoped resources will match this rule.
	// Namespace API objects are cluster-scoped.
	// "Namespaced" means that only namespaced resources will match this rule.
	// "*" means that there are no scope restrictions.
	// Subresources match the scope of their parent resource.
	// Default is "*".
	//
	// +optional
	Scope *ScopeType
}

// ScopeType specifies the type of scope being used
type ScopeType string

const (
	// ClusterScope means that scope is limited to cluster-scoped objects.
	// Namespace objects are cluster-scoped.
	ClusterScope ScopeType = "Cluster"
	// NamespacedScope means that scope is limited to namespaced objects.
	NamespacedScope ScopeType = "Namespaced"
	// AllScopes means that all scopes are included.
	AllScopes ScopeType = "*"
)

// FailurePolicyType specifies the type of failure policy
type FailurePolicyType string

const (
	// Ignore means that an error calling the webhook is ignored.
	Ignore FailurePolicyType = "Ignore"
	// Fail means that an error calling the webhook causes the admission to fail.
	Fail FailurePolicyType = "Fail"
)

// MatchPolicyType specifies the type of match policy
type MatchPolicyType string

const (
	// Exact means requests should only be sent to the webhook if they exactly match a given rule
	Exact MatchPolicyType = "Exact"
	// Equivalent means requests should be sent to the webhook if they modify a resource listed in rules via another API group or version.
	Equivalent MatchPolicyType = "Equivalent"
)

// SideEffectClass denotes the type of side effects resulting from calling the webhook
type SideEffectClass string

const (
	// SideEffectClassUnknown means that no information is known about the side effects of calling the webhook.
	// If a request with the dry-run attribute would trigger a call to this webhook, the request will instead fail.
	SideEffectClassUnknown SideEffectClass = "Unknown"
	// SideEffectClassNone means that calling the webhook will have no side effects.
	SideEffectClassNone SideEffectClass = "None"
	// SideEffectClassSome means that calling the webhook will possibly have side effects.
	// If a request with the dry-run attribute would trigger a call to this webhook, the request will instead fail.
	SideEffectClassSome SideEffectClass = "Some"
	// SideEffectClassNoneOnDryRun means that calling the webhook will possibly have side effects, but if the
	// request being reviewed has the dry-run attribute, the side effects will be suppressed.
	SideEffectClassNoneOnDryRun SideEffectClass = "NoneOnDryRun"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingAdmissionPolicy describes the definition of an admission validation policy that accepts or rejects an object without changing it.
type ValidatingAdmissionPolicy struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta
	// Specification of the desired behavior of the ValidatingAdmissionPolicy.
	Spec ValidatingAdmissionPolicySpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingAdmissionPolicyList is a list of ValidatingAdmissionPolicy.
type ValidatingAdmissionPolicyList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta
	// List of ValidatingAdmissionPolicy.
	Items []ValidatingAdmissionPolicy
}

// ValidatingAdmissionPolicySpec is the specification of the desired behavior of the AdmissionPolicy.
type ValidatingAdmissionPolicySpec struct {
	// ParamKind specifies the kind of resources used to parameterize this policy.
	// If absent, there are no parameters for this policy and the param CEL variable will not be provided to validation expressions.
	// If ParamKind refers to a non-existent kind, this policy definition is mis-configured and the FailurePolicy is applied.
	// If paramKind is specified but paramRef is unset in ValidatingAdmissionPolicyBinding, the params variable will be null.
	// +optional
	ParamKind *ParamKind

	// MatchConstraints specifies what resources this policy is designed to validate.
	// The AdmissionPolicy cares about a request if it matches _all_ Constraint.
	// However, in order to prevent clusters from being put into an unstable state that cannot be recovered from via the API
	// ValidatingAdmissionPolicy cannot match ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding.
	// Required.
	MatchConstraints *MatchResources

	// Validations contain CEL expressions which is used to apply the validation.
	// A minimum of one validation is required for a policy definition.
	// Required.
	Validations []Validation

	// FailurePolicy defines how to handle failures for the admission policy.
	// Failures can occur from invalid or mis-configured policy definitions or bindings.
	// A policy is invalid if spec.paramKind refers to a non-existent Kind.
	// A binding is invalid if spec.paramRef.name refers to a non-existent resource.
	// Allowed values are Ignore or Fail. Defaults to Fail.
	// +optional
	FailurePolicy *FailurePolicyType
}

// ParamKind is a tuple of Group Kind and Version.
type ParamKind struct {
	// APIVersion is the API group version the resources belong to.
	// In format of "group/version".
	// Required.
	APIVersion string

	// Kind is the API kind the resources belong to.
	// Required.
	Kind string
}

// Validation specifies the CEL expression which is used to apply the validation.
type Validation struct {
	// Expression represents the expression which will be evaluated by CEL.
	// ref: https://github.com/google/cel-spec
	// CEL expressions have access to the contents of the Admission request/response, organized into CEL variables as well as some other useful variables:
	//
	//'object' - The object from the incoming request. The value is null for DELETE requests.
	//'oldObject' - The existing object. The value is null for CREATE requests.
	//'request' - Attributes of the admission request([ref](/pkg/apis/admission/types.go#AdmissionRequest)).
	//'params' - Parameter resource referred to by the policy binding being evaluated. Only populated if the policy has a ParamKind.
	//
	// The `apiVersion`, `kind`, `metadata.name` and `metadata.generateName` are always accessible from the root of the
	// object. No other metadata properties are accessible.
	//
	// Only property names of the form `[a-zA-Z_.-/][a-zA-Z0-9_.-/]*` are accessible.
	// Accessible property names are escaped according to the following rules when accessed in the expression:
	// - '__' escapes to '__underscores__'
	// - '.' escapes to '__dot__'
	// - '-' escapes to '__dash__'
	// - '/' escapes to '__slash__'
	// - Property names that exactly match a CEL RESERVED keyword escape to '__{keyword}__'. The keywords are:
	//	  "true", "false", "null", "in", "as", "break", "const", "continue", "else", "for", "function", "if",
	//	  "import", "let", "loop", "package", "namespace", "return".
	// Examples:
	//   - Expression accessing a property named "namespace": {"Expression": "object.__namespace__ > 0"}
	//   - Expression accessing a property named "x-prop": {"Expression": "object.x__dash__prop > 0"}
	//   - Expression accessing a property named "redact__d": {"Expression": "object.redact__underscores__d > 0"}
	//
	// Equality on arrays with list type of 'set' or 'map' ignores element order, i.e. [1, 2] == [2, 1].
	// Concatenation on arrays with x-kubernetes-list-type use the semantics of the list type:
	//   - 'set': `X + Y` performs a union where the array positions of all elements in `X` are preserved and
	//     non-intersecting elements in `Y` are appended, retaining their partial order.
	//   - 'map': `X + Y` performs a merge where the array positions of all keys in `X` are preserved but the values
	//     are overwritten by values in `Y` when the key sets of `X` and `Y` intersect. Elements in `Y` with
	//     non-intersecting keys are appended, retaining their partial order.
	// Required.
	Expression string
	// Message represents the message displayed when validation fails. The message is required if the Expression contains
	// line breaks. The message must not contain line breaks.
	// If unset, the message is "failed rule: {Rule}".
	// e.g. "must be a URL with the host matching spec.host"
	// If ExpressMessage is specified, Message will be ignored
	// If the Expression contains line breaks. Eith Message or ExpressMessage is required.
	// The message must not contain line breaks.
	// If unset, the message is "failed Expression: {Expression}".
	// +optional
	Message string
	// Reason represents a machine-readable description of why this validation failed.
	// If this is the first validation in the list to fail, this reason, as well as the
	// corresponding HTTP response code, are used in the
	// HTTP response to the client.
	// The currently supported reasons are: "Unauthorized", "Forbidden", "Invalid", "RequestEntityTooLarge".
	// If not set, StatusReasonInvalid is used in the response to the client.
	// +optional
	Reason *metav1.StatusReason
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingAdmissionPolicyBinding binds the ValidatingAdmissionPolicy with paramerized resources.
// ValidatingAdmissionPolicyBinding and parameter CRDs together define how cluster administrators configure policies for clusters.
type ValidatingAdmissionPolicyBinding struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta
	// Specification of the desired behavior of the ValidatingAdmissionPolicyBinding.
	Spec ValidatingAdmissionPolicyBindingSpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingAdmissionPolicyBindingList is a list of PolicyBinding.
type ValidatingAdmissionPolicyBindingList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta
	// List of PolicyBinding.
	Items []ValidatingAdmissionPolicyBinding
}

// ValidatingAdmissionPolicyBindingSpec is the specification of the ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingSpec struct {
	// PolicyName references a ValidatingAdmissionPolicy name which the ValidatingAdmissionPolicyBinding binds to.
	// If the referenced resource does not exist, this binding is considered invalid and will be ignored
	// Required.
	PolicyName string

	// ParamRef specifies the parameter resource used to configure the admission control policy.
	// It should point to a resource of the type specified in ParamKind of the bound ValidatingAdmissionPolicy.
	// If the policy specifies a ParamKind and the resource referred to by ParamRef does not exist, this binding is considered mis-configured and the FailurePolicy of the ValidatingAdmissionPolicy applied.
	// +optional
	ParamRef *ParamRef

	// MatchResources declares what resources match this binding and will be validated by it.
	// Note that this is intersected with the policy's matchConstraints, so only requests that are matched by the policy can be selected by this.
	// If this is unset, all resources matched by the policy are validated by this binding
	// When resourceRules is unset, it does not constrain resource matching. If a resource is matched by the other fields of this object, it will be validated.
	// Note that this is differs from ValidatingAdmissionPolicy matchConstraints, where resourceRules are required.
	// +optional
	MatchResources *MatchResources
}

// ParamRef references a parameter resource
type ParamRef struct {
	// Name of the resource being referenced.
	Name string
	// Namespace of the referenced resource.
	// Should be empty for the cluster-scoped resources
	// +optional
	Namespace string
}

// MatchResources decides whether to run the admission control policy on an object based
// on whether it meets the match criteria.
// The exclude rules take precedence over include rules (if a resource matches both, it is excluded)
type MatchResources struct {
	// NamespaceSelector decides whether to run the admission control policy on an object based
	// on whether the namespace for that object matches the selector. If the
	// object itself is a namespace, the matching is performed on
	// object.metadata.labels. If the object is another cluster scoped resource,
	// it never skips the policy.
	//
	// For example, to run the webhook on any objects whose namespace is not
	// associated with "runlevel" of "0" or "1";  you will set the selector as
	// follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "runlevel",
	//       "operator": "NotIn",
	//       "values": [
	//         "0",
	//         "1"
	//       ]
	//     }
	//   ]
	// }
	//
	// If instead you want to only run the policy on any objects whose
	// namespace is associated with the "environment" of "prod" or "staging";
	// you will set the selector as follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "environment",
	//       "operator": "In",
	//       "values": [
	//         "prod",
	//         "staging"
	//       ]
	//     }
	//   ]
	// }
	//
	// See
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
	// for more examples of label selectors.
	//
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	NamespaceSelector *metav1.LabelSelector
	// ObjectSelector decides whether to run the validation based on if the
	// object has matching labels. objectSelector is evaluated against both
	// the oldObject and newObject that would be sent to the cel validation, and
	// is considered to match if either object matches the selector. A null
	// object (oldObject in the case of create, or newObject in the case of
	// delete) or an object that cannot have labels (like a
	// DeploymentRollback or a PodProxyOptions object) is not considered to
	// match.
	// Use the object selector only if the webhook is opt-in, because end
	// users may skip the admission webhook by setting the labels.
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	ObjectSelector *metav1.LabelSelector
	// ResourceRules describes what operations on what resources/subresources the ValidatingAdmissionPolicy matches.
	// The policy cares about an operation if it matches _any_ Rule.
	// +optional
	ResourceRules []NamedRuleWithOperations
	// ExcludeResourceRules describes what operations on what resources/subresources the ValidatingAdmissionPolicy should not care about.
	// The exclude rules take precedence over include rules (if a resource matches both, it is excluded)
	// +optional
	ExcludeResourceRules []NamedRuleWithOperations
	// matchPolicy defines how the "MatchResources" list is used to match incoming requests.
	// Allowed values are "Exact" or "Equivalent".
	//
	// - Exact: match a request only if it exactly matches a specified rule.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// but "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would not be sent to the ValidatingAdmissionPolicy.
	//
	// - Equivalent: match a request if modifies a resource listed in rules, even via another API group or version.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// and "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would be converted to apps/v1 and sent to the ValidatingAdmissionPolicy.
	//
	// Defaults to "Equivalent"
	// +optional
	MatchPolicy *MatchPolicyType
}

// NamedRuleWithOperations is a tuple of Operations and Resources with ResourceNames.
type NamedRuleWithOperations struct {
	// ResourceNames is an optional white list of names that the rule applies to.  An empty set means that everything is allowed.
	// +optional
	ResourceNames []string
	// RuleWithOperations is a tuple of Operations and Resources.
	RuleWithOperations RuleWithOperations
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingWebhookConfiguration describes the configuration of an admission webhook that accepts or rejects and object without changing it.
type ValidatingWebhookConfiguration struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta
	// Webhooks is a list of webhooks and the affected resources and operations.
	// +optional
	Webhooks []ValidatingWebhook
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingWebhookConfigurationList is a list of ValidatingWebhookConfiguration.
type ValidatingWebhookConfigurationList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta
	// List of ValidatingWebhookConfigurations.
	Items []ValidatingWebhookConfiguration
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MutatingWebhookConfiguration describes the configuration of and admission webhook that accept or reject and may change the object.
type MutatingWebhookConfiguration struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta
	// Webhooks is a list of webhooks and the affected resources and operations.
	// +optional
	Webhooks []MutatingWebhook
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MutatingWebhookConfigurationList is a list of MutatingWebhookConfiguration.
type MutatingWebhookConfigurationList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta
	// List of MutatingWebhookConfiguration.
	Items []MutatingWebhookConfiguration
}

// ValidatingWebhook describes an admission webhook and the resources and operations it applies to.
type ValidatingWebhook struct {
	// The name of the admission webhook.
	// Name should be fully qualified, e.g., imagepolicy.kubernetes.io, where
	// "imagepolicy" is the name of the webhook, and kubernetes.io is the name
	// of the organization.
	// Required.
	Name string

	// ClientConfig defines how to communicate with the hook.
	// Required
	ClientConfig WebhookClientConfig

	// Rules describes what operations on what resources/subresources the webhook cares about.
	// The webhook cares about an operation if it matches _any_ Rule.
	Rules []RuleWithOperations

	// FailurePolicy defines how unrecognized errors from the admission endpoint are handled -
	// allowed values are Ignore or Fail. Defaults to Ignore.
	// +optional
	FailurePolicy *FailurePolicyType

	// matchPolicy defines how the "rules" list is used to match incoming requests.
	// Allowed values are "Exact" or "Equivalent".
	//
	// - Exact: match a request only if it exactly matches a specified rule.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// but "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would not be sent to the webhook.
	//
	// - Equivalent: match a request if modifies a resource listed in rules, even via another API group or version.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// and "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would be converted to apps/v1 and sent to the webhook.
	//
	// +optional
	MatchPolicy *MatchPolicyType

	// NamespaceSelector decides whether to run the webhook on an object based
	// on whether the namespace for that object matches the selector. If the
	// object itself is a namespace, the matching is performed on
	// object.metadata.labels. If the object is another cluster scoped resource,
	// it never skips the webhook.
	//
	// For example, to run the webhook on any objects whose namespace is not
	// associated with "runlevel" of "0" or "1";  you will set the selector as
	// follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "runlevel",
	//       "operator": "NotIn",
	//       "values": [
	//         "0",
	//         "1"
	//       ]
	//     }
	//   ]
	// }
	//
	// If instead you want to only run the webhook on any objects whose
	// namespace is associated with the "environment" of "prod" or "staging";
	// you will set the selector as follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "environment",
	//       "operator": "In",
	//       "values": [
	//         "prod",
	//         "staging"
	//       ]
	//     }
	//   ]
	// }
	//
	// See
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
	// for more examples of label selectors.
	//
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	NamespaceSelector *metav1.LabelSelector

	// ObjectSelector decides whether to run the webhook based on if the
	// object has matching labels. objectSelector is evaluated against both
	// the oldObject and newObject that would be sent to the webhook, and
	// is considered to match if either object matches the selector. A null
	// object (oldObject in the case of create, or newObject in the case of
	// delete) or an object that cannot have labels (like a
	// DeploymentRollback or a PodProxyOptions object) is not considered to
	// match.
	// Use the object selector only if the webhook is opt-in, because end
	// users may skip the admission webhook by setting the labels.
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	ObjectSelector *metav1.LabelSelector

	// SideEffects states whether this webhook has side effects.
	// Acceptable values are: Unknown, None, Some, NoneOnDryRun
	// Webhooks with side effects MUST implement a reconciliation system, since a request may be
	// rejected by a future step in the admission chain and the side effects therefore need to be undone.
	// Requests with the dryRun attribute will be auto-rejected if they match a webhook with
	// sideEffects == Unknown or Some. Defaults to Unknown.
	// +optional
	SideEffects *SideEffectClass

	// TimeoutSeconds specifies the timeout for this webhook. After the timeout passes,
	// the webhook call will be ignored or the API call will fail based on the
	// failure policy.
	// The timeout value must be between 1 and 30 seconds.
	// +optional
	TimeoutSeconds *int32

	// AdmissionReviewVersions is an ordered list of preferred `AdmissionReview`
	// versions the Webhook expects. API server will try to use first version in
	// the list which it supports. If none of the versions specified in this list
	// supported by API server, validation will fail for this object.
	// If the webhook configuration has already been persisted with a version apiserver
	// does not understand, calls to the webhook will fail and be subject to the failure policy.
	// +optional
	AdmissionReviewVersions []string
}

// MutatingWebhook describes an admission webhook and the resources and operations it applies to.
type MutatingWebhook struct {
	// The name of the admission webhook.
	// Name should be fully qualified, e.g., imagepolicy.kubernetes.io, where
	// "imagepolicy" is the name of the webhook, and kubernetes.io is the name
	// of the organization.
	// Required.
	Name string

	// ClientConfig defines how to communicate with the hook.
	// Required
	ClientConfig WebhookClientConfig

	// Rules describes what operations on what resources/subresources the webhook cares about.
	// The webhook cares about an operation if it matches _any_ Rule.
	Rules []RuleWithOperations

	// FailurePolicy defines how unrecognized errors from the admission endpoint are handled -
	// allowed values are Ignore or Fail. Defaults to Ignore.
	// +optional
	FailurePolicy *FailurePolicyType

	// matchPolicy defines how the "rules" list is used to match incoming requests.
	// Allowed values are "Exact" or "Equivalent".
	//
	// - Exact: match a request only if it exactly matches a specified rule.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// but "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would not be sent to the webhook.
	//
	// - Equivalent: match a request if modifies a resource listed in rules, even via another API group or version.
	// For example, if deployments can be modified via apps/v1, apps/v1beta1, and extensions/v1beta1,
	// and "rules" only included `apiGroups:["apps"], apiVersions:["v1"], resources: ["deployments"]`,
	// a request to apps/v1beta1 or extensions/v1beta1 would be converted to apps/v1 and sent to the webhook.
	//
	// +optional
	MatchPolicy *MatchPolicyType

	// NamespaceSelector decides whether to run the webhook on an object based
	// on whether the namespace for that object matches the selector. If the
	// object itself is a namespace, the matching is performed on
	// object.metadata.labels. If the object is another cluster scoped resource,
	// it never skips the webhook.
	//
	// For example, to run the webhook on any objects whose namespace is not
	// associated with "runlevel" of "0" or "1";  you will set the selector as
	// follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "runlevel",
	//       "operator": "NotIn",
	//       "values": [
	//         "0",
	//         "1"
	//       ]
	//     }
	//   ]
	// }
	//
	// If instead you want to only run the webhook on any objects whose
	// namespace is associated with the "environment" of "prod" or "staging";
	// you will set the selector as follows:
	// "namespaceSelector": {
	//   "matchExpressions": [
	//     {
	//       "key": "environment",
	//       "operator": "In",
	//       "values": [
	//         "prod",
	//         "staging"
	//       ]
	//     }
	//   ]
	// }
	//
	// See
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
	// for more examples of label selectors.
	//
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	NamespaceSelector *metav1.LabelSelector

	// ObjectSelector decides whether to run the webhook based on if the
	// object has matching labels. objectSelector is evaluated against both
	// the oldObject and newObject that would be sent to the webhook, and
	// is considered to match if either object matches the selector. A null
	// object (oldObject in the case of create, or newObject in the case of
	// delete) or an object that cannot have labels (like a
	// DeploymentRollback or a PodProxyOptions object) is not considered to
	// match.
	// Use the object selector only if the webhook is opt-in, because end
	// users may skip the admission webhook by setting the labels.
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	ObjectSelector *metav1.LabelSelector

	// SideEffects states whether this webhook has side effects.
	// Acceptable values are: Unknown, None, Some, NoneOnDryRun
	// Webhooks with side effects MUST implement a reconciliation system, since a request may be
	// rejected by a future step in the admission chain and the side effects therefore need to be undone.
	// Requests with the dryRun attribute will be auto-rejected if they match a webhook with
	// sideEffects == Unknown or Some. Defaults to Unknown.
	// +optional
	SideEffects *SideEffectClass

	// TimeoutSeconds specifies the timeout for this webhook. After the timeout passes,
	// the webhook call will be ignored or the API call will fail based on the
	// failure policy.
	// The timeout value must be between 1 and 30 seconds.
	// +optional
	TimeoutSeconds *int32

	// AdmissionReviewVersions is an ordered list of preferred `AdmissionReview`
	// versions the Webhook expects. API server will try to use first version in
	// the list which it supports. If none of the versions specified in this list
	// supported by API server, validation will fail for this object.
	// If the webhook configuration has already been persisted with a version apiserver
	// does not understand, calls to the webhook will fail and be subject to the failure policy.
	// +optional
	AdmissionReviewVersions []string

	// reinvocationPolicy indicates whether this webhook should be called multiple times as part of a single admission evaluation.
	// Allowed values are "Never" and "IfNeeded".
	//
	// Never: the webhook will not be called more than once in a single admission evaluation.
	//
	// IfNeeded: the webhook will be called at least one additional time as part of the admission evaluation
	// if the object being admitted is modified by other admission plugins after the initial webhook call.
	// Webhooks that specify this option *must* be idempotent, and hence able to process objects they previously admitted.
	// Note:
	// * the number of additional invocations is not guaranteed to be exactly one.
	// * if additional invocations result in further modifications to the object, webhooks are not guaranteed to be invoked again.
	// * webhooks that use this option may be reordered to minimize the number of additional invocations.
	// * to validate an object after all mutations are guaranteed complete, use a validating admission webhook instead.
	//
	// Defaults to "Never".
	// +optional
	ReinvocationPolicy *ReinvocationPolicyType
}

// ReinvocationPolicyType specifies what type of policy the admission hook uses.
type ReinvocationPolicyType string

var (
	// NeverReinvocationPolicy indicates that the webhook must not be called more than once in a
	// single admission evaluation.
	NeverReinvocationPolicy ReinvocationPolicyType = "Never"
	// IfNeededReinvocationPolicy indicates that the webhook may be called at least one
	// additional time as part of the admission evaluation if the object being admitted is
	// modified by other admission plugins after the initial webhook call.
	IfNeededReinvocationPolicy ReinvocationPolicyType = "IfNeeded"
)

// RuleWithOperations is a tuple of Operations and Resources. It is recommended to make
// sure that all the tuple expansions are valid.
type RuleWithOperations struct {
	// Operations is the operations the admission hook cares about - CREATE, UPDATE, or *
	// for all operations.
	// If '*' is present, the length of the slice must be one.
	// Required.
	Operations []OperationType
	// Rule is embedded, it describes other criteria of the rule, like
	// APIGroups, APIVersions, Resources, etc.
	Rule
}

// OperationType specifies what type of operation the admission hook cares about.
type OperationType string

// The constants should be kept in sync with those defined in k8s.io/kubernetes/pkg/admission/interface.go.
const (
	OperationAll OperationType = "*"
	Create       OperationType = "CREATE"
	Update       OperationType = "UPDATE"
	Delete       OperationType = "DELETE"
	Connect      OperationType = "CONNECT"
)

// WebhookClientConfig contains the information to make a TLS
// connection with the webhook
type WebhookClientConfig struct {
	// `url` gives the location of the webhook, in standard URL form
	// (`scheme://host:port/path`). Exactly one of `url` or `service`
	// must be specified.
	//
	// The `host` should not refer to a service running in the cluster; use
	// the `service` field instead. The host might be resolved via external
	// DNS in some apiservers (e.g., `kube-apiserver` cannot resolve
	// in-cluster DNS as that would be a layering violation). `host` may
	// also be an IP address.
	//
	// Please note that using `localhost` or `127.0.0.1` as a `host` is
	// risky unless you take great care to run this webhook on all hosts
	// which run an apiserver which might need to make calls to this
	// webhook. Such installs are likely to be non-portable, i.e., not easy
	// to turn up in a new cluster.
	//
	// The scheme must be "https"; the URL must begin with "https://".
	//
	// A path is optional, and if present may be any string permissible in
	// a URL. You may use the path to pass an arbitrary string to the
	// webhook, for example, a cluster identifier.
	//
	// Attempting to use a user or basic auth e.g. "user:password@" is not
	// allowed. Fragments ("#...") and query parameters ("?...") are not
	// allowed, either.
	//
	// +optional
	URL *string

	// `service` is a reference to the service for this webhook. Either
	// `service` or `url` must be specified.
	//
	// If the webhook is running within the cluster, then you should use `service`.
	//
	// +optional
	Service *ServiceReference

	// `caBundle` is a PEM encoded CA bundle which will be used to validate the webhook's server certificate.
	// If unspecified, system trust roots on the apiserver are used.
	// +optional
	CABundle []byte
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// `namespace` is the namespace of the service.
	// Required
	Namespace string
	// `name` is the name of the service.
	// Required
	Name string

	// `path` is an optional URL path which will be sent in any request to
	// this service.
	// +optional
	Path *string

	// If specified, the port on the service that hosting webhook.
	// `port` should be a valid port number (1-65535, inclusive).
	// +optional
	Port int32
}
