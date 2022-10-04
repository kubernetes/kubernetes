/*
Copyright 2022 The Kubernetes Authors.

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

package v1alpha1

import (
	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Rule is a tuple of APIGroups, APIVersion, and Resources.It is recommended
// to make sure that all the tuple expansions are valid.
type Rule = v1.Rule

// ScopeType specifies a scope for a Rule.
// +enum
type ScopeType = v1.ScopeType

const (
	// ClusterScope means that scope is limited to cluster-scoped objects.
	// Namespace objects are cluster-scoped.
	ClusterScope ScopeType = v1.ClusterScope
	// NamespacedScope means that scope is limited to namespaced objects.
	NamespacedScope ScopeType = v1.NamespacedScope
	// AllScopes means that all scopes are included.
	AllScopes ScopeType = v1.AllScopes
)

// FailurePolicyType specifies a failure policy that defines how unrecognized errors from the admission endpoint are handled.
// +enum
type FailurePolicyType string

const (
	// Ignore means that an error calling the webhook is ignored.
	Ignore FailurePolicyType = "Ignore"
	// Fail means that an error calling the webhook causes the admission to fail.
	Fail FailurePolicyType = "Fail"
)

// MatchPolicyType specifies the type of match policy.
// +enum
type MatchPolicyType string

const (
	// Exact means requests should only be sent to the webhook if they exactly match a given rule.
	Exact MatchPolicyType = "Exact"
	// Equivalent means requests should be sent to the webhook if they modify a resource listed in rules via another API group or version.
	Equivalent MatchPolicyType = "Equivalent"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ValidatingAdmissionPolicy describes the definition of an admission validation policy that accepts or rejects an object without changing it.
type ValidatingAdmissionPolicy struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Specification of the desired behavior of the ValidatingAdmissionPolicy.
	Spec ValidatingAdmissionPolicySpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ValidatingAdmissionPolicyList is a list of ValidatingAdmissionPolicy.
type ValidatingAdmissionPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of ValidatingAdmissionPolicy.
	Items []ValidatingAdmissionPolicy `json:"items,omitempty" protobuf:"bytes,2,rep,name=items"`
}

// ValidatingAdmissionPolicySpec is the specification of the desired behavior of the AdmissionPolicy.
type ValidatingAdmissionPolicySpec struct {
	// ParamKind specifies the kind of resources used to parameterize this policy.
	// If absent, there are no parameters for this policy and the param CEL variable will not be provided to validation expressions.
	// If ParamKind refers to a non-existent kind, this policy definition is mis-configured and the FailurePolicy is applied.
	// If paramKind is specified but paramRef is unset in ValidatingAdmissionPolicyBinding, the params variable will be null.
	// +optional
	ParamKind *ParamKind `json:"paramKind,omitempty" protobuf:"bytes,1,rep,name=paramKind"`

	// MatchConstraints specifies what resources this policy is designed to validate.
	// The AdmissionPolicy cares about a request if it matches _all_ Constraints.
	// However, in order to prevent clusters from being put into an unstable state that cannot be recovered from via the API
	// ValidatingAdmissionPolicy cannot match ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding.
	// Required.
	MatchConstraints *MatchResources `json:"matchConstraints,omitempty" protobuf:"bytes,2,rep,name=matchConstraints"`

	// Validations contain CEL expressions which is used to apply the validation.
	// A minimum of one validation is required for a policy definition.
	// +listType=atomic
	// Required.
	Validations []Validation `json:"validations" protobuf:"bytes,3,rep,name=validations"`

	// FailurePolicy defines how to handle failures for the admission policy.
	// Failures can occur from invalid or mis-configured policy definitions or bindings.
	// A policy is invalid if spec.paramKind refers to a non-existent Kind.
	// A binding is invalid if spec.paramRef.name refers to a non-existent resource.
	// Allowed values are Ignore or Fail. Defaults to Fail.
	// +optional
	FailurePolicy *FailurePolicyType `json:"failurePolicy,omitempty" protobuf:"bytes,4,opt,name=failurePolicy,casttype=FailurePolicyType"`
}

// ParamKind is a tuple of Group Kind and Version.
// +structType=atomic
type ParamKind struct {
	// APIVersion is the API group version the resources belong to.
	// In format of "group/version".
	// Required.
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,1,rep,name=apiVersion"`

	// Kind is the API kind the resources belong to.
	// Required.
	Kind string `json:"kind,omitempty" protobuf:"bytes,2,rep,name=kind"`
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
	Expression string `json:"expression" protobuf:"bytes,1,opt,name=Expression"`
	// Message represents the message displayed when validation fails. The message is required if the Expression contains
	// line breaks. The message must not contain line breaks.
	// If unset, the message is "failed rule: {Rule}".
	// e.g. "must be a URL with the host matching spec.host"
	// If the Expression contains line breaks. Message is required.
	// The message must not contain line breaks.
	// If unset, the message is "failed Expression: {Expression}".
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,2,opt,name=message"`
	// Reason represents a machine-readable description of why this validation failed.
	// If this is the first validation in the list to fail, this reason, as well as the
	// corresponding HTTP response code, are used in the
	// HTTP response to the client.
	// The currently supported reasons are: "Unauthorized", "Forbidden", "Invalid", "RequestEntityTooLarge".
	// If not set, StatusReasonInvalid is used in the response to the client.
	// +optional
	Reason *metav1.StatusReason `json:"reason,omitempty" protobuf:"bytes,3,opt,name=reason"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ValidatingAdmissionPolicyBinding binds the ValidatingAdmissionPolicy with paramerized resources.
// ValidatingAdmissionPolicyBinding and parameter CRDs together define how cluster administrators configure policies for clusters.
type ValidatingAdmissionPolicyBinding struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Specification of the desired behavior of the ValidatingAdmissionPolicyBinding.
	Spec ValidatingAdmissionPolicyBindingSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ValidatingAdmissionPolicyBindingList is a list of ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of PolicyBinding.
	Items []ValidatingAdmissionPolicyBinding `json:"items,omitempty" protobuf:"bytes,2,rep,name=items"`
}

// ValidatingAdmissionPolicyBindingSpec is the specification of the ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingSpec struct {
	// PolicyName references a ValidatingAdmissionPolicy name which the ValidatingAdmissionPolicyBinding binds to.
	// If the referenced resource does not exist, this binding is considered invalid and will be ignored
	// Required.
	PolicyName string `json:"policyName,omitempty" protobuf:"bytes,1,rep,name=policyName"`

	// ParamRef specifies the parameter resource used to configure the admission control policy.
	// It should point to a resource of the type specified in ParamKind of the bound ValidatingAdmissionPolicy.
	// If the policy specifies a ParamKind and the resource referred to by ParamRef does not exist, this binding is considered mis-configured and the FailurePolicy of the ValidatingAdmissionPolicy applied.
	// +optional
	ParamRef *ParamRef `json:"paramRef,omitempty" protobuf:"bytes,2,rep,name=paramRef"`

	// MatchResources declares what resources match this binding and will be validated by it.
	// Note that this is intersected with the policy's matchConstraints, so only requests that are matched by the policy can be selected by this.
	// If this is unset, all resources matched by the policy are validated by this binding
	// When resourceRules is unset, it does not constrain resource matching. If a resource is matched by the other fields of this object, it will be validated.
	// Note that this is differs from ValidatingAdmissionPolicy matchConstraints, where resourceRules are required.
	// +optional
	MatchResources *MatchResources `json:"matchResources,omitempty" protobuf:"bytes,3,rep,name=matchResources"`
}

// ParamRef references a parameter resource
// +structType=atomic
type ParamRef struct {
	// Name of the resource being referenced.
	Name string `json:"name,omitempty" protobuf:"bytes,1,rep,name=name"`
	// Namespace of the referenced resource.
	// Should be empty for the cluster-scoped resources
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,2,rep,name=namespace"`
}

// MatchResources decides whether to run the admission control policy on an object based
// on whether it meets the match criteria.
// The exclude rules take precedence over include rules (if a resource matches both, it is excluded)
// +structType=atomic
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
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty" protobuf:"bytes,1,opt,name=namespaceSelector"`
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
	ObjectSelector *metav1.LabelSelector `json:"objectSelector,omitempty" protobuf:"bytes,2,opt,name=objectSelector"`
	// ResourceRules describes what operations on what resources/subresources the ValidatingAdmissionPolicy matches.
	// The policy cares about an operation if it matches _any_ Rule.
	// +listType=atomic
	// +optional
	ResourceRules []NamedRuleWithOperations `json:"resourceRules,omitempty" protobuf:"bytes,3,rep,name=resourceRules"`
	// ExcludeResourceRules describes what operations on what resources/subresources the ValidatingAdmissionPolicy should not care about.
	// The exclude rules take precedence over include rules (if a resource matches both, it is excluded)
	// +listType=atomic
	// +optional
	ExcludeResourceRules []NamedRuleWithOperations `json:"excludeResourceRules,omitempty" protobuf:"bytes,4,rep,name=excludeResourceRules"`
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
	MatchPolicy *MatchPolicyType `json:"matchPolicy,omitempty" protobuf:"bytes,7,opt,name=matchPolicy,casttype=MatchPolicyType"`
}

// NamedRuleWithOperations is a tuple of Operations and Resources with ResourceNames.
// +structType=atomic
type NamedRuleWithOperations struct {
	// ResourceNames is an optional white list of names that the rule applies to.  An empty set means that everything is allowed.
	// +listType=atomic
	// +optional
	ResourceNames []string `json:"resourceNames,omitempty" protobuf:"bytes,1,rep,name=resourceNames"`
	// RuleWithOperations is a tuple of Operations and Resources.
	RuleWithOperations `json:",inline" protobuf:"bytes,2,opt,name=ruleWithOperations"`
}

// RuleWithOperations is a tuple of Operations and Resources. It is recommended to make
// sure that all the tuple expansions are valid.
type RuleWithOperations = v1.RuleWithOperations

// OperationType specifies an operation for a request.
// +enum
type OperationType = v1.OperationType

// The constants should be kept in sync with those defined in k8s.io/kubernetes/pkg/admission/interface.go.
const (
	OperationAll OperationType = v1.OperationAll
	Create       OperationType = v1.Create
	Update       OperationType = v1.Update
	Delete       OperationType = v1.Delete
	Connect      OperationType = v1.Connect
)
