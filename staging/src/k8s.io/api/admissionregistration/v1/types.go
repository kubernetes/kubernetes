/*
Copyright 2019 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Rule is a tuple of APIGroups, APIVersion, and Resources.It is recommended
// to make sure that all the tuple expansions are valid.
type Rule struct {
	// APIGroups is the API groups the resources belong to. '*' is all groups.
	// If '*' is present, the length of the slice must be one.
	// Required.
	// +listType=atomic
	APIGroups []string `json:"apiGroups,omitempty" protobuf:"bytes,1,rep,name=apiGroups"`

	// APIVersions is the API versions the resources belong to. '*' is all versions.
	// If '*' is present, the length of the slice must be one.
	// Required.
	// +listType=atomic
	APIVersions []string `json:"apiVersions,omitempty" protobuf:"bytes,2,rep,name=apiVersions"`

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
	// +listType=atomic
	Resources []string `json:"resources,omitempty" protobuf:"bytes,3,rep,name=resources"`

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
	Scope *ScopeType `json:"scope,omitempty" protobuf:"bytes,4,rep,name=scope"`
}

// ScopeType specifies a scope for a Rule.
// +enum
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

// FailurePolicyType specifies a failure policy that defines how unrecognized errors from the admission endpoint are handled.
// +enum
type FailurePolicyType string

const (
	// Ignore means that an error calling the webhook is ignored.
	Ignore FailurePolicyType = "Ignore"
	// Fail means that an error calling the webhook causes the admission to fail.
	Fail FailurePolicyType = "Fail"
)

// ParameterNotFoundActionType specifies a failure policy that defines how a binding
// is evaluated when the param referred by its perNamespaceParamRef is not found.
type ParameterNotFoundActionType string

const (
	// Allow means all requests will be admitted if no param resources
	// could be found.
	AllowAction ParameterNotFoundActionType = "Allow"
	// Deny means all requests will be denied if no param resources are found.
	DenyAction ParameterNotFoundActionType = "Deny"
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

// SideEffectClass specifies the types of side effects a webhook may have.
// +enum
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
// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ValidatingAdmissionPolicy describes the definition of an admission validation policy that accepts or rejects an object without changing it.
type ValidatingAdmissionPolicy struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Specification of the desired behavior of the ValidatingAdmissionPolicy.
	Spec ValidatingAdmissionPolicySpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// The status of the ValidatingAdmissionPolicy, including warnings that are useful to determine if the policy
	// behaves in the expected way.
	// Populated by the system.
	// Read-only.
	// +optional
	Status ValidatingAdmissionPolicyStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ValidatingAdmissionPolicyStatus represents the status of an admission validation policy.
type ValidatingAdmissionPolicyStatus struct {
	// The generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`
	// The results of type checking for each expression.
	// Presence of this field indicates the completion of the type checking.
	// +optional
	TypeChecking *TypeChecking `json:"typeChecking,omitempty" protobuf:"bytes,2,opt,name=typeChecking"`
	// The conditions represent the latest available observations of a policy's current state.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" protobuf:"bytes,3,rep,name=conditions"`
}

// ValidatingAdmissionPolicyConditionType is the condition type of admission validation policy.
type ValidatingAdmissionPolicyConditionType string

// TypeChecking contains results of type checking the expressions in the
// ValidatingAdmissionPolicy
type TypeChecking struct {
	// The type checking warnings for each expression.
	// +optional
	// +listType=atomic
	ExpressionWarnings []ExpressionWarning `json:"expressionWarnings,omitempty" protobuf:"bytes,1,rep,name=expressionWarnings"`
}

// ExpressionWarning is a warning information that targets a specific expression.
type ExpressionWarning struct {
	// The path to the field that refers the expression.
	// For example, the reference to the expression of the first item of
	// validations is "spec.validations[0].expression"
	FieldRef string `json:"fieldRef" protobuf:"bytes,2,opt,name=fieldRef"`
	// The content of type checking information in a human-readable form.
	// Each line of the warning contains the type that the expression is checked
	// against, followed by the type check error from the compiler.
	Warning string `json:"warning" protobuf:"bytes,3,opt,name=warning"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ValidatingAdmissionPolicyList is a list of ValidatingAdmissionPolicy.
type ValidatingAdmissionPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of ValidatingAdmissionPolicy.
	Items []ValidatingAdmissionPolicy `json:"items" protobuf:"bytes,2,rep,name=items"`
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
	// Validations and AuditAnnotations may not both be empty; a minimum of one Validations or AuditAnnotations is
	// required.
	// +listType=atomic
	// +optional
	Validations []Validation `json:"validations,omitempty" protobuf:"bytes,3,rep,name=validations"`

	// failurePolicy defines how to handle failures for the admission policy. Failures can
	// occur from CEL expression parse errors, type check errors, runtime errors and invalid
	// or mis-configured policy definitions or bindings.
	//
	// A policy is invalid if spec.paramKind refers to a non-existent Kind.
	// A binding is invalid if spec.paramRef.name refers to a non-existent resource.
	//
	// failurePolicy does not define how validations that evaluate to false are handled.
	//
	// When failurePolicy is set to Fail, ValidatingAdmissionPolicyBinding validationActions
	// define how failures are enforced.
	//
	// Allowed values are Ignore or Fail. Defaults to Fail.
	// +optional
	FailurePolicy *FailurePolicyType `json:"failurePolicy,omitempty" protobuf:"bytes,4,opt,name=failurePolicy,casttype=FailurePolicyType"`

	// auditAnnotations contains CEL expressions which are used to produce audit
	// annotations for the audit event of the API request.
	// validations and auditAnnotations may not both be empty; a least one of validations or auditAnnotations is
	// required.
	// +listType=atomic
	// +optional
	AuditAnnotations []AuditAnnotation `json:"auditAnnotations,omitempty" protobuf:"bytes,5,rep,name=auditAnnotations"`

	// MatchConditions is a list of conditions that must be met for a request to be validated.
	// Match conditions filter requests that have already been matched by the rules,
	// namespaceSelector, and objectSelector. An empty list of matchConditions matches all requests.
	// There are a maximum of 64 match conditions allowed.
	//
	// If a parameter object is provided, it can be accessed via the `params` handle in the same
	// manner as validation expressions.
	//
	// The exact matching logic is (in order):
	//   1. If ANY matchCondition evaluates to FALSE, the policy is skipped.
	//   2. If ALL matchConditions evaluate to TRUE, the policy is evaluated.
	//   3. If any matchCondition evaluates to an error (but none are FALSE):
	//      - If failurePolicy=Fail, reject the request
	//      - If failurePolicy=Ignore, the policy is skipped
	//
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +optional
	MatchConditions []MatchCondition `json:"matchConditions,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,6,rep,name=matchConditions"`

	// Variables contain definitions of variables that can be used in composition of other expressions.
	// Each variable is defined as a named CEL expression.
	// The variables defined here will be available under `variables` in other expressions of the policy
	// except MatchConditions because MatchConditions are evaluated before the rest of the policy.
	//
	// The expression of a variable can refer to other variables defined earlier in the list but not those after.
	// Thus, Variables must be sorted by the order of first appearance and acyclic.
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +optional
	Variables []Variable `json:"variables,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,7,rep,name=variables"`
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
	// CEL expressions have access to the contents of the API request/response, organized into CEL variables as well as some other useful variables:
	//
	// - 'object' - The object from the incoming request. The value is null for DELETE requests.
	// - 'oldObject' - The existing object. The value is null for CREATE requests.
	// - 'request' - Attributes of the API request([ref](/pkg/apis/admission/types.go#AdmissionRequest)).
	// - 'params' - Parameter resource referred to by the policy binding being evaluated. Only populated if the policy has a ParamKind.
	// - 'namespaceObject' - The namespace object that the incoming object belongs to. The value is null for cluster-scoped resources.
	// - 'variables' - Map of composited variables, from its name to its lazily evaluated value.
	//   For example, a variable named 'foo' can be accessed as 'variables.foo'.
	// - 'authorizer' - A CEL Authorizer. May be used to perform authorization checks for the principal (user or service account) of the request.
	//   See https://pkg.go.dev/k8s.io/apiserver/pkg/cel/library#Authz
	// - 'authorizer.requestResource' - A CEL ResourceCheck constructed from the 'authorizer' and configured with the
	//   request resource.
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
	// messageExpression declares a CEL expression that evaluates to the validation failure message that is returned when this rule fails.
	// Since messageExpression is used as a failure message, it must evaluate to a string.
	// If both message and messageExpression are present on a validation, then messageExpression will be used if validation fails.
	// If messageExpression results in a runtime error, the runtime error is logged, and the validation failure message is produced
	// as if the messageExpression field were unset. If messageExpression evaluates to an empty string, a string with only spaces, or a string
	// that contains line breaks, then the validation failure message will also be produced as if the messageExpression field were unset, and
	// the fact that messageExpression produced an empty string/string with only spaces/string with line breaks will be logged.
	// messageExpression has access to all the same variables as the `expression` except for 'authorizer' and 'authorizer.requestResource'.
	// Example:
	// "object.x must be less than max ("+string(params.max)+")"
	// +optional
	MessageExpression string `json:"messageExpression,omitempty" protobuf:"bytes,4,opt,name=messageExpression"`
}

// Variable is the definition of a variable that is used for composition. A variable is defined as a named expression.
// +structType=atomic
type Variable struct {
	// Name is the name of the variable. The name must be a valid CEL identifier and unique among all variables.
	// The variable can be accessed in other expressions through `variables`
	// For example, if name is "foo", the variable will be available as `variables.foo`
	Name string `json:"name" protobuf:"bytes,1,opt,name=Name"`

	// Expression is the expression that will be evaluated as the value of the variable.
	// The CEL expression has access to the same identifiers as the CEL expressions in Validation.
	Expression string `json:"expression" protobuf:"bytes,2,opt,name=Expression"`
}

// AuditAnnotation describes how to produce an audit annotation for an API request.
type AuditAnnotation struct {
	// key specifies the audit annotation key. The audit annotation keys of
	// a ValidatingAdmissionPolicy must be unique. The key must be a qualified
	// name ([A-Za-z0-9][-A-Za-z0-9_.]*) no more than 63 bytes in length.
	//
	// The key is combined with the resource name of the
	// ValidatingAdmissionPolicy to construct an audit annotation key:
	// "{ValidatingAdmissionPolicy name}/{key}".
	//
	// If an admission webhook uses the same resource name as this ValidatingAdmissionPolicy
	// and the same audit annotation key, the annotation key will be identical.
	// In this case, the first annotation written with the key will be included
	// in the audit event and all subsequent annotations with the same key
	// will be discarded.
	//
	// Required.
	Key string `json:"key" protobuf:"bytes,1,opt,name=key"`

	// valueExpression represents the expression which is evaluated by CEL to
	// produce an audit annotation value. The expression must evaluate to either
	// a string or null value. If the expression evaluates to a string, the
	// audit annotation is included with the string value. If the expression
	// evaluates to null or empty string the audit annotation will be omitted.
	// The valueExpression may be no longer than 5kb in length.
	// If the result of the valueExpression is more than 10kb in length, it
	// will be truncated to 10kb.
	//
	// If multiple ValidatingAdmissionPolicyBinding resources match an
	// API request, then the valueExpression will be evaluated for
	// each binding. All unique values produced by the valueExpressions
	// will be joined together in a comma-separated list.
	//
	// Required.
	ValueExpression string `json:"valueExpression" protobuf:"bytes,2,opt,name=valueExpression"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ValidatingAdmissionPolicyBinding binds the ValidatingAdmissionPolicy with paramerized resources.
// ValidatingAdmissionPolicyBinding and parameter CRDs together define how cluster administrators configure policies for clusters.
//
// For a given admission request, each binding will cause its policy to be
// evaluated N times, where N is 1 for policies/bindings that don't use
// params, otherwise N is the number of parameters selected by the binding.
//
// The CEL expressions of a policy must have a computed CEL cost below the maximum
// CEL budget. Each evaluation of the policy is given an independent CEL cost budget.
// Adding/removing policies, bindings, or params can not affect whether a
// given (policy, binding, param) combination is within its own CEL budget.
type ValidatingAdmissionPolicyBinding struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Specification of the desired behavior of the ValidatingAdmissionPolicyBinding.
	Spec ValidatingAdmissionPolicyBindingSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ValidatingAdmissionPolicyBindingList is a list of ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of PolicyBinding.
	Items []ValidatingAdmissionPolicyBinding `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ValidatingAdmissionPolicyBindingSpec is the specification of the ValidatingAdmissionPolicyBinding.
type ValidatingAdmissionPolicyBindingSpec struct {
	// PolicyName references a ValidatingAdmissionPolicy name which the ValidatingAdmissionPolicyBinding binds to.
	// If the referenced resource does not exist, this binding is considered invalid and will be ignored
	// Required.
	PolicyName string `json:"policyName,omitempty" protobuf:"bytes,1,rep,name=policyName"`

	// paramRef specifies the parameter resource used to configure the admission control policy.
	// It should point to a resource of the type specified in ParamKind of the bound ValidatingAdmissionPolicy.
	// If the policy specifies a ParamKind and the resource referred to by ParamRef does not exist, this binding is considered mis-configured and the FailurePolicy of the ValidatingAdmissionPolicy applied.
	// If the policy does not specify a ParamKind then this field is ignored, and the rules are evaluated without a param.
	// +optional
	ParamRef *ParamRef `json:"paramRef,omitempty" protobuf:"bytes,2,rep,name=paramRef"`

	// MatchResources declares what resources match this binding and will be validated by it.
	// Note that this is intersected with the policy's matchConstraints, so only requests that are matched by the policy can be selected by this.
	// If this is unset, all resources matched by the policy are validated by this binding
	// When resourceRules is unset, it does not constrain resource matching. If a resource is matched by the other fields of this object, it will be validated.
	// Note that this is differs from ValidatingAdmissionPolicy matchConstraints, where resourceRules are required.
	// +optional
	MatchResources *MatchResources `json:"matchResources,omitempty" protobuf:"bytes,3,rep,name=matchResources"`

	// validationActions declares how Validations of the referenced ValidatingAdmissionPolicy are enforced.
	// If a validation evaluates to false it is always enforced according to these actions.
	//
	// Failures defined by the ValidatingAdmissionPolicy's FailurePolicy are enforced according
	// to these actions only if the FailurePolicy is set to Fail, otherwise the failures are
	// ignored. This includes compilation errors, runtime errors and misconfigurations of the policy.
	//
	// validationActions is declared as a set of action values. Order does
	// not matter. validationActions may not contain duplicates of the same action.
	//
	// The supported actions values are:
	//
	// "Deny" specifies that a validation failure results in a denied request.
	//
	// "Warn" specifies that a validation failure is reported to the request client
	// in HTTP Warning headers, with a warning code of 299. Warnings can be sent
	// both for allowed or denied admission responses.
	//
	// "Audit" specifies that a validation failure is included in the published
	// audit event for the request. The audit event will contain a
	// `validation.policy.admission.k8s.io/validation_failure` audit annotation
	// with a value containing the details of the validation failures, formatted as
	// a JSON list of objects, each with the following fields:
	// - message: The validation failure message string
	// - policy: The resource name of the ValidatingAdmissionPolicy
	// - binding: The resource name of the ValidatingAdmissionPolicyBinding
	// - expressionIndex: The index of the failed validations in the ValidatingAdmissionPolicy
	// - validationActions: The enforcement actions enacted for the validation failure
	// Example audit annotation:
	// `"validation.policy.admission.k8s.io/validation_failure": "[{\"message\": \"Invalid value\", {\"policy\": \"policy.example.com\", {\"binding\": \"policybinding.example.com\", {\"expressionIndex\": \"1\", {\"validationActions\": [\"Audit\"]}]"`
	//
	// Clients should expect to handle additional values by ignoring
	// any values not recognized.
	//
	// "Deny" and "Warn" may not be used together since this combination
	// needlessly duplicates the validation failure both in the
	// API response body and the HTTP warning headers.
	//
	// Required.
	// +listType=set
	ValidationActions []ValidationAction `json:"validationActions,omitempty" protobuf:"bytes,4,rep,name=validationActions"`
}

// ParamRef describes how to locate the params to be used as input to
// expressions of rules applied by a policy binding.
// +structType=atomic
type ParamRef struct {
	// name is the name of the resource being referenced.
	//
	// One of `name` or `selector` must be set, but `name` and `selector` are
	// mutually exclusive properties. If one is set, the other must be unset.
	//
	// A single parameter used for all admission requests can be configured
	// by setting the `name` field, leaving `selector` blank, and setting namespace
	// if `paramKind` is namespace-scoped.
	//
	Name string `json:"name,omitempty" protobuf:"bytes,1,rep,name=name"`

	// namespace is the namespace of the referenced resource. Allows limiting
	// the search for params to a specific namespace. Applies to both `name` and
	// `selector` fields.
	//
	// A per-namespace parameter may be used by specifying a namespace-scoped
	// `paramKind` in the policy and leaving this field empty.
	//
	// - If `paramKind` is cluster-scoped, this field MUST be unset. Setting this
	// field results in a configuration error.
	//
	// - If `paramKind` is namespace-scoped, the namespace of the object being
	// evaluated for admission will be used when this field is left unset. Take
	// care that if this is left empty the binding must not match any cluster-scoped
	// resources, which will result in an error.
	//
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,2,rep,name=namespace"`

	// selector can be used to match multiple param objects based on their labels.
	// Supply selector: {} to match all resources of the ParamKind.
	//
	// If multiple params are found, they are all evaluated with the policy expressions
	// and the results are ANDed together.
	//
	// One of `name` or `selector` must be set, but `name` and `selector` are
	// mutually exclusive properties. If one is set, the other must be unset.
	//
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,3,rep,name=selector"`

	// `parameterNotFoundAction` controls the behavior of the binding when the resource
	// exists, and name or selector is valid, but there are no parameters
	// matched by the binding. If the value is set to `Allow`, then no
	// matched parameters will be treated as successful validation by the binding.
	// If set to `Deny`, then no matched parameters will be subject to the
	// `failurePolicy` of the policy.
	//
	// Allowed values are `Allow` or `Deny`
	//
	// Required
	ParameterNotFoundAction *ParameterNotFoundActionType `json:"parameterNotFoundAction,omitempty" protobuf:"bytes,4,rep,name=parameterNotFoundAction"`
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

// ValidationAction specifies a policy enforcement action.
// +enum
type ValidationAction string

const (
	// Deny specifies that a validation failure results in a denied request.
	Deny ValidationAction = "Deny"
	// Warn specifies that a validation failure is reported to the request client
	// in HTTP Warning headers, with a warning code of 299. Warnings can be sent
	// both for allowed or denied admission responses.
	Warn ValidationAction = "Warn"
	// Audit specifies that a validation failure is included in the published
	// audit event for the request. The audit event will contain a
	// `validation.policy.admission.k8s.io/validation_failure` audit annotation
	// with a value containing the details of the validation failure.
	Audit ValidationAction = "Audit"
)

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

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.16

// ValidatingWebhookConfiguration describes the configuration of and admission webhook that accept or reject and object without changing it.
type ValidatingWebhookConfiguration struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Webhooks is a list of webhooks and the affected resources and operations.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	Webhooks []ValidatingWebhook `json:"webhooks,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,2,rep,name=Webhooks"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.16

// ValidatingWebhookConfigurationList is a list of ValidatingWebhookConfiguration.
type ValidatingWebhookConfigurationList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of ValidatingWebhookConfiguration.
	Items []ValidatingWebhookConfiguration `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.16

// MutatingWebhookConfiguration describes the configuration of and admission webhook that accept or reject and may change the object.
type MutatingWebhookConfiguration struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Webhooks is a list of webhooks and the affected resources and operations.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	Webhooks []MutatingWebhook `json:"webhooks,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,2,rep,name=Webhooks"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.16

// MutatingWebhookConfigurationList is a list of MutatingWebhookConfiguration.
type MutatingWebhookConfigurationList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// List of MutatingWebhookConfiguration.
	Items []MutatingWebhookConfiguration `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ValidatingWebhook describes an admission webhook and the resources and operations it applies to.
type ValidatingWebhook struct {
	// The name of the admission webhook.
	// Name should be fully qualified, e.g., imagepolicy.kubernetes.io, where
	// "imagepolicy" is the name of the webhook, and kubernetes.io is the name
	// of the organization.
	// Required.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// ClientConfig defines how to communicate with the hook.
	// Required
	ClientConfig WebhookClientConfig `json:"clientConfig" protobuf:"bytes,2,opt,name=clientConfig"`

	// Rules describes what operations on what resources/subresources the webhook cares about.
	// The webhook cares about an operation if it matches _any_ Rule.
	// However, in order to prevent ValidatingAdmissionWebhooks and MutatingAdmissionWebhooks
	// from putting the cluster in a state which cannot be recovered from without completely
	// disabling the plugin, ValidatingAdmissionWebhooks and MutatingAdmissionWebhooks are never called
	// on admission requests for ValidatingWebhookConfiguration and MutatingWebhookConfiguration objects.
	// +listType=atomic
	Rules []RuleWithOperations `json:"rules,omitempty" protobuf:"bytes,3,rep,name=rules"`

	// FailurePolicy defines how unrecognized errors from the admission endpoint are handled -
	// allowed values are Ignore or Fail. Defaults to Fail.
	// +optional
	FailurePolicy *FailurePolicyType `json:"failurePolicy,omitempty" protobuf:"bytes,4,opt,name=failurePolicy,casttype=FailurePolicyType"`

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
	// Defaults to "Equivalent"
	// +optional
	MatchPolicy *MatchPolicyType `json:"matchPolicy,omitempty" protobuf:"bytes,9,opt,name=matchPolicy,casttype=MatchPolicyType"`

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
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels
	// for more examples of label selectors.
	//
	// Default to the empty LabelSelector, which matches everything.
	// +optional
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty" protobuf:"bytes,5,opt,name=namespaceSelector"`

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
	ObjectSelector *metav1.LabelSelector `json:"objectSelector,omitempty" protobuf:"bytes,10,opt,name=objectSelector"`

	// SideEffects states whether this webhook has side effects.
	// Acceptable values are: None, NoneOnDryRun (webhooks created via v1beta1 may also specify Some or Unknown).
	// Webhooks with side effects MUST implement a reconciliation system, since a request may be
	// rejected by a future step in the admission chain and the side effects therefore need to be undone.
	// Requests with the dryRun attribute will be auto-rejected if they match a webhook with
	// sideEffects == Unknown or Some.
	SideEffects *SideEffectClass `json:"sideEffects" protobuf:"bytes,6,opt,name=sideEffects,casttype=SideEffectClass"`

	// TimeoutSeconds specifies the timeout for this webhook. After the timeout passes,
	// the webhook call will be ignored or the API call will fail based on the
	// failure policy.
	// The timeout value must be between 1 and 30 seconds.
	// Default to 10 seconds.
	// +optional
	TimeoutSeconds *int32 `json:"timeoutSeconds,omitempty" protobuf:"varint,7,opt,name=timeoutSeconds"`

	// AdmissionReviewVersions is an ordered list of preferred `AdmissionReview`
	// versions the Webhook expects. API server will try to use first version in
	// the list which it supports. If none of the versions specified in this list
	// supported by API server, validation will fail for this object.
	// If a persisted webhook configuration specifies allowed versions and does not
	// include any versions known to the API Server, calls to the webhook will fail
	// and be subject to the failure policy.
	// +listType=atomic
	AdmissionReviewVersions []string `json:"admissionReviewVersions" protobuf:"bytes,8,rep,name=admissionReviewVersions"`

	// MatchConditions is a list of conditions that must be met for a request to be sent to this
	// webhook. Match conditions filter requests that have already been matched by the rules,
	// namespaceSelector, and objectSelector. An empty list of matchConditions matches all requests.
	// There are a maximum of 64 match conditions allowed.
	//
	// The exact matching logic is (in order):
	//   1. If ANY matchCondition evaluates to FALSE, the webhook is skipped.
	//   2. If ALL matchConditions evaluate to TRUE, the webhook is called.
	//   3. If any matchCondition evaluates to an error (but none are FALSE):
	//      - If failurePolicy=Fail, reject the request
	//      - If failurePolicy=Ignore, the error is ignored and the webhook is skipped
	//
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +optional
	MatchConditions []MatchCondition `json:"matchConditions,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,11,opt,name=matchConditions"`
}

// MutatingWebhook describes an admission webhook and the resources and operations it applies to.
type MutatingWebhook struct {
	// The name of the admission webhook.
	// Name should be fully qualified, e.g., imagepolicy.kubernetes.io, where
	// "imagepolicy" is the name of the webhook, and kubernetes.io is the name
	// of the organization.
	// Required.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// ClientConfig defines how to communicate with the hook.
	// Required
	ClientConfig WebhookClientConfig `json:"clientConfig" protobuf:"bytes,2,opt,name=clientConfig"`

	// Rules describes what operations on what resources/subresources the webhook cares about.
	// The webhook cares about an operation if it matches _any_ Rule.
	// However, in order to prevent ValidatingAdmissionWebhooks and MutatingAdmissionWebhooks
	// from putting the cluster in a state which cannot be recovered from without completely
	// disabling the plugin, ValidatingAdmissionWebhooks and MutatingAdmissionWebhooks are never called
	// on admission requests for ValidatingWebhookConfiguration and MutatingWebhookConfiguration objects.
	// +listType=atomic
	Rules []RuleWithOperations `json:"rules,omitempty" protobuf:"bytes,3,rep,name=rules"`

	// FailurePolicy defines how unrecognized errors from the admission endpoint are handled -
	// allowed values are Ignore or Fail. Defaults to Fail.
	// +optional
	FailurePolicy *FailurePolicyType `json:"failurePolicy,omitempty" protobuf:"bytes,4,opt,name=failurePolicy,casttype=FailurePolicyType"`

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
	// Defaults to "Equivalent"
	// +optional
	MatchPolicy *MatchPolicyType `json:"matchPolicy,omitempty" protobuf:"bytes,9,opt,name=matchPolicy,casttype=MatchPolicyType"`

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
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty" protobuf:"bytes,5,opt,name=namespaceSelector"`

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
	ObjectSelector *metav1.LabelSelector `json:"objectSelector,omitempty" protobuf:"bytes,11,opt,name=objectSelector"`

	// SideEffects states whether this webhook has side effects.
	// Acceptable values are: None, NoneOnDryRun (webhooks created via v1beta1 may also specify Some or Unknown).
	// Webhooks with side effects MUST implement a reconciliation system, since a request may be
	// rejected by a future step in the admission chain and the side effects therefore need to be undone.
	// Requests with the dryRun attribute will be auto-rejected if they match a webhook with
	// sideEffects == Unknown or Some.
	SideEffects *SideEffectClass `json:"sideEffects" protobuf:"bytes,6,opt,name=sideEffects,casttype=SideEffectClass"`

	// TimeoutSeconds specifies the timeout for this webhook. After the timeout passes,
	// the webhook call will be ignored or the API call will fail based on the
	// failure policy.
	// The timeout value must be between 1 and 30 seconds.
	// Default to 10 seconds.
	// +optional
	TimeoutSeconds *int32 `json:"timeoutSeconds,omitempty" protobuf:"varint,7,opt,name=timeoutSeconds"`

	// AdmissionReviewVersions is an ordered list of preferred `AdmissionReview`
	// versions the Webhook expects. API server will try to use first version in
	// the list which it supports. If none of the versions specified in this list
	// supported by API server, validation will fail for this object.
	// If a persisted webhook configuration specifies allowed versions and does not
	// include any versions known to the API Server, calls to the webhook will fail
	// and be subject to the failure policy.
	// +listType=atomic
	AdmissionReviewVersions []string `json:"admissionReviewVersions" protobuf:"bytes,8,rep,name=admissionReviewVersions"`

	// reinvocationPolicy indicates whether this webhook should be called multiple times as part of a single admission evaluation.
	// Allowed values are "Never" and "IfNeeded".
	//
	// Never: the webhook will not be called more than once in a single admission evaluation.
	//
	// IfNeeded: the webhook will be called at least one additional time as part of the admission evaluation
	// if the object being admitted is modified by other admission plugins after the initial webhook call.
	// Webhooks that specify this option *must* be idempotent, able to process objects they previously admitted.
	// Note:
	// * the number of additional invocations is not guaranteed to be exactly one.
	// * if additional invocations result in further modifications to the object, webhooks are not guaranteed to be invoked again.
	// * webhooks that use this option may be reordered to minimize the number of additional invocations.
	// * to validate an object after all mutations are guaranteed complete, use a validating admission webhook instead.
	//
	// Defaults to "Never".
	// +optional
	ReinvocationPolicy *ReinvocationPolicyType `json:"reinvocationPolicy,omitempty" protobuf:"bytes,10,opt,name=reinvocationPolicy,casttype=ReinvocationPolicyType"`

	// MatchConditions is a list of conditions that must be met for a request to be sent to this
	// webhook. Match conditions filter requests that have already been matched by the rules,
	// namespaceSelector, and objectSelector. An empty list of matchConditions matches all requests.
	// There are a maximum of 64 match conditions allowed.
	//
	// The exact matching logic is (in order):
	//   1. If ANY matchCondition evaluates to FALSE, the webhook is skipped.
	//   2. If ALL matchConditions evaluate to TRUE, the webhook is called.
	//   3. If any matchCondition evaluates to an error (but none are FALSE):
	//      - If failurePolicy=Fail, reject the request
	//      - If failurePolicy=Ignore, the error is ignored and the webhook is skipped
	//
	// +patchMergeKey=name
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=name
	// +optional
	MatchConditions []MatchCondition `json:"matchConditions,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,12,opt,name=matchConditions"`
}

// ReinvocationPolicyType specifies what type of policy is used when other admission plugins also perform
// modifications.
// +enum
type ReinvocationPolicyType string

const (
	// NeverReinvocationPolicy indicates that the mutation must not be called more than once in a
	// single admission evaluation.
	NeverReinvocationPolicy ReinvocationPolicyType = "Never"
	// IfNeededReinvocationPolicy indicates that the mutation may be called at least one
	// additional time as part of the admission evaluation if the object being admitted is
	// modified by other admission plugins after the initial mutation call.
	IfNeededReinvocationPolicy ReinvocationPolicyType = "IfNeeded"
)

// RuleWithOperations is a tuple of Operations and Resources. It is recommended to make
// sure that all the tuple expansions are valid.
type RuleWithOperations struct {
	// Operations is the operations the admission hook cares about - CREATE, UPDATE, DELETE, CONNECT or *
	// for all of those operations and any future admission operations that are added.
	// If '*' is present, the length of the slice must be one.
	// Required.
	// +listType=atomic
	Operations []OperationType `json:"operations,omitempty" protobuf:"bytes,1,rep,name=operations,casttype=OperationType"`
	// Rule is embedded, it describes other criteria of the rule, like
	// APIGroups, APIVersions, Resources, etc.
	Rule `json:",inline" protobuf:"bytes,2,opt,name=rule"`
}

// OperationType specifies an operation for a request.
// +enum
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
	URL *string `json:"url,omitempty" protobuf:"bytes,3,opt,name=url"`

	// `service` is a reference to the service for this webhook. Either
	// `service` or `url` must be specified.
	//
	// If the webhook is running within the cluster, then you should use `service`.
	//
	// +optional
	Service *ServiceReference `json:"service,omitempty" protobuf:"bytes,1,opt,name=service"`

	// `caBundle` is a PEM encoded CA bundle which will be used to validate the webhook's server certificate.
	// If unspecified, system trust roots on the apiserver are used.
	// +optional
	CABundle []byte `json:"caBundle,omitempty" protobuf:"bytes,2,opt,name=caBundle"`
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// `namespace` is the namespace of the service.
	// Required
	Namespace string `json:"namespace" protobuf:"bytes,1,opt,name=namespace"`
	// `name` is the name of the service.
	// Required
	Name string `json:"name" protobuf:"bytes,2,opt,name=name"`

	// `path` is an optional URL path which will be sent in any request to
	// this service.
	// +optional
	Path *string `json:"path,omitempty" protobuf:"bytes,3,opt,name=path"`

	// If specified, the port on the service that hosting webhook.
	// Default to 443 for backward compatibility.
	// `port` should be a valid port number (1-65535, inclusive).
	// +optional
	Port *int32 `json:"port,omitempty" protobuf:"varint,4,opt,name=port"`
}

// MatchCondition represents a condition which must by fulfilled for a request to be sent to a webhook.
type MatchCondition struct {
	// Name is an identifier for this match condition, used for strategic merging of MatchConditions,
	// as well as providing an identifier for logging purposes. A good name should be descriptive of
	// the associated expression.
	// Name must be a qualified name consisting of alphanumeric characters, '-', '_' or '.', and
	// must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or
	// '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]') with an
	// optional DNS subdomain prefix and '/' (e.g. 'example.com/MyName')
	//
	// Required.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// Expression represents the expression which will be evaluated by CEL. Must evaluate to bool.
	// CEL expressions have access to the contents of the AdmissionRequest and Authorizer, organized into CEL variables:
	//
	// 'object' - The object from the incoming request. The value is null for DELETE requests.
	// 'oldObject' - The existing object. The value is null for CREATE requests.
	// 'request' - Attributes of the admission request(/pkg/apis/admission/types.go#AdmissionRequest).
	// 'authorizer' - A CEL Authorizer. May be used to perform authorization checks for the principal (user or service account) of the request.
	//   See https://pkg.go.dev/k8s.io/apiserver/pkg/cel/library#Authz
	// 'authorizer.requestResource' - A CEL ResourceCheck constructed from the 'authorizer' and configured with the
	//   request resource.
	// Documentation on CEL: https://kubernetes.io/docs/reference/using-api/cel/
	//
	// Required.
	Expression string `json:"expression" protobuf:"bytes,2,opt,name=expression"`
}
