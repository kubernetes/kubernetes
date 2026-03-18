/*
Copyright The Kubernetes Authors.

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
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36

// AuthorizationConditionsReview describes a request to evaluate authorization conditions.
type AuthorizationConditionsReview struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard list metadata.
	// In AuthorizationConditionsReview, it must be an empty struct.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// request describes the attributes for the authorization conditions request.
	// +optional
	Request *AuthorizationConditionsRequest `json:"request,omitempty" protobuf:"bytes,2,opt,name=request"`
	// response describes the attributes for the authorization conditions response.
	// +optional
	Response *AuthorizationConditionsResponse `json:"response,omitempty" protobuf:"bytes,3,opt,name=response"`
}

// AuthorizationConditionsRequest describes the authorization conditions request.
type AuthorizationConditionsRequest struct {
	// decision contains the conditional decision the authorizer authored at authorization time.
	// +required
	Decision ConditionsAwareDecision `json:"decision" protobuf:"bytes,1,opt,name=decision"`

	// admissionControlData may contain additional information for evaluating the conditions.
	// +optional
	AdmissionControlData *AuthorizationConditionsTargetAdmissionControl `json:"admissionControlData,omitempty" protobuf:"bytes,2,opt,name=admissionControlData"`
}

// AuthorizationConditionsTargetAdmissionControl contains the data available during admission control,
// against which authorization decisions can be written. It follows the same structure as AdmissionReview.
type AuthorizationConditionsTargetAdmissionControl struct {
	// UID is an identifier for the individual request/response. It allows us to distinguish instances of requests which are
	// otherwise identical (parallel requests, requests when earlier requests did not modify etc)
	// The UID is meant to track the round trip (request/response) between the KAS and the WebHook, not the user request.
	// It is suitable for correlating log entries between the webhook and apiserver, for either auditing or debugging.
	// TODO(luxas): This is kept here in case we want it in the future, and the same proto binding for it is
	// reserved as for AdmissionReview.
	// UID types.UID `json:"uid" protobuf:"bytes,1,opt,name=uid"`

	// These fields are reserved here for future use, in case conditional authorization would need to support something
	// similar to AdmissionReview's matchPolicy: Equivalent. That is not supported to begin with, though.
	// Kind is the fully-qualified type of object being submitted (for example, v1.Pod or autoscaling.v1.Scale)
	// Kind metav1.GroupVersionKind `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Resource is the fully-qualified resource being requested (for example, v1.pods)
	// Resource metav1.GroupVersionResource `json:"resource" protobuf:"bytes,3,opt,name=resource"`
	// SubResource is the subresource being requested, if any (for example, "status" or "scale")
	// +optional
	// SubResource string `json:"subResource,omitempty" protobuf:"bytes,4,opt,name=subResource"`

	// requestKind is the fully-qualified type of the original API request (for example, v1.Pod or autoscaling.v1.Scale).
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
	RequestKind *metav1.GroupVersionKind `json:"requestKind,omitempty" protobuf:"bytes,14,opt,name=requestKind"`
	// requestResource is the fully-qualified resource of the original API request (for example, v1.pods).
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
	RequestResource *metav1.GroupVersionResource `json:"requestResource,omitempty" protobuf:"bytes,15,opt,name=requestResource"`
	// requestSubResource is the name of the subresource of the original API request, if any (for example, "status" or "scale")
	// If this is specified and differs from the value in "subResource", an equivalent match and conversion was performed.
	// See documentation for the "matchPolicy" field in the webhook configuration type.
	// +optional
	RequestSubResource string `json:"requestSubResource,omitempty" protobuf:"bytes,16,opt,name=requestSubResource"`

	// name is the name of the object as presented in the request. On a CREATE operation, the client may omit name and
	// rely on the server to generate the name. If that is the case, this field will contain an empty string.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,5,opt,name=name"`
	// namespace is the namespace associated with the request (if any).
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,6,opt,name=namespace"`
	// operation is the operation being performed. This may be different than the operation
	// requested. e.g. a patch can result in either a CREATE or UPDATE Operation.
	Operation admissionv1.Operation `json:"operation" protobuf:"bytes,7,opt,name=operation"`

	// userInfo is information about the requesting user
	UserInfo authenticationv1.UserInfo `json:"userInfo" protobuf:"bytes,9,opt,name=userInfo"`
	// object is the object from the incoming request.
	// +optional
	Object runtime.RawExtension `json:"object,omitempty" protobuf:"bytes,10,opt,name=object"`
	// oldObject is the existing object. Only populated for DELETE and UPDATE requests.
	// +optional
	OldObject runtime.RawExtension `json:"oldObject,omitempty" protobuf:"bytes,11,opt,name=oldObject"`
	// dryRun indicates that modifications will definitely not be persisted for this request.
	// Defaults to false.
	// +optional
	DryRun *bool `json:"dryRun,omitempty" protobuf:"varint,12,opt,name=dryRun"`
	// options is the operation option structure of the operation being performed.
	// e.g. `meta.k8s.io/v1.DeleteOptions` or `meta.k8s.io/v1.CreateOptions`. This may be
	// different than the options the caller provided. e.g. for a patch request the performed
	// Operation might be a CREATE, in which case the Options will a
	// `meta.k8s.io/v1.CreateOptions` even though the caller provided `meta.k8s.io/v1.PatchOptions`.
	// +optional
	Options runtime.RawExtension `json:"options,omitempty" protobuf:"bytes,13,opt,name=options"`
}

// AuthorizationConditionsResponse describes an authorization conditions response.
type AuthorizationConditionsResponse struct {
	// decision contains the authorizer's decision after seeing the data.
	// +required
	Decision ConditionsAwareDecision `json:"decision" protobuf:"bytes,1,opt,name=decision"`
}

// ConditionEffect specifies how a condition evaluating to
// true should be treated.
// +enum
type ConditionEffect string

const (
	// ConditionEffectAllow means that if this condition
	// evaluates to true, the ConditionsMap evaluates to Allow, unless any
	// Deny/NoOpinion condition also evaluates to true.
	ConditionEffectAllow ConditionEffect = "Allow"

	// ConditionEffectDeny means that if this condition
	// evaluates to true, the ConditionsMap necessarily evaluates to Deny.
	// No further authorizers are consulted.
	ConditionEffectDeny ConditionEffect = "Deny"

	// ConditionEffectNoOpinion means that if this condition
	// evaluates to true, the given authorizer's ConditionsMap cannot evaluate
	// to Allow anymore, but necessarily Deny or NoOpinion.
	ConditionEffectNoOpinion ConditionEffect = "NoOpinion"
)

// Condition represents a single authorization condition to be evaluated against
// data available later in the request chain, e.g. objects available in admission.
type Condition struct {
	// id uniquely identifies this condition within the scope of the authorizer
	// that authored it. Validated as a Kubernetes label key.
	// +required
	ID string `json:"id" protobuf:"bytes,1,opt,name=id"`

	// effect specifies how the condition evaluating to "true" should be treated.
	// +required
	Effect ConditionEffect `json:"effect" protobuf:"bytes,2,opt,name=effect"`

	// condition is a string encoding of the condition to be evaluated.
	// It is a pure, deterministic function from condition data to a boolean (or error).
	// Might or might not be human-readable.
	// Optional, if the ID alone is enough for the authorizer to know how to evaluate the condition.
	// +optional
	Condition string `json:"condition,omitempty" protobuf:"bytes,3,opt,name=condition"`

	// type describes the type (format/encoding/language) of the condition,
	// if there are multiple possibilities. Should be formatted as a Kubernetes label key.
	// Any domain suffix of *.k8s.io or *.kubernetes.io is reserved for Kubernetes use.
	// +optional
	Type string `json:"type,omitempty" protobuf:"bytes,4,opt,name=type"`

	// description is an optional human-friendly description that can be shown
	// as an error message or for debugging.
	// +optional
	Description string `json:"description,omitempty" protobuf:"bytes,5,opt,name=description"`
}

// ConditionsMap represents a map of conditions.
type ConditionsMap struct {
	// conditions is an unordered map of conditions, keyed by ID, that shall be evaluated
	// data available later, to determine whether the authorizer that authored the conditions
	// allows or denies the request.
	// If any ConditionsEffect=Deny condition evaluates to true or errors, the evaluated decision must be Deny.
	// Else if any ConditionsEffect=NoOpinion condition evaluates to true or errors, the evaluated decision must be NoOpinion.
	// Else if any ConditionsEffect=Allow condition evaluates to true, the evaluated decision must be Allow.
	// Else, the evaluated decision must be NoOpinion.
	// +listType=map
	// +listMapKey=id
	// +required
	Conditions []Condition `json:"conditions" protobuf:"bytes,1,rep,name=conditions"` //nolint:kubeapilinter // These are authorization conditions.
}

// ConditionsAwareDecisionType is an enum representing what kind of authorization decision
// the ConditionsAwareDecision represents. The zero value represents the Deny type.
// +enum
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
	// +required
	Type ConditionsAwareDecisionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=ConditionsAwareDecisionType"`

	// reason is optional. It indicates why a request was allowed or denied.
	// Only applicable when type is one of ConditionsAwareDecisionTypeDeny,
	// ConditionsAwareDecisionTypeAllow or ConditionsAwareDecisionTypeNoOpinion.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,2,opt,name=reason"`

	// evaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	// Only applicable when type is one of ConditionsAwareDecisionTypeDeny,
	// ConditionsAwareDecisionTypeAllow or ConditionsAwareDecisionTypeNoOpinion.
	// +optional
	EvaluationError string `json:"evaluationError,omitempty" protobuf:"bytes,3,opt,name=evaluationError"`

	// conditionsMap represents a conditional decision, modelled as a map of conditions.
	// Must be non-null when type == "ConditionsMap", otherwise this field must be unset.
	// +optional
	ConditionsMap *ConditionsMap `json:"conditionsMap,omitempty" protobuf:"bytes,4,opt,name=conditionsMap"`

	// union forms an ordered tree of decisions, where the union decision is represented by
	// an internal node, and all other decision types are leaf nodes. During evaluation, the
	// leaf decisions are evaluated in depth-first order, until an Allow or Deny decision is found.
	// The order of the decisions must match exactly the order of the authorizers in the union authorizer.
	// At least one of the leaves must be of type ConditionsMap, as otherwise the union could be trivially
	// reduced to just a single Allow/Deny/NoOpinion.
	//
	// Must have at least one element when type == "Union", otherwise this field must be unset.
	//
	// +optional
	// +listType=atomic
	Union []ConditionsAwareDecision `json:"union,omitempty" protobuf:"bytes,5,rep,name=union"`
}
