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

package v1beta1

import (
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AdmissionReview describes an admission review request/response.
type AdmissionReview struct {
	metav1.TypeMeta `json:",inline"`
	// Request describes the attributes for the admission request.
	// +optional
	Request *AdmissionRequest `json:"request,omitempty" protobuf:"bytes,1,opt,name=request"`
	// Response describes the attributes for the admission response.
	// +optional
	Response *AdmissionResponse `json:"response,omitempty" protobuf:"bytes,2,opt,name=response"`
}

// AdmissionRequest describes the admission.Attributes for the admission request.
type AdmissionRequest struct {
	// UID is an identifier for the individual request/response. It allows us to distinguish instances of requests which are
	// otherwise identical (parallel requests, requests when earlier requests did not modify etc)
	// The UID is meant to track the round trip (request/response) between the KAS and the WebHook, not the user request.
	// It is suitable for correlating log entries between the webhook and apiserver, for either auditing or debugging.
	UID types.UID `json:"uid" protobuf:"bytes,1,opt,name=uid"`
	// Kind is the fully-qualified type of object being submitted (for example, v1.Pod or autoscaling.v1.Scale)
	Kind metav1.GroupVersionKind `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Resource is the fully-qualified resource being requested (for example, v1.pods)
	Resource metav1.GroupVersionResource `json:"resource" protobuf:"bytes,3,opt,name=resource"`
	// SubResource is the subresource being requested, if any (for example, "status" or "scale")
	// +optional
	SubResource string `json:"subResource,omitempty" protobuf:"bytes,4,opt,name=subResource"`

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
	RequestKind *metav1.GroupVersionKind `json:"requestKind,omitempty" protobuf:"bytes,13,opt,name=requestKind"`
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
	RequestResource *metav1.GroupVersionResource `json:"requestResource,omitempty" protobuf:"bytes,14,opt,name=requestResource"`
	// RequestSubResource is the name of the subresource of the original API request, if any (for example, "status" or "scale")
	// If this is specified and differs from the value in "subResource", an equivalent match and conversion was performed.
	// See documentation for the "matchPolicy" field in the webhook configuration type.
	// +optional
	RequestSubResource string `json:"requestSubResource,omitempty" protobuf:"bytes,15,opt,name=requestSubResource"`

	// Name is the name of the object as presented in the request.  On a CREATE operation, the client may omit name and
	// rely on the server to generate the name.  If that is the case, this field will contain an empty string.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,5,opt,name=name"`
	// Namespace is the namespace associated with the request (if any).
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,6,opt,name=namespace"`
	// Operation is the operation being performed. This may be different than the operation
	// requested. e.g. a patch can result in either a CREATE or UPDATE Operation.
	Operation Operation `json:"operation" protobuf:"bytes,7,opt,name=operation"`
	// UserInfo is information about the requesting user
	UserInfo authenticationv1.UserInfo `json:"userInfo" protobuf:"bytes,8,opt,name=userInfo"`
	// Object is the object from the incoming request.
	// +optional
	Object runtime.RawExtension `json:"object,omitempty" protobuf:"bytes,9,opt,name=object"`
	// OldObject is the existing object. Only populated for DELETE and UPDATE requests.
	// +optional
	OldObject runtime.RawExtension `json:"oldObject,omitempty" protobuf:"bytes,10,opt,name=oldObject"`
	// DryRun indicates that modifications will definitely not be persisted for this request.
	// Defaults to false.
	// +optional
	DryRun *bool `json:"dryRun,omitempty" protobuf:"varint,11,opt,name=dryRun"`
	// Options is the operation option structure of the operation being performed.
	// e.g. `meta.k8s.io/v1.DeleteOptions` or `meta.k8s.io/v1.CreateOptions`. This may be
	// different than the options the caller provided. e.g. for a patch request the performed
	// Operation might be a CREATE, in which case the Options will a
	// `meta.k8s.io/v1.CreateOptions` even though the caller provided `meta.k8s.io/v1.PatchOptions`.
	// +optional
	Options runtime.RawExtension `json:"options,omitempty" protobuf:"bytes,12,opt,name=options"`
}

// AdmissionResponse describes an admission response.
type AdmissionResponse struct {
	// UID is an identifier for the individual request/response.
	// This should be copied over from the corresponding AdmissionRequest.
	UID types.UID `json:"uid" protobuf:"bytes,1,opt,name=uid"`

	// Allowed indicates whether or not the admission request was permitted.
	Allowed bool `json:"allowed" protobuf:"varint,2,opt,name=allowed"`

	// Result contains extra details into why an admission request was denied.
	// This field IS NOT consulted in any way if "Allowed" is "true".
	// +optional
	Result *metav1.Status `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`

	// The patch body. Currently we only support "JSONPatch" which implements RFC 6902.
	// +optional
	Patch []byte `json:"patch,omitempty" protobuf:"bytes,4,opt,name=patch"`

	// The type of Patch. Currently we only allow "JSONPatch".
	// +optional
	PatchType *PatchType `json:"patchType,omitempty" protobuf:"bytes,5,opt,name=patchType"`

	// AuditAnnotations is an unstructured key value map set by remote admission controller (e.g. error=image-blacklisted).
	// MutatingAdmissionWebhook and ValidatingAdmissionWebhook admission controller will prefix the keys with
	// admission webhook name (e.g. imagepolicy.example.com/error=image-blacklisted). AuditAnnotations will be provided by
	// the admission webhook to add additional context to the audit log for this request.
	// +optional
	AuditAnnotations map[string]string `json:"auditAnnotations,omitempty" protobuf:"bytes,6,opt,name=auditAnnotations"`
}

// PatchType is the type of patch being used to represent the mutated object
type PatchType string

// PatchType constants.
const (
	PatchTypeJSONPatch PatchType = "JSONPatch"
)

// Operation is the type of resource operation being checked for admission control
type Operation string

// Operation constants
const (
	Create  Operation = "CREATE"
	Update  Operation = "UPDATE"
	Delete  Operation = "DELETE"
	Connect Operation = "CONNECT"
)
