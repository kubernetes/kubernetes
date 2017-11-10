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

package v1alpha1

import (
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AdmissionReviewRequest describes an admission request.
type AdmissionReviewRequest struct {
	metav1.TypeMeta `json:",inline"`

	// Metadata contains metadata about the request including the UID for the Request.
	Metadata AdmissionReviewMetadata `json:"metadata" protobuf:"bytes,1,opt,name=metadata"`

	// Object is the object from the incoming request prior to default values being applied
	Object runtime.RawExtension `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`

	// OldObject is the existing object. Only populated for UPDATE requests.
	// +optional
	OldObject runtime.RawExtension `json:"oldObject,omitempty" protobuf:"bytes,3,opt,name=oldObject"`
}

// AdmissionReviewMetadata contains information about the request, including a UID to identify the request instance.
type AdmissionReviewMetadata struct {
	// UID is an identifier for the individual request. It allows us to distinguish instances of requests which are
	// otherwise identical (parallel requests, requests when earlier requests did not modify etc)
	UID types.UID `json:"uid,omitempty" protobuf:"bytes,1,opt,name=uid,casttype=k8s.io/apimachinery/pkg/types.UID"`
	// Kind is the type of object being manipulated.  For example: Pod
	Kind metav1.GroupVersionKind `json:"kind,omitempty" protobuf:"bytes,2,opt,name=kind"`
	// Name is the name of the object as presented in the request.  On a CREATE operation, the client may omit name and
	// rely on the server to generate the name.  If that is the case, this method will return the empty string.
	Name string `json:"name,omitempty" protobuf:"bytes,3,opt,name=name"`
	// Namespace is the namespace associated with the request (if any).
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,4,opt,name=namespace"`
	// Operation is the operation being performed
	Operation Operation `json:"operation,omitempty" protobuf:"bytes,5,opt,name=operation"`
	// Resource is the name of the resource being requested.  This is not the kind.  For example: pods
	Resource metav1.GroupVersionResource `json:"resource,omitempty" protobuf:"bytes,6,opt,name=resource"`
	// SubResource is the name of the subresource being requested.  This is a different resource, scoped to the parent
	// resource, but it may have a different kind. For instance, /pods has the resource "pods" and the kind "Pod", while
	// /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod" (because status operates on
	// pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource
	// "binding", and kind "Binding".
	SubResource string `json:"subResource,omitempty" protobuf:"bytes,7,opt,name=subResource"`
	// UserInfo is information about the requesting user
	UserInfo authenticationv1.UserInfo `json:"userInfo,omitempty" protobuf:"bytes,8,opt,name=userInfo"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AdmissionReviewResponse describes an admission response.
type AdmissionReviewResponse struct {
	metav1.TypeMeta `json:",inline"`

	// Metadata contains metadata about the request including the UID from the Request.
	Metadata AdmissionReviewMetadata `json:"metadata" protobuf:"bytes,1,opt,name=metadata"`

	// Status is filled in by the webhook and indicates whether the admission
	// request should be permitted.
	// +optional
	Status AdmissionReviewStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`

	// The patch body. Currently we only support "JSONPatch" which implements RFC 6902.
	// +optional
	Patch []byte `json:"patch,omitempty" protobuf:"bytes,3,opt,name=patch"`

	// The type of Patch. Currently we only allow "JSONPatch".
	PatchType string `json:"patchType" protobuf:"bytes,4,opt,name=patchType"`
}

// AdmissionReviewStatus describes the status of the admission request.
type AdmissionReviewStatus struct {
	// Allowed indicates whether or not the admission request was permitted.
	Allowed bool `json:"allowed" protobuf:"varint,1,opt,name=allowed"`
	// Result contains extra details into why an admission request was denied.
	// This field IS NOT consulted in any way if "Allowed" is "true".
	// +optional
	Result *metav1.Status `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// Operation is the type of resource operation being checked for admission control
type Operation string

// Operation constants
const (
	Create  Operation = "CREATE"
	Update  Operation = "UPDATE"
	Delete  Operation = "DELETE"
	Connect Operation = "CONNECT"
)
