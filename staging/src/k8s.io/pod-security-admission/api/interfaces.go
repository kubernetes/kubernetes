package api

import (
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Attributes exposes the admission request parameters consumed by the PodSecurity admission controller.
type Attributes interface {
	// GetName is the name of the object associated with the request.
	GetName() string
	// GetNamespace is the namespace associated with the request (if any)
	GetNamespace() string
	// GetResource is the name of the resource being requested.  This is not the kind.  For example: pods
	GetResource() schema.GroupVersionResource
	// GetSubresource is the name of the subresource being requested.  This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	GetSubresource() string
	// GetOperation is the operation being performed
	GetOperation() admissionv1.Operation

	// GetObject returns the typed Object from incoming request.
	// For objects in the core API group, the result must use the v1 API.
	GetObject() (runtime.Object, error)
	// GetOldObject returns the typed existing object. Only populated for UPDATE requests.
	// For objects in the core API group, the result must use the v1 API.
	GetOldObject() (runtime.Object, error)
	// GetUserName is the requesting user's authenticated name.
	GetUserName() string
}

