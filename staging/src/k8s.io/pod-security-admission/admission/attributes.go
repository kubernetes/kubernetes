/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"fmt"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Attributes interface {
	// GetNamespace is the namespace associated with the request (if any)
	GetNamespace() string
	// GetResource is the name of the resource being requested.  This is not the kind.  For example: pods
	GetResource() schema.GroupVersionResource
	// GetSubresource is the name of the subresource being requested.  This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	GetSubresource() string
	// GetOperation is the operation being performed
	GetOperation() admissionv1.Operation // Operation

	// GetObject is the object from the incoming request prior to default values being applied
	GetObject() (runtime.Object, error)
	// GetOldObject is the existing object. Only populated for UPDATE requests.
	GetOldObject() (runtime.Object, error)
	// GetUserName is the requesting user's authenticated name.
	GetUserName() string
}

// AttributesRecord is a simple struct implementing the Attributes interface.
type AttributesRecord struct {
	Namespace   string
	Resource    schema.GroupVersionResource
	Subresource string
	Operation   admissionv1.Operation
	Object      runtime.Object
	OldObject   runtime.Object
	Username    string
}

func (a *AttributesRecord) GetNamespace() string {
	return a.Namespace
}
func (a *AttributesRecord) GetResource() schema.GroupVersionResource {
	return a.Resource
}
func (a *AttributesRecord) GetSubresource() string {
	return a.Subresource
}
func (a *AttributesRecord) GetOperation() admissionv1.Operation {
	return a.Operation
}
func (a *AttributesRecord) GetUserName() string {
	return a.Username
}
func (a *AttributesRecord) GetObject() (runtime.Object, error) {
	return a.Object, nil
}
func (a *AttributesRecord) GetOldObject() (runtime.Object, error) {
	return a.OldObject, nil
}

// Decoder is an interface for decoding serialized objects from an admission request.
type Decoder interface {
	// DecodeRaw decodes a RawExtension object into the passed-in runtime.Object.
	DecodeRaw(rawObj runtime.RawExtension, into runtime.Object) error
}

// RequestAttributes adapts an admission.Request to the Attributes interface.
func RequestAttributes(request admissionv1.AdmissionRequest, decoder Decoder) Attributes {
	return &attributes{
		r:       request,
		decoder: decoder,
	}
}

// attributes is an interface used by AdmissionController to get information about a request
// that is used to make an admission decision.
type attributes struct {
	r       admissionv1.AdmissionRequest
	decoder Decoder
}

func (a *attributes) GetNamespace() string {
	return a.r.Namespace
}
func (a *attributes) GetResource() schema.GroupVersionResource {
	return schema.GroupVersionResource(*a.r.RequestResource)
}
func (a *attributes) GetSubresource() string {
	return a.r.RequestSubResource
}
func (a *attributes) GetOperation() admissionv1.Operation {
	return a.r.Operation
}
func (a *attributes) GetUserName() string {
	return a.r.UserInfo.Username
}
func (a *attributes) GetObject() (runtime.Object, error) {
	return a.decode(a.r.Object)
}
func (a *attributes) GetOldObject() (runtime.Object, error) {
	return a.decode(a.r.OldObject)
}
func (a *attributes) decode(in runtime.RawExtension) (runtime.Object, error) {
	var out runtime.Object
	switch a.GetResource().GroupResource() {
	case corev1.Resource("namespaces"):
		out = &corev1.Namespace{}
	case corev1.Resource("pods"):
		out = &corev1.Pod{}
	case corev1.Resource("replicationcontrollers"):
		out = &corev1.ReplicationController{}
	case corev1.Resource("podtemplates"):
		out = &corev1.PodTemplate{}
	case corev1.Resource("replicasets"):
		out = &appsv1.ReplicaSet{}
	case appsv1.Resource("deployments"):
		out = &appsv1.Deployment{}
	case appsv1.Resource("statefulsets"):
		out = &appsv1.StatefulSet{}
	case appsv1.Resource("daemonsets"):
		out = &appsv1.DaemonSet{}
	case batchv1.Resource("jobs"):
		out = &batchv1.Job{}
	case batchv1.Resource("cronjobs"):
		out = &batchv1.CronJob{}
	default:
		return nil, fmt.Errorf("unexpected resource: %s", a.GetResource())
	}
	if err := a.decoder.DecodeRaw(in, out); err != nil {
		return nil, err
	}
	return out, nil
}
