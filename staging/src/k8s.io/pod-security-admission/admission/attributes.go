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
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/pod-security-admission/api"
)

// AttributesRecord is a simple struct implementing the Attributes interface.
type AttributesRecord struct {
	Name        string
	Namespace   string
	Resource    schema.GroupVersionResource
	Subresource string
	Operation   admissionv1.Operation
	Object      runtime.Object
	OldObject   runtime.Object
	Username    string
}

func (a *AttributesRecord) GetName() string {
	return a.Name
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

// RequestAttributes adapts an admission.Request to the Attributes interface.
func RequestAttributes(request *admissionv1.AdmissionRequest, decoder runtime.Decoder) api.Attributes {
	return &attributes{
		r:       request,
		decoder: decoder,
	}
}

// attributes is an interface used by AdmissionController to get information about a request
// that is used to make an admission decision.
type attributes struct {
	r       *admissionv1.AdmissionRequest
	decoder runtime.Decoder
}

func (a *attributes) GetName() string {
	return a.r.Name
}
func (a *attributes) GetNamespace() string {
	return a.r.Namespace
}
func (a *attributes) GetResource() schema.GroupVersionResource {
	return schema.GroupVersionResource(a.r.Resource)
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
	if in.Raw == nil {
		return nil, nil
	}
	gvk := schema.GroupVersionKind(a.r.Kind)
	out, _, err := a.decoder.Decode(in.Raw, &gvk, nil)
	return out, err
}
