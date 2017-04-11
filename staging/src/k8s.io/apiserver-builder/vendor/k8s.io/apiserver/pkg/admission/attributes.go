/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
)

type attributesRecord struct {
	kind        schema.GroupVersionKind
	namespace   string
	name        string
	resource    schema.GroupVersionResource
	subresource string
	operation   Operation
	object      runtime.Object
	oldObject   runtime.Object
	userInfo    user.Info
}

func NewAttributesRecord(object runtime.Object, oldObject runtime.Object, kind schema.GroupVersionKind, namespace, name string, resource schema.GroupVersionResource, subresource string, operation Operation, userInfo user.Info) Attributes {
	return &attributesRecord{
		kind:        kind,
		namespace:   namespace,
		name:        name,
		resource:    resource,
		subresource: subresource,
		operation:   operation,
		object:      object,
		oldObject:   oldObject,
		userInfo:    userInfo,
	}
}

func (record *attributesRecord) GetKind() schema.GroupVersionKind {
	return record.kind
}

func (record *attributesRecord) GetNamespace() string {
	return record.namespace
}

func (record *attributesRecord) GetName() string {
	return record.name
}

func (record *attributesRecord) GetResource() schema.GroupVersionResource {
	return record.resource
}

func (record *attributesRecord) GetSubresource() string {
	return record.subresource
}

func (record *attributesRecord) GetOperation() Operation {
	return record.operation
}

func (record *attributesRecord) GetObject() runtime.Object {
	return record.object
}

func (record *attributesRecord) GetOldObject() runtime.Object {
	return record.oldObject
}

func (record *attributesRecord) GetUserInfo() user.Info {
	return record.userInfo
}
