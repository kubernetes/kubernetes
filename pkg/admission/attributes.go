/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/runtime"
)

type attributesRecord struct {
	kind        unversioned.GroupKind
	namespace   string
	name        string
	resource    unversioned.GroupResource
	subresource string
	operation   Operation
	object      runtime.Object
	userInfo    user.Info
}

func NewAttributesRecord(object runtime.Object, kind unversioned.GroupKind, namespace, name string, resource unversioned.GroupResource, subresource string, operation Operation, userInfo user.Info) Attributes {
	return &attributesRecord{
		kind:        kind,
		namespace:   namespace,
		name:        name,
		resource:    resource,
		subresource: subresource,
		operation:   operation,
		object:      object,
		userInfo:    userInfo,
	}
}

func (record *attributesRecord) GetKind() unversioned.GroupKind {
	return record.kind
}

func (record *attributesRecord) GetNamespace() string {
	return record.namespace
}

func (record *attributesRecord) GetName() string {
	return record.name
}

func (record *attributesRecord) GetResource() unversioned.GroupResource {
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

func (record *attributesRecord) GetUserInfo() user.Info {
	return record.userInfo
}
