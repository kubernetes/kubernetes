/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type attributesRecord struct {
	client    client.Interface
	namespace string
	kind      string
	operation string
	object    runtime.Object
}

func NewAttributesRecord(client client.Interface, object runtime.Object, namespace, kind, operation string) Attributes {
	return &attributesRecord{
		client:    client,
		namespace: namespace,
		kind:      kind,
		operation: operation,
		object:    object,
	}
}

func (record *attributesRecord) GetClient() client.Interface {
	return record.client
}

func (record *attributesRecord) GetNamespace() string {
	return record.namespace
}

func (record *attributesRecord) GetKind() string {
	return record.kind
}

func (record *attributesRecord) GetOperation() string {
	return record.operation
}

func (record *attributesRecord) GetObject() runtime.Object {
	return record.object
}
