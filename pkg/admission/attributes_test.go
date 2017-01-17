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

package admission

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
)

type FakeAdmissionObject struct{}

func (obj *FakeAdmissionObject) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}

func TestAttributesRecord(t *testing.T) {
	expectedKind := schema.GroupVersionKind{
		Group:   "attributes-record-group",
		Version: "attributes-record-version",
		Kind:    "attributes-record-kind",
	}

	expectedNamespace := "attributes-record-namespace"
	expectedName := "attributes-record-name"
	expectedResource := schema.GroupVersionResource{
		Group:    "attributes-record-group",
		Version:  "attributes-record-version",
		Resource: "attributes-record-resource",
	}
	expectedSubResource := "attributes-record-subresource"
	expectedOperation := Operation("CREATE")
	expectedObject := &FakeAdmissionObject{}
	expectedOldObject := &FakeAdmissionObject{}
	expectedUserInfo := &user.DefaultInfo{
		Name:   "user-info-name",
		UID:    "user-info-UID",
		Groups: []string{"user-info-groups1", "user-info-groups2"},
	}

	attributesRecord := NewAttributesRecord(expectedObject, expectedOldObject, expectedKind, expectedNamespace, expectedName,
		expectedResource, expectedSubResource, expectedOperation, expectedUserInfo)

	if attributesRecord.GetName() != expectedName {
		t.Fatalf("get attributes record name error, expected: %s, got: %s", expectedName, attributesRecord.GetName())
	}

	if attributesRecord.GetNamespace() != expectedNamespace {
		t.Fatalf("get attributes record namespace error, expected: %s, got: %s", expectedNamespace, attributesRecord.GetNamespace())
	}

	if attributesRecord.GetSubresource() != expectedSubResource {
		t.Fatalf("get attributes record subresource error, expected: %s, got: %s", expectedSubResource, attributesRecord.GetSubresource())
	}

	if !reflect.DeepEqual(attributesRecord.GetKind(), expectedKind) {
		t.Fatalf("get attributes record kind error, expected: %v, got: %v", expectedKind, attributesRecord.GetKind())
	}

	if !reflect.DeepEqual(attributesRecord.GetResource(), expectedResource) {
		t.Fatalf("get attributes record resource error, expected: %v, got: %v", expectedResource, attributesRecord.GetResource())
	}

	if !reflect.DeepEqual(attributesRecord.GetObject(), expectedObject) {
		t.Fatalf("get attributes record object error, expected: %v, got: %v", expectedObject, attributesRecord.GetObject())
	}

	if !reflect.DeepEqual(attributesRecord.GetOldObject(), expectedOldObject) {
		t.Fatalf("get attributes record oldObject error, expected: %v, got: %v", expectedOldObject, attributesRecord.GetOldObject())
	}

	if !reflect.DeepEqual(attributesRecord.GetUserInfo(), expectedUserInfo) {
		t.Fatalf("get attributes record user info error, expected: %v, got: %v", expectedUserInfo, attributesRecord.GetUserInfo())
	}

	if !reflect.DeepEqual(attributesRecord.GetOperation(), expectedOperation) {
		t.Fatalf("get attributes record operation error, expected: %v, got: %v", expectedOperation, attributesRecord.GetOperation())
	}
}
