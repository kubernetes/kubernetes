/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// makes sure that we never get:
// Internal error occurred: [some error, object does not implement the Object interfaces]
func TestNewForbidden(t *testing.T) {
	attributes := NewAttributesRecord(
		&fakeObj{},
		nil,
		schema.GroupVersionKind{Group: "foo", Version: "bar", Kind: "Baz"},
		"",
		"",
		schema.GroupVersionResource{Group: "foo", Version: "bar", Resource: "baz"},
		"",
		Create,
		nil)
	err := errors.New("some error")
	expectedErr := `baz.foo "Unknown/errorGettingName" is forbidden: some error`

	actualErr := NewForbidden(attributes, err)
	if actualErr.Error() != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, actualErr)
	}
}

type fakeObj struct{}
type fakeObjKind struct{}

func (f *fakeObj) GetObjectKind() schema.ObjectKind {
	return &fakeObjKind{}
}
func (f *fakeObj) DeepCopyObject() runtime.Object {
	return f
}

func (f *fakeObjKind) SetGroupVersionKind(kind schema.GroupVersionKind) {}
func (f *fakeObjKind) GroupVersionKind() schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "foo", Version: "bar", Kind: "Baz"}
}
