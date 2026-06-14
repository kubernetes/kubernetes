/*
Copyright The Kubernetes Authors.

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

package managedfields

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	"k8s.io/apimachinery/pkg/runtime"
)

func managedFields() []metav1.ManagedFieldsEntry {
	return []metav1.ManagedFieldsEntry{{
		Manager:    "test",
		Operation:  metav1.ManagedFieldsOperationApply,
		APIVersion: "v1",
	}}
}

func TestRemoveSingleObject(t *testing.T) {
	original := &testapigroup.Carp{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ManagedFields: managedFields()},
	}

	got := Remove(original)

	if carp, ok := got.(*testapigroup.Carp); !ok || len(carp.ManagedFields) != 0 {
		t.Fatalf("expected managedFields to be cleared, got %#v", got)
	}
	if got == runtime.Object(original) {
		t.Errorf("expected a copy, got the same instance")
	}
	if len(original.ManagedFields) == 0 {
		t.Errorf("input object was mutated")
	}
}

func TestRemoveList(t *testing.T) {
	original := &testapigroup.CarpList{
		Items: []testapigroup.Carp{
			{ObjectMeta: metav1.ObjectMeta{Name: "a", ManagedFields: managedFields()}},
			{ObjectMeta: metav1.ObjectMeta{Name: "b"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "c", ManagedFields: managedFields()}},
		},
	}

	got := Remove(original).(*testapigroup.CarpList)

	for i := range got.Items {
		if len(got.Items[i].ManagedFields) != 0 {
			t.Errorf("item %d still has managedFields", i)
		}
	}
	if got == original {
		t.Errorf("expected a copy of the list")
	}
	if len(original.Items[0].ManagedFields) == 0 {
		t.Errorf("input list was mutated")
	}
}

func TestRemoveNonMetaObject(t *testing.T) {
	// Objects without object metadata must be handled without panicking.
	original := &metav1.Status{Status: metav1.StatusSuccess}

	got := Remove(original)

	if status, ok := got.(*metav1.Status); !ok || status.Status != metav1.StatusSuccess {
		t.Errorf("expected an equivalent Status, got %#v", got)
	}
}

func TestRemoveNil(t *testing.T) {
	if got := Remove(nil); got != nil {
		t.Errorf("expected nil, got %#v", got)
	}
}

func TestRemoveInPlace(t *testing.T) {
	obj := &testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "foo", ManagedFields: managedFields()}}
	RemoveInPlace(obj)
	if len(obj.ManagedFields) != 0 {
		t.Errorf("expected managedFields cleared in place, got %#v", obj.ManagedFields)
	}

	list := &testapigroup.CarpList{Items: []testapigroup.Carp{
		{ObjectMeta: metav1.ObjectMeta{Name: "a", ManagedFields: managedFields()}},
		{ObjectMeta: metav1.ObjectMeta{Name: "b", ManagedFields: managedFields()}},
	}}
	RemoveInPlace(list)
	for i := range list.Items {
		if len(list.Items[i].ManagedFields) != 0 {
			t.Errorf("item %d managedFields not cleared in place", i)
		}
	}
}
