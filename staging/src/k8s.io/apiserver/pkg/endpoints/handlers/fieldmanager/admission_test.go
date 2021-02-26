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

package fieldmanager_test

import (
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
)

func TestAdmission(t *testing.T) {
	wrap := &mockAdmissionController{}
	ac := fieldmanager.NewManagedFieldsValidatingAdmissionController(wrap)

	tests := []struct {
		beforeAdmission []metav1.ManagedFieldsEntry
		afterAdmission  []metav1.ManagedFieldsEntry
		expected        []metav1.ManagedFieldsEntry
	}{
		{
			beforeAdmission: []metav1.ManagedFieldsEntry{
				{
					Manager: "test",
				},
			},
			afterAdmission: []metav1.ManagedFieldsEntry{
				{
					Manager: "",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager: "test",
				},
			},
		},
		{
			beforeAdmission: []metav1.ManagedFieldsEntry{
				{
					APIVersion: "test",
				},
			},
			afterAdmission: []metav1.ManagedFieldsEntry{
				{
					APIVersion: "",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					APIVersion: "test",
				},
			},
		},
		{
			beforeAdmission: []metav1.ManagedFieldsEntry{
				{
					FieldsType: "FieldsV1",
				},
			},
			afterAdmission: []metav1.ManagedFieldsEntry{
				{
					FieldsType: "test",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					FieldsType: "FieldsV1",
				},
			},
		},
	}

	for _, test := range tests {
		obj := &unstructured.Unstructured{}
		obj.SetManagedFields(test.beforeAdmission)
		wrap.admit = replaceManagedFields(test.afterAdmission)

		attrs := admission.NewAttributesRecord(obj, obj, schema.GroupVersionKind{}, "default", "", schema.GroupVersionResource{}, "", admission.Update, nil, false, nil)
		if err := ac.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(obj.GetManagedFields(), test.expected) {
			t.Fatalf("expected: \n%v\ngot:\n%v", test.expected, obj.GetManagedFields())
		}
	}
}

func replaceManagedFields(with []metav1.ManagedFieldsEntry) func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
		objectMeta, err := meta.Accessor(a.GetObject())
		if err != nil {
			return err
		}
		objectMeta.SetManagedFields(with)
		return nil
	}
}

type mockAdmissionController struct {
	admit func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error
}

func (c *mockAdmissionController) Handles(operation admission.Operation) bool {
	return true
}

func (c *mockAdmissionController) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return c.admit(ctx, a, o)
}
