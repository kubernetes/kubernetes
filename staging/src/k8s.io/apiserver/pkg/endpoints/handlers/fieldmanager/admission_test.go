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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

func TestAdmission(t *testing.T) {
	wrap := &mockAdmissionController{}
	ac := fieldmanager.NewManagedFieldsValidatingAdmissionController(wrap)
	now := metav1.Now()

	validFieldsV1 := metav1.FieldsV1{}
	var err error
	validFieldsV1.Raw, err = fieldpath.NewSet(fieldpath.MakePathOrDie("metadata", "labels", "test-label")).ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	validManagedFieldsEntry := metav1.ManagedFieldsEntry{
		APIVersion: "v1",
		Operation:  metav1.ManagedFieldsOperationApply,
		Time:       &now,
		Manager:    "test",
		FieldsType: "FieldsV1",
		FieldsV1:   &validFieldsV1,
	}

	managedFieldsMutators := map[string]func(in metav1.ManagedFieldsEntry) (out metav1.ManagedFieldsEntry, shouldReset bool){
		"invalid APIVersion": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.APIVersion = ""
			return managedFields, true
		},
		"invalid Operation": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.Operation = "invalid operation"
			return managedFields, true
		},
		"invalid fieldsType": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.FieldsType = "invalid fieldsType"
			return managedFields, true
		},
		"invalid fieldsV1": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.FieldsV1 = &metav1.FieldsV1{Raw: []byte("{invalid}")}
			return managedFields, true
		},
		"invalid manager": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.Manager = ""
			return managedFields, false
		},
	}

	for name, mutate := range managedFieldsMutators {
		t.Run(name, func(t *testing.T) {
			mutated, shouldReset := mutate(validManagedFieldsEntry)
			validEntries := []metav1.ManagedFieldsEntry{validManagedFieldsEntry}
			mutatedEntries := []metav1.ManagedFieldsEntry{mutated}

			obj := &v1.ConfigMap{}
			obj.SetManagedFields(validEntries)

			wrap.admit = replaceManagedFields(mutatedEntries)

			attrs := admission.NewAttributesRecord(obj, obj, schema.GroupVersionKind{}, "default", "", schema.GroupVersionResource{}, "", admission.Update, nil, false, nil)
			if err := ac.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
				t.Fatal(err)
			}

			if shouldReset && !reflect.DeepEqual(obj.GetManagedFields(), validEntries) {
				t.Fatalf("expected: \n%v\ngot:\n%v", validEntries, obj.GetManagedFields())
			}
			if !shouldReset && reflect.DeepEqual(obj.GetManagedFields(), validEntries) {
				t.Fatalf("expected: \n%v\ngot:\n%v", mutatedEntries, obj.GetManagedFields())
			}
		})
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
