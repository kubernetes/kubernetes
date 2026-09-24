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
	_ "embed"
	"reflect"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/warning"
)

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

func TestAdmission(t *testing.T) {
	wrap := &mockAdmissionController{}
	ac := fieldmanager.NewManagedFieldsValidatingAdmissionController(wrap)
	now := metav1.Now()

	validFieldsV1 := metav1.FieldsV1{}
	raw, err := fieldpath.NewSet(fieldpath.MakePathOrDie("metadata", "labels", "test-label")).ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	validFieldsV1.SetRawBytes(raw)
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
			managedFields.FieldsV1 = metav1.NewFieldsV1("{invalid}")
			return managedFields, true
		},
		"invalid manager": func(managedFields metav1.ManagedFieldsEntry) (metav1.ManagedFieldsEntry, bool) {
			managedFields.Manager = ""
			return managedFields, false
		},
	}

	mutationStyles := []struct {
		name      string
		admitWith func(metav1.ManagedFieldsEntry) admitFunc
	}{
		{name: "replaceSlice", admitWith: replaceManagedFields},
		{name: "overwriteElement", admitWith: overwriteManagedFieldsElement},
	}

	for name, mutate := range managedFieldsMutators {
		for _, style := range mutationStyles {
			t.Run(name+"/"+style.name, func(t *testing.T) {
				mutated, shouldReset := mutate(validManagedFieldsEntry)
				validEntries := []metav1.ManagedFieldsEntry{validManagedFieldsEntry}

				obj := &v1.ConfigMap{}
				obj.SetManagedFields([]metav1.ManagedFieldsEntry{validManagedFieldsEntry})

				wrap.admit = style.admitWith(mutated)

				attrs := admission.NewAttributesRecord(obj, obj, schema.GroupVersionKind{}, "default", "", schema.GroupVersionResource{}, "", admission.Update, nil, false, nil)
				if err := ac.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
					t.Fatal(err)
				}

				if shouldReset && !reflect.DeepEqual(obj.GetManagedFields(), validEntries) {
					t.Fatalf("expected: \n%v\ngot:\n%v", validEntries, obj.GetManagedFields())
				}
				if !shouldReset && reflect.DeepEqual(obj.GetManagedFields(), validEntries) {
					t.Fatalf("expected: \n%v\ngot:\n%v", []metav1.ManagedFieldsEntry{mutated}, obj.GetManagedFields())
				}
			})
		}
	}
}

func TestAdmissionSkipsValidationWhenUnchanged(t *testing.T) {
	wrap := &mockAdmissionController{admit: func(context.Context, admission.Attributes, admission.ObjectInterfaces) error { return nil }}
	ac := fieldmanager.NewManagedFieldsValidatingAdmissionController(wrap)

	obj := &v1.ConfigMap{}
	obj.SetManagedFields([]metav1.ManagedFieldsEntry{{Manager: "test", Operation: "invalid operation"}})

	rec := &warningRecorder{}
	ctx := warning.WithWarningRecorder(context.TODO(), rec)
	attrs := admission.NewAttributesRecord(obj, obj, schema.GroupVersionKind{}, "default", "", schema.GroupVersionResource{}, "", admission.Update, nil, false, nil)
	if err := ac.(admission.MutationInterface).Admit(ctx, attrs, nil); err != nil {
		t.Fatal(err)
	}
	if len(rec.warnings) != 0 {
		t.Errorf("managedFields were revalidated although admission did not change them: %v", rec.warnings)
	}
}

func BenchmarkAdmission(b *testing.B) {
	pod := &v1.Pod{}
	if err := yaml.Unmarshal(exemplarPodYAML, pod); err != nil {
		b.Fatal(err)
	}
	entries := pod.ManagedFields
	// Same content, distinct pointers, so the wrapper sees a change and decodes.
	copied := pod.DeepCopy().ManagedFields

	for _, tc := range []struct {
		name  string
		admit admitFunc
	}{
		{
			name:  "unchanged",
			admit: func(context.Context, admission.Attributes, admission.ObjectInterfaces) error { return nil },
		},
		{
			name: "replaced",
			admit: func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
				objectMeta, err := meta.Accessor(a.GetObject())
				if err != nil {
					return err
				}
				if &objectMeta.GetManagedFields()[0] == &entries[0] {
					objectMeta.SetManagedFields(copied)
				} else {
					objectMeta.SetManagedFields(entries)
				}
				return nil
			},
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ac := fieldmanager.NewManagedFieldsValidatingAdmissionController(&mockAdmissionController{admit: tc.admit})
			obj := pod.DeepCopy()
			obj.SetManagedFields(entries)
			attrs := admission.NewAttributesRecord(obj, obj, schema.GroupVersionKind{}, "default", "", schema.GroupVersionResource{}, "", admission.Update, nil, false, nil)
			b.ReportAllocs()
			for b.Loop() {
				if err := ac.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

type admitFunc = func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error

type warningRecorder struct {
	warnings []string
}

func (r *warningRecorder) AddWarning(_, text string) {
	r.warnings = append(r.warnings, text)
}

func overwriteManagedFieldsElement(to metav1.ManagedFieldsEntry) admitFunc {
	return func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
		objectMeta, err := meta.Accessor(a.GetObject())
		if err != nil {
			return err
		}
		objectMeta.GetManagedFields()[0] = to
		return nil
	}
}

func replaceManagedFields(with metav1.ManagedFieldsEntry) admitFunc {
	return func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
		objectMeta, err := meta.Accessor(a.GetObject())
		if err != nil {
			return err
		}
		objectMeta.SetManagedFields([]metav1.ManagedFieldsEntry{with})
		return nil
	}
}

type mockAdmissionController struct {
	admit admitFunc
}

func (c *mockAdmissionController) Handles(operation admission.Operation) bool {
	return true
}

func (c *mockAdmissionController) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return c.admit(ctx, a, o)
}
