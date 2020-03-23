/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"sigs.k8s.io/structured-merge-diff/v3/fieldpath"
)

type fakeManager struct{}

var _ fieldmanager.Manager = &fakeManager{}

func (*fakeManager) Update(_, newObj runtime.Object, managed fieldmanager.Managed, _ string) (runtime.Object, fieldmanager.Managed, error) {
	return newObj, managed, nil
}

func (*fakeManager) Apply(_, _ runtime.Object, _ fieldmanager.Managed, _ string, force bool) (runtime.Object, fieldmanager.Managed, error) {
	panic("not implemented")
	return nil, nil, nil
}

func TestCapManagersManagerMergesEntries(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
	f.fieldManager = fieldmanager.NewCapManagersManager(f.fieldManager, 3)

	podWithLabels := func(labels ...string) runtime.Object {
		labelMap := map[string]interface{}{}
		for _, key := range labels {
			labelMap[key] = "true"
		}
		obj := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": labelMap,
				},
			},
		}
		obj.SetKind("Pod")
		obj.SetAPIVersion("v1")
		return obj
	}

	if err := f.Update(podWithLabels("one"), "fieldmanager_test_update_1"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	expectIdempotence(t, f)

	if err := f.Update(podWithLabels("one", "two"), "fieldmanager_test_update_2"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	expectIdempotence(t, f)

	if err := f.Update(podWithLabels("one", "two", "three"), "fieldmanager_test_update_3"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	expectIdempotence(t, f)

	if err := f.Update(podWithLabels("one", "two", "three", "four"), "fieldmanager_test_update_4"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	expectIdempotence(t, f)

	if e, a := 3, len(f.ManagedFields()); e != a {
		t.Fatalf("exected %v entries in managedFields, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "ancient-changes", f.ManagedFields()[0].Manager; e != a {
		t.Fatalf("exected first manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "fieldmanager_test_update_3", f.ManagedFields()[1].Manager; e != a {
		t.Fatalf("exected second manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "fieldmanager_test_update_4", f.ManagedFields()[2].Manager; e != a {
		t.Fatalf("exected third manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}

	expectManagesField(t, f, "ancient-changes", fieldpath.MakePathOrDie("metadata", "labels", "one"))
	expectManagesField(t, f, "ancient-changes", fieldpath.MakePathOrDie("metadata", "labels", "two"))
	expectManagesField(t, f, "fieldmanager_test_update_3", fieldpath.MakePathOrDie("metadata", "labels", "three"))
	expectManagesField(t, f, "fieldmanager_test_update_4", fieldpath.MakePathOrDie("metadata", "labels", "four"))
}

func TestCapUpdateManagers(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
	f.fieldManager = fieldmanager.NewCapManagersManager(&fakeManager{}, 3)

	set := func(fields ...string) *metav1.FieldsV1 {
		s := fieldpath.NewSet()
		for _, f := range fields {
			s.Insert(fieldpath.MakePathOrDie(f))
		}
		b, err := s.ToJSON()
		if err != nil {
			panic(fmt.Sprintf("error building ManagedFieldsEntry for test: %v", err))
		}
		return &metav1.FieldsV1{Raw: b}
	}

	entry := func(name string, version string, order int, fields *metav1.FieldsV1) metav1.ManagedFieldsEntry {
		return metav1.ManagedFieldsEntry{
			Manager:    name,
			APIVersion: version,
			Operation:  "Update",
			FieldsType: "FieldsV1",
			FieldsV1:   fields,
			Time:       &metav1.Time{Time: time.Time{}.Add(time.Hour * time.Duration(order))},
		}
	}

	testCases := []struct {
		name     string
		input    []metav1.ManagedFieldsEntry
		expected []metav1.ManagedFieldsEntry
	}{
		{
			name: "one version, no ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("update-manager1", "v1", 1, set("a")),
				entry("update-manager2", "v1", 2, set("b")),
				entry("update-manager3", "v1", 3, set("c")),
				entry("update-manager4", "v1", 4, set("d")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 2, set("a", "b")),
				entry("update-manager3", "v1", 3, set("c")),
				entry("update-manager4", "v1", 4, set("d")),
			},
		}, {
			name: "one version, one ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 2, set("a", "b")),
				entry("update-manager3", "v1", 3, set("c")),
				entry("update-manager4", "v1", 4, set("d")),
				entry("update-manager5", "v1", 5, set("e")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 3, set("a", "b", "c")),
				entry("update-manager4", "v1", 4, set("d")),
				entry("update-manager5", "v1", 5, set("e")),
			},
		}, {
			name: "two versions, no ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("update-manager1", "v1", 1, set("a")),
				entry("update-manager2", "v2", 2, set("b")),
				entry("update-manager3", "v1", 3, set("c")),
				entry("update-manager4", "v1", 4, set("d")),
				entry("update-manager5", "v1", 5, set("e")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("update-manager2", "v2", 2, set("b")),
				entry("ancient-changes", "v1", 4, set("a", "c", "d")),
				entry("update-manager5", "v1", 5, set("e")),
			},
		}, {
			name: "three versions, one ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("update-manager2", "v2", 2, set("b")),
				entry("ancient-changes", "v1", 4, set("a", "c", "d")),
				entry("update-manager5", "v1", 5, set("e")),
				entry("update-manager6", "v3", 6, set("f")),
				entry("update-manager7", "v2", 7, set("g")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 5, set("a", "c", "d", "e")),
				entry("update-manager6", "v3", 6, set("f")),
				entry("ancient-changes", "v2", 7, set("b", "g")),
			},
		}, {
			name: "three versions, two ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 5, set("a", "c", "d", "e")),
				entry("update-manager6", "v3", 6, set("f")),
				entry("ancient-changes", "v2", 7, set("b", "g")),
				entry("update-manager8", "v3", 8, set("h")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 5, set("a", "c", "d", "e")),
				entry("ancient-changes", "v2", 7, set("b", "g")),
				entry("ancient-changes", "v3", 8, set("f", "h")),
			},
		}, {
			name: "four versions, two ancient changes",
			input: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 5, set("a", "c", "d", "e")),
				entry("update-manager6", "v3", 6, set("f")),
				entry("ancient-changes", "v2", 7, set("b", "g")),
				entry("update-manager8", "v4", 8, set("h")),
			},
			expected: []metav1.ManagedFieldsEntry{
				entry("ancient-changes", "v1", 5, set("a", "c", "d", "e")),
				entry("update-manager6", "v3", 6, set("f")),
				entry("ancient-changes", "v2", 7, set("b", "g")),
				entry("update-manager8", "v4", 8, set("h")),
			},
		},
	}

	for _, tc := range testCases {
		f.Reset()
		accessor, err := meta.Accessor(f.liveObj)
		if err != nil {
			t.Fatalf("%v: couldn't get accessor: %v", tc.name, err)
		}
		accessor.SetManagedFields(tc.input)
		if err := f.Update(f.liveObj, "no-op-update"); err != nil {
			t.Fatalf("%v: failed to do no-op update to object: %v", tc.name, err)
		}

		if e, a := tc.expected, f.ManagedFields(); !apiequality.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected value for managedFields:\nexpected: %v\n but got: %v", tc.name, mustMarshal(e), mustMarshal(a))
		}
		expectIdempotence(t, f)
	}
}

// expectIdempotence does a no-op update and ensures that managedFields doesn't change by calling capUpdateManagers.
func expectIdempotence(t *testing.T, f TestFieldManager) {
	before := []metav1.ManagedFieldsEntry{}
	for _, m := range f.ManagedFields() {
		before = append(before, *m.DeepCopy())
	}

	if err := f.Update(f.liveObj, "no-op-update"); err != nil {
		t.Fatalf("failed to do no-op update to object: %v", err)
	}

	if after := f.ManagedFields(); !apiequality.Semantic.DeepEqual(before, after) {
		t.Fatalf("exected idempotence, but managedFields changed:\nbefore: %v\n after: %v", mustMarshal(before), mustMarshal(after))
	}
}

// expectManagesField ensures that manager m currently manages field path p.
func expectManagesField(t *testing.T, f TestFieldManager, m string, p fieldpath.Path) {
	for _, e := range f.ManagedFields() {
		if e.Manager == m {
			var s fieldpath.Set
			err := s.FromJSON(bytes.NewReader(e.FieldsV1.Raw))
			if err != nil {
				t.Fatalf("error parsing managedFields for %v: %v: %#v", m, err, f.ManagedFields())
			}
			if !s.Has(p) {
				t.Fatalf("expected managedFields for %v to contain %v, but got:\n%v", m, p.String(), s.String())
			}
			return
		}
	}
	t.Fatalf("exected to find manager name %v, but got: %#v", m, f.ManagedFields())
}

func mustMarshal(i interface{}) string {
	b, err := json.MarshalIndent(i, "", "  ")
	if err != nil {
		panic(fmt.Sprintf("error marshalling %v to json: %v", i, err))
	}
	return string(b)
}
