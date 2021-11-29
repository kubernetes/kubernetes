/*
Copyright 2020 The Kubernetes Authors.

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

package fieldmanager

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/yaml"
)

func TestNoManagedFieldsUpdateDoesntUpdateTime(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"), "", nil)

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "pod",
			"labels": {"app": "nginx"}
		},
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	if err := f.Update(obj, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	managed := f.ManagedFields()
	obj2 := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "pod",
			"labels": {"app": "nginx2"}
		},
	}`), &obj2.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	time.Sleep(time.Second)
	if err := f.Update(obj2, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	if !reflect.DeepEqual(managed, f.ManagedFields()) {
		t.Errorf("ManagedFields changed:\nBefore:\n%v\nAfter:\n%v", managed, f.ManagedFields())
	}
}

type NoopManager struct{}

func (NoopManager) Apply(liveObj, appliedObj runtime.Object, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error) {
	return nil, managed, nil
}

func (NoopManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	return nil, nil, nil
}

// Ensures that if ManagedFieldsUpdater gets a nil value from its nested manager
// chain (meaning the operation was a no-op), then the ManagedFieldsUpdater
// itself will return a copy of the input live object, with its managed fields
// removed
func TestNilNewObjectReplacedWithDeepCopyExcludingManagedFields(t *testing.T) {
	// Initialize our "live object" with some managed fields
	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "pod",
			"labels": {"app": "nginx"},
			"managedFields": [
				{
					"apiVersion": "v1",
					"fieldsType": "FieldsV1",
					"fieldsV1": {
						"f:metadata": {
							"f:labels": {
								"f:app": {}
							}
						}
					},
					"manager": "fieldmanager_test",
					"operation": "Apply",
					"time": "2021-11-11T18:41:17Z"
				}
			]
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		t.Fatalf("couldn't get accessor: %v", err)
	}

	// Decode the managed fields in the live object, since it isn't allowed in the patch.
	managed, err := DecodeManagedFields(accessor.GetManagedFields())
	if err != nil {
		t.Fatalf("failed to decode managed fields: %v", err)
	}

	updater := NewManagedFieldsUpdater(NoopManager{})

	newObject, _, err := updater.Apply(obj, obj.DeepCopyObject(), managed, "some_manager", false)
	if err != nil {
		t.Fatalf("failed to apply configuration %v", err)
	}

	if newObject == obj {
		t.Fatalf("returned newObject must not be the same instance as the passed in liveObj")
	}

	// Rip off managed fields of live, and check that it is deeply
	// equal to newObject
	liveWithoutManaged := obj.DeepCopyObject()
	internal.RemoveObjectManagedFields(liveWithoutManaged)

	if !reflect.DeepEqual(liveWithoutManaged, newObject) {
		t.Fatalf("returned newObject must be deeply equal to the input live object, without managed fields")
	}
}
