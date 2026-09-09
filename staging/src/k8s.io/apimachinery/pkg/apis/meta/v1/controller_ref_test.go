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

package v1

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type metaObj struct {
	ObjectMeta
	TypeMeta
}

func TestNewControllerRef(t *testing.T) {
	gvk := schema.GroupVersionKind{
		Group:   "group",
		Version: "v1",
		Kind:    "Kind",
	}
	obj1 := &metaObj{
		ObjectMeta: ObjectMeta{
			Name: "name",
			UID:  "uid1",
		},
	}
	controllerRef := NewControllerRef(obj1, gvk)
	if controllerRef.UID != obj1.UID {
		t.Errorf("Incorrect UID: %s", controllerRef.UID)
	}
	if controllerRef.Controller == nil || *controllerRef.Controller != true {
		t.Error("Controller must be set to true")
	}
	if controllerRef.BlockOwnerDeletion == nil || *controllerRef.BlockOwnerDeletion != true {
		t.Error("BlockOwnerDeletion must be set to true")
	}
	if controllerRef.APIVersion == "" ||
		controllerRef.Kind == "" ||
		controllerRef.Name == "" {
		t.Errorf("All controllerRef fields must be set: %v", controllerRef)
	}
}

func TestGetControllerOf(t *testing.T) {
	gvk := schema.GroupVersionKind{
		Group:   "group",
		Version: "v1",
		Kind:    "Kind",
	}
	obj1 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID:  "uid1",
			Name: "name1",
		},
	}
	controllerRef := NewControllerRef(obj1, gvk)
	controllerRef.BlockOwnerDeletion = nil
	var falseRef = false
	obj2 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID:  "uid2",
			Name: "name1",
			OwnerReferences: []OwnerReference{
				{
					Name:       "owner1",
					Controller: &falseRef,
				},
				*controllerRef,
				{
					Name:       "owner2",
					Controller: &falseRef,
				},
			},
		},
	}

	if GetControllerOf(obj1) != nil {
		t.Error("GetControllerOf must return null")
	}
	c := GetControllerOf(obj2)
	if c.Name != controllerRef.Name || c.UID != controllerRef.UID {
		t.Errorf("Incorrect result of GetControllerOf: %v", c)
	}

	// test that all pointers are also deep copied
	if (c.Controller == controllerRef.Controller) ||
		(c.BlockOwnerDeletion != nil && c.BlockOwnerDeletion == controllerRef.BlockOwnerDeletion) {
		t.Errorf("GetControllerOf did not return deep copy: %v", c)
	}
}

func BenchmarkGetControllerOf(b *testing.B) {
	gvk := schema.GroupVersionKind{
		Group:   "group",
		Version: "v1",
		Kind:    "Kind",
	}
	obj1 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID:  "9d0cdf8a-dedc-11e9-bf91-42010a800167",
			Name: "my-object",
		},
	}
	controllerRef := NewControllerRef(obj1, gvk)
	controllerRef2 := *controllerRef
	controllerRef2.Controller = nil
	obj2 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID:             "uid2",
			Name:            "name1",
			OwnerReferences: []OwnerReference{controllerRef2, controllerRef2, *controllerRef},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		c := GetControllerOf(obj2)
		if c.Name != controllerRef.Name || c.UID != controllerRef.UID {
			b.Errorf("Incorrect result of GetControllerOf: %v", c)
		}
	}
}

func TestIsControlledBy(t *testing.T) {
	gvk := schema.GroupVersionKind{
		Group:   "group",
		Version: "v1",
		Kind:    "Kind",
	}
	obj1 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID: "uid1",
		},
	}
	obj2 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID: "uid2",
			OwnerReferences: []OwnerReference{
				*NewControllerRef(obj1, gvk),
			},
		},
	}
	obj3 := &metaObj{
		ObjectMeta: ObjectMeta{
			UID: "uid3",
			OwnerReferences: []OwnerReference{
				*NewControllerRef(obj2, gvk),
			},
		},
	}
	if !IsControlledBy(obj2, obj1) || !IsControlledBy(obj3, obj2) {
		t.Error("Incorrect IsControlledBy result: false")
	}
	if IsControlledBy(obj3, obj1) {
		t.Error("Incorrect IsControlledBy result: true")
	}
}
