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

package tester

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
)

type testObject struct {
	runtime.TypeMeta
	DefaultedField string
}

func (obj *testObject) DeepCopyObject() runtime.Object {
	ret := *obj
	return &ret
}

type testDefaulter struct{}

func (d *testDefaulter) Default(obj runtime.Object) {
	if t, ok := obj.(*testObject); ok {
		if t.DefaultedField == "" {
			t.DefaultedField = "defaulted"
		}
	}
}

func TestWithDefaulter(t *testing.T) {
	defaulter := &testDefaulter{}

	tester := &Tester{}
	tester.WithDefaulter(defaulter)

	if tester.defaulter == nil {
		t.Fatal("expected defaulter to be set")
	}

	result := tester.WithDefaulter(defaulter)
	if result != tester {
		t.Fatal("expected WithDefaulter to return the same Tester instance for chaining")
	}
}

func TestCreateObjectAppliesDefaults(t *testing.T) {
	defaulter := &testDefaulter{}
	tester := &Tester{defaulter: defaulter}

	obj := &testObject{}

	if obj.DefaultedField != "" {
		t.Fatalf("expected DefaultedField to be empty, got %q", obj.DefaultedField)
	}

	if tester.defaulter != nil {
		tester.defaulter.Default(obj)
	}

	if obj.DefaultedField != "defaulted" {
		t.Fatalf("expected DefaultedField to be %q, got %q", "defaulted", obj.DefaultedField)
	}
}

func TestSetObjectsForListAppliesDefaults(t *testing.T) {
	defaulter := &testDefaulter{}
	tester := &Tester{defaulter: defaulter}

	objects := []runtime.Object{
		&testObject{},
		&testObject{DefaultedField: "kept"},
	}

	if tester.defaulter != nil {
		for i := range objects {
			tester.defaulter.Default(objects[i])
		}
	}

	expected := []string{"defaulted", "kept"}
	for i, obj := range objects {
		tObj := obj.(*testObject)
		if tObj.DefaultedField != expected[i] {
			t.Errorf("object %d: expected DefaultedField to be %q, got %q", i, expected[i], tObj.DefaultedField)
		}
	}
}

func TestNoDefaulterDoesNotApplyDefaults(t *testing.T) {
	tester := &Tester{}

	obj := &testObject{}

	if tester.defaulter != nil {
		tester.defaulter.Default(obj)
	}

	if obj.DefaultedField != "" {
		t.Fatalf("expected DefaultedField to be empty when no defaulter is set, got %q", obj.DefaultedField)
	}
}
