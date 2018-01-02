/*
Copyright 2016 The Kubernetes Authors.

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

package wholepkg

import (
	"reflect"
	"testing"

	fuzz "github.com/google/gofuzz"
)

func TestDeepCopyPrimitives(t *testing.T) {
	x := Struct_Primitives{}
	y := Struct_Primitives{}

	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal to start, but are not")
	}

	fuzzer := fuzz.New()
	fuzzer.Fuzz(&x)
	fuzzer.Fuzz(&y)

	if reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should not be equal, but are")
	}

	x.DeepCopyInto(&y)
	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal, but are not")
	}
}

func TestDeepCopyInterfaceFields(t *testing.T) {
	x := Struct_Interfaces{}
	y := Struct_Interfaces{}

	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal to start, but are not")
	}

	fuzzer := fuzz.New()

	obj := Struct_ExplicitObject{}
	fuzzer.Fuzz(&obj)
	x.ObjectField = &obj

	sel := Struct_ExplicitSelectorExplicitObject{}
	fuzzer.Fuzz(&sel)
	x.SelectorField = &sel

	if reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should not be equal, but are")
	}

	x.DeepCopyInto(&y)
	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal, but are not")
	}
}

func TestNilCopy(t *testing.T) {
	var x *Struct_B
	y := x.DeepCopy()
	if y != nil {
		t.Error("Expected nil as deepcopy of nil, got %+v", y)
	}
}

func assertMethod(t *testing.T, typ reflect.Type, name string) {
	if _, found := typ.MethodByName(name); !found {
		t.Errorf("Struct_ExplicitObject must have %v method", name)
	}
}

func assertNotMethod(t *testing.T, typ reflect.Type, name string) {
	if _, found := typ.MethodByName(name); found {
		t.Errorf("%v must not have %v method", typ, name)
	}
}

func TestInterfaceTypes(t *testing.T) {
	explicitObject := reflect.TypeOf(&Struct_ExplicitObject{})
	assertMethod(t, explicitObject, "DeepCopyObject")

	typeMeta := reflect.TypeOf(&Struct_TypeMeta{})
	assertNotMethod(t, typeMeta, "DeepCopy")

	objectAndList := reflect.TypeOf(&Struct_ObjectAndList{})
	assertMethod(t, objectAndList, "DeepCopyObject")
	assertMethod(t, objectAndList, "DeepCopyList")

	objectAndObject := reflect.TypeOf(&Struct_ObjectAndObject{})
	assertMethod(t, objectAndObject, "DeepCopyObject")

	explicitSelectorExplicitObject := reflect.TypeOf(&Struct_ExplicitSelectorExplicitObject{})
	assertMethod(t, explicitSelectorExplicitObject, "DeepCopySelector")
	assertMethod(t, explicitSelectorExplicitObject, "DeepCopyObject")
}

func TestInterfaceDeepCopy(t *testing.T) {
	x := Struct_ExplicitObject{}

	fuzzer := fuzz.New()
	fuzzer.Fuzz(&x)

	y_obj := x.DeepCopyObject()
	y, ok := y_obj.(*Struct_ExplicitObject)
	if !ok {
		t.Fatalf("epxected Struct_ExplicitObject from Struct_ExplicitObject.DeepCopyObject, got: %t", y_obj)
	}
	if !reflect.DeepEqual(y, &x) {
		t.Error("objects should be equal, but are not")
	}
}

func TestInterfaceNonPointerDeepCopy(t *testing.T) {
	x := Struct_NonPointerExplicitObject{}

	fuzzer := fuzz.New()
	fuzzer.Fuzz(&x)

	y_obj := x.DeepCopyObject()
	y, ok := y_obj.(Struct_NonPointerExplicitObject)
	if !ok {
		t.Fatalf("epxected Struct_NonPointerExplicitObject from Struct_NonPointerExplicitObject.DeepCopyObject, got: %t", y_obj)
	}
	if !reflect.DeepEqual(y, x) {
		t.Error("objects should be equal, but are not")
	}
}
