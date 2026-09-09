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

	"sigs.k8s.io/randfill"
)

func TestDeepCopyPrimitives(t *testing.T) {
	x := StructPrimitives{}
	y := StructPrimitives{}

	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal to start, but are not")
	}

	fuzzer := randfill.New()
	fuzzer.Fill(&x)
	fuzzer.Fill(&y)

	if reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should not be equal, but are")
	}

	x.DeepCopyInto(&y)
	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal, but are not")
	}
}

func TestDeepCopyInterfaceFields(t *testing.T) {
	x := StructInterfaces{}
	y := StructInterfaces{}

	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal to start, but are not")
	}

	fuzzer := randfill.New()

	obj := StructExplicitObject{}
	fuzzer.Fill(&obj)
	x.ObjectField = &obj

	sel := StructExplicitSelectorExplicitObject{}
	fuzzer.Fill(&sel)
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
	var x *StructB
	y := x.DeepCopy()
	if y != nil {
		t.Errorf("Expected nil as deepcopy of nil, got %+v", y)
	}
}

func assertMethod(t *testing.T, typ reflect.Type, name string) {
	if _, found := typ.MethodByName(name); !found {
		t.Errorf("StructExplicitObject must have %v method", name)
	}
}

func assertNotMethod(t *testing.T, typ reflect.Type, name string) {
	if _, found := typ.MethodByName(name); found {
		t.Errorf("%v must not have %v method", typ, name)
	}
}

func TestInterfaceTypes(t *testing.T) {
	explicitObject := reflect.TypeOf(&StructExplicitObject{})
	assertMethod(t, explicitObject, "DeepCopyObject")

	typeMeta := reflect.TypeOf(&StructTypeMeta{})
	assertNotMethod(t, typeMeta, "DeepCopy")

	objectAndList := reflect.TypeOf(&StructObjectAndList{})
	assertMethod(t, objectAndList, "DeepCopyObject")
	assertMethod(t, objectAndList, "DeepCopyList")

	objectAndObject := reflect.TypeOf(&StructObjectAndObject{})
	assertMethod(t, objectAndObject, "DeepCopyObject")

	explicitSelectorExplicitObject := reflect.TypeOf(&StructExplicitSelectorExplicitObject{})
	assertMethod(t, explicitSelectorExplicitObject, "DeepCopySelector")
	assertMethod(t, explicitSelectorExplicitObject, "DeepCopyObject")
}

func TestInterfaceDeepCopy(t *testing.T) {
	x := StructExplicitObject{}

	fuzzer := randfill.New()
	fuzzer.Fill(&x)

	yObj := x.DeepCopyObject()
	y, ok := yObj.(*StructExplicitObject)
	if !ok {
		t.Fatalf("epxected StructExplicitObject from StructExplicitObject.DeepCopyObject, got: %t", yObj)
	}
	if !reflect.DeepEqual(y, &x) {
		t.Error("objects should be equal, but are not")
	}
}

func TestInterfaceNonPointerDeepCopy(t *testing.T) {
	x := StructNonPointerExplicitObject{}

	fuzzer := randfill.New()
	fuzzer.Fill(&x)

	yObj := x.DeepCopyObject()
	y, ok := yObj.(StructNonPointerExplicitObject)
	if !ok {
		t.Fatalf("epxected StructNonPointerExplicitObject from StructNonPointerExplicitObject.DeepCopyObject, got: %t", yObj)
	}
	if !reflect.DeepEqual(y, x) {
		t.Error("objects should be equal, but are not")
	}
}
