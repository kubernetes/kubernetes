/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"testing"
)

func TestTypes_ListType(t *testing.T) {
	list := NewListType(StringType, -1)
	if !list.IsList() {
		t.Error("list type not identifiable as list")
	}
	if list.TypeName() != "list" {
		t.Errorf("got %s, wanted list", list.TypeName())
	}
	if list.DefaultValue() == nil {
		t.Error("got nil zero value for list type")
	}
	if list.ElemType.TypeName() != "string" {
		t.Errorf("got %s, wanted elem type of string", list.ElemType.TypeName())
	}
	expT, err := list.ExprType()
	if err != nil {
		t.Errorf("fail to get cel type: %s", err)
	}
	if expT.GetListType() == nil {
		t.Errorf("got %v, wanted CEL list type", expT)
	}
}

func TestTypes_MapType(t *testing.T) {
	mp := NewMapType(StringType, IntType, -1)
	if !mp.IsMap() {
		t.Error("map type not identifiable as map")
	}
	if mp.TypeName() != "map" {
		t.Errorf("got %s, wanted map", mp.TypeName())
	}
	if mp.DefaultValue() == nil {
		t.Error("got nil zero value for map type")
	}
	if mp.KeyType.TypeName() != "string" {
		t.Errorf("got %s, wanted key type of string", mp.KeyType.TypeName())
	}
	if mp.ElemType.TypeName() != "int" {
		t.Errorf("got %s, wanted elem type of int", mp.ElemType.TypeName())
	}
	expT, err := mp.ExprType()
	if err != nil {
		t.Errorf("fail to get cel type: %s", err)
	}
	if expT.GetMapType() == nil {
		t.Errorf("got %v, wanted CEL map type", expT)
	}
}

func testValue(t *testing.T, id int64, val interface{}) *DynValue {
	t.Helper()
	dv, err := NewDynValue(id, val)
	if err != nil {
		t.Fatalf("NewDynValue(%d, %v) failed: %v", id, val, err)
	}
	return dv
}
