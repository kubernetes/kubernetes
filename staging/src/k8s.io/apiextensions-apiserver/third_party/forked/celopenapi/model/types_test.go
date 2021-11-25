// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

func TestTypes_ListType(t *testing.T) {
	list := NewListType(StringType)
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
	if list.ExprType().GetListType() == nil {
		t.Errorf("got %v, wanted CEL list type", list.ExprType())
	}
}

func TestTypes_MapType(t *testing.T) {
	mp := NewMapType(StringType, IntType)
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
	if mp.ExprType().GetMapType() == nil {
		t.Errorf("got %v, wanted CEL map type", mp.ExprType())
	}
}

func TestTypes_RuleTypesFieldMapping(t *testing.T) {
	stdEnv, _ := cel.NewEnv()
	reg := NewRegistry(stdEnv)
	rt, err := NewRuleTypes("CustomObject", testSchema(), true, reg)
	if err != nil {
		t.Fatal(err)
	}
	rt.TypeProvider = stdEnv.TypeProvider()
	nestedFieldType, found := rt.FindFieldType("CustomObject", "nested")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested', wanted found")
	}
	if nestedFieldType.Type.GetMessageType() != "CustomObject.nested" {
		t.Errorf("got field type %v, wanted mock_template.nested", nestedFieldType.Type)
	}
	subnameFieldType, found := rt.FindFieldType("CustomObject.nested", "subname")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.subname', wanted found")
	}
	if subnameFieldType.Type.GetPrimitive() != exprpb.Type_STRING {
		t.Errorf("got field type %v, wanted string", subnameFieldType.Type)
	}
	flagsFieldType, found := rt.FindFieldType("CustomObject.nested", "flags")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.flags', wanted found")
	}
	if flagsFieldType.Type.GetMapType() == nil {
		t.Errorf("got field type %v, wanted map", flagsFieldType.Type)
	}
	flagFieldType, found := rt.FindFieldType("CustomObject.nested.flags", "my_flag")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.flags.my_flag', wanted found")
	}
	if flagFieldType.Type.GetPrimitive() != exprpb.Type_BOOL {
		t.Errorf("got field type %v, wanted bool", flagFieldType.Type)
	}

	// Manually constructed instance of the schema.
	name := NewField(1, "name")
	name.Ref = testValue(t, 2, "test-instance")
	nestedVal := NewMapValue()
	flags := NewField(5, "flags")
	flagsVal := NewMapValue()
	myFlag := NewField(6, "my_flag")
	myFlag.Ref = testValue(t, 7, true)
	flagsVal.AddField(myFlag)
	flags.Ref = testValue(t, 8, flagsVal)
	dates := NewField(9, "dates")
	dates.Ref = testValue(t, 10, NewListValue())
	nestedVal.AddField(flags)
	nestedVal.AddField(dates)
	nested := NewField(3, "nested")
	nested.Ref = testValue(t, 4, nestedVal)
	mapVal := NewMapValue()
	mapVal.AddField(name)
	mapVal.AddField(nested)
	//rule := rt.ConvertToRule(testValue(t, 11, mapVal))
	//if rule == nil {
	//	t.Error("map could not be converted to rule")
	//}
	//if rule.GetID() != 11 {
	//	t.Errorf("got %d as the rule id, wanted 11", rule.GetID())
	//}
	//ruleVal := rt.NativeToValue(rule)
	//if ruleVal == nil {
	//	t.Error("got CEL rule value of nil, wanted non-nil")
	//}

	opts, err := rt.EnvOptions(stdEnv.TypeProvider())
	if err != nil {
		t.Fatal(err)
	}
	ruleEnv, err := stdEnv.Extend(opts...)
	if err != nil {
		t.Fatal(err)
	}
	helloVal := ruleEnv.TypeAdapter().NativeToValue("hello")
	if helloVal.Equal(types.String("hello")) != types.True {
		t.Errorf("got %v, wanted types.String('hello')", helloVal)
	}
}

func testValue(t *testing.T, id int64, val interface{}) *DynValue {
	t.Helper()
	dv, err := NewDynValue(id, val)
	if err != nil {
		t.Fatalf("model.NewDynValue(%d, %v) failed: %v", id, val, err)
	}
	return dv
}
