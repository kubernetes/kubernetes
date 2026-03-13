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

package model

import (
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

func TestTypes_RuleTypesFieldMapping(t *testing.T) {
	stdEnv, _ := cel.NewEnv()
	rt := apiservercel.NewDeclTypeProvider(SchemaDeclType(testSchema(), true).MaybeAssignTypeName("CustomObject"))
	nestedFieldType, found := rt.FindStructFieldType("CustomObject", "nested")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested', wanted found")
	}
	if nestedFieldType.Type.DeclaredTypeName() != "CustomObject.nested" {
		t.Errorf("got field type %v, wanted mock_template.nested", nestedFieldType.Type)
	}
	subnameFieldType, found := rt.FindStructFieldType("CustomObject.nested", "subname")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.subname', wanted found")
	}
	if subnameFieldType.Type.TypeName() != "string" {
		t.Errorf("got field type %v, wanted string", subnameFieldType.Type)
	}
	flagsFieldType, found := rt.FindStructFieldType("CustomObject.nested", "flags")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.flags', wanted found")
	}
	if flagsFieldType.Type.Kind() != types.MapKind {
		t.Errorf("got field type %v, wanted map", flagsFieldType.Type)
	}
	flagFieldType, found := rt.FindStructFieldType("CustomObject.nested.flags", "my_flag")
	if !found {
		t.Fatal("got field not found for 'CustomObject.nested.flags.my_flag', wanted found")
	}
	if flagFieldType.Type.Kind() != types.BoolKind {
		t.Errorf("got field type %v, wanted bool", flagFieldType.Type)
	}

	// Manually constructed instance of the schema.
	name := apiservercel.NewField(1, "name")
	name.Ref = testValue(t, 2, "test-instance")
	nestedVal := apiservercel.NewMapValue()
	flags := apiservercel.NewField(5, "flags")
	flagsVal := apiservercel.NewMapValue()
	myFlag := apiservercel.NewField(6, "my_flag")
	myFlag.Ref = testValue(t, 7, true)
	flagsVal.AddField(myFlag)
	flags.Ref = testValue(t, 8, flagsVal)
	dates := apiservercel.NewField(9, "dates")
	dates.Ref = testValue(t, 10, apiservercel.NewListValue())
	nestedVal.AddField(flags)
	nestedVal.AddField(dates)
	nested := apiservercel.NewField(3, "nested")
	nested.Ref = testValue(t, 4, nestedVal)
	mapVal := apiservercel.NewMapValue()
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

	opts, err := rt.EnvOptions(stdEnv.CELTypeProvider())
	if err != nil {
		t.Fatal(err)
	}
	ruleEnv, err := stdEnv.Extend(opts...)
	if err != nil {
		t.Fatal(err)
	}
	helloVal := ruleEnv.CELTypeAdapter().NativeToValue("hello")
	if helloVal.Equal(types.String("hello")) != types.True {
		t.Errorf("got %v, wanted types.String('hello')", helloVal)
	}
}

func testValue(t *testing.T, id int64, val interface{}) *apiservercel.DynValue {
	t.Helper()
	dv, err := apiservercel.NewDynValue(id, val)
	if err != nil {
		t.Fatalf("NewDynValue(%d, %v) failed: %v", id, val, err)
	}
	return dv
}
