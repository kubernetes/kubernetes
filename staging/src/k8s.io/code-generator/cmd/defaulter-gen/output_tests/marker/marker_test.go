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

package marker

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external"
	externalexternal "k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external/external"
	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external2"
	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external3"
)

func getPointerFromString(s string) *string {
	return &s
}

var (
	defaultInt32 int32 = 32
	defaultInt64 int64 = 64
)

func Test_Marker(t *testing.T) {
	testcases := []struct {
		name string
		in   Defaulted
		out  Defaulted
	}{
		{
			name: "default",
			in:   Defaulted{},
			out: Defaulted{
				StringDefault:      "bar",
				StringEmptyDefault: "",
				StringEmpty:        "",
				StringPointer:      getPointerFromString("default"),
				Int64:              &defaultInt64,
				Int32:              &defaultInt32,
				IntDefault:         1,
				IntEmptyDefault:    0,
				IntEmpty:           0,
				FloatDefault:       0.5,
				FloatEmptyDefault:  0.0,
				FloatEmpty:         0.0,
				List: []Item{
					getPointerFromString("foo"),
					getPointerFromString("bar"),
				},
				Sub: &SubStruct{
					S: "foo",
					I: 5,
				},
				OtherSub: SubStruct{
					S: "",
					I: 1,
				},
				StructList: []SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				PtrStructList: []*SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				StringList: []string{
					"foo",
				},
				Map: map[string]Item{
					"foo": getPointerFromString("bar"),
				},
				StructMap: map[string]SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				PtrStructMap: map[string]*SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				AliasPtr: getPointerFromString("banana"),
			},
		},
		{
			name: "values-omitempty",
			in: Defaulted{
				StringDefault: "changed",
				IntDefault:    5,
			},
			out: Defaulted{
				StringDefault:      "changed",
				StringEmptyDefault: "",
				StringEmpty:        "",
				StringPointer:      getPointerFromString("default"),
				Int64:              &defaultInt64,
				Int32:              &defaultInt32,
				IntDefault:         5,
				IntEmptyDefault:    0,
				IntEmpty:           0,
				FloatDefault:       0.5,
				FloatEmptyDefault:  0.0,
				FloatEmpty:         0.0,
				List: []Item{
					getPointerFromString("foo"),
					getPointerFromString("bar"),
				},
				Sub: &SubStruct{
					S: "foo",
					I: 5,
				},
				StructList: []SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				PtrStructList: []*SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				StringList: []string{
					"foo",
				},
				OtherSub: SubStruct{
					S: "",
					I: 1,
				},
				Map: map[string]Item{
					"foo": getPointerFromString("bar"),
				},
				StructMap: map[string]SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				PtrStructMap: map[string]*SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				AliasPtr: getPointerFromString("banana"),
			},
		},
		{
			name: "lists",
			in: Defaulted{
				List: []Item{
					nil,
					getPointerFromString("bar"),
				},
			},
			out: Defaulted{
				StringDefault:      "bar",
				StringEmptyDefault: "",
				StringEmpty:        "",
				StringPointer:      getPointerFromString("default"),
				Int64:              &defaultInt64,
				Int32:              &defaultInt32,
				IntDefault:         1,
				IntEmptyDefault:    0,
				IntEmpty:           0,
				FloatDefault:       0.5,
				FloatEmptyDefault:  0.0,
				FloatEmpty:         0.0,
				List: []Item{
					getPointerFromString("apple"),
					getPointerFromString("bar"),
				},
				Sub: &SubStruct{
					S: "foo",
					I: 5,
				},
				StructList: []SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				PtrStructList: []*SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				StringList: []string{
					"foo",
				},
				OtherSub: SubStruct{
					S: "",
					I: 1,
				},
				Map: map[string]Item{
					"foo": getPointerFromString("bar"),
				},
				StructMap: map[string]SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				PtrStructMap: map[string]*SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				AliasPtr: getPointerFromString("banana"),
			},
		},
		{
			name: "stringmap",
			in: Defaulted{
				Map: map[string]Item{
					"foo": nil,
					"bar": getPointerFromString("banana"),
				},
			},
			out: Defaulted{
				StringDefault:      "bar",
				StringEmptyDefault: "",
				StringEmpty:        "",
				StringPointer:      getPointerFromString("default"),
				Int64:              &defaultInt64,
				Int32:              &defaultInt32,
				IntDefault:         1,
				IntEmptyDefault:    0,
				IntEmpty:           0,
				FloatDefault:       0.5,
				FloatEmptyDefault:  0.0,
				FloatEmpty:         0.0,
				List: []Item{
					getPointerFromString("foo"),
					getPointerFromString("bar"),
				},
				Sub: &SubStruct{
					S: "foo",
					I: 5,
				},
				StructList: []SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				PtrStructList: []*SubStruct{
					{
						S: "foo1",
						I: 1,
					},
					{
						S: "foo2",
						I: 1,
					},
				},
				StringList: []string{
					"foo",
				},
				OtherSub: SubStruct{
					S: "",
					I: 1,
				},
				Map: map[string]Item{
					"foo": getPointerFromString("apple"),
					"bar": getPointerFromString("banana"),
				},
				StructMap: map[string]SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				PtrStructMap: map[string]*SubStruct{
					"foo": {
						S: "string",
						I: 1,
					},
				},
				AliasPtr: getPointerFromString("banana"),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			SetObjectDefaults_Defaulted(&tc.in)
			if diff := cmp.Diff(tc.out, tc.in); len(diff) > 0 {
				t.Errorf("Error: Expected and actual output are different \n %s\n", diff)
			}
		})
	}
}

func Test_DefaultingFunction(t *testing.T) {
	in := DefaultedWithFunction{}
	SetObjectDefaults_DefaultedWithFunction(&in)
	out := DefaultedWithFunction{
		S1: "default_function",
		S2: "default_marker",
	}
	if diff := cmp.Diff(out, in); len(diff) > 0 {
		t.Errorf("Error: Expected and actual output are different \n %s\n", diff)
	}

}

func Test_DefaultingReference(t *testing.T) {
	dv := DefaultedValueItem(SomeValue)
	SomeDefault := SomeDefault
	SomeValue := SomeValue

	ptrVar9 := string(SomeValue)
	ptrVar8 := &ptrVar9
	ptrVar7 := (*B1)(&ptrVar8)
	ptrVar6 := (*B2)(&ptrVar7)
	ptrVar5 := &ptrVar6
	ptrVar4 := &ptrVar5
	ptrVar3 := &ptrVar4
	ptrVar2 := (*B3)(&ptrVar3)
	ptrVar1 := &ptrVar2

	var external2Str = external2.String(SomeValue)

	testcases := []struct {
		name string
		in   DefaultedWithReference
		out  DefaultedWithReference
	}{
		{
			name: "default",
			in:   DefaultedWithReference{},
			out: DefaultedWithReference{
				AliasPointerInside:              Item(&SomeDefault),
				AliasOverride:                   Item(&SomeDefault),
				AliasConvertDefaultPointer:      &dv,
				AliasPointerDefault:             &dv,
				PointerAliasDefault:             Item(getPointerFromString("apple")),
				AliasNonPointer:                 SomeValue,
				AliasPointer:                    &SomeValue,
				SymbolReference:                 SomeDefault,
				SameNamePackageSymbolReference1: external.AConstant,
				SameNamePackageSymbolReference2: externalexternal.AnotherConstant,
				PointerConversion:               (*B4)(&ptrVar1),
				PointerConversionValue:          (B4)(ptrVar1),
				FullyQualifiedLocalSymbol:       string(SomeValue),
				ImportFromAliasCast:             external3.StringPointer(&external2Str),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			SetObjectDefaults_DefaultedWithReference(&tc.in)
			if diff := cmp.Diff(tc.out, tc.in); len(diff) > 0 {
				t.Errorf("Error: Expected and actual output are different \n %s\n", diff)
			}
		})
	}
}
