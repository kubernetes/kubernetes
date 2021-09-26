// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package delta_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/delta"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
)

var oContent exports.Content
var nContent exports.Content
var oBreaking exports.Content
var nBreaking exports.Content

func init() {
	oContent, _ = exports.Get("./testdata/nonbreaking/old")
	nContent, _ = exports.Get("./testdata/nonbreaking/new")
	oBreaking, _ = exports.Get("./testdata/breaking/old")
	nBreaking, _ = exports.Get("./testdata/breaking/new")
}

func Test_GetAddedExports(t *testing.T) {
	aContent := delta.GetExports(oContent, nContent)

	// const

	if l := len(aContent.Consts); l != 4 {
		t.Logf("wrong number of consts added, have %v, want %v", l, 4)
		t.Fail()
	}

	cAdded := map[string]exports.Const{
		"Blue":    {Type: "Color", Value: "Blue"},
		"Green":   {Type: "Color", Value: "Green"},
		"Red":     {Type: "Color", Value: "Red"},
		"Holiday": {Type: "DayOfWeek", Value: "Holiday"},
	}

	for k, v := range cAdded {
		t.Run(fmt.Sprintf("const %s", k), func(t *testing.T) {
			if c, ok := aContent.Consts[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else if c.Type != v.Type {
				t.Logf("mismatched const type, have %s, want %s", aContent.Consts[k].Type, v.Type)
				t.Fail()
			}
		})
	}

	// func

	if l := len(aContent.Funcs); l != 6 {
		t.Logf("wrong number of funcs added, have %v, want %v", l, 6)
		t.Fail()
	}

	fAdded := map[string]exports.Func{
		"DoNothing2":                 {},
		"Client.ExportData":          {Params: strPtr("context.Context,string,string,ExportRDBParameters"), Returns: strPtr("ExportDataFuture,error")},
		"Client.ExportDataPreparer":  {Params: strPtr("context.Context,string,string,ExportRDBParameters"), Returns: strPtr("*http.Request,error")},
		"Client.ExportDataSender":    {Params: strPtr("*http.Request"), Returns: strPtr("ExportDataFuture,error")},
		"Client.ExportDataResponder": {Params: strPtr("*http.Response"), Returns: strPtr("autorest.Response,error")},
		"ExportDataFuture.Result":    {Params: strPtr("Client"), Returns: strPtr("autorest.Response,error")},
	}

	for k, v := range fAdded {
		t.Run(fmt.Sprintf("func %s", k), func(t *testing.T) {
			if f, ok := aContent.Funcs[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else if !reflect.DeepEqual(f, v) {
				t.Logf("mismatched func type, have %+v, want %+v", v, f)
				t.Fail()
			}
		})
	}

	// interface

	if l := len(aContent.Interfaces); l != 2 {
		t.Logf("wrong number of interfaces added, have %v, want %v", l, 2)
		t.Fail()
	}

	iAdded := map[string]exports.Interface{
		"NewInterface": {Methods: map[string]exports.Func{
			"One": {Params: strPtr("int")},
			"Two": {Returns: strPtr("error")},
		}},
		"SomeInterface": {Methods: map[string]exports.Func{
			"NewMethod": {Params: strPtr("string"), Returns: strPtr("bool,error")},
		}},
	}

	for k, v := range iAdded {
		t.Run(fmt.Sprintf("interface %s", k), func(t *testing.T) {
			if i, ok := aContent.Interfaces[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else if !reflect.DeepEqual(i, v) {
				t.Logf("mismatched interface type, have %+v, want %+v", i, v)
				t.Fail()
			}
		})
	}

	// struct

	if l := len(aContent.Structs); l != 4 {
		t.Logf("wrong number of structs added, have %v, want %v", l, 4)
		t.Fail()
	}

	sAdded := map[string]exports.Struct{
		"ExportDataFuture": {
			AnonymousFields: []string{"azure.Future"},
			Fields:          map[string]string{"NewField": "string"},
		},
		"ExportRDBParameters": {
			Fields: map[string]string{
				"Format":    "*string",
				"Prefix":    "*string",
				"Container": "*string",
			},
		},
		"CreateProperties": {
			Fields: map[string]string{
				"NewField": "*float64",
			},
		},
		"DeleteFuture": {
			Fields: map[string]string{
				"NewField": "string",
			},
		},
	}

	for k, v := range sAdded {
		t.Run(fmt.Sprintf("struct %s", k), func(t *testing.T) {
			if s, ok := aContent.Structs[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else if !reflect.DeepEqual(s, v) {
				t.Logf("mismatched struct type, have %+v, want %+v", v, s)
				t.Fail()
			}
		})
	}
}

func Test_GetAddedStructFields(t *testing.T) {
	nf := delta.GetStructFields(oContent, nContent)

	if l := len(nf); l != 2 {
		t.Logf("wrong number of structs with new fields, have %v, want %v", l, 2)
		t.Fail()
	}

	added := map[string]exports.Struct{
		"CreateProperties": {
			Fields: map[string]string{"NewField": "*float64"},
		},
		"DeleteFuture": {
			Fields: map[string]string{"NewField": "string"},
		},
	}

	if !reflect.DeepEqual(added, nf) {
		t.Logf("mismatched fields added, have %+v, want %+v", nf, added)
		t.Fail()
	}
}

func Test_GetAddedInterfaceMethods(t *testing.T) {
	ni := delta.GetInterfaceMethods(oContent, nContent)

	if l := len(ni); l != 1 {
		t.Logf("wrong number of interfaces with new methods, have %v, want %v", l, 1)
		t.Fail()
	}

	added := map[string]exports.Interface{
		"SomeInterface": {
			Methods: map[string]exports.Func{
				"NewMethod": {Params: strPtr("string"), Returns: strPtr("bool,error")},
			},
		},
	}

	if !reflect.DeepEqual(added, ni) {
		t.Logf("mismatched methods added, have %+v, want %+v", ni, added)
		t.Fail()
	}
}

func Test_GetNoChanges(t *testing.T) {
	nc := delta.GetExports(nContent, nContent)
	if !reflect.DeepEqual(nc, delta.NewContent()) {
		t.Log("expected empty exports")
		t.Fail()
	}

	ni := delta.GetInterfaceMethods(nContent, nContent)
	if !reflect.DeepEqual(ni, map[string]exports.Interface{}) {
		t.Log("expected no new interfaces")
		t.Fail()
	}

	nf := delta.GetStructFields(oContent, oContent)
	if !reflect.DeepEqual(nf, map[string]exports.Struct{}) {
		t.Log("expected no new struct fields")
		t.Fail()
	}
}

func Test_GetConstTypeChanges(t *testing.T) {
	cc := delta.GetConstTypeChanges(oBreaking, nBreaking)

	if l := len(cc); l != 8 {
		t.Logf("wrong number of const changed, have %v, want %v", l, 8)
		t.Fail()
	}

	change := delta.Signature{
		From: "DayOfWeek",
		To:   "Day",
	}
	changed := map[string]delta.Signature{
		"Friday":    change,
		"Monday":    change,
		"Saturday":  change,
		"Sunday":    change,
		"Thursday":  change,
		"Tuesday":   change,
		"Wednesday": change,
		"Weekend":   change,
	}

	if !reflect.DeepEqual(cc, changed) {
		t.Logf("mismatched changes, have %+v, want %+v", cc, changed)
		t.Fail()
	}
}

func Test_GetFuncSigChanges(t *testing.T) {
	fsc := delta.GetFuncSigChanges(oBreaking, nBreaking)

	if l := len(fsc); l != 6 {
		t.Logf("wrong number of func sigs changed, have %v want %v", l, 6)
		t.Fail()
	}

	changed := map[string]delta.FuncSig{
		"DoNothing": {
			Params: &delta.Signature{From: delta.None, To: "string"},
		},
		"DoNothingWithParam": {
			Params: &delta.Signature{From: "int", To: delta.None},
		},
		"Client.List": {
			Params:  &delta.Signature{From: "context.Context", To: "context.Context,string"},
			Returns: &delta.Signature{From: "ListResultPage,error", To: "ListResult,error"},
		},
		"Client.ListPreparer": {
			Params: &delta.Signature{From: "context.Context", To: "context.Context,string"},
		},
		"Client.Delete": {
			Params: &delta.Signature{From: "context.Context,string,string", To: "context.Context,string"},
		},
		"Client.DeletePreparer": {
			Params: &delta.Signature{From: "context.Context,string,string", To: "context.Context,string"},
		},
	}

	for k, v := range changed {
		t.Run(fmt.Sprintf("func %s", k), func(t *testing.T) {
			if f, ok := fsc[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else {
				if !reflect.DeepEqual(v, f) {
					t.Logf("mismatched changes, have %+v, want %+v", f, v)
					t.Fail()
				}
			}
		})
	}
}

func Test_GetInterfaceMethodSigChanges(t *testing.T) {
	isc := delta.GetInterfaceMethodSigChanges(oBreaking, nBreaking)

	if l := len(isc); l != 1 {
		t.Logf("wrong number of interfaces with method sig changes, have %v, want %v", l, 1)
		t.Fail()
	}

	changed := map[string]delta.InterfaceDef{
		"SomeInterface": {
			MethodSigs: map[string]delta.FuncSig{
				"One": {Params: &delta.Signature{From: delta.None, To: "string"}},
				"Two": {Params: &delta.Signature{From: "bool", To: "bool,int"}},
			},
		},
	}

	for k, v := range changed {
		t.Run(fmt.Sprintf("interface %s", k), func(t *testing.T) {
			if i, ok := isc[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else {
				if !reflect.DeepEqual(v, i) {
					t.Logf("mismatched changes, have %+v, want %+v", i, v)
					t.Fail()
				}
			}
		})
	}
}

func Test_GetStructFieldChanges(t *testing.T) {
	sfc := delta.GetStructFieldChanges(oBreaking, nBreaking)

	if l := len(sfc); l != 2 {
		t.Logf("wrong number of structs with field changes, have %v, want %v", l, 2)
		t.Fail()
	}

	changed := map[string]delta.StructDef{
		"CreateProperties": {
			Fields: map[string]delta.Signature{
				"SubnetID":           {From: "*string", To: "*int"},
				"RedisConfiguration": {From: "map[string]*string", To: "interface{}"},
			},
		},
		"ListResult": {
			Fields: map[string]delta.Signature{
				"NextLink": {From: "*string", To: "string"},
			},
		},
	}

	for k, v := range changed {
		t.Run(fmt.Sprintf("struct %s", k), func(t *testing.T) {
			if s, ok := sfc[k]; !ok {
				t.Log("missing")
				t.Fail()
			} else {
				if !reflect.DeepEqual(v, s) {
					t.Logf("mismatched changes, have %+v, want %+v", s, v)
					t.Fail()
				}
			}
		})
	}
}

func strPtr(s string) *string {
	return &s
}
