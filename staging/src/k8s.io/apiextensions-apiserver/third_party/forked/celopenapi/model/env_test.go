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
)

func TestEnv_Vars(t *testing.T) {
	env := NewEnv("test.v1.Environment")
	env.Container = "test.v1"
	env.Vars = []*Var{
		NewVar("greeting", StringType),
		NewVar("replies", NewListType(StringType)),
	}
	expr := `greeting == 'hello' && replies.size() > 0`
	stdEnv, _ := cel.NewEnv()
	ast, iss := stdEnv.Compile(expr)
	if iss.Err() == nil {
		t.Errorf("got ast %v, expected error", ast)
	}
	custEnv, err := stdEnv.Extend(env.ExprEnvOptions()...)
	if err != nil {
		t.Fatal(err)
	}
	_, iss = custEnv.Compile(expr)
	if iss.Err() != nil {
		t.Errorf("got error %v, wanted ast", iss)
	}
}

func TestEnv_Funcs(t *testing.T) {
	env := NewEnv("test.v1.Environment")
	env.Container = "test.v1"
	env.Functions = []*Function{
		NewFunction("greeting",
			NewOverload("string_greeting_string", StringType, StringType, BoolType),
			NewFreeFunctionOverload("greeting_string", StringType, BoolType)),
		NewFunction("getOrDefault",
			NewOverload("map_get_or_default_param",
				NewMapType(NewTypeParam("K"), NewTypeParam("V")),
				NewTypeParam("K"), NewTypeParam("V"),
				NewTypeParam("V"))),
	}
	expr := `greeting('hello') && 'jim'.greeting('hello') && {'a': 0}.getOrDefault('b', 1) == 1`
	stdEnv, _ := cel.NewEnv()
	ast, iss := stdEnv.Compile(expr)
	if iss.Err() == nil {
		t.Errorf("got ast %v, expected error", ast)
	}
	custEnv, err := stdEnv.Extend(env.ExprEnvOptions()...)
	if err != nil {
		t.Fatal(err)
	}
	_, iss = custEnv.Compile(expr)
	if iss.Err() != nil {
		t.Errorf("got error %v, wanted ast", iss)
	}
}
