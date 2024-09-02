/*
Copyright 2023 The Kubernetes Authors.

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

package lazy

import (
	"fmt"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"

	"k8s.io/apimachinery/pkg/util/version"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

func TestLazyMapType(t *testing.T) {
	env, variablesType, err := buildTestEnv()
	if err != nil {
		t.Fatal(err)
	}
	variablesMap := NewMapValue(variablesType)
	activation := &testActivation{variables: variablesMap}

	// add foo as a string
	variablesType.Fields["foo"] = apiservercel.NewDeclField("foo", apiservercel.StringType, true, nil, nil)
	variablesMap.Append("foo", func(_ *MapValue) ref.Val {
		return types.String("foo-string")
	})

	exp := "variables.foo == 'foo-string'"
	v, err := compileAndRun(env, activation, exp)
	if err != nil {
		t.Fatalf("%q: %v", exp, err)
	}
	if !v.Value().(bool) {
		t.Errorf("expected equal but got non-equal")
	}

	evalCounter := 0
	// add dict as a map constructed from an expression
	variablesType.Fields["dict"] = apiservercel.NewDeclField("dict", apiservercel.DynType, true, nil, nil)
	variablesMap.Append("dict", func(_ *MapValue) ref.Val {
		evalCounter++
		v, err := compileAndRun(env, activation, `{"a": "a"}`)
		if err != nil {
			return types.NewErr(err.Error())
		}
		return v
	})

	// iterate the map with .all
	exp = `variables.all(n, n != "")`
	v, err = compileAndRun(env, activation, exp)
	if err != nil {
		t.Fatalf("%q: %v", exp, err)
	}
	if v.Value().(bool) != true {
		t.Errorf("%q: wrong result: %v", exp, v.Value())
	}

	// add unused as a string
	variablesType.Fields["unused"] = apiservercel.NewDeclField("unused", apiservercel.StringType, true, nil, nil)
	variablesMap.Append("unused", func(_ *MapValue) ref.Val {
		t.Fatalf("unused variable must not be evaluated")
		return nil
	})

	exp = "variables.dict.a + ' ' + variables.dict.a + ' ' + variables.foo"
	v, err = compileAndRun(env, activation, exp)
	if err != nil {
		t.Fatalf("%q: %v", exp, err)
	}
	if v.Value().(string) != "a a foo-string" {
		t.Errorf("%q: wrong result: %v", exp, v.Value())
	}
	if evalCounter != 1 {
		t.Errorf("expected eval %d times but got %d", 1, evalCounter)
	}

	// unused due to boolean short-circuiting
	// if `variables.unused` is evaluated, the whole test will have a fatal error and exit.
	exp = "variables.dict.a == 'wrong' && variables.unused == 'unused'"
	v, err = compileAndRun(env, activation, exp)
	if err != nil {
		t.Fatalf("%q: %v", exp, err)
	}
	if v.Value().(bool) != false {
		t.Errorf("%q: wrong result: %v", exp, v.Value())
	}
}

type testActivation struct {
	variables *MapValue
}

func compileAndRun(env *cel.Env, activation *testActivation, exp string) (ref.Val, error) {
	ast, issues := env.Compile(exp)
	if issues != nil {
		return nil, fmt.Errorf("fail to compile: %v", issues)
	}
	prog, err := env.Program(ast)
	if err != nil {
		return nil, fmt.Errorf("cannot create program: %w", err)
	}
	v, _, err := prog.Eval(activation)
	if err != nil {
		return nil, fmt.Errorf("cannot eval program: %w", err)
	}
	return v, nil
}

func buildTestEnv() (*cel.Env, *apiservercel.DeclType, error) {
	variablesType := apiservercel.NewMapType(apiservercel.StringType, apiservercel.AnyType, 0)
	variablesType.Fields = make(map[string]*apiservercel.DeclField)
	envSet, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true).Extend(
		environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 28),
			EnvOptions: []cel.EnvOption{
				cel.Variable("variables", variablesType.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				variablesType,
			},
		})
	if err != nil {
		return nil, nil, err
	}
	env, err := envSet.Env(environment.NewExpressions)
	return env, variablesType, err
}

func (a *testActivation) ResolveName(name string) (any, bool) {
	switch name {
	case "variables":
		return a.variables, true
	default:
		return nil, false
	}
}

func (a *testActivation) Parent() interpreter.Activation {
	return nil
}
