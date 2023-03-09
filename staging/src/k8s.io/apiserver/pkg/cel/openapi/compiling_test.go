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

package openapi

import (
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestMultipleTypes(t *testing.T) {
	env, err := buildTestEnv()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		expression         string
		expectCompileError bool
		expectEvalResult   bool
	}{
		{
			expression:       "foo.foo == bar.bar",
			expectEvalResult: true,
		},
		{
			expression:         "foo.bar == 'value'",
			expectCompileError: true,
		},
		{
			expression:       "foo.foo == 'value'",
			expectEvalResult: true,
		},
		{
			expression:       "bar.bar == 'value'",
			expectEvalResult: true,
		},
		{
			expression:       "foo.common + bar.common <= 2",
			expectEvalResult: false, // 3 > 2
		},
		{
			expression:         "foo.confusion == bar.confusion",
			expectCompileError: true,
		},
	} {
		t.Run(tc.expression, func(t *testing.T) {
			ast, issues := env.Compile(tc.expression)
			if issues != nil {
				if tc.expectCompileError {
					return
				}
				t.Fatalf("compile error: %v", issues)
			}
			if issues != nil {
				t.Fatal(issues)
			}
			p, err := env.Program(ast)
			if err != nil {
				t.Fatal(err)
			}
			ret, _, err := p.Eval(&simpleActivation{
				foo: map[string]any{"foo": "value", "common": 1, "confusion": "114514"},
				bar: map[string]any{"bar": "value", "common": 2, "confusion": 114514},
			})
			if err != nil {
				t.Fatal(err)
			}
			if ret.Type() != types.BoolType {
				t.Errorf("bad result type: %v", ret.Type())
			}
			if res := ret.Value().(bool); tc.expectEvalResult != res {
				t.Errorf("expectEvalResult expression evaluates to %v, got %v", tc.expectEvalResult, res)
			}
		})
	}

}

// buildTestEnv sets up an environment that contains two variables, "foo" and
// "bar".
// foo is an object with a string field "foo", an integer field "common", and a string field "confusion"
// bar is an object with a string field "bar", an integer field "common", and an integer field "confusion"
func buildTestEnv() (*cel.Env, error) {
	var opts []cel.EnvOption
	opts = append(opts, cel.HomogeneousAggregateLiterals())
	opts = append(opts, cel.EagerlyValidateDeclarations(true), cel.DefaultUTCTimeZone(true))
	opts = append(opts, library.ExtensionLibs...)
	env, err := cel.NewEnv(opts...)
	if err != nil {
		return nil, err
	}
	reg := apiservercel.NewRegistry(env)

	declType := common.SchemaDeclType(simpleMapSchema("foo", spec.StringProperty()), true)
	fooRT, err := apiservercel.NewRuleTypes("fooType", declType, reg)
	if err != nil {
		return nil, err
	}
	fooRT, err = fooRT.WithTypeProvider(env.TypeProvider())
	if err != nil {
		return nil, err
	}
	fooType, _ := fooRT.FindDeclType("fooType")

	declType = common.SchemaDeclType(simpleMapSchema("bar", spec.Int64Property()), true)
	barRT, err := apiservercel.NewRuleTypes("barType", declType, reg)
	if err != nil {
		return nil, err
	}
	barRT, err = barRT.WithTypeProvider(env.TypeProvider())
	if err != nil {
		return nil, err
	}
	barType, _ := barRT.FindDeclType("barType")

	opts = append(opts, cel.CustomTypeProvider(&apiservercel.CompositedTypeProvider{Providers: []ref.TypeProvider{fooRT, barRT}}))
	opts = append(opts, cel.CustomTypeAdapter(&apiservercel.CompositedTypeAdapter{Adapters: []ref.TypeAdapter{fooRT, barRT}}))
	opts = append(opts, cel.Variable("foo", fooType.CelType()))
	opts = append(opts, cel.Variable("bar", barType.CelType()))
	return env.Extend(opts...)
}

func simpleMapSchema(fieldName string, confusionSchema *spec.Schema) common.Schema {
	return &Schema{Schema: &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type: []string{"object"},
			Properties: map[string]spec.Schema{
				fieldName:   *spec.StringProperty(),
				"common":    *spec.Int64Property(),
				"confusion": *confusionSchema,
			},
		},
	}}
}

type simpleActivation struct {
	foo any
	bar any
}

func (a *simpleActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case "foo":
		return a.foo, true
	case "bar":
		return a.bar, true
	default:
		return nil, false
	}
}

func (a *simpleActivation) Parent() interpreter.Activation {
	return nil
}
