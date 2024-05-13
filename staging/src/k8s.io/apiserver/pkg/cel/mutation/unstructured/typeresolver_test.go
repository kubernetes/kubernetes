/*
Copyright 2024 The Kubernetes Authors.

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

package unstructured

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/mutation"
)

func TestTypeProvider(t *testing.T) {
	for _, tc := range []struct {
		name          string
		expression    string
		expectedValue any
	}{
		{
			name:          "not an object",
			expression:    `string(114514)`,
			expectedValue: "114514",
		},
		{
			name:          "empty",
			expression:    "Object{}",
			expectedValue: map[string]any{},
		},
		{
			name:       "Object.spec",
			expression: "Object{spec: Object.spec{replicas: 3}}",
			expectedValue: map[string]any{
				"spec": map[string]any{
					// an integer maps to int64
					"replicas": int64(3),
				},
			},
		},
		{
			// list literal does not require new path code of the type provider
			// comparing to the object literal.
			// This test case serves as a note of "supported syntax"
			name: "Object.spec.template.containers",
			expression: `Object{
				spec: Object.spec{
					template: Object.spec.template{
						containers: [
							Object.spec.template.containers.item{
								name: "nginx",
								image: "nginx",
								args: ["-g"]
							}
						]
					}
				}
			}`,
			expectedValue: map[string]any{
				"spec": map[string]any{
					"template": map[string]any{
						"containers": []any{
							map[string]any{
								"name":  "nginx",
								"image": "nginx",
								"args":  []any{"-g"},
							},
						},
					},
				},
			},
		},
		{
			name: "list of ints",
			expression: `Object{
				intList: [1, 2, 3]
			}`,
			expectedValue: map[string]any{
				"intList": []any{int64(1), int64(2), int64(3)},
			},
		},
		{
			name: "map string-to-string",
			expression: `Object{
				annotations: {"foo": "bar"}
			}`,
			expectedValue: map[string]any{
				"annotations": map[string]any{
					"foo": "bar",
				},
			},
		},
		{
			name: "field access",
			expression: `Object{
				intList: [1, 2, 3]
			}.intList.sum()`,
			expectedValue: int64(6),
		},
		{
			name:          "equality check",
			expression:    "Object{spec: Object.spec{replicas: 3}} == Object{spec: Object.spec{replicas: 1 + 2}}",
			expectedValue: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, option := mutation.NewTypeProviderAndEnvOption(&TypeResolver{})
			env := mustCreateEnv(t, option)
			ast, issues := env.Compile(tc.expression)
			if issues != nil {
				t.Fatalf("unexpected issues during compilation: %v", issues)
			}
			program, err := env.Program(ast)
			if err != nil {
				t.Fatalf("unexpected error while creating program: %v", err)
			}
			r, _, err := program.Eval(map[string]any{})
			if err != nil {
				t.Fatalf("unexpected error during evaluation: %v", err)
			}
			if v := r.Value(); !reflect.DeepEqual(v, tc.expectedValue) {
				t.Errorf("expected %v but got %v", tc.expectedValue, v)
			}
		})
	}
}
func mustCreateEnv(t testing.TB, envOptions ...cel.EnvOption) *cel.Env {
	envSet, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true).
		Extend(environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 30),
			EnvOptions:        envOptions,
		})
	if err != nil {
		t.Fatalf("fail to create env set: %v", err)
	}
	env, err := envSet.Env(environment.StoredExpressions)
	if err != nil {
		t.Fatalf("fail to setup env: %v", env)
	}
	return env
}
