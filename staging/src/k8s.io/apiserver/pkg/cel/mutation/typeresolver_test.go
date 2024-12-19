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

package mutation

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/mutation/dynamic"
)

func TestTypeResolver(t *testing.T) {
	for _, tc := range []struct {
		name               string
		expression         string
		expectedValue      any
		expectCompileError string
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
			// list literal does not require new Path code of the type provider
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
		{
			name:               "invalid type",
			expression:         "Invalid{}",
			expectCompileError: "undeclared reference to 'Invalid'",
		},
		{
			name:          "logic around JSONPatch",
			expression:    `true ? JSONPatch{op: "add", path: "/spec/replicas", value: 3} : JSONPatch{op: "remove", path: "/spec/replicas"}`,
			expectedValue: &JSONPatchVal{Op: "add", Path: "/spec/replicas", Val: types.Int(3)},
		},
		{
			name:          "JSONPatch add",
			expression:    `JSONPatch{op: "add", path: "/spec/replicas", value: 3}`,
			expectedValue: &JSONPatchVal{Op: "add", Path: "/spec/replicas", Val: types.Int(3)},
		},
		{
			name:          "JSONPatch remove",
			expression:    `JSONPatch{op: "remove", path: "/spec/replicas"}`,
			expectedValue: &JSONPatchVal{Op: "remove", Path: "/spec/replicas"},
		},
		{
			name:          "JSONPatch replace",
			expression:    `JSONPatch{op: "replace", path: "/spec/replicas", value: 3}`,
			expectedValue: &JSONPatchVal{Op: "replace", Path: "/spec/replicas", Val: types.Int(3)},
		},
		{
			name:          "JSONPatch move",
			expression:    `JSONPatch{op: "move", from: "/spec/replicas", path: "/spec/replicas"}`,
			expectedValue: &JSONPatchVal{Op: "move", From: "/spec/replicas", Path: "/spec/replicas"},
		},
		{
			name:          "JSONPatch copy",
			expression:    `JSONPatch{op: "copy", from: "/spec/replicas", path: "/spec/replicas"}`,
			expectedValue: &JSONPatchVal{Op: "copy", From: "/spec/replicas", Path: "/spec/replicas"},
		},
		{
			name:          "JSONPatch test",
			expression:    `JSONPatch{op: "test", path: "/spec/replicas", value: 3}`,
			expectedValue: &JSONPatchVal{Op: "test", Path: "/spec/replicas", Val: types.Int(3)},
		},
		{
			name:          "JSONPatch invalid op",
			expression:    `JSONPatch{op: "invalid", path: "/spec/replicas", value: 3}`,
			expectedValue: &JSONPatchVal{Op: "invalid", Path: "/spec/replicas", Val: types.Int(3)},
			// no error because the values are not checked in compilation.
		},
		{
			name:          "JSONPatch missing path",
			expression:    `JSONPatch{op: "add", value: 3}`,
			expectedValue: &JSONPatchVal{Op: "add", Val: types.Int(3)},
			// no error because the values are not checked in compilation.
		},
		{
			name:          "JSONPatch missing value",
			expression:    `JSONPatch{op: "add", path: "/spec/replicas"}`,
			expectedValue: &JSONPatchVal{Op: "add", Path: "/spec/replicas"},
			// no error because the values are not checked in compilation.
		},
		{
			name:               "JSONPatch invalid value",
			expression:         `JSONPatch{op: "add", path: "/spec/replicas", value: Invalid{}}`,
			expectCompileError: "undeclared reference to 'Invalid'",
		},
		{
			name:               "JSONPatch invalid path",
			expression:         `JSONPatch{op: "add", path: Invalid{}, value: 3}`,
			expectCompileError: "undeclared reference to 'Invalid'",
		},
		{
			name:               "JSONPatch invalid from",
			expression:         `JSONPatch{op: "move", from: Invalid{}, path: "/spec/replicas"}`,
			expectCompileError: "undeclared reference to 'Invalid'",
		},
		{
			name:               "JSONPatch invalid op type",
			expression:         `JSONPatch{op: 1, path: "/spec/replicas", value: 3}`,
			expectCompileError: "expected type of field 'op' is 'string' but provided type is 'int'",
		},
		{
			name:               "JSONPatch invalid path type",
			expression:         `JSONPatch{op: "add", path: 1, value: 3}`,
			expectCompileError: "expected type of field 'path' is 'string' but provided type is 'int'",
		},
		{
			name:               "JSONPatch invalid from type",
			expression:         `JSONPatch{op: "move", from: 1, path: "/spec/replicas"}`,
			expectCompileError: "expected type of field 'from' is 'string' but provided type is 'int'",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, option := common.NewResolverTypeProviderAndEnvOption(&DynamicTypeResolver{})
			env := mustCreateEnv(t, option)
			ast, issues := env.Compile(tc.expression)
			if len(tc.expectCompileError) > 0 {
				if issues == nil {
					t.Fatalf("expected error %v but got no error", tc.expectCompileError)
				}
				if !strings.Contains(issues.String(), tc.expectCompileError) {
					t.Fatalf("expected error %v but got %v", tc.expectCompileError, issues.String())
				}
				return
			}

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
				t.Errorf("expected %#v but got %#v", tc.expectedValue, v)
			}
		})
	}
}

// TestCELOptional is an exploration test to demonstrate how CEL optional library
// behave for the use cases that the mutation library requires.
func TestCELOptional(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		expression           string
		expectedVal          ref.Val
		expectedCompileError string
	}{
		{
			name: "construct existing field with none, empty object",
			expression: `Object{
				?existing: optional.none()
			}`,
			expectedVal: dynamic.NewObjectVal(types.NewObjectType("Object"), map[string]ref.Val{
				// "existing" field was not set.
			}),
		},
		{
			name:        "object of zero value, ofNonZeroValue",
			expression:  `Object{?spec: optional.ofNonZeroValue(Object.spec{?replicas: Object{}.?replicas})}`,
			expectedVal: dynamic.NewObjectVal(types.NewObjectType("Object"), map[string]ref.Val{
				// "existing" field was not set.
			}),
		},
		{
			name:        "access existing field, return none",
			expression:  `Object{}.?existing`,
			expectedVal: types.OptionalNone,
		},
		{
			name:        "map non-existing field, return none",
			expression:  `{"foo": 1}[?"bar"]`,
			expectedVal: types.OptionalNone,
		},
		{
			name:        "map existing field, return actual value",
			expression:  `{"foo": 1}[?"foo"]`,
			expectedVal: types.OptionalOf(types.Int(1)),
		},
		{
			// Map has a different behavior than Object
			//
			// Quoting from
			// https://github.com/google/cel-spec/blob/master/doc/langdef.md#field-selection
			//
			// To test for the presence of a field, the boolean-valued macro has(e.f) can be used.
			//
			// 1. If e evaluates to a map, then has(e.f) indicates whether the string f is
			// a key in the map (note that f must syntactically be an identifier).
			//
			name: "has on a map, de-sugared, non-existing field, returns false",
			// has marco supports only the dot access syntax.
			expression:  `has({"foo": 1}.bar)`,
			expectedVal: types.False,
		},
		{
			name: "has on a map, de-sugared, existing field, returns true",
			// has marco supports only the dot access syntax.
			expression:  `has({"foo": 1}.foo)`,
			expectedVal: types.True,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, option := common.NewResolverTypeProviderAndEnvOption(&DynamicTypeResolver{})
			env := mustCreateEnvWithOptional(t, option)
			ast, issues := env.Compile(tc.expression)
			if len(tc.expectedCompileError) > 0 {
				if issues == nil {
					t.Fatalf("expected error %v but got no error", tc.expectedCompileError)
				}
				if !strings.Contains(issues.String(), tc.expectedCompileError) {
					t.Fatalf("expected error %v but got %v", tc.expectedCompileError, issues.String())
				}
				return
			}
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
			if equals := tc.expectedVal.Equal(r); equals.Value() != true {
				t.Errorf("expected %#+v but got %#+v", tc.expectedVal, r)
			}
		})
	}
}

// mustCreateEnv creates the default env for testing, with given option.
// it fatally fails the test if the env fails to set up.
func mustCreateEnv(t testing.TB, envOptions ...cel.EnvOption) *cel.Env {
	envSet, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true).
		Extend(environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 0), // Always enabled. This is just for test.
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

// mustCreateEnvWithOptional creates the default env for testing, with given option,
// and set up the optional library with default configuration.
// it fatally fails the test if the env fails to set up.
func mustCreateEnvWithOptional(t testing.TB, envOptions ...cel.EnvOption) *cel.Env {
	return mustCreateEnv(t, append(envOptions, cel.OptionalTypes())...)
}
