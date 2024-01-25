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
	"strings"
	"testing"

	celtypes "github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apiserver/pkg/cel/mutation/common"
)

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
			// question mark syntax still requires the field to exist in object construction
			name: "construct non-existing field, compile error",
			expression: `Object{
				?nonExisting: optional.none()
			}`,
			expectedCompileError: `undefined field 'nonExisting'`,
		},
		{
			// The root cause of the behavior above is that, has on an object (or Message in the Language Def),
			// still require the field to be declared in the schema.
			//
			// Quoting from
			// https://github.com/google/cel-spec/blob/master/doc/langdef.md#field-selection
			//
			// To test for the presence of a field, the boolean-valued macro has(e.f) can be used.
			//
			// 2. If e evaluates to a message and f is not a declared field for the message,
			// has(e.f) raises a no_such_field error.
			name:                 "has(Object{}), de-sugared, compile error",
			expression:           "has(Object{}.nonExisting)",
			expectedCompileError: `undefined field 'nonExisting'`,
		},
		{
			name: "construct existing field with none, empty object",
			expression: `Object{
				?existing: optional.none()
			}`,
			expectedVal: common.NewObjectVal(nil, map[string]ref.Val{
				// "existing" field was not set.
			}),
		},
		{
			name:        "object of zero value, ofNonZeroValue",
			expression:  `Object{?spec: optional.ofNonZeroValue(Object.spec{?replicas: Object{}.?replicas})}`,
			expectedVal: common.NewObjectVal(nil, map[string]ref.Val{
				// "existing" field was not set.
			}),
		},
		{
			name:                 "access non-existing field, return none",
			expression:           `Object{}.?nonExisting`,
			expectedCompileError: `undefined field 'nonExisting'`,
		},
		{
			name:        "access existing field, return none",
			expression:  `Object{}.?existing`,
			expectedVal: celtypes.OptionalNone,
		},
		{
			name:        "map non-existing field, return none",
			expression:  `{"foo": 1}[?"bar"]`,
			expectedVal: celtypes.OptionalNone,
		},
		{
			name:        "map existing field, return actual value",
			expression:  `{"foo": 1}[?"foo"]`,
			expectedVal: celtypes.OptionalOf(celtypes.Int(1)),
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
			expectedVal: celtypes.False,
		},
		{
			name: "has on a map, de-sugared, existing field, returns true",
			// has marco supports only the dot access syntax.
			expression:  `has({"foo": 1}.foo)`,
			expectedVal: celtypes.True,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, option := NewTypeProviderAndEnvOption(&mockTypeResolverForOptional{
				mockTypeResolver: &mockTypeResolver{},
			})
			env := mustCreateEnvWithOptional(t, option)
			ast, issues := env.Compile(tc.expression)
			if issues != nil {
				if tc.expectedCompileError == "" {
					t.Fatalf("unexpected issues during compilation: %v", issues)
				} else if !strings.Contains(issues.String(), tc.expectedCompileError) {
					t.Fatalf("unexpected compile error, want to contain %q but got %v", tc.expectedCompileError, issues)
				}
				return
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
				t.Errorf("expected %v but got %v", tc.expectedVal, r)
			}
		})
	}
}
