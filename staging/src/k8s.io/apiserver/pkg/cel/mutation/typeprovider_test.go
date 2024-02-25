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
	"testing"
)

func TestTypeProvider(t *testing.T) {
	for _, tc := range []struct {
		name          string
		expression    string
		expectedValue any
	}{
		{
			name:          "not an object",
			expression:    `2 * 31 * 1847`,
			expectedValue: int64(114514), // type resolver should not interfere.
		},
		{
			name:          "empty",
			expression:    "Object{}",
			expectedValue: map[string]any{},
		},
		{
			name:          "Object.spec",
			expression:    "Object{spec: Object.spec{replicas: 3}}",
			expectedValue: map[string]any{"spec": map[string]any{"replicas": int64(3)}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, option := NewTypeProviderAndEnvOption(&mockTypeResolver{})
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
			if v := r.Value(); !reflect.DeepEqual(tc.expectedValue, v) {
				t.Errorf("expected %v but got %v", tc.expectedValue, v)
			}
		})
	}
}
