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

package common

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/mutation/dynamic"
)

func TestTypeProvider(t *testing.T) {
	for _, tc := range []struct {
		name               string
		expression         string
		expectedValue      any
		expectCompileError string
	}{
		{
			name:          "not an object",
			expression:    `2 * 31 * 1847`,
			expectedValue: int64(114514), // type resolver should not interfere.
		},
		{
			name:          "empty",
			expression:    "Test{}",
			expectedValue: map[string]any{},
		},
		{
			name:          "simple",
			expression:    "Test{x: 1}",
			expectedValue: map[string]any{"x": int64(1)},
		},
		{
			name:               "invalid type",
			expression:         "Objectfoo{}",
			expectCompileError: "undeclared reference to 'Objectfoo'",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			option := ResolverEnvOption(&mockTypeResolver{})
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
			if v := r.Value(); !reflect.DeepEqual(tc.expectedValue, v) {
				t.Errorf("expected %v but got %v", tc.expectedValue, v)
			}
		})
	}
}

// mockTypeResolver is a mock implementation of DynamicTypeResolver that
// allows the object to contain any field.
type mockTypeResolver struct {
}

func (m *mockTypeResolver) Resolve(name string) (ResolvedType, bool) {
	if name == "Test" {
		return newMockResolvedType(m, name), true
	}
	return nil, false
}

// mockResolvedType is a mock implementation of ResolvedType that
// contains any field.
type mockResolvedType struct {
	objectType *types.Type
	resolver   TypeResolver
}

func newMockResolvedType(resolver TypeResolver, name string) *mockResolvedType {
	objectType := types.NewObjectType(name)
	return &mockResolvedType{
		objectType: objectType,
		resolver:   resolver,
	}
}

func (m *mockResolvedType) HasTrait(trait int) bool {
	return m.objectType.HasTrait(trait)
}

func (m *mockResolvedType) TypeName() string {
	return m.objectType.TypeName()
}

func (m *mockResolvedType) Type() *types.Type {
	return m.objectType
}

func (m *mockResolvedType) TypeType() *types.Type {
	return types.NewTypeTypeWithParam(m.objectType)
}

func (m *mockResolvedType) Field(name string) (*types.FieldType, bool) {
	return &types.FieldType{
		Type: types.DynType,
		IsSet: func(target any) bool {
			return true
		},
		GetFrom: func(target any) (any, error) {
			return nil, nil
		},
	}, true
}

func (m *mockResolvedType) FieldNames() ([]string, bool) {
	return nil, true
}

func (m *mockResolvedType) Val(fields map[string]ref.Val) ref.Val {
	return dynamic.NewObjectVal(m.objectType, fields)
}

// mustCreateEnv creates the default env for testing, with given option.
// it fatally fails the test if the env fails to set up.
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
