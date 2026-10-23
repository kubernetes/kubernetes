/*
Copyright 2026 The Kubernetes Authors.

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

package cel

import (
	"context"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/utils/ptr"
)

func TestCompileCELExpressionWithVariables(t *testing.T) {
	compiler := GetCompiler(Features{})
	envType := environment.NewExpressions

	device := Device{
		Driver: "gpu.example.com",
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"gpu.example.com/model": {StringValue: ptr.To("A100")},
			"gpu.example.com/numa":  {IntValue: ptr.To(int64(1))},
		},
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"gpu.example.com/memory": {Value: resource.MustParse("40Gi")},
		},
	}

	tests := []struct {
		name       string
		expression string
		variables  []NamedVariable
		wantMatch  bool
		wantErr    string
	}{
		{
			name:       "basic variables",
			expression: `device.driver == "gpu.example.com" && variables.attrs.model == "A100" && variables.memOK`,
			variables: []NamedVariable{
				{Name: "attrs", Expression: `device.attributes["gpu.example.com"]`},
				{Name: "memOK", Expression: `device.capacity["gpu.example.com"].memory.isGreaterThan(quantity("39Gi"))`},
			},
			wantMatch: true,
		},
		{
			name:       "variable references earlier variable",
			expression: `variables.model == "A100"`,
			variables: []NamedVariable{
				{Name: "attrs", Expression: `device.attributes["gpu.example.com"]`},
				{Name: "model", Expression: `variables.attrs.model`},
			},
			wantMatch: true,
		},
		{
			name:       "forward reference rejected",
			expression: `variables.later`,
			variables: []NamedVariable{
				{Name: "later", Expression: `variables.earlier`},
				{Name: "earlier", Expression: `true`},
			},
			wantErr: "variable \"later\"",
		},
		{
			name:       "lazy unused variable with missing attribute does not fail",
			expression: `device.driver == "other.example.com" && variables.missing.model == "x"`,
			variables: []NamedVariable{
				{Name: "missing", Expression: `device.attributes["gpu.example.com"].noSuchKey`},
			},
			wantMatch: false,
		},
		{
			name:       "referenced missing attribute fails",
			expression: `variables.missing.model == "x"`,
			variables: []NamedVariable{
				{Name: "missing", Expression: `device.attributes["gpu.example.com"].noSuchKey`},
			},
			wantErr: "no such key",
		},
		{
			name:       "main expression must be bool",
			expression: `variables.attrs`,
			variables: []NamedVariable{
				{Name: "attrs", Expression: `device.attributes["gpu.example.com"]`},
			},
			wantErr: "must evaluate to bool",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := compiler.CompileCELExpressionWithVariables(tc.expression, tc.variables, Options{EnvType: &envType})
			if tc.wantErr != "" && result.Error != nil {
				if !strings.Contains(result.Error.Detail, tc.wantErr) && !strings.Contains(result.Error.Error(), tc.wantErr) {
					t.Fatalf("compile error %q does not contain %q", result.Error.Detail, tc.wantErr)
				}
				return
			}
			if result.Error != nil {
				if tc.wantErr != "" {
					// runtime error path below
				} else {
					t.Fatalf("unexpected compile error: %v", result.Error)
				}
			}
			if result.Error != nil {
				t.Fatalf("unexpected compile error: %v", result.Error)
			}

			match, _, err := result.DeviceMatches(context.Background(), device)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected runtime error containing %q", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("runtime error %q does not contain %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("DeviceMatches: %v", err)
			}
			if match != tc.wantMatch {
				t.Fatalf("match=%v, want %v", match, tc.wantMatch)
			}
		})
	}
}

func TestCompileCELExpressionWithVariablesCostAggregation(t *testing.T) {
	compiler := GetCompiler(Features{})
	envType := environment.NewExpressions
	result := compiler.CompileCELExpressionWithVariables(
		`variables.a && variables.b`,
		[]NamedVariable{
			{Name: "a", Expression: `true`},
			{Name: "b", Expression: `true`},
		},
		Options{EnvType: &envType},
	)
	if result.Error != nil {
		t.Fatalf("compile: %v", result.Error)
	}
	plain := compiler.CompileCELExpression(`true && true`, Options{EnvType: &envType})
	if plain.Error != nil {
		t.Fatalf("plain compile: %v", plain.Error)
	}
	if result.MaxCost < plain.MaxCost {
		t.Fatalf("composed MaxCost %d should be >= plain MaxCost %d", result.MaxCost, plain.MaxCost)
	}
}

func TestCacheKeyIncludesVariables(t *testing.T) {
	cache := NewCache(10, Features{})
	plain := cache.GetOrCompileSelector(`true`, nil)
	withVars := cache.GetOrCompileSelector(`variables.a`, []NamedVariable{{Name: "a", Expression: `true`}})
	if plain.Error != nil {
		t.Fatalf("plain: %v", plain.Error)
	}
	if withVars.Error != nil {
		t.Fatalf("withVars: %v", withVars.Error)
	}
	// Different programs must not share a cache entry keyed only on expression.
	if plain.Program == withVars.Program {
		t.Fatal("expected distinct compiled programs for plain vs variables selector")
	}
	again := cache.GetOrCompileSelector(`variables.a`, []NamedVariable{{Name: "a", Expression: `true`}})
	if again.Program != withVars.Program {
		t.Fatal("expected cache hit for identical variables selector")
	}
}
