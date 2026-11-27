/*
Copyright The Kubernetes Authors.

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

package generators

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	gotypes "go/types"
	"strings"
	"testing"

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

func TestWithFuncsShadowedField(t *testing.T) {
	base := &types.Type{
		Name: types.Name{Package: "example.com/api", Name: "Base"},
		Kind: types.Struct,
		Members: []types.Member{
			{Name: "Enabled", Type: types.Bool, Tags: `json:"enabled,omitempty"`},
			{Name: "Mode", Type: types.String, Tags: `json:"mode,omitempty"`},
		},
	}
	embedded := types.Member{Name: "Base", Type: base, Embedded: true, Tags: `json:",inline"`}
	direct := types.Member{Name: "Enabled", Type: &types.Type{Kind: types.Pointer, Elem: types.Bool}, Tags: `json:"enabled,omitempty"`}
	middle := &types.Type{Name: types.Name{Package: "example.com/api", Name: "Middle"}, Kind: types.Struct, Members: []types.Member{embedded}}
	nested := types.Member{Name: "Middle", Type: middle, Embedded: true, Tags: `json:",inline"`}
	middleOverride := &types.Type{Name: types.Name{Package: "example.com/api", Name: "MiddleOverride"}, Kind: types.Struct, Members: []types.Member{embedded, direct}}
	for _, tc := range []struct {
		name         string
		members      []types.Member
		wantSelector string
	}{
		{name: "embedded first", members: []types.Member{embedded, direct}, wantSelector: "b.Enabled"},
		{name: "direct first", members: []types.Member{direct, embedded}, wantSelector: "b.Enabled"},
		{name: "unshadowed", members: []types.Member{embedded}, wantSelector: "b.BaseApplyConfiguration.Enabled"},
		{name: "ignored direct field", members: []types.Member{embedded, {Name: "Enabled", Type: types.Bool, Tags: `json:"-"`}}, wantSelector: "b.BaseApplyConfiguration.Enabled"},
		{name: "nested embedded first", members: []types.Member{nested, direct}, wantSelector: "b.Enabled"},
		{name: "nested direct first", members: []types.Member{direct, nested}, wantSelector: "b.Enabled"},
		{name: "intermediate direct field", members: []types.Member{{Name: "MiddleOverride", Type: middleOverride, Embedded: true, Tags: `json:",inline"`}}, wantSelector: "b.MiddleOverrideApplyConfiguration.Enabled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := &types.Type{Name: types.Name{Package: "example.com/api", Name: "Override"}, Kind: types.Struct, Members: tc.members}
			g := &applyConfigurationGenerator{
				localPkg:    "example.com/ac",
				imports:     generator.NewImportTrackerForPackage("example.com/ac"),
				applyConfig: applyConfig{Type: input, ApplyConfiguration: types.Ref("example.com/ac", "OverrideApplyConfiguration")},
				refGraph:    refGraph{base.Name: "example.com/ac", middle.Name: "example.com/ac", middleOverride.Name: "example.com/ac", input.Name: "example.com/ac"},
			}
			context := &generator.Context{Namers: NameSystems()}
			for name, system := range g.Namers(context) {
				context.Namers[name] = system
			}
			var output bytes.Buffer
			output.WriteString("package ac\n")
			if err := g.GenerateType(context, input, &output); err != nil {
				t.Fatal(err)
			}
			file, err := parser.ParseFile(token.NewFileSet(), "generated.go", output.Bytes(), 0)
			if err != nil {
				t.Fatal(err)
			}
			setters, assignments := 0, 0
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Name.Name != "WithEnabled" {
					continue
				}
				setters++
				ast.Inspect(fn.Body, func(node ast.Node) bool {
					assign, ok := node.(*ast.AssignStmt)
					if !ok {
						return true
					}
					assignments++
					var selector bytes.Buffer
					if err := format.Node(&selector, token.NewFileSet(), assign.Lhs[0]); err != nil {
						t.Fatal(err)
					}
					if got := selector.String(); got != tc.wantSelector {
						t.Errorf("WithEnabled assigns %s, want %s", got, tc.wantSelector)
					}
					return true
				})
			}
			if setters != 1 {
				t.Errorf("got %d WithEnabled methods, want 1", setters)
			}
			if assignments != 1 {
				t.Errorf("got %d WithEnabled assignments, want 1", assignments)
			}
			if !strings.Contains(output.String(), "b.BaseApplyConfiguration.Mode = &value") {
				t.Error("unshadowed Mode setter was lost")
			}
			// Type-check the generated declarations together, including embedded configurations.
			for _, embeddedType := range []*types.Type{base, middle, middleOverride} {
				g.applyConfig = applyConfig{Type: embeddedType, ApplyConfiguration: types.Ref("example.com/ac", embeddedType.Name.Name+ApplyConfigurationTypeSuffix)}
				if err := g.GenerateType(context, embeddedType, &output); err != nil {
					t.Fatal(err)
				}
			}
			fset := token.NewFileSet()
			file, err = parser.ParseFile(fset, "generated.go", output.Bytes(), 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := new(gotypes.Config).Check("example.com/ac", fset, []*ast.File{file}, nil); err != nil {
				t.Fatal(err)
			}
		})
	}
}
