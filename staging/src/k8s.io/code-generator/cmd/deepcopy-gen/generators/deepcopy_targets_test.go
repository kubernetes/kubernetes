/*
Copyright 2025 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"testing"

	"k8s.io/code-generator/cmd/deepcopy-gen/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

// copyableStruct returns a struct type with a single primitive int32 field.
// copyableType in deepcopy.go accepts a non-private struct kind.
func copyableStruct(pkgPath, name string, comments []string) *types.Type {
	return &types.Type{
		Name:         types.Name{Package: pkgPath, Name: name},
		Kind:         types.Struct,
		CommentLines: comments,
		Members: []types.Member{
			{Name: "X", Type: types.Int32},
		},
	}
}

func TestGetTargets(t *testing.T) {
	type pkgSpec struct {
		path     string
		comments []string
		// types maps type-name to its CommentLines (a copyable struct is
		// synthesized for each entry). nil means no types in the package.
		types map[string][]string
	}

	cases := []struct {
		name     string
		pkgs     []pkgSpec
		wantPkgs []string
		// wantAllTypes maps PkgPath -> expected genDeepCopy.allTypes value.
		wantAllTypes map[string]bool
		// wantRegister maps PkgPath -> expected genDeepCopy.registerTypes value.
		wantRegister map[string]bool
	}{
		{
			name: "package tag with copyable struct activates",
			pkgs: []pkgSpec{
				{
					path:     "example.com/pkg/a",
					comments: []string{"+k8s:deepcopy-gen=package"},
					types:    map[string][]string{"T": nil},
				},
			},
			wantPkgs:     []string{"example.com/pkg/a"},
			wantAllTypes: map[string]bool{"example.com/pkg/a": true},
			wantRegister: map[string]bool{"example.com/pkg/a": false},
		},
		{
			name: "package tag with register=false activates with register=false",
			pkgs: []pkgSpec{
				{
					path:     "example.com/pkg/b",
					comments: []string{"+k8s:deepcopy-gen=package,register=false"},
					types:    map[string][]string{"T": nil},
				},
			},
			wantPkgs:     []string{"example.com/pkg/b"},
			wantAllTypes: map[string]bool{"example.com/pkg/b": true},
			wantRegister: map[string]bool{"example.com/pkg/b": false},
		},
		{
			name: "package tag with register=true activates with register=true",
			pkgs: []pkgSpec{
				{
					path:     "example.com/pkg/c",
					comments: []string{"+k8s:deepcopy-gen=package,register=true"},
					types:    map[string][]string{"T": nil},
				},
			},
			wantPkgs:     []string{"example.com/pkg/c"},
			wantAllTypes: map[string]bool{"example.com/pkg/c": true},
			wantRegister: map[string]bool{"example.com/pkg/c": true},
		},
		{
			name: "package opted out is skipped",
			pkgs: []pkgSpec{
				{
					path:     "example.com/pkg/d",
					comments: []string{"+k8s:deepcopy-gen=false"},
					types:    map[string][]string{"T": nil},
				},
			},
			wantPkgs: nil,
		},
		{
			name: "no package tag but type opts in activates",
			pkgs: []pkgSpec{
				{
					path: "example.com/pkg/e",
					types: map[string][]string{
						"T": {"+k8s:deepcopy-gen=true"},
					},
				},
			},
			wantPkgs:     []string{"example.com/pkg/e"},
			wantAllTypes: map[string]bool{"example.com/pkg/e": false},
			wantRegister: map[string]bool{"example.com/pkg/e": false},
		},
		{
			name: "package tag but no copyable types is skipped",
			pkgs: []pkgSpec{
				{
					path:     "example.com/pkg/f",
					comments: []string{"+k8s:deepcopy-gen=package"},
					types:    nil,
				},
			},
			wantPkgs: nil,
		},
		{
			name: "no tag and no opt-in types is skipped",
			pkgs: []pkgSpec{
				{
					path:  "example.com/pkg/g",
					types: map[string][]string{"T": nil},
				},
			},
			wantPkgs: nil,
		},
		// Ecosystem regression: a third-party generator's tag in the same
		// doc.go must NOT cause deepcopy-gen to fail. The deepcopy-gen
		// tag still activates as expected.
		{
			name: "foreign third-party generator tag is ignored",
			pkgs: []pkgSpec{
				{
					path: "example.com/pkg/h",
					comments: []string{
						"+k8s:my-custom-gen=value",
						"+k8s:deepcopy-gen=package",
					},
					types: map[string][]string{"T": nil},
				},
			},
			wantPkgs:     []string{"example.com/pkg/h"},
			wantAllTypes: map[string]bool{"example.com/pkg/h": true},
			wantRegister: map[string]bool{"example.com/pkg/h": false},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			universe := types.Universe{}
			var inputs []string
			for _, ps := range tc.pkgs {
				pkg := &types.Package{
					Path:     ps.path,
					Dir:      ps.path,
					Name:     "pkg",
					Comments: ps.comments,
					Types:    map[string]*types.Type{},
				}
				for tname, tcomments := range ps.types {
					pkg.Types[tname] = copyableStruct(ps.path, tname, tcomments)
				}
				universe[ps.path] = pkg
				inputs = append(inputs, ps.path)
			}

			ctx := &generator.Context{
				Universe: universe,
				Inputs:   inputs,
			}

			result := GetTargets(ctx, args.New())

			var gotPkgs []string
			for _, tgt := range result {
				gotPkgs = append(gotPkgs, tgt.Path())
			}
			sort.Strings(gotPkgs)
			want := append([]string(nil), tc.wantPkgs...)
			sort.Strings(want)
			if !reflect.DeepEqual(gotPkgs, want) {
				t.Errorf("PkgPaths = %v, want %v", gotPkgs, want)
			}

			for _, tgt := range result {
				gens := tgt.Generators(ctx)
				if len(gens) != 1 {
					t.Errorf("pkg %q: got %d generators, want 1", tgt.Path(), len(gens))
					continue
				}
				gdc, ok := gens[0].(*genDeepCopy)
				if !ok {
					t.Errorf("pkg %q: generator type = %T, want *genDeepCopy", tgt.Path(), gens[0])
					continue
				}
				if want, ok := tc.wantAllTypes[tgt.Path()]; ok && gdc.allTypes != want {
					t.Errorf("pkg %q: allTypes = %v, want %v", tgt.Path(), gdc.allTypes, want)
				}
				if want, ok := tc.wantRegister[tgt.Path()]; ok && gdc.registerTypes != want {
					t.Errorf("pkg %q: registerTypes = %v, want %v", tgt.Path(), gdc.registerTypes, want)
				}
			}
		})
	}
}
