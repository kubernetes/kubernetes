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
	"os"
	"path/filepath"
	"sort"
	"testing"

	"k8s.io/code-generator/cmd/register-gen/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

func TestGetTargets(t *testing.T) {
	type pkgSpec struct {
		path     string
		dir      string // optional; will use t.TempDir() if needRegisterFile
		comments []string
		// typeMembers describes a single struct type in the package. nil means no types.
		typeMembers []types.Member
		// needRegisterFile, when true, creates dir/register.go before invoking GetTargets.
		needRegisterFile bool
	}

	externalTypeMeta := []types.Member{
		{Name: "TypeMeta", Embedded: true, Tags: `json:",inline"`},
	}
	internalTypeMeta := []types.Member{
		{Name: "TypeMeta", Embedded: true},
	}

	cases := []struct {
		name        string
		pkgs        []pkgSpec
		wantPaths   []string          // package paths expected in returned targets, sorted
		wantGroups  map[string]string // pkgPath -> expected group
		wantHasType map[string]bool   // pkgPath -> whether typesToGenerate is non-empty
	}{
		{
			name: "external pkg with +groupName activates",
			pkgs: []pkgSpec{{
				path:        "k8s.io/api/foo/v1",
				comments:    []string{"+groupName=foo.k8s.io"},
				typeMembers: externalTypeMeta,
			}},
			wantPaths:   []string{"k8s.io/api/foo/v1"},
			wantGroups:  map[string]string{"k8s.io/api/foo/v1": "foo.k8s.io"},
			wantHasType: map[string]bool{"k8s.io/api/foo/v1": true},
		},
		{
			name: "external pkg with +k8s:register-gen=false is opted out",
			pkgs: []pkgSpec{{
				path:        "k8s.io/api/foo/v1",
				comments:    []string{"+k8s:register-gen=false"},
				typeMembers: externalTypeMeta,
			}},
			wantPaths: nil,
		},
		{
			name: "opt-out wins over +groupName",
			pkgs: []pkgSpec{{
				path:        "k8s.io/api/foo/v1",
				comments:    []string{"+groupName=foo.k8s.io", "+k8s:register-gen=false"},
				typeMembers: externalTypeMeta,
			}},
			wantPaths: nil,
		},
		{
			name: "no +groupName and no +k8s:register-gen tag is skipped",
			pkgs: []pkgSpec{{
				path:        "k8s.io/api/foo/v1",
				typeMembers: externalTypeMeta,
			}},
			wantPaths: nil,
		},
		{
			// isInternal returns an error when the package has no TypeMeta-
			// bearing types at all, and GetTargets treats that error as a skip.
			name: "+groupName but no TypeMeta types is skipped (isInternal errors)",
			pkgs: []pkgSpec{{
				path:     "k8s.io/api/foo/v1",
				comments: []string{"+groupName=foo.k8s.io"},
			}},
			wantPaths: nil,
		},
		{
			name: "internal pkg (TypeMeta without json tag) is skipped",
			pkgs: []pkgSpec{{
				path:        "k8s.io/api/foo/v1",
				comments:    []string{"+groupName=foo"},
				typeMembers: internalTypeMeta,
			}},
			wantPaths: nil,
		},
		{
			name: "pkg with existing register.go is skipped",
			pkgs: []pkgSpec{{
				path:             "k8s.io/api/foo/v1",
				comments:         []string{"+groupName=foo.k8s.io"},
				typeMembers:      externalTypeMeta,
				needRegisterFile: true,
			}},
			wantPaths: nil,
		},
		// Ecosystem regression: a third-party generator's tag in the same
		// doc.go must NOT cause register-gen to fail. The +groupName= still
		// activates as expected.
		{
			name: "foreign third-party generator tag is ignored",
			pkgs: []pkgSpec{{
				path: "k8s.io/api/foo/v1",
				comments: []string{
					"+k8s:my-custom-gen=value",
					"+groupName=foo.k8s.io",
				},
				typeMembers: externalTypeMeta,
			}},
			wantPaths:   []string{"k8s.io/api/foo/v1"},
			wantGroups:  map[string]string{"k8s.io/api/foo/v1": "foo.k8s.io"},
			wantHasType: map[string]bool{"k8s.io/api/foo/v1": true},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			universe := types.Universe{}
			ctx := &generator.Context{
				Universe: universe,
			}
			for i, ps := range tc.pkgs {
				dir := ps.dir
				if dir == "" {
					dir = t.TempDir()
				}
				if ps.needRegisterFile {
					f, err := os.Create(filepath.Join(dir, "register.go"))
					if err != nil {
						t.Fatalf("creating register.go for pkg[%d]: %v", i, err)
					}
					if err := f.Close(); err != nil {
						t.Fatalf("closing register.go for pkg[%d]: %v", i, err)
					}
				}
				p := universe.Package(ps.path)
				p.Name = filepath.Base(ps.path)
				p.Dir = dir
				p.Comments = ps.comments
				if ps.typeMembers != nil {
					p.Types["MyType"] = &types.Type{
						Name:    types.Name{Package: ps.path, Name: "MyType"},
						Kind:    types.Struct,
						Members: ps.typeMembers,
					}
				}
				ctx.Inputs = append(ctx.Inputs, ps.path)
			}

			a := &args.Args{OutputFile: "zz_generated.register.go"}
			got := GetTargets(ctx, a)

			gotPaths := make([]string, 0, len(got))
			for _, g := range got {
				gotPaths = append(gotPaths, g.Path())
			}
			sort.Strings(gotPaths)
			wantPaths := append([]string(nil), tc.wantPaths...)
			sort.Strings(wantPaths)
			if !equalStrSlice(gotPaths, wantPaths) {
				t.Fatalf("target paths: got %v, want %v", gotPaths, wantPaths)
			}

			// Verify per-package group and typesToGenerate by exercising the
			// generator function the SimpleTarget would run.
			for _, tgt := range got {
				st, ok := tgt.(*generator.SimpleTarget)
				if !ok {
					t.Fatalf("%s: target is %T, want *generator.SimpleTarget", tgt.Path(), tgt)
				}
				gens := st.GeneratorsFunc(ctx)
				if len(gens) != 1 {
					t.Fatalf("%s: got %d generators, want 1", tgt.Path(), len(gens))
				}
				rg, ok := gens[0].(*registerExternalGenerator)
				if !ok {
					t.Fatalf("%s: generator is %T, want *registerExternalGenerator", tgt.Path(), gens[0])
				}
				if want, ok := tc.wantGroups[tgt.Path()]; ok {
					if string(rg.gv.Group) != want {
						t.Errorf("%s: group = %q, want %q", tgt.Path(), rg.gv.Group, want)
					}
				}
				if want, ok := tc.wantHasType[tgt.Path()]; ok {
					hasType := len(rg.typesToGenerate) > 0
					if hasType != want {
						t.Errorf("%s: hasType = %v, want %v (typesToGenerate=%v)", tgt.Path(), hasType, want, rg.typesToGenerate)
					}
				}
			}
		})
	}
}

func equalStrSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
