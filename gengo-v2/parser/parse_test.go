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

package parser

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"testing"

	"golang.org/x/tools/go/packages"
	"k8s.io/gengo/v2/types"
)

func TestNew(t *testing.T) {
	b := New()
	if b.goPkgs == nil {
		t.Errorf("expected .goPkgs to be initialized")
	}
	if b.userRequested == nil {
		t.Errorf("expected .userRequested to be initialized")
	}
	if b.fullyProcessed == nil {
		t.Errorf("expected .fullyProcessed to be initialized")
	}
	if b.fset == nil {
		t.Errorf("expected .fset to be initialized")
	}
	if b.endLineToCommentGroup == nil {
		t.Errorf("expected .endLineToCommentGroup to be initialized")
	}
}

func sorted(in ...string) []string {
	out := make([]string, len(in))
	copy(out, in)
	sort.Strings(out)
	return out
}

func sliceEq(a, b []string) bool {
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

func pkgPathsFromSlice(pkgs []*packages.Package) []string {
	paths := []string{}
	for _, pkg := range pkgs {
		paths = append(paths, pkg.PkgPath)
	}
	return sorted(paths...)
}

func pkgPathsFromMap(pkgs map[importPathString]*packages.Package) []string {
	paths := []string{}
	for _, pkg := range pkgs {
		paths = append(paths, pkg.PkgPath)
	}
	return sorted(paths...)
}

func pkgPathsFromUniverse(u types.Universe) []string {
	paths := []string{}
	for path := range u {
		paths = append(paths, path)
	}
	return sorted(paths...)
}

func pretty(in []string) string {
	size := 0
	oneline := true
	for _, s := range in {
		size += len(s)
		if size > 60 {
			oneline = false
			break
		}
	}
	var jb []byte
	var err error
	if oneline {
		jb, err = json.Marshal(in)
	} else {
		jb, err = json.MarshalIndent(in, "", "  ")
	}
	if err != nil {
		panic(fmt.Sprintf("JSON marshal failed: %v", err))
	}
	return string(jb)
}

func keys[T any](m map[string]T) []string {
	ret := []string{}
	for k := range m {
		ret = append(ret, k)
	}
	return sorted(ret...)
}

func TestAddBuildTags(t *testing.T) {
	testTags := []string{"foo", "bar", "qux"}

	b := New()
	if len(b.buildTags) != 0 {
		t.Errorf("expected no default build tags, got %v", b.buildTags)
	}
	b.AddBuildTags(testTags[0])
	if want, got := testTags[0:1], b.buildTags; !sliceEq(want, got) {
		t.Errorf("wrong build tags:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
	}
	b.AddBuildTags(testTags[1], testTags[2])
	if want, got := testTags, b.buildTags; !sliceEq(want, got) {
		t.Errorf("wrong build tags:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
	}
	b.AddBuildTags()
	if want, got := testTags, b.buildTags; !sliceEq(want, got) {
		t.Errorf("wrong build tags:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
	}
}

func TestFindPackages(t *testing.T) {
	b := New()

	// Proper packages with deps.
	if pkgs, err := b.FindPackages("./testdata/root1", "./testdata/root2", "./testdata/roots345/..."); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		expected := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1",
			"k8s.io/gengo/v2/parser/testdata/root2",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3/lib3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4/lib4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5/lib5",
		)
		if want, got := expected, sorted(pkgs...); !sliceEq(want, got) {
			t.Errorf("wrong pkgs:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
		if len(b.goPkgs) != 0 {
			t.Errorf("expected no added .goPkgs, got %v", pretty(pkgPathsFromMap(b.goPkgs)))
		}
	}

	// Non-existent packages should be an error.
	if pkgs, err := b.FindPackages("./testdata/does-not-exist"); err == nil {
		t.Errorf("unexpected success: %v", pkgs)
	}

	// Packages without .go files should be an error.
	if pkgs, err := b.FindPackages("./testdata/has-no-gofiles"); err == nil {
		t.Errorf("unexpected success: %v", pkgs)
	}

	// Invalid go files are not an error.
	if pkgs, err := b.FindPackages("./testdata/does-not-parse"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		expected := []string{
			"k8s.io/gengo/v2/parser/testdata/does-not-parse",
		}
		if want, got := expected, pkgs; !sliceEq(want, got) {
			t.Errorf("wrong pkgs:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
	}
}

func TestAlreadyLoaded(t *testing.T) {
	newPkg := func(path string) *packages.Package {
		return &packages.Package{
			ID:      path,
			PkgPath: path,
			Name:    filepath.Base(path),
		}
	}

	b := New()

	// Test loading something we don't have.
	if existing, netNew, err := b.alreadyLoaded("./testdata/root1"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		if len(existing) > 0 {
			t.Errorf("unexpected existing pkg(s): %v", existing)
		}
		if len(netNew) != 1 {
			t.Errorf("expected 1 net-new, got: %v", netNew)
		} else if want, got := "k8s.io/gengo/v2/parser/testdata/root1", netNew[0]; want != got {
			t.Errorf("wrong net-new, want %v, got %v", want, got)
		}
	}

	// Test loading something already present.
	b.goPkgs["k8s.io/gengo/v2/parser/testdata/root1"] = newPkg("k8s.io/gengo/v2/parser/testdata/root1")
	if existing, netNew, err := b.alreadyLoaded("./testdata/root1"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		if len(existing) != 1 {
			t.Errorf("expected 1 existing, got: %v", existing)
		} else if want, got := "k8s.io/gengo/v2/parser/testdata/root1", existing[0].PkgPath; want != got {
			t.Errorf("wrong existing, want %v, got %v", want, got)
		}
		if len(netNew) > 0 {
			t.Errorf("unexpected net-new pkg(s): %v", netNew)
		}
	}

	// Test loading something partly present.
	if existing, netNew, err := b.alreadyLoaded("./testdata/root1/..."); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		if len(existing) != 1 {
			t.Errorf("expected 1 existing, got: %v", existing)
		} else if want, got := "k8s.io/gengo/v2/parser/testdata/root1", existing[0].PkgPath; want != got {
			t.Errorf("wrong existing, want %v, got %v", want, got)
		}
		if len(netNew) != 1 {
			t.Errorf("expected 1 net-new, got: %v", netNew)
		} else if want, got := "k8s.io/gengo/v2/parser/testdata/root1/lib1", netNew[0]; want != got {
			t.Errorf("wrong net-new, want %v, got %v", want, got)
		}
	}
}

func TestLoadPackagesInternal(t *testing.T) {
	b := New()

	// Proper packages with deps.
	if pkgs, err := b.loadPackages("./testdata/root1", "./testdata/root2", "./testdata/roots345/..."); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		expectedDirect := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1",
			"k8s.io/gengo/v2/parser/testdata/root2",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3/lib3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4/lib4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5/lib5",
		)
		expectedIndirect := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1/lib1",
			"k8s.io/gengo/v2/parser/testdata/root2/lib2",
			"k8s.io/gengo/v2/parser/testdata/rootpeer",
			"k8s.io/gengo/v2/parser/testdata/rootpeer/sub1",
			"k8s.io/gengo/v2/parser/testdata/rootpeer/sub2",
		)
		expectedAll := sorted(append(expectedDirect, expectedIndirect...)...)

		if want, got := expectedDirect, pkgPathsFromSlice(pkgs); !sliceEq(want, got) {
			t.Errorf("wrong pkgs returned:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
		if want, got := expectedAll, pkgPathsFromMap(b.goPkgs); !sliceEq(want, got) {
			t.Errorf("wrong pkgs in .goPkgs:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}

		for _, path := range expectedDirect {
			if !b.userRequested[importPathString(path)] {
				t.Errorf("expected .userRequested[%q] to be set", path)
			}
			if b.fullyProcessed[importPathString(path)] {
				t.Errorf("expected .fullyProcessed[%q] to be unset", path)
			}
		}
		for _, path := range expectedIndirect {
			if b.userRequested[importPathString(path)] {
				t.Errorf("expected .userRequested[%q] to be unset", path)
			}
			if b.fullyProcessed[importPathString(path)] {
				t.Errorf("expected .fullyProcessed[%q] to be unset", path)
			}
		}

		// There is a comment is at this fixed location.
		pos := fileLine{b.goPkgs["k8s.io/gengo/v2/parser/testdata/root1"].GoFiles[0], 9}
		if b.endLineToCommentGroup[pos] == nil {
			t.Errorf("expected a comment-group ending at %v", pos)
			t.Errorf("%v", b.endLineToCommentGroup)
		}
	}

	// Non-existent packages should be an error.
	if pkgs, err := b.loadPackages("./testdata/does-not-exist"); err == nil {
		t.Errorf("unexpected success: %v", pkgs)
	}

	// Packages without .go files should be an error.
	if pkgs, err := b.loadPackages("./testdata/has-no-gofiles"); err == nil {
		t.Errorf("unexpected success: %v", pkgs)
	}

	// Invalid go files are an error.
	if pkgs, err := b.loadPackages("./testdata/does-not-parse"); err == nil {
		t.Errorf("unexpected success: %v", pkgs)
	}

	// Packages which parse but do not compile are NOT an error.
	if pkgs, err := b.loadPackages("./testdata/does-not-compile"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		expected := []string{
			"k8s.io/gengo/v2/parser/testdata/does-not-compile",
		}
		if want, got := expected, pkgPathsFromSlice(pkgs); !sliceEq(want, got) {
			t.Errorf("wrong pkgs:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
		if b.goPkgs[importPathString(expected[0])] == nil {
			t.Errorf("package not found in .goPkgs: %v", expected[0])
		}
	}
}

func TestLoadPackagesTo(t *testing.T) {
	b := New()
	u := types.Universe{}

	// Proper packages with deps.
	if pkgs, err := b.LoadPackagesTo(&u, "./testdata/root1", "./testdata/root2", "./testdata/roots345/..."); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		expectedDirect := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1",
			"k8s.io/gengo/v2/parser/testdata/root2",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3/lib3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4/lib4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5/lib5",
		)
		expectedIndirect := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1/lib1",
			"k8s.io/gengo/v2/parser/testdata/root2/lib2",
			"k8s.io/gengo/v2/parser/testdata/rootpeer",
			"k8s.io/gengo/v2/parser/testdata/rootpeer/sub1",
			"k8s.io/gengo/v2/parser/testdata/rootpeer/sub2",
		)
		expectedAll := expectedDirect
		expectedAll = append(expectedAll, expectedIndirect...)
		expectedAll = sorted(append(expectedAll, "")...) // This is the "builtin" pkg

		if want, got := len(expectedDirect), len(pkgs); want != got {
			t.Errorf("wrong number of pkgs returned: want: %d got: %d", want, got)
		}
		if want, got := expectedAll, pkgPathsFromUniverse(u); !sliceEq(want, got) {
			t.Errorf("wrong pkgs in universe:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
	}
}

func TestForEachPackageRecursive(t *testing.T) {
	newPkg := func(path string, imports ...*packages.Package) *packages.Package {
		pkg := &packages.Package{
			ID:      path,
			PkgPath: path,
			Name:    filepath.Base(path),
			Imports: map[string]*packages.Package{},
		}
		for _, imp := range imports {
			pkg.Imports[imp.PkgPath] = imp
		}
		return pkg
	}

	pkgs := []*packages.Package{
		newPkg("example.com/root1",
			newPkg("example.com/dep1", newPkg("example.com/dep11"), newPkg("example.com/dep12")),
			newPkg("example.com/dep2", newPkg("example.com/dep21")),
		),
		newPkg("example.com/root2",
			newPkg("example.com/dep3"),
		),
		newPkg("example.com/root3"),
	}

	// Test success.
	expect := sorted(
		"example.com/root1",
		"example.com/dep1",
		"example.com/dep11",
		"example.com/dep12",
		"example.com/dep2",
		"example.com/dep21",
		"example.com/root2",
		"example.com/dep3",
		"example.com/root3",
	)
	visited := []string{}
	visit := func(pkg *packages.Package) error {
		visited = append(visited, pkg.PkgPath)
		return nil
	}
	if err := forEachPackageRecursive(pkgs, visit); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if want, got := expect, sorted(visited...); !sliceEq(want, got) {
		t.Errorf("want: %v\ngot:  %v", pretty(want), pretty(got))
	}

	// Test errors.
	fail := func(pkg *packages.Package) error {
		return fmt.Errorf("%s", pkg.PkgPath)
	}
	if err := forEachPackageRecursive(pkgs, fail); err == nil {
		t.Errorf("unexpected success")
	} else if wrapped, ok := err.(interface{ Unwrap() []error }); !ok {
		t.Errorf("expected unwrappable error, got %v", err)
	} else {
		visited := []string{}
		for _, err := range wrapped.Unwrap() {
			visited = append(visited, err.Error())
		}
		if want, got := expect, sorted(visited...); !sliceEq(want, got) {
			t.Errorf("wrong errors:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
	}
}

func TestUserRequestedPackages(t *testing.T) {
	b := New()

	// Proper packages with deps.
	if err := b.LoadPackages("./testdata/root1", "./testdata/root2", "./testdata/roots345/..."); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		want := sorted(
			"k8s.io/gengo/v2/parser/testdata/root1",
			"k8s.io/gengo/v2/parser/testdata/root2",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root3/lib3",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root4/lib4",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5",
			"k8s.io/gengo/v2/parser/testdata/roots345/root5/lib5",
		)
		got := b.UserRequestedPackages() // should be sorted!

		if !sliceEq(want, got) {
			t.Errorf("wrong pkgs returned:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}
	}
}

func TestAddOnePkgToUniverse(t *testing.T) {
	b := New()

	// Proper packages with deps.
	if pkgs, err := b.loadPackages("./testdata/root2"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else {
		direct := "k8s.io/gengo/v2/parser/testdata/root2"
		indirect := "k8s.io/gengo/v2/parser/testdata/root2/lib2"

		if want, got := []string{direct}, pkgPathsFromSlice(pkgs); !sliceEq(want, got) {
			t.Errorf("wrong pkgs returned:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
		}

		u := types.Universe{}
		if err := b.addPkgToUniverse(pkgs[0], &u); err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		// TODO: This is not an exhaustive test.  There are lots of types
		// and combinations of things that are not covered.

		// verify the depth of processing
		if !b.fullyProcessed[importPathString(direct)] {
			t.Errorf("expected .fullyProcessed[%q] to be set", direct)
		}
		if b.fullyProcessed[importPathString(indirect)] {
			t.Errorf("expected .fullyProcessed[%q] to be unset", indirect)
		}

		// verify their existence
		pd := b.goPkgs[importPathString(direct)]
		if pd == nil {
			t.Fatalf("expected non-nil from .goPkgs")
		}
		pi := b.goPkgs[importPathString(indirect)]
		if pi == nil {
			t.Fatalf("expected non-nil from .goPkgs")
		}
		ud := u[direct]
		if ud == nil {
			t.Fatalf("expected non-nil from universe")
		}
		ui := u[indirect]
		if ui == nil {
			t.Fatalf("expected non-nil from universe")
		}

		// verify metadata
		if want, got := pd.PkgPath, ud.Path; want != got {
			t.Errorf("expected .Path %q, got %q", want, got)
		}
		if want, got := filepath.Dir(pd.GoFiles[0]), ud.SourcePath; want != got {
			t.Errorf("expected .SourcePath %q, got %q", want, got)
		}
		if want, got := pi.PkgPath, ui.Path; want != got {
			t.Errorf("expected .Path %q, got %q", want, got)
		}
		if want, got := filepath.Dir(pi.GoFiles[0]), ui.SourcePath; want != got {
			t.Errorf("expected .SourcePath %q, got %q", want, got)
		}

		// verify doc.go handling
		if len(ud.DocComments) != 2 { // Fixed value from the testdata
			t.Errorf("expected 2 doc-comment lines, got: %v", pretty(ud.DocComments))
		}
		if len(ud.Comments) != 6 { // Fixed value from the testdata
			t.Errorf("expected 3 comments, 2 lines each, got: %v", pretty(ud.Comments))
		}

		// verify types
		if len(ui.Types) != 0 {
			t.Errorf("expected zero types in indirect package, got %d", len(ui.Types))
		}
		if len(ud.Types) == 0 {
			t.Errorf("expected non-zero types in direct package")
		} else {
			type testcase struct {
				kind       types.Kind
				elem       string // just the type name
				key        string // just the type name
				underlying *testcase
			}
			cases := map[string]testcase{
				"Int": {
					kind: types.Alias,
					underlying: &testcase{
						kind: types.Builtin,
					},
				},
				"String": {
					kind: types.Alias,
					underlying: &testcase{
						kind: types.Builtin,
					},
				},
				"EmptyStruct": {
					kind: types.Struct,
				},
				"Struct": {
					kind: types.Struct,
				},
				"M": {
					kind: types.Alias,
					underlying: &testcase{
						kind: types.Map,
						elem: "*k8s.io/gengo/v2/parser/testdata/root2.Struct",
						key:  types.String.String(),
					},
				},
			}

			want := keys(cases)
			got := keys(ud.Types)

			if !sliceEq(want, got) {
				t.Errorf("wrong types found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			} else {
				for name, obj := range ud.Types {
					n := types.Name{Package: ud.Path, Name: name}
					if obj.Name != n {
						t.Errorf("wrong name for type %s: %v", name, obj.Name)
					}
					comment1 := fmt.Sprintf("%s comment", name)
					if want, got := []string{comment1}, obj.CommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for type %s:\nwant: %v\ngot:  %v", name, want, got)
					}
					comment2 := fmt.Sprintf("SecondClosest %s comment", name)
					if want, got := []string{comment2}, obj.SecondClosestCommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for type %s:\nwant: %v\ngot:  %v", name, want, got)
					}
				}

				// Declare, then define because it is recursive.
				var vrfy func(name string, tc *testcase, obj *types.Type)
				vrfy = func(name string, tc *testcase, obj *types.Type) {
					if want, got := tc.kind, obj.Kind; want != got {
						t.Errorf("wrong .Kind for type %s:\nwant: %v, got: %v", name, want, got)
					} else if obj.Kind == types.Alias {
						vrfy(name+"^", tc.underlying, obj.Underlying)
					}
					if want, got := tc.elem, obj.Elem.String(); want != got {
						t.Errorf("wrong .Elem for type %s:\nwant: %v, got: %v", name, want, got)
					}
					if want, got := tc.key, obj.Key.String(); want != got {
						t.Errorf("wrong .Key for type %s:\nwant: %v, got: %v", name, want, got)
					}
					// TODO: Members, Methods, Len
				}
				for name, tc := range cases {
					obj := ud.Types[name]
					vrfy(name, &tc, obj)
				}
			}
		}

		// verify functions
		if len(ui.Functions) != 0 {
			t.Errorf("expected zero functions in indirect package, got %d", len(ui.Functions))
		}
		if len(ud.Functions) == 0 {
			t.Errorf("expected non-zero functions in direct package")
		} else {
			type testcase struct {
				kind types.Kind
			}
			cases := map[string]testcase{
				"PublicFunc": {
					kind: types.DeclarationOf,
				},
				"privateFunc": {
					kind: types.DeclarationOf,
				},
			}

			want := keys(cases)
			got := keys(ud.Functions)

			if !sliceEq(want, got) {
				t.Errorf("wrong functions found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			} else {
				for name, obj := range ud.Functions {
					n := types.Name{Package: ud.Path, Name: name}
					if obj.Name != n {
						t.Errorf("wrong name for function %s: %v", name, obj.Name)
					}
					comment1 := fmt.Sprintf("%s comment", name)
					if want, got := []string{comment1}, obj.CommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for function %s:\nwant: %v\ngot:  %v", name, want, got)
					}
					comment2 := fmt.Sprintf("SecondClosest %s comment", name)
					if want, got := []string{comment2}, obj.SecondClosestCommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for function %s:\nwant: %v\ngot:  %v", name, want, got)
					}
				}

				for name, tc := range cases {
					obj := ud.Functions[name]
					if want, got := tc.kind, obj.Kind; want != got {
						t.Errorf("wrong .Kind for function %s:\nwant: %v, got: %v", name, want, got)
					}
					// TODO: Signature
				}
			}
		}

		// verify variables
		if len(ui.Variables) != 0 {
			t.Errorf("expected zero variables in indirect package, got %d", len(ui.Variables))
		}
		if len(ud.Variables) == 0 {
			t.Errorf("expected non-zero variables in direct package")
		} else {
			type testcase struct {
				kind types.Kind
			}
			cases := map[string]testcase{
				"X": {
					kind: types.DeclarationOf,
				},
			}

			want := keys(cases)
			got := keys(ud.Variables)

			if !sliceEq(want, got) {
				t.Errorf("wrong variables found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			} else {
				for name, obj := range ud.Variables {
					n := types.Name{Package: ud.Path, Name: name}
					if obj.Name != n {
						t.Errorf("wrong name for variable %s: %v", name, obj.Name)
					}
					comment1 := fmt.Sprintf("%s comment", name)
					if want, got := []string{comment1}, obj.CommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for variable %s:\nwant: %v\ngot:  %v", name, want, got)
					}
					comment2 := fmt.Sprintf("SecondClosest %s comment", name)
					if want, got := []string{comment2}, obj.SecondClosestCommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for variable %s:\nwant: %v\ngot:  %v", name, want, got)
					}
				}

				for name, tc := range cases {
					obj := ud.Variables[name]
					if want, got := tc.kind, obj.Kind; want != got {
						t.Errorf("wrong .Kind for variable %s:\nwant: %v, got: %v", name, want, got)
					}
					// TODO: Underlying
				}
			}
		}

		// verify constants
		if len(ui.Constants) != 0 {
			t.Errorf("expected zero constants in indirect package, got %d", len(ui.Constants))
		}
		if len(ud.Constants) == 0 {
			t.Errorf("expected non-zero constants in direct package")
		} else {
			type testcase struct {
				kind types.Kind
			}
			cases := map[string]testcase{
				"Y": {
					kind: types.DeclarationOf,
				},
			}

			want := keys(cases)
			got := keys(ud.Constants)

			if !sliceEq(want, got) {
				t.Errorf("wrong constants found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			} else {
				for name, obj := range ud.Constants {
					n := types.Name{Package: ud.Path, Name: name}
					if obj.Name != n {
						t.Errorf("wrong name for constant %s: %v", name, obj.Name)
					}
					comment1 := fmt.Sprintf("%s comment", name)
					if want, got := []string{comment1}, obj.CommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for constant %s:\nwant: %v\ngot:  %v", name, want, got)
					}
					comment2 := fmt.Sprintf("SecondClosest %s comment", name)
					if want, got := []string{comment2}, obj.SecondClosestCommentLines; !sliceEq(want, got) {
						t.Errorf("wrong comments for constant %s:\nwant: %v\ngot:  %v", name, want, got)
					}
				}

				for name, tc := range cases {
					obj := ud.Constants[name]
					if want, got := tc.kind, obj.Kind; want != got {
						t.Errorf("wrong .Kind for constant %s:\nwant: %v, got: %v", name, want, got)
					}
					// TODO: Underlying
				}
			}
		}

		// verify imports
		if len(ui.Imports) != 0 {
			t.Errorf("expected zero imports in indirect package, got %d", len(ui.Imports))
		}
		if len(ud.Imports) == 0 {
			t.Errorf("expected non-zero imports in direct package")
		} else {
			want := sorted(
				"k8s.io/gengo/v2/parser/testdata/root2/lib2",
				"k8s.io/gengo/v2/parser/testdata/rootpeer",
			)
			got := keys(ud.Imports)

			if !sliceEq(want, got) {
				t.Errorf("wrong imports found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			}
		}

		// verify types in the "" pkg
		if builtins := u[""]; len(builtins.Types) == 0 {
			t.Errorf("expected non-zero types in the \"\" package")
		} else {
			// NOTE: this captures how the code behaved at the time of this
			// test's creation, but does not speak to why it does this or
			// whether it is correct or optimal.
			want := sorted(
				"int",
				"*int",
				"*string",
				"string",
				"untyped string",
				"func()",
				"*k8s.io/gengo/v2/parser/testdata/root2.Int",
				"*k8s.io/gengo/v2/parser/testdata/root2.String",
				"*k8s.io/gengo/v2/parser/testdata/root2.EmptyStruct",
				"*k8s.io/gengo/v2/parser/testdata/root2.Struct",
				"func (k8s.io/gengo/v2/parser/testdata/root2.Struct).PublicMethod()",
				"func (k8s.io/gengo/v2/parser/testdata/root2.Struct).privateMethod()",
				"map[string]*k8s.io/gengo/v2/parser/testdata/root2.Struct",
			)
			got := keys(builtins.Types)

			if !sliceEq(want, got) {
				t.Errorf("wrong types found:\nwant: %v\ngot:  %v", pretty(want), pretty(got))
			}
		}
	}
}
