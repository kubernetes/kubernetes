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

package main

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

func TestRemoveLastDir(t *testing.T) {
	table := map[string]struct{ newPath, removedDir string }{
		"a/b/c": {"a/c", "b"},
	}
	for slashInput, expect := range table {
		input := filepath.FromSlash(slashInput)

		gotPath, gotRemoved := removeLastDir(input)
		if e, a := filepath.FromSlash(expect.newPath), gotPath; e != a {
			t.Errorf("%v: wanted %v, got %v", input, e, a)
		}
		if e, a := filepath.FromSlash(expect.removedDir), gotRemoved; e != a {
			t.Errorf("%v: wanted %v, got %v", input, e, a)
		}
	}
}

func TestTransitiveClosure(t *testing.T) {
	cases := []struct {
		name     string
		in       map[string][]string
		expected map[string][]string
	}{
		{
			name: "no transition",
			in: map[string][]string{
				"a": {"b"},
				"c": {"d"},
			},
			expected: map[string][]string{
				"a": {"b"},
				"c": {"d"},
			},
		},
		{
			name: "simple",
			in: map[string][]string{
				"a": {"b"},
				"b": {"c"},
				"c": {"d"},
			},
			expected: map[string][]string{
				"a": {"b", "c", "d"},
				"b": {"c", "d"},
				"c": {"d"},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := transitiveClosure(c.in)
			if !reflect.DeepEqual(c.expected, out) {
				t.Errorf("expected: %#v, got %#v", c.expected, out)
			}
		})
	}
}

func TestHasTestFiles(t *testing.T) {
	cases := []struct {
		input  []string
		expect bool
	}{{
		input:  nil,
		expect: false,
	}, {
		input:  []string{},
		expect: false,
	}, {
		input:  []string{"foo.go"},
		expect: false,
	}, {
		input:  []string{"foo.go", "bar.go"},
		expect: false,
	}, {
		input:  []string{"foo_test.go"},
		expect: true,
	}, {
		input:  []string{"foo.go", "foo_test.go"},
		expect: true,
	}, {
		input:  []string{"foo.go", "foo_test.go", "bar.go", "bar_test.go"},
		expect: true,
	}}

	for _, tc := range cases {
		ret := hasTestFiles(tc.input)
		if ret != tc.expect {
			t.Errorf("expected %v, got %v: %q", tc.expect, ret, tc.input)
		}
	}
}

func TestPackageDir(t *testing.T) {
	cases := []struct {
		input  *packages.Package
		expect string
	}{{
		input: &packages.Package{
			PkgPath:      "example.com/foo/bar/qux",
			GoFiles:      []string{"/src/prj/file.go"},
			IgnoredFiles: []string{"/otherdir/file.go"},
		},
		expect: filepath.Clean("/src/prj"),
	}, {
		input: &packages.Package{
			PkgPath:      "example.com/foo/bar/qux",
			IgnoredFiles: []string{"/src/prj/file.go"},
		},
		expect: filepath.Clean("/src/prj"),
	}, {
		input: &packages.Package{
			PkgPath: "example.com/foo/bar/qux",
		},
		expect: "",
	}}

	for i, tc := range cases {
		ret := packageDir(tc.input)
		if ret != tc.expect {
			t.Errorf("[%d] expected %v, got %v: %q", i, tc.expect, ret, tc.input)
		}
	}
}

func TestHasPathPrefix(t *testing.T) {
	cases := []struct {
		base   string
		pfx    string
		expect bool
	}{{
		base:   "",
		pfx:    "",
		expect: true,
	}, {
		base:   "/foo/bar",
		pfx:    "",
		expect: true,
	}, {
		base:   "",
		pfx:    "/foo",
		expect: false,
	}, {
		base:   "/foo",
		pfx:    "/foo",
		expect: true,
	}, {
		base:   "/foo/bar",
		pfx:    "/foo",
		expect: true,
	}, {
		base:   "/foobar/qux",
		pfx:    "/foo",
		expect: false,
	}, {
		base:   "/foo/bar/bat/qux/zrb",
		pfx:    "/foo/bar/bat",
		expect: true,
	}}

	for _, tc := range cases {
		ret := hasPathPrefix(tc.base, tc.pfx)
		if ret != tc.expect {
			t.Errorf("expected %v, got %v: (%q, %q)", tc.expect, ret, tc.base, tc.pfx)
		}
	}
}

func checkAllErrorStrings(t *testing.T, errs []error, expect []string) {
	t.Helper()
	if len(errs) != len(expect) {
		t.Fatalf("expected %d errors, got %d: %q", len(expect), len(errs), errs)
	}

	for _, str := range expect {
		found := false
		for _, err := range errs {
			if strings.HasPrefix(err.Error(), str) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("did not find error %q", str)
			t.Logf("\tseek: %s\n\t  in:", str)
			for _, err := range errs {
				t.Logf("\t      %s", err.Error())
			}
		}
	}
}

func TestSimpleForward(t *testing.T) {
	pkgs, err := loadPkgs("./testdata/simple-fwd/aaa")
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	if len(pkgs) != 1 {
		t.Fatalf("expected 1 pkg result, got %d", len(pkgs))
	}
	if pkgs[0].PkgPath != "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/aaa" {
		t.Fatalf("wrong PkgPath: %q", pkgs[0].PkgPath)
	}

	boss := newBoss(pkgs)
	errs := boss.Verify(pkgs[0])

	expect := []string{
		`"k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/forbidden" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/forbidden/f1" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/neither" did not match any rule`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/neither/n1" did not match any rule`,
	}

	checkAllErrorStrings(t, errs, expect)
}

func TestNestedForward(t *testing.T) {
	pkgs, err := loadPkgs("./testdata/nested-fwd/aaa")
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	if len(pkgs) != 1 {
		t.Fatalf("expected 1 pkg result, got %d", len(pkgs))
	}
	if pkgs[0].PkgPath != "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/aaa" {
		t.Fatalf("wrong PkgPath: %q", pkgs[0].PkgPath)
	}

	boss := newBoss(pkgs)
	errs := boss.Verify(pkgs[0])

	expect := []string{
		`"k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-both" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-root" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-sub" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/aaa" -> "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/neither/n1" did not match any rule`,
	}

	checkAllErrorStrings(t, errs, expect)
}

func TestInverse(t *testing.T) {
	pkgs, err := loadPkgs("./testdata/inverse/...")
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	if len(pkgs) != 10 {
		t.Fatalf("expected 10 pkg results, got %d", len(pkgs))
	}

	boss := newBoss(pkgs)

	var errs []error
	for _, pkg := range pkgs {
		errs = append(errs, boss.Verify(pkg)...)
	}

	expect := []string{
		`"k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden" <- "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/aaa" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden/f1" <- "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/aaa" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed/a2" <- "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed" did not match any rule`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden/f2" <- "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed" did not match any rule`,
	}

	checkAllErrorStrings(t, errs, expect)
}

func TestTransitive(t *testing.T) {
	pkgs, err := loadPkgs("./testdata/transitive/...")
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	if len(pkgs) != 10 {
		t.Fatalf("expected 10 pkg results, got %d", len(pkgs))
	}

	boss := newBoss(pkgs)

	var errs []error
	for _, pkg := range pkgs {
		errs = append(errs, boss.Verify(pkg)...)
	}

	expect := []string{
		`"k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden" <- "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/aaa" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden/f1" <- "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/aaa" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden/f2" <-- "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/aaa" is forbidden`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed/a2" <- "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed" did not match any rule`,
		`"k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden/f2" <- "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed" did not match any rule`,
	}

	checkAllErrorStrings(t, errs, expect)
}
