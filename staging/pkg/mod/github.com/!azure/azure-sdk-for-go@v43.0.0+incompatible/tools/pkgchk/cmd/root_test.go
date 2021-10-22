// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"path/filepath"
	"strings"
	"testing"
)

const (
	testRoot = "../../testpkgs"
)

var (
	expected = map[string]string{
		"/scenrioa/foo":                    "foo",
		"/scenriob/foo":                    "foo",
		"/scenriob/foo/v2":                 "foo",
		"/scenrioc/mgmt/2019-10-11/foo":    "foo",
		"/scenriod/mgmt/2019-10-11/foo":    "foo",
		"/scenriod/mgmt/2019-10-11/foo/v2": "foo",
		"/scenrioe/mgmt/2019-10-11/foo":    "foo",
		"/scenrioe/mgmt/2019-10-11/foo/v2": "foo",
		"/scenrioe/mgmt/2019-10-11/foo/v3": "foo",
	}
)

func Test_getPkgs(t *testing.T) {
	rootDir, err := filepath.Abs(testRoot)
	if err != nil {
		t.Fatalf("failed to get absolute path: %+v", err)
	}
	pkgs, err := getPkgs(rootDir)
	if err != nil {
		t.Fatalf("failed to get packages: %+v", err)
	}
	if len(pkgs) != len(expected) {
		t.Fatalf("expected %d packages, but got %d", len(expected), len(pkgs))
	}
	for _, pkg := range pkgs {
		if pkgName, ok := expected[pkg.d]; !ok {
			t.Fatalf("got pkg path '%s', but not found in expected", pkg.d)
		} else if !strings.EqualFold(pkgName, pkg.p.Name) {
			t.Fatalf("expected package of '%s' in path '%s', but got '%s'", pkgName, pkg.d, pkg.p.Name)
		}
	}
}

func Test_verifyDirectoryStructure(t *testing.T) {
	rootDir, err := filepath.Abs(testRoot)
	if err != nil {
		t.Fatalf("failed to get absolute path: %+v", err)
	}
	pkgs, err := getPkgs(rootDir)
	if err != nil {
		t.Fatalf("failed to get packages: %+v", err)
	}
	for _, pkg := range pkgs {
		if err := verifyDirectoryStructure(pkg); err != nil {
			t.Fatalf("failed to verify directory structure: %+v", err)
		}
	}
}

func Test_verifyPkgMatchesDir(t *testing.T) {
	rootDir, err := filepath.Abs(testRoot)
	if err != nil {
		t.Fatalf("failed to get absolute path: %+v", err)
	}
	pkgs, err := getPkgs(rootDir)
	if err != nil {
		t.Fatalf("failed to get packages: %+v", err)
	}
	for _, pkg := range pkgs {
		if err := verifyPkgMatchesDir(pkg); err != nil {
			t.Fatalf("failed to verify package directory name: %+v", err)
		}
	}
}
