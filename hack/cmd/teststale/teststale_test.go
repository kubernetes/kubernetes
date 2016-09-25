/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"testing"
	"time"
)

const (
	// seed for rand.Source to generate data for files
	seed int64 = 42

	// 1K binary file
	binLen = 1024

	// Directory of the test package relative to $GOPATH
	testImportDir = "example.com/proj/pkg"
)

var (
	pastHour = time.Now().Add(-1 * time.Hour)

	// The test package we are testing against
	testPkg = path.Join(testImportDir, "test")
)

// fakegolist implements the `golist` interface providing fake package information for testing.
type fakegolist struct {
	dir       string
	importMap map[string]pkg
	testFiles []string
	binfile   string
}

func newFakegolist() (*fakegolist, error) {
	dir, err := ioutil.TempDir("", "teststale")
	if err != nil {
		// test can't proceed without a temp directory.
		return nil, fmt.Errorf("failed to create a temp directory for testing: %v", err)
	}

	// Set the temp directory as the $GOPATH
	if err := os.Setenv("GOPATH", dir); err != nil {
		// can't proceed without pointing the $GOPATH to the temp directory.
		return nil, fmt.Errorf("failed to set \"$GOPATH\" pointing to %q: %v", dir, err)
	}

	// Setup $GOPATH directory layout.
	// Yeah! I am bored of repeatedly writing "if err != nil {}"!
	if os.MkdirAll(filepath.Join(dir, "bin"), 0750) != nil ||
		os.MkdirAll(filepath.Join(dir, "pkg", "linux_amd64"), 0750) != nil ||
		os.MkdirAll(filepath.Join(dir, "src"), 0750) != nil {
		return nil, fmt.Errorf("failed to setup the $GOPATH directory structure")
	}

	// Create a temp file to represent the test binary.
	binfile, err := ioutil.TempFile("", "testbin")
	if err != nil {
		return nil, fmt.Errorf("failed to create the temp file to represent the test binary: %v", err)
	}

	// Could have used crypto/rand instead, but it doesn't matter.
	rr := rand.New(rand.NewSource(42))
	bin := make([]byte, binLen)
	if _, err = rr.Read(bin); err != nil {
		return nil, fmt.Errorf("couldn't read from the random source: %v", err)
	}
	if _, err := binfile.Write(bin); err != nil {
		return nil, fmt.Errorf("couldn't write to the binary file %q: %v", binfile.Name(), err)
	}
	if err := binfile.Close(); err != nil {
		// It is arguable whether this should be fatal.
		return nil, fmt.Errorf("failed to close the binary file %q: %v", binfile.Name(), err)
	}

	if err := os.Chtimes(binfile.Name(), time.Now(), time.Now()); err != nil {
		return nil, fmt.Errorf("failed to modify the mtime of the binary file %q: %v", binfile.Name(), err)
	}

	// Create test source files directory.
	testdir := filepath.Join(dir, "src", testPkg)
	if err := os.MkdirAll(testdir, 0750); err != nil {
		return nil, fmt.Errorf("failed to create test source directory %q: %v", testdir, err)
	}

	fgl := &fakegolist{
		dir: dir,
		importMap: map[string]pkg{
			"example.com/proj/pkg/test": {
				Dir:        path.Join(dir, "src", testPkg),
				ImportPath: testPkg,
				Target:     path.Join(dir, "pkg", "linux_amd64", testImportDir, "test.a"),
				Stale:      false,
				TestGoFiles: []string{
					"foo_test.go",
					"bar_test.go",
				},
				TestImports: []string{
					"example.com/proj/pkg/p1",
					"example.com/proj/pkg/p1/c11",
					"example.com/proj/pkg/p2",
					"example.com/proj/cmd/p3/c12/c23",
					"strings",
					"testing",
				},
				XTestGoFiles: []string{
					"xfoo_test.go",
					"xbar_test.go",
					"xbaz_test.go",
				},
				XTestImports: []string{
					"example.com/proj/pkg/test",
					"example.com/proj/pkg/p1",
					"example.com/proj/cmd/p3/c12/c23",
					"os",
					"testing",
				},
			},
			"example.com/proj/pkg/p1":         {Stale: false},
			"example.com/proj/pkg/p1/c11":     {Stale: false},
			"example.com/proj/pkg/p2":         {Stale: false},
			"example.com/proj/cmd/p3/c12/c23": {Stale: false},
			"strings":                         {Stale: false},
			"testing":                         {Stale: false},
			"os":                              {Stale: false},
		},
		testFiles: []string{
			"foo_test.go",
			"bar_test.go",
			"xfoo_test.go",
			"xbar_test.go",
			"xbaz_test.go",
		},
		binfile: binfile.Name(),
	}

	// Create test source files.
	for _, fn := range fgl.testFiles {
		fp := filepath.Join(testdir, fn)
		if _, err := os.Create(fp); err != nil {
			return nil, fmt.Errorf("failed to create the test file %q: %v", fp, err)
		}
		if err := os.Chtimes(fp, time.Now(), pastHour); err != nil {
			return nil, fmt.Errorf("failed to modify the mtime of the test file %q: %v", binfile.Name(), err)
		}
	}

	return fgl, nil
}

func (fgl *fakegolist) pkgInfo(pkgPaths []string) ([]pkg, error) {
	var pkgs []pkg
	for _, path := range pkgPaths {
		p, ok := fgl.importMap[path]
		if !ok {
			return nil, fmt.Errorf("package %q not found", path)
		}
		pkgs = append(pkgs, p)
	}
	return pkgs, nil
}

func (fgl *fakegolist) chMtime(filename string, mtime time.Time) error {
	for _, fn := range fgl.testFiles {
		if fn == filename {
			fp := filepath.Join(fgl.dir, "src", testPkg, fn)
			if err := os.Chtimes(fp, time.Now(), mtime); err != nil {
				return fmt.Errorf("failed to modify the mtime of %q: %v", filename, err)
			}
			return nil
		}
	}
	return fmt.Errorf("file %q not found", filename)
}

func (fgl *fakegolist) chStale(pkg string, stale bool) error {
	if p, ok := fgl.importMap[pkg]; ok {
		p.Stale = stale
		fgl.importMap[pkg] = p
		return nil
	}
	return fmt.Errorf("package %q not found", pkg)
}

func (fgl *fakegolist) cleanup() {
	os.RemoveAll(fgl.dir)
	os.Remove(fgl.binfile)
}

func TestIsTestStale(t *testing.T) {
	cases := []struct {
		fileMtime    map[string]time.Time
		pkgStaleness map[string]bool
		result       bool
	}{
		// Basic test: binary is fresh, all modifications were before the binary was built.
		{
			result: false,
		},
		// A local test file is new, hence binary must be stale.
		{
			fileMtime: map[string]time.Time{
				"foo_test.go": time.Now().Add(1 * time.Hour),
			},
			result: true,
		},
		// Test package is new, so binary must be stale.
		{
			pkgStaleness: map[string]bool{
				"example.com/proj/pkg/test": true,
			},
			result: true,
		},
		// Test package dependencies are new, so binary must be stale.
		{
			pkgStaleness: map[string]bool{
				"example.com/proj/cmd/p3/c12/c23": true,
				"strings":                         true,
			},
			result: true,
		},
		// External test files are new, hence binary must be stale.
		{
			fileMtime: map[string]time.Time{
				"xfoo_test.go": time.Now().Add(1 * time.Hour),
				"xbar_test.go": time.Now().Add(2 * time.Hour),
			},
			result: true,
		},
		// External test dependency is new, so binary must be stale.
		{
			pkgStaleness: map[string]bool{
				"os": true,
			},
			result: true,
		},
		// Multiple source files and dependencies are new, so binary must be stale.
		{
			fileMtime: map[string]time.Time{
				"foo_test.go":  time.Now().Add(1 * time.Hour),
				"xfoo_test.go": time.Now().Add(2 * time.Hour),
				"xbar_test.go": time.Now().Add(3 * time.Hour),
			},
			pkgStaleness: map[string]bool{
				"example.com/proj/pkg/p1":         true,
				"example.com/proj/pkg/p1/c11":     true,
				"example.com/proj/pkg/p2":         true,
				"example.com/proj/cmd/p3/c12/c23": true,
				"strings":                         true,
				"os":                              true,
			},
			result: true,
		},
		// Everything is new, so binary must be stale.
		{
			fileMtime: map[string]time.Time{
				"foo_test.go":  time.Now().Add(3 * time.Hour),
				"bar_test.go":  time.Now().Add(1 * time.Hour),
				"xfoo_test.go": time.Now().Add(2 * time.Hour),
				"xbar_test.go": time.Now().Add(1 * time.Hour),
				"xbaz_test.go": time.Now().Add(2 * time.Hour),
			},
			pkgStaleness: map[string]bool{
				"example.com/proj/pkg/p1":         true,
				"example.com/proj/pkg/p1/c11":     true,
				"example.com/proj/pkg/p2":         true,
				"example.com/proj/cmd/p3/c12/c23": true,
				"example.com/proj/pkg/test":       true,
				"strings":                         true,
				"testing":                         true,
				"os":                              true,
			},
			result: true,
		},
	}

	for _, tc := range cases {
		fgl, err := newFakegolist()
		if err != nil {
			t.Fatalf("failed to setup the test: %v", err)
		}
		defer fgl.cleanup()

		for fn, mtime := range tc.fileMtime {
			if err := fgl.chMtime(fn, mtime); err != nil {
				t.Fatalf("failed to change the mtime of %q: %v", fn, err)
			}
		}

		for pkg, stale := range tc.pkgStaleness {
			if err := fgl.chStale(pkg, stale); err != nil {
				t.Fatalf("failed to change the staleness of %q: %v", pkg, err)
			}
		}

		if tc.result != isTestStale(fgl, fgl.binfile, testPkg) {
			if tc.result {
				t.Errorf("Expected test package %q to be stale", testPkg)
			} else {
				t.Errorf("Expected test package %q to be not stale", testPkg)
			}
		}
	}
}
