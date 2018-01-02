// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// gen runs go generate on Unicode- and CLDR-related package in the text
// repositories, taking into account dependencies and versions.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"go/format"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/internal/gen"
)

var (
	verbose     = flag.Bool("v", false, "verbose output")
	force       = flag.Bool("force", false, "ignore failing dependencies")
	doCore      = flag.Bool("core", false, "force an update to core")
	excludeList = flag.String("exclude", "",
		"comma-separated list of packages to exclude")

	// The user can specify a selection of packages to build on the command line.
	args []string
)

func exclude(pkg string) bool {
	if len(args) > 0 {
		return !contains(args, pkg)
	}
	return contains(strings.Split(*excludeList, ","), pkg)
}

// TODO:
// - Better version handling.
// - Generate tables for the core unicode package?
// - Add generation for encodings. This requires some retooling here and there.
// - Running repo-wide "long" tests.

var vprintf = fmt.Printf

func main() {
	gen.Init()
	args = flag.Args()
	if !*verbose {
		// Set vprintf to a no-op.
		vprintf = func(string, ...interface{}) (int, error) { return 0, nil }
	}

	// TODO: create temporary cache directory to load files and create and set
	// a "cache" option if the user did not specify the UNICODE_DIR environment
	// variable. This will prevent duplicate downloads and also will enable long
	// tests, which really need to be run after each generated package.

	updateCore := *doCore
	if gen.UnicodeVersion() != unicode.Version {
		fmt.Printf("Requested Unicode version %s; core unicode version is %s.\n",
			gen.UnicodeVersion(),
			unicode.Version)
		// TODO: use collate to compare. Simple comparison will work, though,
		// until Unicode reaches version 10. To avoid circular dependencies, we
		// could use the NumericWeighter without using package collate using a
		// trivial Weighter implementation.
		if gen.UnicodeVersion() < unicode.Version && !*force {
			os.Exit(2)
		}
		updateCore = true
	}

	var unicode = &dependency{}
	if updateCore {
		fmt.Printf("Updating core to version %s...\n", gen.UnicodeVersion())
		unicode = generate("unicode")

		// Test some users of the unicode packages, especially the ones that
		// keep a mirrored table. These may need to be corrected by hand.
		generate("regexp", unicode)
		generate("strconv", unicode) // mimics Unicode table
		generate("strings", unicode)
		generate("testing", unicode) // mimics Unicode table
	}

	var (
		cldr       = generate("./unicode/cldr", unicode)
		language   = generate("./language", cldr)
		internal   = generate("./internal", unicode, language)
		norm       = generate("./unicode/norm", unicode)
		rangetable = generate("./unicode/rangetable", unicode)
		cases      = generate("./cases", unicode, norm, language, rangetable)
		width      = generate("./width", unicode)
		bidi       = generate("./unicode/bidi", unicode, norm, rangetable)
		mib        = generate("./encoding/internal/identifier", unicode)
		_          = generate("./encoding/htmlindex", unicode, language, mib)
		_          = generate("./encoding/ianaindex", unicode, language, mib)
		_          = generate("./secure/precis", unicode, norm, rangetable, cases, width, bidi)
		_          = generate("./currency", unicode, cldr, language, internal)
		_          = generate("./internal/number", unicode, cldr, language, internal)
		_          = generate("./feature/plural", unicode, cldr, language, internal)
		_          = generate("./internal/export/idna", unicode, bidi, norm)
		_          = generate("./language/display", unicode, cldr, language, internal)
		_          = generate("./collate", unicode, norm, cldr, language, rangetable)
		_          = generate("./search", unicode, norm, cldr, language, rangetable)
	)
	all.Wait()

	// Copy exported packages to the destination golang.org repo.
	copyExported("golang.org/x/net/idna")

	if updateCore {
		copyVendored()
	}

	if hasErrors {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	vprintf("SUCCESS\n")
}

var (
	all       sync.WaitGroup
	hasErrors bool
)

type dependency struct {
	sync.WaitGroup
	hasErrors bool
}

func generate(pkg string, deps ...*dependency) *dependency {
	var wg dependency
	if exclude(pkg) {
		return &wg
	}
	wg.Add(1)
	all.Add(1)
	go func() {
		defer wg.Done()
		defer all.Done()
		// Wait for dependencies to finish.
		for _, d := range deps {
			d.Wait()
			if d.hasErrors && !*force {
				fmt.Printf("--- ABORT: %s\n", pkg)
				wg.hasErrors = true
				return
			}
		}
		vprintf("=== GENERATE %s\n", pkg)
		args := []string{"generate"}
		if *verbose {
			args = append(args, "-v")
		}
		args = append(args, pkg)
		cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), args...)
		w := &bytes.Buffer{}
		cmd.Stderr = w
		cmd.Stdout = w
		if err := cmd.Run(); err != nil {
			fmt.Printf("--- FAIL: %s:\n\t%v\n\tError: %v\n", pkg, indent(w), err)
			hasErrors = true
			wg.hasErrors = true
			return
		}

		vprintf("=== TEST %s\n", pkg)
		args[0] = "test"
		cmd = exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), args...)
		wt := &bytes.Buffer{}
		cmd.Stderr = wt
		cmd.Stdout = wt
		if err := cmd.Run(); err != nil {
			fmt.Printf("--- FAIL: %s:\n\t%v\n\tError: %v\n", pkg, indent(wt), err)
			hasErrors = true
			wg.hasErrors = true
			return
		}
		vprintf("--- SUCCESS: %s\n\t%v\n", pkg, indent(w))
		fmt.Print(wt.String())
	}()
	return &wg
}

// copyExported copies a package in x/text/internal/export to the
// destination repository.
func copyExported(p string) {
	copyPackage(
		filepath.Join("internal", "export", path.Base(p)),
		filepath.Join("..", filepath.FromSlash(p[len("golang.org/x"):])),
		"golang.org/x/text/internal/export/"+path.Base(p),
		p)
}

// copyVendored copies packages used by Go core into the vendored directory.
func copyVendored() {
	root := filepath.Join(build.Default.GOROOT, filepath.FromSlash("src/vendor/golang_org/x"))

	err := filepath.Walk(root, func(dir string, info os.FileInfo, err error) error {
		if err != nil || !info.IsDir() || root == dir {
			return err
		}
		src := dir[len(root)+1:]
		const slash = string(filepath.Separator)
		if c := strings.Split(src, slash); c[0] == "text" {
			// Copy a text repo package from its normal location.
			src = strings.Join(c[1:], slash)
		} else {
			// Copy the vendored package if it exists in the export directory.
			src = filepath.Join("internal", "export", filepath.Base(src))
		}
		copyPackage(src, dir, "golang.org", "golang_org")
		return nil
	})
	if err != nil {
		fmt.Printf("Seeding directory %s has failed %v:", root, err)
		os.Exit(1)
	}
}

// goGenRE is used to remove go:generate lines.
var goGenRE = regexp.MustCompile("//go:generate[^\n]*\n")

// copyPackage copies relevant files from a directory in x/text to the
// destination package directory. The destination package is assumed to have
// the same name. For each copied file go:generate lines are removed and
// and package comments are rewritten to the new path.
func copyPackage(dirSrc, dirDst, search, replace string) {
	err := filepath.Walk(dirSrc, func(file string, info os.FileInfo, err error) error {
		base := filepath.Base(file)
		if err != nil || info.IsDir() ||
			!strings.HasSuffix(base, ".go") ||
			strings.HasSuffix(base, "_test.go") && !strings.HasPrefix(base, "example") ||
			// Don't process subdirectories.
			filepath.Dir(file) != dirSrc {
			return nil
		}
		b, err := ioutil.ReadFile(file)
		if err != nil || bytes.Contains(b, []byte("\n// +build ignore")) {
			return err
		}
		// Fix paths.
		b = bytes.Replace(b, []byte(search), []byte(replace), -1)
		// Remove go:generate lines.
		b = goGenRE.ReplaceAllLiteral(b, nil)
		comment := "// Code generated by running \"go generate\" in golang.org/x/text. DO NOT EDIT.\n\n"
		if *doCore {
			comment = "// Code generated by running \"go run gen.go -core\" in golang.org/x/text. DO NOT EDIT.\n\n"
		}
		if !bytes.HasPrefix(b, []byte(comment)) {
			b = append([]byte(comment), b...)
		}
		if b, err = format.Source(b); err != nil {
			fmt.Println("Failed to format file:", err)
			os.Exit(1)
		}
		file = filepath.Join(dirDst, base)
		vprintf("=== COPY %s\n", file)
		return ioutil.WriteFile(file, b, 0666)
	})
	if err != nil {
		fmt.Println("Copying exported files failed:", err)
		os.Exit(1)
	}
}

func contains(a []string, s string) bool {
	for _, e := range a {
		if s == e {
			return true
		}
	}
	return false
}

func indent(b *bytes.Buffer) string {
	return strings.Replace(strings.TrimSpace(b.String()), "\n", "\n\t", -1)
}
