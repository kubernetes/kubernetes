// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// Test cases for mkmerge.go.
// Usage:
//     $ go test mkmerge.go mkmerge_test.go
package main

import (
	"bytes"
	"fmt"
	"go/parser"
	"go/token"
	"html/template"
	"strings"
	"testing"
)

func TestImports(t *testing.T) {
	t.Run("importName", func(t *testing.T) {
		cases := []struct {
			src   string
			ident string
		}{
			{`"syscall"`, "syscall"},
			{`. "foobar"`, "."},
			{`"go/ast"`, "ast"},
			{`moo "go/format"`, "moo"},
			{`. "go/token"`, "."},
			{`"golang.org/x/sys/unix"`, "unix"},
			{`nix "golang.org/x/sys/unix"`, "nix"},
			{`_ "golang.org/x/sys/unix"`, "_"},
		}

		for _, c := range cases {
			pkgSrc := fmt.Sprintf("package main\nimport %s", c.src)

			f, err := parser.ParseFile(token.NewFileSet(), "", pkgSrc, parser.ImportsOnly)
			if err != nil {
				t.Error(err)
				continue
			}
			if len(f.Imports) != 1 {
				t.Errorf("Got %d imports, expected 1", len(f.Imports))
				continue
			}

			got, err := importName(f.Imports[0])
			if err != nil {
				t.Fatal(err)
			}
			if got != c.ident {
				t.Errorf("Got %q, expected %q", got, c.ident)
			}
		}
	})

	t.Run("filterImports", func(t *testing.T) {
		cases := []struct{ before, after string }{
			{`package test

			import (
				"foo"
				"bar"
			)`,
				"package test\n"},
			{`package test

			import (
				"foo"
				"bar"
			)

			func useFoo() { foo.Usage() }`,
				`package test

import (
	"foo"
)

func useFoo() { foo.Usage() }
`},
		}
		for _, c := range cases {
			got, err := filterImports([]byte(c.before))
			if err != nil {
				t.Error(err)
			}

			if string(got) != c.after {
				t.Errorf("Got:\n%s\nExpected:\n%s\n", got, c.after)
			}
		}
	})
}

func TestMerge(t *testing.T) {
	// Input architecture files
	inTmpl := template.Must(template.New("input").Parse(`
// Package comments

// build directives for arch{{.}}

// +build goos,arch{{.}}

package main

/*
#include <stdint.h>
#include <stddef.h>
int utimes(uintptr_t, uintptr_t);
int utimensat(int, uintptr_t, uintptr_t, int);
*/
import "C"

// The imports
import (
	"commonDep"
	"uniqueDep{{.}}"
)

// Vars
var (
	commonVar = commonDep.Use("common")

	uniqueVar{{.}} = "unique{{.}}"
)

// Common free standing comment

// Common comment
const COMMON_INDEPENDENT = 1234
const UNIQUE_INDEPENDENT_{{.}} = "UNIQUE_INDEPENDENT_{{.}}"

// Group comment
const (
	COMMON_GROUP = "COMMON_GROUP"
	UNIQUE_GROUP_{{.}} = "UNIQUE_GROUP_{{.}}"
)

// Group2 comment
const (
	UNIQUE_GROUP21_{{.}} = "UNIQUE_GROUP21_{{.}}"
	UNIQUE_GROUP22_{{.}} = "UNIQUE_GROUP22_{{.}}"
)

// Group3 comment
const (
	sub1Common1 = 11
	sub1Unique2{{.}} = 12
	sub1Common3_LONG = 13

	sub2Unique1{{.}} = 21
	sub2Common2 = 22
	sub2Common3 = 23
	sub2Unique4{{.}} = 24
)

type commonInt int

type uniqueInt{{.}} int

func commonF() string {
	return commonDep.Use("common")
	}

func uniqueF() string {
	C.utimes(0, 0)
	return uniqueDep{{.}}.Use("{{.}}")
	}

// Group4 comment
const (
	sub3Common1 = 31
	sub3Unique2{{.}} = 32
	sub3Unique3{{.}} = 33
	sub3Common4 = 34

	sub4Common1, sub4Unique2{{.}} = 41, 42
	sub4Unique3{{.}}, sub4Common4 = 43, 44
)
`))

	// Filtered architecture files
	outTmpl := template.Must(template.New("output").Parse(`// Package comments

// build directives for arch{{.}}

// +build goos,arch{{.}}

package main

/*
#include <stdint.h>
#include <stddef.h>
int utimes(uintptr_t, uintptr_t);
int utimensat(int, uintptr_t, uintptr_t, int);
*/
import "C"

// The imports
import (
	"commonDep"
	"uniqueDep{{.}}"
)

// Vars
var (
	commonVar = commonDep.Use("common")

	uniqueVar{{.}} = "unique{{.}}"
)

const UNIQUE_INDEPENDENT_{{.}} = "UNIQUE_INDEPENDENT_{{.}}"

// Group comment
const (
	UNIQUE_GROUP_{{.}} = "UNIQUE_GROUP_{{.}}"
)

// Group2 comment
const (
	UNIQUE_GROUP21_{{.}} = "UNIQUE_GROUP21_{{.}}"
	UNIQUE_GROUP22_{{.}} = "UNIQUE_GROUP22_{{.}}"
)

// Group3 comment
const (
	sub1Unique2{{.}} = 12

	sub2Unique1{{.}} = 21
	sub2Unique4{{.}} = 24
)

type uniqueInt{{.}} int

func uniqueF() string {
	C.utimes(0, 0)
	return uniqueDep{{.}}.Use("{{.}}")
}

// Group4 comment
const (
	sub3Unique2{{.}} = 32
	sub3Unique3{{.}} = 33

	sub4Common1, sub4Unique2{{.}} = 41, 42
	sub4Unique3{{.}}, sub4Common4 = 43, 44
)
`))

	const mergedFile = `// Package comments

package main

// The imports
import (
	"commonDep"
)

// Common free standing comment

// Common comment
const COMMON_INDEPENDENT = 1234

// Group comment
const (
	COMMON_GROUP = "COMMON_GROUP"
)

// Group3 comment
const (
	sub1Common1      = 11
	sub1Common3_LONG = 13

	sub2Common2 = 22
	sub2Common3 = 23
)

type commonInt int

func commonF() string {
	return commonDep.Use("common")
}

// Group4 comment
const (
	sub3Common1 = 31
	sub3Common4 = 34
)
`

	// Generate source code for different "architectures"
	var inFiles, outFiles []srcFile
	for _, arch := range strings.Fields("A B C D") {
		buf := new(bytes.Buffer)
		err := inTmpl.Execute(buf, arch)
		if err != nil {
			t.Fatal(err)
		}
		inFiles = append(inFiles, srcFile{"file" + arch, buf.Bytes()})

		buf = new(bytes.Buffer)
		err = outTmpl.Execute(buf, arch)
		if err != nil {
			t.Fatal(err)
		}
		outFiles = append(outFiles, srcFile{"file" + arch, buf.Bytes()})
	}

	t.Run("getCodeSet", func(t *testing.T) {
		got, err := getCodeSet(inFiles[0].src)
		if err != nil {
			t.Fatal(err)
		}

		expectedElems := []codeElem{
			{token.COMMENT, "Package comments\n"},
			{token.COMMENT, "build directives for archA\n"},
			{token.COMMENT, "+build goos,archA\n"},
			{token.CONST, `COMMON_INDEPENDENT = 1234`},
			{token.CONST, `UNIQUE_INDEPENDENT_A = "UNIQUE_INDEPENDENT_A"`},
			{token.CONST, `COMMON_GROUP = "COMMON_GROUP"`},
			{token.CONST, `UNIQUE_GROUP_A = "UNIQUE_GROUP_A"`},
			{token.CONST, `UNIQUE_GROUP21_A = "UNIQUE_GROUP21_A"`},
			{token.CONST, `UNIQUE_GROUP22_A = "UNIQUE_GROUP22_A"`},
			{token.CONST, `sub1Common1 = 11`},
			{token.CONST, `sub1Unique2A = 12`},
			{token.CONST, `sub1Common3_LONG = 13`},
			{token.CONST, `sub2Unique1A = 21`},
			{token.CONST, `sub2Common2 = 22`},
			{token.CONST, `sub2Common3 = 23`},
			{token.CONST, `sub2Unique4A = 24`},
			{token.CONST, `sub3Common1 = 31`},
			{token.CONST, `sub3Unique2A = 32`},
			{token.CONST, `sub3Unique3A = 33`},
			{token.CONST, `sub3Common4 = 34`},
			{token.CONST, `sub4Common1, sub4Unique2A = 41, 42`},
			{token.CONST, `sub4Unique3A, sub4Common4 = 43, 44`},
			{token.TYPE, `commonInt int`},
			{token.TYPE, `uniqueIntA int`},
			{token.FUNC, `func commonF() string {
	return commonDep.Use("common")
}`},
			{token.FUNC, `func uniqueF() string {
	C.utimes(0, 0)
	return uniqueDepA.Use("A")
}`},
		}
		expected := newCodeSet()
		for _, d := range expectedElems {
			expected.add(d)
		}

		if len(got.set) != len(expected.set) {
			t.Errorf("Got %d codeElems, expected %d", len(got.set), len(expected.set))
		}
		for expElem := range expected.set {
			if !got.has(expElem) {
				t.Errorf("Didn't get expected codeElem %#v", expElem)
			}
		}
		for gotElem := range got.set {
			if !expected.has(gotElem) {
				t.Errorf("Got unexpected codeElem %#v", gotElem)
			}
		}
	})

	t.Run("getCommonSet", func(t *testing.T) {
		got, err := getCommonSet(inFiles)
		if err != nil {
			t.Fatal(err)
		}

		expected := newCodeSet()
		expected.add(codeElem{token.COMMENT, "Package comments\n"})
		expected.add(codeElem{token.CONST, `COMMON_INDEPENDENT = 1234`})
		expected.add(codeElem{token.CONST, `COMMON_GROUP = "COMMON_GROUP"`})
		expected.add(codeElem{token.CONST, `sub1Common1 = 11`})
		expected.add(codeElem{token.CONST, `sub1Common3_LONG = 13`})
		expected.add(codeElem{token.CONST, `sub2Common2 = 22`})
		expected.add(codeElem{token.CONST, `sub2Common3 = 23`})
		expected.add(codeElem{token.CONST, `sub3Common1 = 31`})
		expected.add(codeElem{token.CONST, `sub3Common4 = 34`})
		expected.add(codeElem{token.TYPE, `commonInt int`})
		expected.add(codeElem{token.FUNC, `func commonF() string {
	return commonDep.Use("common")
}`})

		if len(got.set) != len(expected.set) {
			t.Errorf("Got %d codeElems, expected %d", len(got.set), len(expected.set))
		}
		for expElem := range expected.set {
			if !got.has(expElem) {
				t.Errorf("Didn't get expected codeElem %#v", expElem)
			}
		}
		for gotElem := range got.set {
			if !expected.has(gotElem) {
				t.Errorf("Got unexpected codeElem %#v", gotElem)
			}
		}
	})

	t.Run("filter(keepCommon)", func(t *testing.T) {
		commonSet, err := getCommonSet(inFiles)
		if err != nil {
			t.Fatal(err)
		}

		got, err := filter(inFiles[0].src, commonSet.keepCommon)
		expected := []byte(mergedFile)

		if !bytes.Equal(got, expected) {
			t.Errorf("Got:\n%s\nExpected:\n%s", addLineNr(got), addLineNr(expected))
			diffLines(t, got, expected)
		}
	})

	t.Run("filter(keepArchSpecific)", func(t *testing.T) {
		commonSet, err := getCommonSet(inFiles)
		if err != nil {
			t.Fatal(err)
		}

		for i := range inFiles {
			got, err := filter(inFiles[i].src, commonSet.keepArchSpecific)
			if err != nil {
				t.Fatal(err)
			}

			expected := outFiles[i].src

			if !bytes.Equal(got, expected) {
				t.Errorf("Got:\n%s\nExpected:\n%s", addLineNr(got), addLineNr(expected))
				diffLines(t, got, expected)
			}
		}
	})
}

func TestMergedName(t *testing.T) {
	t.Run("getValidGOOS", func(t *testing.T) {
		testcases := []struct {
			filename, goos string
			ok             bool
		}{
			{"zerrors_aix.go", "aix", true},
			{"zerrors_darwin.go", "darwin", true},
			{"zerrors_dragonfly.go", "dragonfly", true},
			{"zerrors_freebsd.go", "freebsd", true},
			{"zerrors_linux.go", "linux", true},
			{"zerrors_netbsd.go", "netbsd", true},
			{"zerrors_openbsd.go", "openbsd", true},
			{"zerrors_solaris.go", "solaris", true},
			{"zerrors_multics.go", "", false},
		}
		for _, tc := range testcases {
			goos, ok := getValidGOOS(tc.filename)
			if goos != tc.goos {
				t.Errorf("got GOOS %q, expected %q", goos, tc.goos)
			}
			if ok != tc.ok {
				t.Errorf("got ok %v, expected %v", ok, tc.ok)
			}
		}
	})
}

// Helper functions to diff test sources

func diffLines(t *testing.T, got, expected []byte) {
	t.Helper()

	gotLines := bytes.Split(got, []byte{'\n'})
	expLines := bytes.Split(expected, []byte{'\n'})

	i := 0
	for i < len(gotLines) && i < len(expLines) {
		if !bytes.Equal(gotLines[i], expLines[i]) {
			t.Errorf("Line %d: Got:\n%q\nExpected:\n%q", i+1, gotLines[i], expLines[i])
			return
		}
		i++
	}

	if i < len(gotLines) && i >= len(expLines) {
		t.Errorf("Line %d: got %q, expected EOF", i+1, gotLines[i])
	}
	if i >= len(gotLines) && i < len(expLines) {
		t.Errorf("Line %d: got EOF, expected %q", i+1, gotLines[i])
	}
}

func addLineNr(src []byte) []byte {
	lines := bytes.Split(src, []byte("\n"))
	for i, line := range lines {
		lines[i] = []byte(fmt.Sprintf("%d: %s", i+1, line))
	}
	return bytes.Join(lines, []byte("\n"))
}
