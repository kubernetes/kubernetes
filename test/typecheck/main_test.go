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
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var packageCases = []struct {
	code string
	errs map[string]string
}{
	// Empty: no problems!
	{"", map[string]string{"linux/amd64": ""}},
	// Slightly less empty: no problems!
	{"func getRandomNumber() int { return 4; }", map[string]string{"darwin/386": ""}},
	// Fixed in #59243
	{`import "golang.org/x/sys/unix"
	func f(err error) {
		if err != unix.ENXIO {
			panic("woops")
		}
	}`, map[string]string{"linux/amd64": "", "windows/amd64": "test.go:4:18: ENXIO not declared by package unix"}},
	// Fixed in #51984
	{`import "golang.org/x/sys/unix"
	const linuxHugetlbfsMagic = 0x958458f6
	func IsHugeTlbfs() bool {
		buf := unix.Statfs_t{}
		unix.Statfs("/tmp/", &buf)
		return buf.Type == linuxHugetlbfsMagic
	}`, map[string]string{
		"linux/amd64": "",
		"linux/386":   "test.go:7:22: linuxHugetlbfsMagic (untyped int constant 2508478710) overflows int32",
	}},
	// Fixed in #51873
	{`var a = map[string]interface{}{"num1": 9223372036854775807}`,
		map[string]string{"linux/arm": "test.go:2:40: 9223372036854775807 (untyped int constant) overflows int"}},
}

var testFiles = map[string]string{
	"golang.org/x/sys/unix/empty.go": `package unix`,
	"golang.org/x/sys/unix/errno_linux.go": `// +build linux
	package unix

	type Errno string
	func (e Errno) Error() string { return string(e) }

	var ENXIO = Errno("3")`,
	"golang.org/x/sys/unix/ztypes_linux_amd64.go": `// +build amd64,linux
	package unix
	type Statfs_t struct {
		Type int64
	}
	func Statfs(path string, statfs *Statfs_t) {}
	`,
	"golang.org/x/sys/unix/ztypes_linux_386.go": `// +build i386,linux
	package unix
	type Statfs_t struct {
		Type int32
	}
	func Statfs(path string, statfs *Statfs_t) {}
	`,
}

func TestHandlePackage(t *testing.T) {
	// When running in Bazel, we don't have access to Go source code. Fake it instead!
	tmpDir, err := ioutil.TempDir("", "test_typecheck")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	for path, data := range testFiles {
		path := filepath.Join(tmpDir, "src", path)
		err := os.MkdirAll(filepath.Dir(path), 0755)
		if err != nil {
			t.Fatal(err)
		}
		err = ioutil.WriteFile(path, []byte(data), 0644)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(path)
	}

	for _, test := range packageCases {
		for platform, expectedErr := range test.errs {
			a := newAnalyzer(platform)
			// Make Imports happen relative to our faked up GOROOT.
			a.ctx.GOROOT = tmpDir
			a.ctx.GOPATH = ""

			errs := []string{}
			a.conf.Error = func(err error) {
				errs = append(errs, err.Error())
			}

			code := "package test\n" + test.code
			parsed, err := parser.ParseFile(a.fset, "test.go", strings.NewReader(code), parser.AllErrors)
			if err != nil {
				t.Fatal(err)
			}
			a.typeCheck(tmpDir, []*ast.File{parsed})

			if expectedErr == "" {
				if len(errs) > 0 {
					t.Errorf("code:\n%s\ngot  %v\nwant %v",
						code, errs, expectedErr)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("code:\n%s\ngot  %v\nwant %v",
						code, errs, expectedErr)
				} else {
					if errs[0] != expectedErr {
						t.Errorf("code:\n%s\ngot  %v\nwant %v",
							code, errs[0], expectedErr)
					}
				}
			}
		}
	}
}

func TestHandlePath(t *testing.T) {
	c := collector{}
	e := errors.New("ex")
	i, _ := os.Stat(".") // i.IsDir() == true
	if c.handlePath("foo", nil, e) != e {
		t.Error("handlePath not returning errors")
	}
	if c.handlePath("vendor", i, nil) != filepath.SkipDir {
		t.Error("should skip vendor")
	}
}

func TestDedupeErrors(t *testing.T) {
	testcases := []struct {
		nPlatforms int
		results    []analyzerResult
		expected   string
	}{
		{1, []analyzerResult{}, ""},
		{1, []analyzerResult{{"linux/arm", "test", nil}}, ""},
		{1, []analyzerResult{
			{"linux/arm", "test", []string{"a"}}},
			"ERROR(linux/arm) a\n"},
		{3, []analyzerResult{
			{"linux/arm", "test", []string{"a"}},
			{"windows/386", "test", []string{"b"}},
			{"windows/amd64", "test", []string{"b", "c"}}},
			"ERROR(linux/arm) a\n" +
				"ERROR(windows) b\n" +
				"ERROR(windows/amd64) c\n"},
	}
	for _, tc := range testcases {
		out := &bytes.Buffer{}
		results := make(chan analyzerResult, len(tc.results))
		for _, res := range tc.results {
			results <- res
		}
		close(results)
		dedupeErrors(out, results, len(tc.results)/tc.nPlatforms, tc.nPlatforms)
		outString := out.String()
		if outString != tc.expected {
			t.Errorf("dedupeErrors(%v) = '%s', expected '%s'",
				tc.results, outString, tc.expected)
		}
	}
}
