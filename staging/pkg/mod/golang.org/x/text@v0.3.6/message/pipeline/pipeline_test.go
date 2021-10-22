// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pipeline

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

var genFiles = flag.Bool("gen", false, "generate output files instead of comparing")

// setHelper is testing.T.Helper on Go 1.9+, overridden by go19_test.go.
var setHelper = func(t *testing.T) {}

func TestFullCycle(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("cannot load outside packages on android")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skipf("skipping because 'go' command is unavailable: %v", err)
	}

	GOPATH, err := ioutil.TempDir("", "pipeline_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(GOPATH)
	testdata := filepath.Join(GOPATH, "src", "testdata")

	// Copy the testdata contents into a new module.
	copyTestdata(t, testdata)
	initTestdataModule(t, testdata)

	// Several places hard-code the use of build.Default.
	// Adjust it to match the test's temporary GOPATH.
	defer func(prev string) { build.Default.GOPATH = prev }(build.Default.GOPATH)
	build.Default.GOPATH = GOPATH + string(filepath.ListSeparator) + build.Default.GOPATH
	if wd := reflect.ValueOf(&build.Default).Elem().FieldByName("WorkingDir"); wd.IsValid() {
		defer func(prev string) { wd.SetString(prev) }(wd.String())
		wd.SetString(testdata)
	}

	// To work around https://golang.org/issue/34860, execute the commands
	// that (transitively) use go/build in the working directory of the
	// corresponding module.
	wd, _ := os.Getwd()
	defer os.Chdir(wd)

	dirs, err := ioutil.ReadDir(testdata)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range dirs {
		if !f.IsDir() {
			continue
		}
		t.Run(f.Name(), func(t *testing.T) {
			chk := func(t *testing.T, err error) {
				setHelper(t)
				if err != nil {
					t.Fatal(err)
				}
			}
			dir := filepath.Join(testdata, f.Name())
			pkgPath := "testdata/" + f.Name()
			config := Config{
				SourceLanguage: language.AmericanEnglish,
				Packages:       []string{pkgPath},
				Dir:            filepath.Join(dir, "locales"),
				GenFile:        "catalog_gen.go",
				GenPackage:     pkgPath,
			}

			os.Chdir(dir)

			// TODO: load config if available.
			s, err := Extract(&config)
			chk(t, err)
			chk(t, s.Import())
			chk(t, s.Merge())
			// TODO:
			//  for range s.Config.Actions {
			//  	//  TODO: do the actions.
			//  }
			chk(t, s.Export())
			chk(t, s.Generate())

			os.Chdir(wd)

			writeJSON(t, filepath.Join(dir, "extracted.gotext.json"), s.Extracted)
			checkOutput(t, dir, f.Name())
		})
	}
}

func copyTestdata(t *testing.T, dst string) {
	err := filepath.Walk("testdata", func(p string, f os.FileInfo, err error) error {
		if p == "testdata" || strings.HasSuffix(p, ".want") {
			return nil
		}

		rel := strings.TrimPrefix(p, "testdata"+string(filepath.Separator))
		if f.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0755)
		}

		data, err := ioutil.ReadFile(p)
		if err != nil {
			return err
		}
		return ioutil.WriteFile(filepath.Join(dst, rel), data, 0644)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func initTestdataModule(t *testing.T, dst string) {
	xTextDir, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}

	goMod := fmt.Sprintf(`module testdata
go 1.11
require golang.org/x/text v0.0.0-00010101000000-000000000000
replace golang.org/x/text v0.0.0-00010101000000-000000000000 => %s
`, xTextDir)
	if err := ioutil.WriteFile(filepath.Join(dst, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	data, err := ioutil.ReadFile(filepath.Join(xTextDir, "go.sum"))
	if err := ioutil.WriteFile(filepath.Join(dst, "go.sum"), data, 0644); err != nil {
		t.Fatal(err)
	}
}

func checkOutput(t *testing.T, gen string, testdataDir string) {
	err := filepath.Walk(gen, func(gotFile string, f os.FileInfo, err error) error {
		if f.IsDir() {
			return nil
		}
		rel := strings.TrimPrefix(gotFile, gen+string(filepath.Separator))

		wantFile := filepath.Join("testdata", testdataDir, rel+".want")
		if _, err := os.Stat(wantFile); os.IsNotExist(err) {
			return nil
		}

		got, err := ioutil.ReadFile(gotFile)
		if err != nil {
			t.Errorf("failed to read %q", gotFile)
			return nil
		}
		if *genFiles {
			if err := ioutil.WriteFile(wantFile, got, 0644); err != nil {
				t.Fatal(err)
			}
		}
		want, err := ioutil.ReadFile(wantFile)
		if err != nil {
			t.Errorf("failed to read %q", wantFile)
		} else {
			scanGot := bufio.NewScanner(bytes.NewReader(got))
			scanWant := bufio.NewScanner(bytes.NewReader(want))
			line := 0
			clean := func(s string) string {
				if i := strings.LastIndex(s, "//"); i != -1 {
					s = s[:i]
				}
				return path.Clean(filepath.ToSlash(s))
			}
			for scanGot.Scan() && scanWant.Scan() {
				got := clean(scanGot.Text())
				want := clean(scanWant.Text())
				if got != want {
					t.Errorf("file %q differs from .want file at line %d:\n\t%s\n\t%s", gotFile, line, got, want)
					break
				}
				line++
			}
			if scanGot.Scan() || scanWant.Scan() {
				t.Errorf("file %q differs from .want file at line %d.", gotFile, line)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func writeJSON(t *testing.T, path string, x interface{}) {
	data, err := json.MarshalIndent(x, "", "    ")
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}
