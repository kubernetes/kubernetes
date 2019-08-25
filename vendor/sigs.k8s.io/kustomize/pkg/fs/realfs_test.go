/*
Copyright 2018 The Kubernetes Authors.

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

package fs

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"testing"
)

func makeTestDir(t *testing.T) (FileSystem, string) {
	x := MakeRealFS()
	td, err := ioutil.TempDir("", "kustomize_testing_dir")
	if err != nil {
		t.Fatalf("unexpected error %s", err)
	}
	testDir, err := filepath.EvalSymlinks(td)
	if err != nil {
		t.Fatalf("unexpected error %s", err)
	}
	if !x.Exists(testDir) {
		t.Fatalf("expected existence")
	}
	if !x.IsDir(testDir) {
		t.Fatalf("expected directory")
	}
	return x, testDir
}

func TestCleanedAbs_1(t *testing.T) {
	x, testDir := makeTestDir(t)
	defer os.RemoveAll(testDir)

	d, f, err := x.CleanedAbs("")
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	if d.String() != wd {
		t.Fatalf("unexpected d=%s", d)
	}
	if f != "" {
		t.Fatalf("unexpected f=%s", f)
	}
}

func TestCleanedAbs_2(t *testing.T) {
	x, testDir := makeTestDir(t)
	defer os.RemoveAll(testDir)

	d, f, err := x.CleanedAbs("/")
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	if d != "/" {
		t.Fatalf("unexpected d=%s", d)
	}
	if f != "" {
		t.Fatalf("unexpected f=%s", f)
	}
}

func TestCleanedAbs_3(t *testing.T) {
	x, testDir := makeTestDir(t)
	defer os.RemoveAll(testDir)

	err := x.WriteFile(
		filepath.Join(testDir, "foo"), []byte(`foo`))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}

	d, f, err := x.CleanedAbs(filepath.Join(testDir, "foo"))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	if d.String() != testDir {
		t.Fatalf("unexpected d=%s", d)
	}
	if f != "foo" {
		t.Fatalf("unexpected f=%s", f)
	}

}

func TestCleanedAbs_4(t *testing.T) {
	x, testDir := makeTestDir(t)
	defer os.RemoveAll(testDir)

	err := x.MkdirAll(filepath.Join(testDir, "d1", "d2"))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	err = x.WriteFile(
		filepath.Join(testDir, "d1", "d2", "bar"),
		[]byte(`bar`))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}

	d, f, err := x.CleanedAbs(
		filepath.Join(testDir, "d1", "d2"))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	if d.String() != filepath.Join(testDir, "d1", "d2") {
		t.Fatalf("unexpected d=%s", d)
	}
	if f != "" {
		t.Fatalf("unexpected f=%s", f)
	}

	d, f, err = x.CleanedAbs(
		filepath.Join(testDir, "d1", "d2", "bar"))
	if err != nil {
		t.Fatalf("unexpected err=%v", err)
	}
	if d.String() != filepath.Join(testDir, "d1", "d2") {
		t.Fatalf("unexpected d=%s", d)
	}
	if f != "bar" {
		t.Fatalf("unexpected f=%s", f)
	}
}

func TestReadFilesRealFS(t *testing.T) {
	x, testDir := makeTestDir(t)
	defer os.RemoveAll(testDir)

	err := x.WriteFile(path.Join(testDir, "foo"), []byte(`foo`))
	if err != nil {
		t.Fatalf("unexpected error %s", err)
	}
	if !x.Exists(path.Join(testDir, "foo")) {
		t.Fatalf("expected foo")
	}
	if x.IsDir(path.Join(testDir, "foo")) {
		t.Fatalf("expected foo not to be a directory")
	}

	err = x.WriteFile(path.Join(testDir, "bar"), []byte(`bar`))
	if err != nil {
		t.Fatalf("unexpected error %s", err)
	}

	files, err := x.Glob(path.Join("testDir", "*"))
	expected := []string{
		path.Join(testDir, "bar"),
		path.Join(testDir, "foo"),
	}
	if err != nil {
		t.Fatalf("expected no error")
	}
	if reflect.DeepEqual(files, expected) {
		t.Fatalf("incorrect files found by glob: %v", files)
	}
}
