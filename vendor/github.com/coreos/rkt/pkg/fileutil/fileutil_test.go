// Copyright 2015 The rkt Authors
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

package fileutil

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/pkg/uid"
)

const tstprefix = "fileutil-test"

func touch(t *testing.T, name string) {
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

type tree struct {
	path string
	dir  bool
}

func createTree(t *testing.T, dir string, tr []tree) {
	for _, f := range tr {
		if f.dir {
			if err := os.MkdirAll(filepath.Join(dir, f.path), 0755); err != nil {
				t.Fatal(err)
			}
		} else {
			touch(t, filepath.Join(dir, f.path))
		}
	}
}

func checkTree(t *testing.T, dir string, tr []tree) {
	for _, f := range tr {
		if _, err := os.Stat(filepath.Join(dir, f.path)); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCopyTree(t *testing.T) {
	td, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(td)

	src := filepath.Join(td, "src")
	dst := filepath.Join(td, "dst")
	if err := os.MkdirAll(filepath.Join(td, "src"), 0755); err != nil {
		t.Fatal(err)
	}

	tr := []tree{
		{
			path: "dir1",
			dir:  true,
		},
		{
			path: "dir2",
			dir:  true,
		},
		{
			path: "dir1/foo",
			dir:  false,
		},
		{
			path: "dir1/bar",
			dir:  false,
		},
	}

	createTree(t, src, tr)

	// absolute paths
	if err := CopyTree(src, dst, uid.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)

	// relative paths
	if err := os.Chdir(td); err != nil {
		t.Fatal(err)
	}

	dst = "dst-rel1"
	if err := CopyTree("././src/", dst, uid.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)

	dst = "./dst-rel2"
	if err := CopyTree("./src", dst, uid.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)
}
