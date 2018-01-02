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

	"github.com/coreos/rkt/pkg/user"
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
		panic(err)
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
	if err := CopyTree(src, dst, user.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)

	// relative paths
	if err := os.Chdir(td); err != nil {
		t.Fatal(err)
	}

	dst = "dst-rel1"
	if err := CopyTree("././src/", dst, user.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)

	dst = "./dst-rel2"
	if err := CopyTree("./src", dst, user.NewBlankUidRange()); err != nil {
		t.Fatal(err)
	}
	checkTree(t, dst, tr)
}

func TestFileIsExecutable(t *testing.T) {
	tempDir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	testCases := []struct {
		Permission   os.FileMode
		IsExecutable bool
	}{
		{0200, false},
		{0400, false},
		{0600, false},
		{0100, true},
		{0300, true},
		{0500, true},
		{0700, true},

		{0002, false},
		{0004, false},
		{0006, false},
		{0001, true},
		{0003, true},
		{0005, true},
		{0007, true},

		{0020, false},
		{0040, false},
		{0060, false},
		{0010, true},
		{0030, true},
		{0050, true},
		{0070, true},

		{0000, false},
		{0222, false},
		{0444, false},
		{0666, false},

		{0146, true},
		{0661, true},
	}

	for _, tc := range testCases {
		f, err := ioutil.TempFile(tempDir, "")
		if err != nil {
			panic(err)
		}

		if err := f.Chmod(tc.Permission); err != nil {
			panic(err)
		}

		if err := f.Close(); err != nil {
			panic(err)
		}

		path := f.Name()

		if tc.IsExecutable != IsExecutable(path) {
			t.Errorf("fileutil.IsExecutable(%q) with permissions %q, expected %v", path, tc.Permission, tc.IsExecutable)
		}
	}
}

func TestDeviceInfo(t *testing.T) {
	// First, test the main call
	{
		kind, major, minor, err := GetDeviceInfo("/dev/null")
		if kind != 'c' || major != 1 || minor != 3 || err != nil {
			t.Errorf("GetDeviceInfo(/dev/null) wrong result")
		}
	}

	{
		_, _, _, err := GetDeviceInfo("/usr")
		if err == nil {
			t.Errorf("GetDeviceInfo(/usr) should return err")
		}
	}

	// Then test the logic more specifically
	for i, tt := range []struct {
		mode  os.FileMode
		rdev  uint64
		kind  rune
		major uint64
		minor uint64
	}{
		// /dev/null
		{
			69206454,
			259,
			'c',
			1,
			3,
		},
		// /dev/sda1
		{
			67109296,
			2049,
			'b',
			8,
			1,
		},
		// /dev/pts/3
		{
			69206416,
			34819,
			'c',
			136,
			3,
		},
		// /dev/dm-0
		{
			67109296,
			64768,
			'b',
			253,
			0,
		},
	} {
		kind, major, minor, err := getDeviceInfo(tt.mode, tt.rdev)
		if err != nil {
			t.Errorf("getDeviceInfo %d not as expected, got err %s", i, err)
		}
		// don't care about result when err
		if err != nil {
			continue
		}

		if tt.kind != kind {
			t.Errorf("getDeviceInfo %d kind expected %v got %v", i, tt.kind, kind)
		}
		if tt.major != major {
			t.Errorf("getDeviceInfo %d major expected %v got %v", i, tt.major, major)
		}
		if tt.minor != minor {
			t.Errorf("getDeviceInfo %d minor expected %v got %v", i, tt.minor, minor)
		}
	}
}
