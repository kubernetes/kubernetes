// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package users

import (
	"io/fs"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func checkFilePermissions(t *testing.T, path string, uid, gid int64, permissions uint32) {
	t.Helper()
	fInfo, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fInfo.Mode().Perm() != fs.FileMode(permissions) {
		t.Errorf("want file mode %s, got file mode %s", fs.FileMode(permissions).String(), fInfo.Mode().Perm().String())
	}
	statInfo, ok := fInfo.Sys().(*syscall.Stat_t)
	if !ok {
		t.Fatalf("could not cast fInfo.Sys to *syscall.Stat_t")
	}
	if statInfo.Uid != uint32(uid) {
		t.Errorf("want uid %d, got uid %d", uid, statInfo.Uid)
	}
	if statInfo.Gid != uint32(gid) {
		t.Errorf("want uid %d, got uid %d", gid, statInfo.Gid)
	}
}

func TestUpdateFileOwnership(t *testing.T) {
	file, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	if err := UpdateFileOwnership(file.Name(), 1000, 1100, 0700); err != nil {
		t.Fatal(err)
	}
	checkFilePermissions(t, file.Name(), 1000, 1100, 0700)
}

func TestUpdateDirectoryOwnership(t *testing.T) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	innerDir := dir + "/inner"
	if err := os.Mkdir(innerDir, 0777); err != nil {
		t.Fatal(err)
	}
	if _, err := ioutil.TempFile(innerDir, ""); err != nil {
		t.Fatal(err)
	}
	if err := UpdateDirectoryOwnership(dir, 1000, 1100, 0700); err != nil {
		t.Fatalf("UpdateDirectoryOwnership(%s, 1000, 1100, 0700) failed: %v", dir, err)
	}
	err = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		checkFilePermissions(t, path, 1000, 1100, 0700)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
