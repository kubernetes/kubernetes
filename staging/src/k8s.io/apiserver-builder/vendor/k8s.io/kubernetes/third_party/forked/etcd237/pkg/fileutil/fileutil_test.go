// Copyright 2015 CoreOS, Inc.
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
	"os/user"
	"path/filepath"
	"reflect"
	"testing"
)

func TestIsDirWriteable(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("unexpected ioutil.TempDir error: %v", err)
	}
	defer os.RemoveAll(tmpdir)
	if err = IsDirWriteable(tmpdir); err != nil {
		t.Fatalf("unexpected IsDirWriteable error: %v", err)
	}
	if err = os.Chmod(tmpdir, 0444); err != nil {
		t.Fatalf("unexpected os.Chmod error: %v", err)
	}
	me, err := user.Current()
	if err != nil {
		// err can be non-nil when cross compiled
		// http://stackoverflow.com/questions/20609415/cross-compiling-user-current-not-implemented-on-linux-amd64
		t.Skipf("failed to get current user: %v", err)
	}
	if me.Name == "root" || me.Name == "Administrator" {
		// ideally we should check CAP_DAC_OVERRIDE.
		// but it does not matter for tests.
		t.Skipf("running as a superuser")
	}
	if err := IsDirWriteable(tmpdir); err == nil {
		t.Fatalf("expected IsDirWriteable to error")
	}
}

func TestReadDir(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	defer os.RemoveAll(tmpdir)
	if err != nil {
		t.Fatalf("unexpected ioutil.TempDir error: %v", err)
	}
	files := []string{"def", "abc", "xyz", "ghi"}
	for _, f := range files {
		var fh *os.File
		fh, err = os.Create(filepath.Join(tmpdir, f))
		if err != nil {
			t.Fatalf("error creating file: %v", err)
		}
		if err = fh.Close(); err != nil {
			t.Fatalf("error closing file: %v", err)
		}
	}
	fs, err := ReadDir(tmpdir)
	if err != nil {
		t.Fatalf("error calling ReadDir: %v", err)
	}
	wfs := []string{"abc", "def", "ghi", "xyz"}
	if !reflect.DeepEqual(fs, wfs) {
		t.Fatalf("ReadDir: got %v, want %v", fs, wfs)
	}
}

func TestExist(t *testing.T) {
	f, err := ioutil.TempFile(os.TempDir(), "fileutil")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	if g := Exist(f.Name()); g != true {
		t.Errorf("exist = %v, want true", g)
	}

	os.Remove(f.Name())
	if g := Exist(f.Name()); g != false {
		t.Errorf("exist = %v, want false", g)
	}
}
