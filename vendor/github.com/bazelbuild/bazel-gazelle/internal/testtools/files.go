/* Copyright 2018 The Bazel Authors. All rights reserved.

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

package testtools

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// FileSpec specifies the content of a test file.
type FileSpec struct {
	// Path is a slash-separated path relative to the test directory. If Path
	// ends with a slash, it indicates a directory should be created
	// instead of a file.
	Path string

	// Symlink is a slash-separated path relative to the test directory. If set,
	// it indicates a symbolic link should be created with this path instead of a
	// file.
	Symlink string

	// Content is the content of the test file.
	Content string
}

// CreateFiles creates a directory of test files. This is a more compact
// alternative to testdata directories. CreateFiles returns a canonical path
// to the directory and a function to call to clean up the directory
// after the test.
func CreateFiles(t *testing.T, files []FileSpec) (dir string, cleanup func()) {
	dir, err := ioutil.TempDir(os.Getenv("TEST_TEMPDIR"), "gazelle_test")
	if err != nil {
		t.Fatal(err)
	}
	dir, err = filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range files {
		path := filepath.Join(dir, filepath.FromSlash(f.Path))
		if strings.HasSuffix(f.Path, "/") {
			if err := os.MkdirAll(path, 0700); err != nil {
				os.RemoveAll(dir)
				t.Fatal(err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			os.RemoveAll(dir)
			t.Fatal(err)
		}
		if f.Symlink != "" {
			if err := os.Symlink(f.Symlink, path); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := ioutil.WriteFile(path, []byte(f.Content), 0600); err != nil {
			os.RemoveAll(dir)
			t.Fatal(err)
		}
	}

	return dir, func() { os.RemoveAll(dir) }
}

// CheckFiles checks that files in "dir" exist and have the content specified
// in "files". Files not listed in "files" are not tested, so extra files
// are allowed.
func CheckFiles(t *testing.T, dir string, files []FileSpec) {
	for _, f := range files {
		path := filepath.Join(dir, f.Path)
		if strings.HasSuffix(f.Path, "/") {
			if st, err := os.Stat(path); err != nil {
				t.Errorf("could not stat %s: %v", f.Path, err)
			} else if !st.IsDir() {
				t.Errorf("not a directory: %s", f.Path)
			}
		} else {
			want := strings.TrimSpace(f.Content)
			gotBytes, err := ioutil.ReadFile(filepath.Join(dir, f.Path))
			if err != nil {
				t.Errorf("could not read %s: %v", f.Path, err)
				continue
			}
			got := strings.TrimSpace(string(gotBytes))
			if got != want {
				t.Errorf("%s: got:\n%s\nwant:\n %s", f.Path, gotBytes, f.Content)
			}
		}
	}
}
