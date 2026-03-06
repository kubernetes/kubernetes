/*
Copyright The Kubernetes Authors.

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

package deviceattribute

import (
	"io/fs"
	"path/filepath"
	"testing"
	"testing/fstest"
)

func TestWithFS(t *testing.T) {
	mc := machine{}
	path := "dra.test"
	content := "DRA deviceattribute"
	mode := fs.FileMode(0o640) // any random mode pattern which is not the default
	fakefs := fstest.MapFS{
		path: &fstest.MapFile{
			Data: []byte(content),
			Mode: mode,
		},
	}

	mod := WithFS(fakefs)
	mod(&mc)

	got, err := mc.sysfs.Lstat(path)
	if err != nil {
		t.Fatalf("cannot Lstat %q: %v", path, err)
	}
	if got.Size() != int64(len(content)) {
		t.Fatalf("did not found content: got %v expected %v", got.Size(), content)
	}
	if got.Mode() != mode {
		t.Fatalf("found mismatched file mode: got %v expected %v", got.Mode(), mode)
	}
}

func TestWithFSFromRoot(t *testing.T) {
	mc := machine{}
	// we are gonna using real files from sysfs pseudofs, but we are using files which
	// are standard since pretty much the inception of sysfs
	mod := WithFSFromRoot("/sys")
	mod(&mc)

	srcPath := filepath.Join("devices", "system", "cpu", "online")
	got, err := fs.ReadFile(mc.sysfs, srcPath)
	if err != nil {
		t.Fatalf("cannot read %q: %v", srcPath, err)
	}
	if len(got) == 0 {
		t.Fatalf("empty content in %q", srcPath)
	}
}
