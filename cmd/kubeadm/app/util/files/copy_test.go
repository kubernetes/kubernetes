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

package files

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCopyDir(t *testing.T) {
	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "src")
	dst := filepath.Join(tmpDir, "dst")
	file := filepath.Join("member", "db")

	if err := os.MkdirAll(filepath.Join(src, "member"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, file), []byte("data"), 0600); err != nil {
		t.Fatal(err)
	}
	// dst already exists, as it does for both the etcd backup and restore.
	if err := os.Mkdir(dst, 0700); err != nil {
		t.Fatal(err)
	}

	if output, err := CopyDir(src, dst); err != nil {
		t.Fatalf("CopyDir failed: %v, output: %q", err, output)
	}

	content, err := os.ReadFile(filepath.Join(dst, file))
	if err != nil {
		t.Fatalf("expected the contents of src to be copied into dst: %v", err)
	}
	if string(content) != "data" {
		t.Errorf("expected %q, got %q", "data", string(content))
	}
	if _, err := os.Stat(filepath.Join(dst, "src")); !os.IsNotExist(err) {
		t.Errorf("expected src not to be nested in dst, got: %v", err)
	}
}
