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

package util

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCopyDir(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		tmp := t.TempDir()
		srcDir := filepath.Join(tmp, "srcdir")
		dstDir := filepath.Join(tmp, "dstdir")

		if err := os.MkdirAll(srcDir, 0755); err != nil {
			t.Fatalf("failed to create src dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(srcDir, "a.txt"), []byte("a"), 0644); err != nil {
			t.Fatalf("failed to write file: %v", err)
		}
		if err := os.WriteFile(filepath.Join(srcDir, "b.txt"), []byte("b"), 0644); err != nil {
			t.Fatalf("failed to write file: %v", err)
		}

		if err := CopyDir(srcDir, dstDir); err != nil {
			t.Fatalf("CopyDir() error = %v", err)
		}

		got, err := os.ReadFile(filepath.Join(dstDir, "a.txt"))
		if err != nil {
			t.Fatalf("failed to read copied file: %v", err)
		}
		if string(got) != "a" {
			t.Errorf("file a.txt content mismatch: got %q, want %q", got, "a")
		}
	})

	t.Run("existing destination", func(t *testing.T) {
		tmp := t.TempDir()
		src := filepath.Join(tmp, "src")
		dst := filepath.Join(tmp, "dst")
		file := filepath.Join("member", "db")

		if err := os.MkdirAll(filepath.Join(src, "member"), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(src, file), []byte("data"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(dst, 0700); err != nil {
			t.Fatal(err)
		}

		if err := CopyDir(src, dst); err != nil {
			t.Fatalf("CopyDir failed: %v", err)
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
	})

	t.Run("source does not exist", func(t *testing.T) {
		tmp := t.TempDir()
		if err := CopyDir(filepath.Join(tmp, "noexist"), filepath.Join(tmp, "dst")); err == nil {
			t.Error("expected error for non-existent source, got nil")
		}
	})
}
