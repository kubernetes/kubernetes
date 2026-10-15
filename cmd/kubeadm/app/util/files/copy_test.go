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
	"strings"
	"testing"
)

func TestCopyFile(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T, tmp string) (src, dest string)
		wantErr bool
	}{
		{
			name: "success",
			setup: func(t *testing.T, tmp string) (string, string) {
				src := filepath.Join(tmp, "src.txt")
				dest := filepath.Join(tmp, "dest.txt")
				if err := os.WriteFile(src, []byte("hello world"), 0644); err != nil {
					t.Fatalf("failed to write source: %v", err)
				}
				return src, dest
			},
		},
		{
			name: "source does not exist",
			setup: func(t *testing.T, tmp string) (string, string) {
				return filepath.Join(tmp, "noexist.txt"), filepath.Join(tmp, "dest.txt")
			},
			wantErr: true,
		},
		{
			name: "dest in non-existent directory",
			setup: func(t *testing.T, tmp string) (string, string) {
				src := filepath.Join(tmp, "src.txt")
				if err := os.WriteFile(src, []byte("test"), 0644); err != nil {
					t.Fatalf("failed to write source: %v", err)
				}
				return src, filepath.Join(tmp, "nonexistent", "dest.txt")
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmp := t.TempDir()
			src, dest := tt.setup(t, tmp)
			err := CopyFile(src, dest)
			if (err != nil) != tt.wantErr {
				t.Errorf("CopyFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCopyFile_ContentAndMode(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "src.txt")
	dest := filepath.Join(tmp, "dest.txt")
	content := []byte("hello kubeadm")

	if err := os.WriteFile(src, content, 0644); err != nil {
		t.Fatalf("failed to write source: %v", err)
	}

	if err := CopyFile(src, dest); err != nil {
		t.Fatalf("CopyFile() error = %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("failed to read dest: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("content mismatch: got %q, want %q", got, content)
	}

	srcInfo, _ := os.Stat(src)
	destInfo, _ := os.Stat(dest)
	if srcInfo.Mode() != destInfo.Mode() {
		t.Errorf("mode not preserved: src %v, dest %v", srcInfo.Mode(), destInfo.Mode())
	}
}

func TestMoveFile(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T, tmp string) (src, dest string)
		wantErr bool
	}{
		{
			name: "success",
			setup: func(t *testing.T, tmp string) (string, string) {
				src := filepath.Join(tmp, "src.txt")
				dest := filepath.Join(tmp, "dest.txt")
				if err := os.WriteFile(src, []byte("move me"), 0644); err != nil {
					t.Fatalf("failed to write source: %v", err)
				}
				return src, dest
			},
		},
		{
			name: "source does not exist",
			setup: func(t *testing.T, tmp string) (string, string) {
				return filepath.Join(tmp, "noexist.txt"), filepath.Join(tmp, "dest.txt")
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmp := t.TempDir()
			src, dest := tt.setup(t, tmp)
			err := MoveFile(src, dest)
			if (err != nil) != tt.wantErr {
				t.Errorf("MoveFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMoveFile_ContentPreserved(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "src.txt")
	dest := filepath.Join(tmp, "dest.txt")
	content := []byte("move and verify")

	if err := os.WriteFile(src, content, 0644); err != nil {
		t.Fatalf("failed to write source: %v", err)
	}

	if err := MoveFile(src, dest); err != nil {
		t.Fatalf("MoveFile() error = %v", err)
	}

	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Error("source file should not exist after move")
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("failed to read dest: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("content mismatch: got %q, want %q", got, content)
	}
}

// TestMoveFile_CrossDevice exercises the fallback copy+delete path
// triggered when os.Rename fails with "invalid cross-device link".
// On Linux the error string contains "invalid cross-device link",
// which triggers the fallback. On macOS the string is "cross-device link"
// (no "invalid"), so the fallback is not entered and the raw error is returned.
func TestMoveFile_CrossDevice(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "src.txt")
	// /dev is on devfs, a different filesystem from tmp.
	dest := "/dev/kubeadm_test_copy"

	if err := os.WriteFile(src, []byte("cross-device"), 0644); err != nil {
		t.Fatalf("failed to write source: %v", err)
	}

	err := MoveFile(src, dest)
	if err == nil {
		_ = os.Remove(dest)
		t.Skip("cross-device copy succeeded unexpectedly; skipping error assertion")
	}

	// On Linux: the fallback path triggers, CopyFile fails (can't write to /dev),
	// and the error is wrapped with "failed to copy file".
	// On macOS: the fallback path is not triggered (different error string),
	// and the raw rename error is returned.
	if !strings.Contains(err.Error(), "failed to copy file") &&
		!strings.Contains(err.Error(), "cross-device link") {
		t.Errorf("expected cross-device or wrapped copy error, got: %v", err)
	}
}

func TestCopyFile_SourceNotReadable(t *testing.T) {
	// os.Stat succeeds but os.Open fails due to insufficient permissions.
	// Skipped when running as root (root bypasses permission checks).
	if os.Geteuid() == 0 {
		t.Skip("running as root; permission test is not meaningful")
	}
	tmp := t.TempDir()
	src := filepath.Join(tmp, "noperm.txt")
	dest := filepath.Join(tmp, "dest.txt")
	if err := os.WriteFile(src, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to write source: %v", err)
	}
	if err := os.Chmod(src, 0); err != nil {
		t.Fatalf("failed to chmod source: %v", err)
	}
	defer func() {
		_ = os.Chmod(src, 0644)
	}()

	err := CopyFile(src, dest)
	if err == nil {
		t.Error("expected error for unreadable source, got nil")
	}
}

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

		out, err := CopyDir(srcDir, dstDir)
		if err != nil {
			t.Fatalf("CopyDir() error = %v, output: %s", err, out)
		}

		got, err := os.ReadFile(filepath.Join(dstDir, "a.txt"))
		if err != nil {
			t.Fatalf("failed to read copied file: %v", err)
		}
		if string(got) != "a" {
			t.Errorf("file a.txt content mismatch: got %q, want %q", got, "a")
		}
	})

	t.Run("source does not exist", func(t *testing.T) {
		tmp := t.TempDir()
		out, err := CopyDir(filepath.Join(tmp, "noexist"), filepath.Join(tmp, "dst"))
		if err == nil {
			t.Error("expected error for non-existent source, got nil")
		}
		if len(out) == 0 {
			t.Error("expected non-empty output on error")
		}
	})
}
