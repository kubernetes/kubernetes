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

func TestCopyFile(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T, tmp string) (src, dest string)
		wantErr bool
		check   func(t *testing.T, src, dest string)
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
			check: func(t *testing.T, src, dest string) {
				want := []byte("hello world")
				got, err := os.ReadFile(dest)
				if err != nil {
					t.Fatalf("failed to read dest: %v", err)
				}
				if string(got) != string(want) {
					t.Errorf("content mismatch: got %q, want %q", got, want)
				}

				srcInfo, _ := os.Stat(src)
				destInfo, _ := os.Stat(dest)
				if srcInfo.Mode() != destInfo.Mode() {
					t.Errorf("mode not preserved: src %v, dest %v", srcInfo.Mode(), destInfo.Mode())
				}
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
			if err == nil && tt.check != nil {
				tt.check(t, src, dest)
			}
		})
	}
}

func TestMoveFile(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T, tmp string) (src, dest string)
		wantErr bool
		check   func(t *testing.T, src, dest string)
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
			check: func(t *testing.T, src, dest string) {
				if _, err := os.Stat(src); !os.IsNotExist(err) {
					t.Error("source file should not exist after move")
				}

				want := []byte("move me")
				got, err := os.ReadFile(dest)
				if err != nil {
					t.Fatalf("failed to read dest: %v", err)
				}
				if string(got) != string(want) {
					t.Errorf("content mismatch: got %q, want %q", got, want)
				}
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
			if err == nil && tt.check != nil {
				tt.check(t, src, dest)
			}
		})
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
