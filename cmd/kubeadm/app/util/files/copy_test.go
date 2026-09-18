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
	dir := t.TempDir()
	src := filepath.Join(dir, "source.conf")
	dest := filepath.Join(dir, "dest.conf")
	content := []byte("kubeadm-config")

	if err := os.WriteFile(src, content, 0640); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	if err := CopyFile(src, dest); err != nil {
		t.Fatalf("CopyFile returned an error: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("failed to read copied file: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("copied content = %q, want %q", got, content)
	}

	srcInfo, err := os.Stat(src)
	if err != nil {
		t.Fatalf("failed to stat source file: %v", err)
	}
	destInfo, err := os.Stat(dest)
	if err != nil {
		t.Fatalf("failed to stat dest file: %v", err)
	}
	if srcInfo.Mode() != destInfo.Mode() {
		t.Errorf("dest mode = %v, want %v", destInfo.Mode(), srcInfo.Mode())
	}

	if _, err := os.ReadFile(src); err != nil {
		t.Errorf("source file should still exist after CopyFile: %v", err)
	}
}

func TestCopyFileOverwritesDestination(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "source.conf")
	dest := filepath.Join(dir, "dest.conf")

	if err := os.WriteFile(src, []byte("new"), 0640); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}
	if err := os.WriteFile(dest, []byte("stale-content-longer-than-new"), 0640); err != nil {
		t.Fatalf("failed to create dest file: %v", err)
	}

	if err := CopyFile(src, dest); err != nil {
		t.Fatalf("CopyFile returned an error: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("failed to read copied file: %v", err)
	}
	if string(got) != "new" {
		t.Errorf("dest content = %q, want %q", got, "new")
	}
}

func TestCopyFileMissingSource(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "missing.conf")
	dest := filepath.Join(dir, "dest.conf")

	if err := CopyFile(src, dest); err == nil {
		t.Fatal("expected an error copying a nonexistent source file, got nil")
	}
}

func TestCopyFileInvalidDestDir(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "source.conf")
	dest := filepath.Join(dir, "missing-dir", "dest.conf")

	if err := os.WriteFile(src, []byte("data"), 0640); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	if err := CopyFile(src, dest); err == nil {
		t.Fatal("expected an error copying into a nonexistent directory, got nil")
	}
}

func TestMoveFile(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "source.conf")
	dest := filepath.Join(dir, "dest.conf")
	content := []byte("kubeadm-flags")

	if err := os.WriteFile(src, content, 0640); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	if err := MoveFile(src, dest); err != nil {
		t.Fatalf("MoveFile returned an error: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("failed to read moved file: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("moved content = %q, want %q", got, content)
	}

	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Errorf("expected source file to be gone after MoveFile, stat err: %v", err)
	}
}

func TestMoveFileMissingSource(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "missing.conf")
	dest := filepath.Join(dir, "dest.conf")

	if err := MoveFile(src, dest); err == nil {
		t.Fatal("expected an error moving a nonexistent source file, got nil")
	}
}
