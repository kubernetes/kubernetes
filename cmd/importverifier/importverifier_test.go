/*
Copyright 2026 The Kubernetes Authors.

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

package main

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadImportRestrictions(t *testing.T) {
	t.Run("missing config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "does-not-exist.json")
		_, err := loadImportRestrictions(path)
		if err == nil {
			t.Fatal("expected an error for a missing config file, got nil")
		}
		// %w keeps the cause inspectable, so callers can match on it directly.
		if !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("expected error to wrap fs.ErrNotExist, got: %v", err)
		}
		if !strings.Contains(err.Error(), "failed to load configuration from") {
			t.Errorf("expected error to mention the config path, got: %v", err)
		}
	})

	t.Run("malformed config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.yaml")
		if err := os.WriteFile(path, []byte("this is not a list"), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := loadImportRestrictions(path)
		if err == nil {
			t.Fatal("expected an error for a malformed config file, got nil")
		}
		if !strings.Contains(err.Error(), "failed to unmarshal from") {
			t.Errorf("expected an unmarshal error, got: %v", err)
		}
		if errors.Unwrap(err) == nil {
			t.Error("expected the unmarshal error to be wrapped, but errors.Unwrap returned nil")
		}
	})

	t.Run("valid config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "good.yaml")
		content := `- baseImportPath: k8s.io/kubernetes/pkg/
  allowedImports:
  - k8s.io/kubernetes/pkg/
`
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		restrictions, err := loadImportRestrictions(path)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(restrictions) != 1 {
			t.Fatalf("expected 1 import restriction, got %d", len(restrictions))
		}
		if got := restrictions[0].BaseDir; got != "k8s.io/kubernetes/pkg/" {
			t.Errorf("unexpected BaseDir: got %q", got)
		}
	})
}
