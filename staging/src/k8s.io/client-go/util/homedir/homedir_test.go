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

package homedir

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestHomeDir(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Run("windows kubeconfig priority", func(t *testing.T) {
			tmp := t.TempDir()
			kubeDir := filepath.Join(tmp, ".kube")
			if err := os.MkdirAll(kubeDir, 0755); err != nil {
				t.Fatalf("failed to create .kube dir: %v", err)
			}
			cfgFile := filepath.Join(kubeDir, "config")
			if err := os.WriteFile(cfgFile, []byte(""), 0644); err != nil {
				t.Fatalf("failed to write config: %v", err)
			}

			t.Setenv("HOME", tmp)
			t.Setenv("HOMEDRIVE", "")
			t.Setenv("HOMEPATH", "")
			t.Setenv("USERPROFILE", "")

			if got := HomeDir(); got != tmp {
				t.Errorf("HomeDir() = %q, want %q", got, tmp)
			}
		})

		t.Run("windows fallback to writeable directory", func(t *testing.T) {
			tmp := t.TempDir()
			t.Setenv("HOME", "")
			t.Setenv("HOMEDRIVE", "")
			t.Setenv("HOMEPATH", "")
			t.Setenv("USERPROFILE", tmp)

			if got := HomeDir(); got != tmp {
				t.Errorf("HomeDir() = %q, want %q", got, tmp)
			}
		})

		t.Run("windows empty env", func(t *testing.T) {
			t.Setenv("HOME", "")
			t.Setenv("HOMEDRIVE", "")
			t.Setenv("HOMEPATH", "")
			t.Setenv("USERPROFILE", "")

			if got := HomeDir(); got != "" {
				t.Errorf("HomeDir() = %q, want empty string", got)
			}
		})
	} else {
		t.Run("unix home set", func(t *testing.T) {
			fakeHome := t.TempDir()
			t.Setenv("HOME", fakeHome)

			if got := HomeDir(); got != fakeHome {
				t.Errorf("HomeDir() = %q, want %q", got, fakeHome)
			}
		})

		t.Run("unix home unset", func(t *testing.T) {
			t.Setenv("HOME", "")

			if got := HomeDir(); got != "" {
				t.Errorf("HomeDir() = %q, want empty string", got)
			}
		})
	}
}
