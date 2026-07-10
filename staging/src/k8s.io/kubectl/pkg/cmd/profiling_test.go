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

package cmd

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestHeapProfilingCreatesOutputWithOwnerOnlyPermissions(t *testing.T) {
	oldProfileName, oldProfileOutput := profileName, profileOutput
	t.Cleanup(func() {
		profileName = oldProfileName
		profileOutput = oldProfileOutput
	})

	oldUmask := syscall.Umask(0022)
	t.Cleanup(func() {
		syscall.Umask(oldUmask)
	})

	profileName = "heap"
	profileOutput = filepath.Join(t.TempDir(), "heap.pprof")

	if err := flushProfiling(nil); err != nil {
		t.Fatalf("unexpected error flushing profile: %v", err)
	}

	fileInfo, err := os.Stat(profileOutput)
	if err != nil {
		t.Fatalf("unexpected error stating profile output: %v", err)
	}
	if fileInfo.Mode().Perm() != 0600 {
		t.Errorf("expected file mode: %v, saw: %v", os.FileMode(0600), fileInfo.Mode().Perm())
	}
}
