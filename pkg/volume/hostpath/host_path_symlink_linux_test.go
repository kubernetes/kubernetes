//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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

package hostpath

import (
	"os"
	"path/filepath"
	"testing"
)

// TestHostPathOrCreateBlocksParentSymlinkRedirect verifies that *OrCreate helpers
// do not create files under a symlink target outside the declared path prefix.
func TestHostPathOrCreateBlocksParentSymlinkRedirect(t *testing.T) {
	trap := t.TempDir()
	proof := t.TempDir()
	if err := os.Symlink(proof, filepath.Join(trap, "esc")); err != nil {
		t.Fatal(err)
	}

	target := filepath.Join(trap, "esc", "marker")
	err := makeFile(target)
	if err == nil {
		if _, statErr := os.Stat(filepath.Join(proof, "marker")); statErr == nil {
			t.Fatal("makeFile followed symlink redirect to create outside trap prefix")
		}
	}
}
