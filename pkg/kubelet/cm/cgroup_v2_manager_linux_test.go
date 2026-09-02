//go:build linux

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

package cm

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"github.com/stretchr/testify/require"
)

func TestWriteChangedCgroupFiles(t *testing.T) {
	// Let WriteFile use a non-cgroupfs directory and truncate on write, as
	// cgroupfs replaces a file's content.
	libcontainercgroups.TestMode = true
	defer func() { libcontainercgroups.TestMode = false }()
	// Backdate each fixture file so a newer modification time marks a write.
	unwritten := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	tests := []struct {
		name        string
		files       map[string]string
		values      map[string]string
		wantFiles   map[string]string
		wantWritten []string
		wantErr     bool
	}{
		{
			name:        "writes files whose content differs",
			files:       map[string]string{"cgroup.max.descendants": "max\n", "cgroup.max.depth": "max\n"},
			values:      map[string]string{"cgroup.max.descendants": "250", "cgroup.max.depth": "50"},
			wantFiles:   map[string]string{"cgroup.max.descendants": "250", "cgroup.max.depth": "50"},
			wantWritten: []string{"cgroup.max.descendants", "cgroup.max.depth"},
		},
		{
			name:      "leaves files that already hold the value",
			files:     map[string]string{"cgroup.max.descendants": "250\n", "cgroup.max.depth": "50\n"},
			values:    map[string]string{"cgroup.max.descendants": "250", "cgroup.max.depth": "50"},
			wantFiles: map[string]string{"cgroup.max.descendants": "250\n", "cgroup.max.depth": "50\n"},
		},
		{
			name:        "writes only the file that differs",
			files:       map[string]string{"cgroup.max.descendants": "250\n", "cgroup.max.depth": "max\n"},
			values:      map[string]string{"cgroup.max.descendants": "250", "cgroup.max.depth": "50"},
			wantFiles:   map[string]string{"cgroup.max.descendants": "250\n", "cgroup.max.depth": "50"},
			wantWritten: []string{"cgroup.max.depth"},
		},
		{
			name:    "fails when a file cannot be read",
			files:   map[string]string{},
			values:  map[string]string{"cgroup.max.descendants": "250"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for file, content := range tt.files {
				path := filepath.Join(dir, file)
				require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
				require.NoError(t, os.Chtimes(path, unwritten, unwritten))
			}

			err := writeChangedCgroupFiles(dir, tt.values)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)

			var written []string
			for file, want := range tt.wantFiles {
				path := filepath.Join(dir, file)
				got, err := os.ReadFile(path)
				require.NoError(t, err)
				require.Equal(t, want, string(got), file)
				info, err := os.Stat(path)
				require.NoError(t, err)
				if !info.ModTime().Equal(unwritten) {
					written = append(written, file)
				}
			}
			require.ElementsMatch(t, tt.wantWritten, written)
		})
	}
}
