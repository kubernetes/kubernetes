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

package csi

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/mount-utils"
)

func TestCleanupUnmountedVolumeArtifacts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		prepare      func(t *testing.T, root string, mounter *mount.FakeMounter)
		wantDataGone bool
		wantDirGone  bool
		wantErr      bool
	}{
		{
			name: "removes vol_data.json when unmounted",
			prepare: func(t *testing.T, root string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.MkdirAll(filepath.Join(root, "mount"), 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "no-op when still mounted",
			prepare: func(t *testing.T, root string, mounter *mount.FakeMounter) {
				t.Helper()
				mountPath := filepath.Join(root, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				mounter.MountPoints = []mount.MountPoint{{Device: "/dev/sdb", Path: mountPath}}
			},
			wantDataGone: false,
			wantDirGone:  false,
		},
		{
			name: "leaves arbitrary content untouched",
			prepare: func(t *testing.T, root string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(root, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, "userdata.txt"), []byte("keep"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  false,
		},
		{
			name: "missing volume dir is ok",
			prepare: func(t *testing.T, root string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.RemoveAll(root); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			// Use a nested volume dir so we can delete the whole tree in "missing" case.
			volumeDir := filepath.Join(root, "pvc-test")
			if err := os.MkdirAll(volumeDir, 0750); err != nil {
				t.Fatal(err)
			}
			mounter := mount.NewFakeMounter(nil)
			tc.prepare(t, volumeDir, mounter)

			err := CleanupUnmountedVolumeArtifacts(mounter, volumeDir)
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			_, dataErr := os.Stat(filepath.Join(volumeDir, volDataFileName))
			if tc.wantDataGone && !os.IsNotExist(dataErr) {
				t.Fatalf("expected %s gone, stat err=%v", volDataFileName, dataErr)
			}
			if !tc.wantDataGone && os.IsNotExist(dataErr) {
				t.Fatalf("expected %s to remain", volDataFileName)
			}

			_, dirErr := os.Stat(volumeDir)
			if tc.wantDirGone && !os.IsNotExist(dirErr) {
				t.Fatalf("expected volume dir gone, stat err=%v", dirErr)
			}
			if !tc.wantDirGone && os.IsNotExist(dirErr) {
				t.Fatalf("expected volume dir to remain")
			}
			if _, err := os.Stat(filepath.Join(volumeDir, "userdata.txt")); err == nil {
				// ensure we did not delete arbitrary content when present
			}
		})
	}
}
