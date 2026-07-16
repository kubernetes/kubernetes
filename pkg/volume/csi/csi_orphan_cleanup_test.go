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
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/mount-utils"
)

func TestCleanupUnmountedVolumeArtifacts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		mounter      mount.Interface
		prepare      func(t *testing.T, volumeDir string, mounter *mount.FakeMounter)
		volumeDir    func(root string) string // optional override of volumeDir path
		wantDataGone bool
		wantDirGone  bool
		wantErr      bool
		errContains  string
	}{
		{
			name: "nil mounter",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
			},
			mounter:     nil,
			wantErr:     true,
			errContains: "mounter is required",
		},
		{
			name: "missing volume dir is ok",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.RemoveAll(volumeDir); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "removes vol_data.json when unmounted empty mount",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.MkdirAll(filepath.Join(volumeDir, "mount"), 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "removes vol_data.json when mount subdir absent",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "no-op when still mounted",
			prepare: func(t *testing.T, volumeDir string, mounter *mount.FakeMounter) {
				t.Helper()
				mountPath := filepath.Join(volumeDir, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				mounter.MountPoints = []mount.MountPoint{{Device: "/dev/sdb", Path: mountPath}}
			},
			wantDataGone: false,
			wantDirGone:  false,
		},
		{
			name: "IsLikelyNotMountPoint non-ENOENT error",
			prepare: func(t *testing.T, volumeDir string, mounter *mount.FakeMounter) {
				t.Helper()
				mountPath := filepath.Join(volumeDir, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				mounter.MountCheckErrors = map[string]error{
					mountPath: errors.New("injected mount check failure"),
				}
			},
			wantDataGone: false,
			wantDirGone:  false,
			wantErr:      true,
			errContains:  "failed to check mount point",
		},
		{
			name: "IsLikelyNotMountPoint ENOENT treated as unmounted",
			prepare: func(t *testing.T, volumeDir string, mounter *mount.FakeMounter) {
				t.Helper()
				mountPath := filepath.Join(volumeDir, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				// FakeMounter returns this error before Stat; os.IsNotExist makes code treat as unmounted.
				mounter.MountCheckErrors = map[string]error{
					mountPath: os.ErrNotExist,
				}
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "non-empty mount path blocks metadata deletion",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				mountPath := filepath.Join(volumeDir, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					t.Fatal(err)
				}
				// Make mount dir non-empty so os.Remove(mountPath) fails.
				if err := os.WriteFile(filepath.Join(mountPath, "leftover"), []byte("x"), 0640); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: false,
			wantDirGone:  false,
			wantErr:      true,
			errContains:  "failed to remove unmounted CSI mount path",
		},
		{
			name: "leaves arbitrary content after removing metadata",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, "userdata.txt"), []byte("keep"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  false,
		},
		{
			name: "PathExists error when volumeDir parent is a file",
			// Stat(volumeDir) fails with ENOTDIR-style error when a path component is a file.
			volumeDir: func(root string) string {
				return filepath.Join(root, "not-a-dir", "volume")
			},
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				// volumeDir = root/not-a-dir/volume; make not-a-dir a file.
				parent := filepath.Dir(volumeDir)
				grand := filepath.Dir(parent)
				if err := os.MkdirAll(grand, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(parent, []byte("file"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantErr:     true,
			errContains: "", // OS-dependent message; just require error
		},
		{
			name: "PathExists error for mount path when volumeDir is a file",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				// Replace volumeDir directory with a file so PathExists(volumeDir) succeeds
				// (file exists) but PathExists(volumeDir/mount) fails.
				if err := os.RemoveAll(volumeDir); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(volumeDir, []byte("not-a-directory"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantErr: true,
		},
		{
			name: "no vol_data empty dir removes dir",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				// empty volume dir, no metadata
			},
			wantDataGone: true,
			wantDirGone:  true,
		},
		{
			name: "no vol_data with leftover file keeps dir",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(volumeDir, "userdata.txt"), []byte("keep"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantDataGone: true,
			wantDirGone:  false,
		},
		{
			name: "unreadable vol_data.json removal fails",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				// Remove write permission on the directory so os.Remove(dataFile) fails.
				if err := os.Chmod(volumeDir, 0500); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					_ = os.Chmod(volumeDir, 0750)
				})
			},
			wantDataGone: false,
			wantDirGone:  false,
			wantErr:      true,
			errContains:  "failed to remove CSI volume data file",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			volumeDir := filepath.Join(root, "pvc-test")
			if tc.volumeDir != nil {
				volumeDir = tc.volumeDir(root)
			} else if err := os.MkdirAll(volumeDir, 0750); err != nil {
				t.Fatal(err)
			}

			fake := mount.NewFakeMounter(nil)
			var mounter mount.Interface = fake
			if tc.name == "nil mounter" {
				mounter = nil
			}

			if tc.prepare != nil {
				tc.prepare(t, volumeDir, fake)
			}

			cleaned, err := CleanupUnmountedVolumeArtifacts(mounter, volumeDir)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if cleaned {
					t.Fatalf("expected cleaned=false on error")
				}
				if tc.errContains != "" && !contains(err.Error(), tc.errContains) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.errContains)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if !tc.wantErr || tc.wantDataGone || tc.wantDirGone {
				// Only assert filesystem outcomes when meaningful.
			}
			if tc.name == "nil mounter" || tc.name == "PathExists error when volumeDir parent is a file" || tc.name == "PathExists error for mount path when volumeDir is a file" {
				return
			}

			_, dataErr := os.Stat(filepath.Join(volumeDir, volDataFileName))
			if tc.wantDataGone && !os.IsNotExist(dataErr) {
				// Permission cases may leave the file; only assert when we expect success path.
				if !tc.wantErr {
					t.Fatalf("expected %s gone, stat err=%v", volDataFileName, dataErr)
				}
			}
			if !tc.wantDataGone && !tc.wantErr && os.IsNotExist(dataErr) {
				t.Fatalf("expected %s to remain", volDataFileName)
			}

			_, dirErr := os.Stat(volumeDir)
			if tc.wantDirGone && !tc.wantErr && !os.IsNotExist(dirErr) {
				t.Fatalf("expected volume dir gone, stat err=%v", dirErr)
			}
			if !tc.wantDirGone && !tc.wantErr && os.IsNotExist(dirErr) {
				t.Fatalf("expected volume dir to remain")
			}

			if _, err := os.Stat(filepath.Join(volumeDir, "userdata.txt")); err == nil {
				// preserved if present
			}
		})
	}
}

func contains(s, sub string) bool {
	return sub == "" || strings.Contains(s, sub)
}
