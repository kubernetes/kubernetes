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
		nilMounter   bool
		volumeDir    func(root string) string
		prepare      func(t *testing.T, volumeDir string, mounter *mount.FakeMounter)
		wantCleaned  bool
		wantDataGone bool
		wantDirGone  bool
		wantErr      bool
		errContains  string
		skipFSAssert bool
	}{
		{
			name:         "nil mounter",
			nilMounter:   true,
			wantErr:      true,
			errContains:  "mounter is required",
			skipFSAssert: true,
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
			wantCleaned:  true,
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
			wantCleaned:  true,
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
		},
		{
			name: "IsMountPoint non-ENOENT error",
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
			wantErr:     true,
			errContains: "failed to check mount point",
		},
		{
			name: "IsMountPoint ENOENT treated as unmounted",
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
					mountPath: os.ErrNotExist,
				}
			},
			wantCleaned:  true,
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
				if err := os.WriteFile(filepath.Join(mountPath, "leftover"), []byte("x"), 0640); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantErr:     true,
			errContains: "failed to remove unmounted CSI mount path",
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
			wantCleaned:  true,
			wantDataGone: true,
		},
		{
			name: "PathExists error when volumeDir parent is a file",
			volumeDir: func(root string) string {
				return filepath.Join(root, "not-a-dir", "volume")
			},
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				parent := filepath.Dir(volumeDir)
				grand := filepath.Dir(parent)
				if err := os.MkdirAll(grand, 0750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(parent, []byte("file"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantErr:      true,
			skipFSAssert: true,
		},
		{
			name: "PathExists error for mount path when volumeDir is a file",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.RemoveAll(volumeDir); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(volumeDir, []byte("not-a-directory"), 0640); err != nil {
					t.Fatal(err)
				}
			},
			wantErr:      true,
			skipFSAssert: true,
		},
		{
			name:         "no vol_data empty dir removes dir",
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
		},
		{
			name: "unreadable vol_data.json removal fails",
			prepare: func(t *testing.T, volumeDir string, _ *mount.FakeMounter) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(volumeDir, volDataFileName), []byte(`{"vol":"x"}`), 0640); err != nil {
					t.Fatal(err)
				}
				if err := os.Chmod(volumeDir, 0500); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					_ = os.Chmod(volumeDir, 0750)
				})
			},
			wantErr:     true,
			errContains: "failed to remove CSI volume data file",
		},
	}

	for _, tc := range tests {
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
			if tc.nilMounter {
				mounter = nil
			}
			if tc.prepare != nil {
				tc.prepare(t, volumeDir, fake)
			}

			cleaned, err := CleanupUnmountedVolumeArtifacts(mounter, volumeDir)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if cleaned {
					t.Fatal("expected cleaned=false on error")
				}
				if tc.errContains != "" && !strings.Contains(err.Error(), tc.errContains) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.errContains)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			} else if cleaned != tc.wantCleaned {
				t.Fatalf("cleaned=%v, want %v", cleaned, tc.wantCleaned)
			}

			if tc.skipFSAssert {
				return
			}

			_, dataErr := os.Stat(filepath.Join(volumeDir, volDataFileName))
			dataGone := os.IsNotExist(dataErr)
			if !tc.wantErr {
				if tc.wantDataGone != dataGone {
					t.Fatalf("vol_data gone=%v, want %v (stat err=%v)", dataGone, tc.wantDataGone, dataErr)
				}
			} else if !tc.wantDataGone && dataGone {
				t.Fatalf("expected %s to remain on error", volDataFileName)
			}

			_, dirErr := os.Stat(volumeDir)
			dirGone := os.IsNotExist(dirErr)
			if !tc.wantErr && tc.wantDirGone != dirGone {
				t.Fatalf("volume dir gone=%v, want %v (stat err=%v)", dirGone, tc.wantDirGone, dirErr)
			}
		})
	}
}
