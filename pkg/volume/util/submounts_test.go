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

package util

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"k8s.io/mount-utils"
)

// recordingMounter records every Unmount call in order. It does not delegate
// to the fake mounter so that stacked mounts on the same path stay visible.
type recordingMounter struct {
	*mount.FakeMounter
	listErr   error
	unmountFn func(target string) error
	unmounted []string
}

func newRecordingMounter(mountPoints []mount.MountPoint) *recordingMounter {
	return &recordingMounter{FakeMounter: mount.NewFakeMounter(mountPoints)}
}

func (m *recordingMounter) List() ([]mount.MountPoint, error) {
	if m.listErr != nil {
		return nil, m.listErr
	}
	return m.FakeMounter.List()
}

func (m *recordingMounter) Unmount(target string) error {
	m.unmounted = append(m.unmounted, target)
	if m.unmountFn != nil {
		return m.unmountFn(target)
	}
	return nil
}

// resolvedTempDir returns a temp dir with symlinks resolved, mirroring the
// canonical paths found in the mount table.
func resolvedTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("failed to resolve temp dir: %v", err)
	}
	return dir
}

func TestUnmountSubmounts(t *testing.T) {
	// A real volume dir on disk; the helper resolves it before looking at
	// the mount table.
	tmpDir := resolvedTempDir(t)
	volDir := filepath.Join(tmpDir, "vol")
	if err := os.MkdirAll(volDir, 0755); err != nil {
		t.Fatalf("failed to create volume dir: %v", err)
	}

	// Each case is a fake mount table and the exact unmount calls, in order,
	// that the helper must make for volDir.
	testCases := map[string]struct {
		mountPoints     []mount.MountPoint
		expectedUnmount []string
	}{
		"no mounts": {},
		"mounts outside the volume dir are left alone": {
			mountPoints: []mount.MountPoint{
				{Path: "/other/path/file.txt", Device: "/dev/sda"},
				{Path: tmpDir, Device: "/dev/sdb"},
				// Shares a prefix with volDir but is a sibling, not a child.
				{Path: volDir + "-other", Device: "/dev/sdc"},
			},
		},
		"the volume dir itself is left alone": {
			mountPoints: []mount.MountPoint{
				{Path: volDir, Device: "tmpfs"},
			},
		},
		"a single file submount is unmounted": {
			mountPoints: []mount.MountPoint{
				{Path: filepath.Join(volDir, "log.txt"), Device: "/dev/sda"},
			},
			expectedUnmount: []string{filepath.Join(volDir, "log.txt")},
		},
		"nested submounts are unmounted deepest first": {
			mountPoints: []mount.MountPoint{
				{Path: filepath.Join(volDir, "a"), Device: "/dev/a"},
				{Path: filepath.Join(volDir, "a", "b", "c"), Device: "/dev/c"},
				{Path: filepath.Join(volDir, "a", "b"), Device: "/dev/b"},
			},
			expectedUnmount: []string{
				filepath.Join(volDir, "a", "b", "c"),
				filepath.Join(volDir, "a", "b"),
				filepath.Join(volDir, "a"),
			},
		},
		"stacked mounts on the same path are each unmounted": {
			mountPoints: []mount.MountPoint{
				{Path: filepath.Join(volDir, "log.txt"), Device: "/dev/a"},
				{Path: filepath.Join(volDir, "log.txt"), Device: "/dev/b"},
			},
			expectedUnmount: []string{
				filepath.Join(volDir, "log.txt"),
				filepath.Join(volDir, "log.txt"),
			},
		},
		"mix of mounts inside and outside the volume": {
			mountPoints: []mount.MountPoint{
				{Path: "/other/path", Device: "/dev/other"},
				{Path: filepath.Join(volDir, "inside.txt"), Device: "/dev/a"},
				{Path: volDir, Device: "tmpfs"},
			},
			expectedUnmount: []string{filepath.Join(volDir, "inside.txt")},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// Load the fake mount table, run the helper, compare the unmount
			// calls it made against the expected order.
			mounter := newRecordingMounter(tc.mountPoints)
			if err := UnmountSubmounts(mounter, volDir); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !slices.Equal(mounter.unmounted, tc.expectedUnmount) {
				t.Errorf("unmounted paths = %v, want %v", mounter.unmounted, tc.expectedUnmount)
			}
		})
	}
}

func TestUnmountSubmountsResolvesSymlinks(t *testing.T) {
	// Build real-root/vol on disk and a link-root symlink pointing at
	// real-root, mimicking a symlinked kubelet root directory.
	tmpDir := resolvedTempDir(t)
	realRoot := filepath.Join(tmpDir, "real-root")
	volDir := filepath.Join(realRoot, "vol")
	if err := os.MkdirAll(volDir, 0755); err != nil {
		t.Fatalf("failed to create volume dir: %v", err)
	}
	linkRoot := filepath.Join(tmpDir, "link-root")
	if err := os.Symlink(realRoot, linkRoot); err != nil {
		t.Fatalf("failed to create symlink: %v", err)
	}

	// The mount table lists the canonical path; the caller uses the symlink.
	leaked := filepath.Join(volDir, "log.txt")
	mounter := newRecordingMounter([]mount.MountPoint{{Path: leaked, Device: "/dev/a"}})

	// Call through the symlink and expect the canonical path to be unmounted.
	if err := UnmountSubmounts(mounter, filepath.Join(linkRoot, "vol")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := []string{leaked}; !slices.Equal(mounter.unmounted, want) {
		t.Errorf("unmounted paths = %v, want %v", mounter.unmounted, want)
	}
}

func TestUnmountSubmountsMissingDir(t *testing.T) {
	// A volume dir that no longer exists has nothing to unmount, even if the
	// mount table still lists something under it.
	mounter := newRecordingMounter([]mount.MountPoint{
		{Path: "/does/not/exist/vol/log.txt", Device: "/dev/a"},
	})
	if err := UnmountSubmounts(mounter, "/does/not/exist/vol"); err != nil {
		t.Fatalf("expected no error for a missing dir, got: %v", err)
	}
	if len(mounter.unmounted) != 0 {
		t.Errorf("expected no unmounts for a missing dir, got %v", mounter.unmounted)
	}
}

func TestUnmountSubmountsCollectsErrors(t *testing.T) {
	// Two submounts, and every unmount fails with EBUSY.
	volDir := resolvedTempDir(t)
	first := filepath.Join(volDir, "first")
	second := filepath.Join(volDir, "second")
	mounter := newRecordingMounter([]mount.MountPoint{
		{Path: first, Device: "/dev/a"},
		{Path: second, Device: "/dev/b"},
	})
	busy := errors.New("device or resource busy")
	mounter.unmountFn = func(string) error { return busy }

	// The returned error must wrap the cause and name both paths.
	err := UnmountSubmounts(mounter, volDir)
	if err == nil {
		t.Fatal("expected an error when unmount fails, got nil")
	}
	if !errors.Is(err, busy) {
		t.Errorf("expected error to wrap the unmount error, got: %v", err)
	}
	for _, p := range []string{first, second} {
		if !strings.Contains(err.Error(), p) {
			t.Errorf("expected error to mention %s, got: %v", p, err)
		}
	}
	// A failure on one submount must not stop the others from being tried.
	if want := []string{second, first}; !slices.Equal(mounter.unmounted, want) {
		t.Errorf("unmounted paths = %v, want %v", mounter.unmounted, want)
	}
}

func TestUnmountSubmountsListError(t *testing.T) {
	// If the mount table cannot be read, that error is returned as-is.
	volDir := resolvedTempDir(t)
	listErr := errors.New("cannot read mount table")
	mounter := newRecordingMounter(nil)
	mounter.listErr = listErr

	err := UnmountSubmounts(mounter, volDir)
	if !errors.Is(err, listErr) {
		t.Errorf("expected error to wrap the list error, got: %v", err)
	}
}
