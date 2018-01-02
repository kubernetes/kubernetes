package fs

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/containerd/containerd/fs/fstest"
	"github.com/pkg/errors"
)

// TODO: Additional tests
// - capability test (requires privilege)
// - chown test (requires privilege)
// - symlink test
// - hardlink test

func TestSimpleDiff(t *testing.T) {
	l1 := fstest.Apply(
		fstest.CreateDir("/etc", 0755),
		fstest.CreateFile("/etc/hosts", []byte("mydomain 10.0.0.1"), 0644),
		fstest.CreateFile("/etc/profile", []byte("PATH=/usr/bin"), 0644),
		fstest.CreateFile("/etc/unchanged", []byte("PATH=/usr/bin"), 0644),
		fstest.CreateFile("/etc/unexpected", []byte("#!/bin/sh"), 0644),
	)
	l2 := fstest.Apply(
		fstest.CreateFile("/etc/hosts", []byte("mydomain 10.0.0.120"), 0644),
		fstest.CreateFile("/etc/profile", []byte("PATH=/usr/bin"), 0666),
		fstest.CreateDir("/root", 0700),
		fstest.CreateFile("/root/.bashrc", []byte("PATH=/usr/sbin:/usr/bin"), 0644),
		fstest.Remove("/etc/unexpected"),
	)
	diff := []TestChange{
		Modify("/etc/hosts"),
		Modify("/etc/profile"),
		Delete("/etc/unexpected"),
		Add("/root"),
		Add("/root/.bashrc"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}

func TestDirectoryReplace(t *testing.T) {
	l1 := fstest.Apply(
		fstest.CreateDir("/dir1", 0755),
		fstest.CreateFile("/dir1/f1", []byte("#####"), 0644),
		fstest.CreateDir("/dir1/f2", 0755),
		fstest.CreateFile("/dir1/f2/f3", []byte("#!/bin/sh"), 0644),
	)
	l2 := fstest.Apply(
		fstest.CreateFile("/dir1/f11", []byte("#New file here"), 0644),
		fstest.RemoveAll("/dir1/f2"),
		fstest.CreateFile("/dir1/f2", []byte("Now file"), 0666),
	)
	diff := []TestChange{
		Add("/dir1/f11"),
		Modify("/dir1/f2"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}

func TestRemoveDirectoryTree(t *testing.T) {
	l1 := fstest.Apply(
		fstest.CreateDir("/dir1/dir2/dir3", 0755),
		fstest.CreateFile("/dir1/f1", []byte("f1"), 0644),
		fstest.CreateFile("/dir1/dir2/f2", []byte("f2"), 0644),
	)
	l2 := fstest.Apply(
		fstest.RemoveAll("/dir1"),
	)
	diff := []TestChange{
		Delete("/dir1"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}

func TestFileReplace(t *testing.T) {
	l1 := fstest.Apply(
		fstest.CreateFile("/dir1", []byte("a file, not a directory"), 0644),
	)
	l2 := fstest.Apply(
		fstest.Remove("/dir1"),
		fstest.CreateDir("/dir1/dir2", 0755),
		fstest.CreateFile("/dir1/dir2/f1", []byte("also a file"), 0644),
	)
	diff := []TestChange{
		Modify("/dir1"),
		Add("/dir1/dir2"),
		Add("/dir1/dir2/f1"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}

func TestParentDirectoryPermission(t *testing.T) {
	l1 := fstest.Apply(
		fstest.CreateDir("/dir1", 0700),
		fstest.CreateDir("/dir2", 0751),
		fstest.CreateDir("/dir3", 0777),
	)
	l2 := fstest.Apply(
		fstest.CreateDir("/dir1/d", 0700),
		fstest.CreateFile("/dir1/d/f", []byte("irrelevant"), 0644),
		fstest.CreateFile("/dir1/f", []byte("irrelevant"), 0644),
		fstest.CreateFile("/dir2/f", []byte("irrelevant"), 0644),
		fstest.CreateFile("/dir3/f", []byte("irrelevant"), 0644),
	)
	diff := []TestChange{
		Add("/dir1/d"),
		Add("/dir1/d/f"),
		Add("/dir1/f"),
		Add("/dir2/f"),
		Add("/dir3/f"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}
func TestUpdateWithSameTime(t *testing.T) {
	tt := time.Now().Truncate(time.Second)
	t1 := tt.Add(5 * time.Nanosecond)
	t2 := tt.Add(6 * time.Nanosecond)
	l1 := fstest.Apply(
		fstest.CreateFile("/file-modified-time", []byte("1"), 0644),
		fstest.Chtime("/file-modified-time", t1),
		fstest.CreateFile("/file-no-change", []byte("1"), 0644),
		fstest.Chtime("/file-no-change", t1),
		fstest.CreateFile("/file-same-time", []byte("1"), 0644),
		fstest.Chtime("/file-same-time", t1),
		fstest.CreateFile("/file-truncated-time-1", []byte("1"), 0644),
		fstest.Chtime("/file-truncated-time-1", t1),
		fstest.CreateFile("/file-truncated-time-2", []byte("1"), 0644),
		fstest.Chtime("/file-truncated-time-2", tt),
	)
	l2 := fstest.Apply(
		fstest.CreateFile("/file-modified-time", []byte("2"), 0644),
		fstest.Chtime("/file-modified-time", t2),
		fstest.CreateFile("/file-no-change", []byte("1"), 0644),
		fstest.Chtime("/file-no-change", tt), // use truncated time, should be regarded as no change
		fstest.CreateFile("/file-same-time", []byte("2"), 0644),
		fstest.Chtime("/file-same-time", t1),
		fstest.CreateFile("/file-truncated-time-1", []byte("2"), 0644),
		fstest.Chtime("/file-truncated-time-1", tt),
		fstest.CreateFile("/file-truncated-time-2", []byte("2"), 0644),
		fstest.Chtime("/file-truncated-time-2", tt),
	)
	diff := []TestChange{
		// "/file-same-time" excluded because matching non-zero nanosecond values
		Modify("/file-modified-time"),
		Modify("/file-truncated-time-1"),
		Modify("/file-truncated-time-2"),
	}

	if err := testDiffWithBase(l1, l2, diff); err != nil {
		t.Fatalf("Failed diff with base: %+v", err)
	}
}

func testDiffWithBase(base, diff fstest.Applier, expected []TestChange) error {
	t1, err := ioutil.TempDir("", "diff-with-base-lower-")
	if err != nil {
		return errors.Wrap(err, "failed to create temp dir")
	}
	defer os.RemoveAll(t1)
	t2, err := ioutil.TempDir("", "diff-with-base-upper-")
	if err != nil {
		return errors.Wrap(err, "failed to create temp dir")
	}
	defer os.RemoveAll(t2)

	if err := base.Apply(t1); err != nil {
		return errors.Wrap(err, "failed to apply base filesystem")
	}

	if err := CopyDir(t2, t1); err != nil {
		return errors.Wrap(err, "failed to copy base directory")
	}

	if err := diff.Apply(t2); err != nil {
		return errors.Wrap(err, "failed to apply diff filesystem")
	}

	changes, err := collectChanges(t1, t2)
	if err != nil {
		return errors.Wrap(err, "failed to collect changes")
	}

	return checkChanges(t2, changes, expected)
}

func TestBaseDirectoryChanges(t *testing.T) {
	apply := fstest.Apply(
		fstest.CreateDir("/etc", 0755),
		fstest.CreateFile("/etc/hosts", []byte("mydomain 10.0.0.1"), 0644),
		fstest.CreateFile("/etc/profile", []byte("PATH=/usr/bin"), 0644),
		fstest.CreateDir("/root", 0700),
		fstest.CreateFile("/root/.bashrc", []byte("PATH=/usr/sbin:/usr/bin"), 0644),
	)
	changes := []TestChange{
		Add("/etc"),
		Add("/etc/hosts"),
		Add("/etc/profile"),
		Add("/root"),
		Add("/root/.bashrc"),
	}

	if err := testDiffWithoutBase(apply, changes); err != nil {
		t.Fatalf("Failed diff without base: %+v", err)
	}
}

func testDiffWithoutBase(apply fstest.Applier, expected []TestChange) error {
	tmp, err := ioutil.TempDir("", "diff-without-base-")
	if err != nil {
		return errors.Wrap(err, "failed to create temp dir")
	}
	defer os.RemoveAll(tmp)

	if err := apply.Apply(tmp); err != nil {
		return errors.Wrap(err, "failed to apply filesytem changes")
	}

	changes, err := collectChanges("", tmp)
	if err != nil {
		return errors.Wrap(err, "failed to collect changes")
	}

	return checkChanges(tmp, changes, expected)
}

func checkChanges(root string, changes, expected []TestChange) error {
	if len(changes) != len(expected) {
		return errors.Errorf("Unexpected number of changes:\n%s", diffString(changes, expected))
	}
	for i := range changes {
		if changes[i].Path != expected[i].Path || changes[i].Kind != expected[i].Kind {
			return errors.Errorf("Unexpected change at %d:\n%s", i, diffString(changes, expected))
		}
		if changes[i].Kind != ChangeKindDelete {
			filename := filepath.Join(root, changes[i].Path)
			efi, err := os.Stat(filename)
			if err != nil {
				return errors.Wrapf(err, "failed to stat %q", filename)
			}
			afi := changes[i].FileInfo
			if afi.Size() != efi.Size() {
				return errors.Errorf("Unexpected change size %d, %q has size %d", afi.Size(), filename, efi.Size())
			}
			if afi.Mode() != efi.Mode() {
				return errors.Errorf("Unexpected change mode %s, %q has mode %s", afi.Mode(), filename, efi.Mode())
			}
			if afi.ModTime() != efi.ModTime() {
				return errors.Errorf("Unexpected change modtime %s, %q has modtime %s", afi.ModTime(), filename, efi.ModTime())
			}
			if expected := filepath.Join(root, changes[i].Path); changes[i].Source != expected {
				return errors.Errorf("Unexpected source path %s, expected %s", changes[i].Source, expected)
			}
		}
	}

	return nil
}

type TestChange struct {
	Kind     ChangeKind
	Path     string
	FileInfo os.FileInfo
	Source   string
}

func collectChanges(a, b string) ([]TestChange, error) {
	changes := []TestChange{}
	err := Changes(context.Background(), a, b, func(k ChangeKind, p string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		changes = append(changes, TestChange{
			Kind:     k,
			Path:     p,
			FileInfo: f,
			Source:   filepath.Join(b, p),
		})
		return nil
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to compute changes")
	}

	return changes, nil
}

func diffString(c1, c2 []TestChange) string {
	return fmt.Sprintf("got(%d):\n%s\nexpected(%d):\n%s", len(c1), changesString(c1), len(c2), changesString(c2))

}

func changesString(c []TestChange) string {
	strs := make([]string, len(c))
	for i := range c {
		strs[i] = fmt.Sprintf("\t%s\t%s", c[i].Kind, c[i].Path)
	}
	return strings.Join(strs, "\n")
}

func Add(p string) TestChange {
	return TestChange{
		Kind: ChangeKindAdd,
		Path: p,
	}
}

func Delete(p string) TestChange {
	return TestChange{
		Kind: ChangeKindDelete,
		Path: p,
	}
}

func Modify(p string) TestChange {
	return TestChange{
		Kind: ChangeKindModify,
		Path: p,
	}
}
