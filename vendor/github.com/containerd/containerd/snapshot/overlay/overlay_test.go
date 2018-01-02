// +build linux

package overlay

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/snapshot"
	"github.com/containerd/containerd/snapshot/storage"
	"github.com/containerd/containerd/snapshot/testsuite"
	"github.com/containerd/containerd/testutil"
)

func newSnapshotter(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error) {
	snapshotter, err := NewSnapshotter(root)
	if err != nil {
		return nil, nil, err
	}

	return snapshotter, nil, nil
}

func TestOverlay(t *testing.T) {
	testutil.RequiresRoot(t)
	testsuite.SnapshotterSuite(t, "Overlay", newSnapshotter)
}

func TestOverlayMounts(t *testing.T) {
	ctx := context.TODO()
	root, err := ioutil.TempDir("", "overlay")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	o, _, err := newSnapshotter(ctx, root)
	if err != nil {
		t.Error(err)
		return
	}
	mounts, err := o.Prepare(ctx, "/tmp/test", "")
	if err != nil {
		t.Error(err)
		return
	}
	if len(mounts) != 1 {
		t.Errorf("should only have 1 mount but received %d", len(mounts))
	}
	m := mounts[0]
	if m.Type != "bind" {
		t.Errorf("mount type should be bind but received %q", m.Type)
	}
	expected := filepath.Join(root, "snapshots", "1", "fs")
	if m.Source != expected {
		t.Errorf("expected source %q but received %q", expected, m.Source)
	}
	if m.Options[0] != "rw" {
		t.Errorf("expected mount option rw but received %q", m.Options[0])
	}
	if m.Options[1] != "rbind" {
		t.Errorf("expected mount option rbind but received %q", m.Options[1])
	}
}

func TestOverlayCommit(t *testing.T) {
	ctx := context.TODO()
	root, err := ioutil.TempDir("", "overlay")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	o, _, err := newSnapshotter(ctx, root)
	if err != nil {
		t.Error(err)
		return
	}
	key := "/tmp/test"
	mounts, err := o.Prepare(ctx, key, "")
	if err != nil {
		t.Error(err)
		return
	}
	m := mounts[0]
	if err := ioutil.WriteFile(filepath.Join(m.Source, "foo"), []byte("hi"), 0660); err != nil {
		t.Error(err)
		return
	}
	if err := o.Commit(ctx, "base", key); err != nil {
		t.Error(err)
		return
	}
}

func TestOverlayOverlayMount(t *testing.T) {
	ctx := context.TODO()
	root, err := ioutil.TempDir("", "overlay")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	o, _, err := newSnapshotter(ctx, root)
	if err != nil {
		t.Error(err)
		return
	}
	key := "/tmp/test"
	if _, err = o.Prepare(ctx, key, ""); err != nil {
		t.Error(err)
		return
	}
	if err := o.Commit(ctx, "base", key); err != nil {
		t.Error(err)
		return
	}
	var mounts []mount.Mount
	if mounts, err = o.Prepare(ctx, "/tmp/layer2", "base"); err != nil {
		t.Error(err)
		return
	}
	if len(mounts) != 1 {
		t.Errorf("should only have 1 mount but received %d", len(mounts))
	}
	m := mounts[0]
	if m.Type != "overlay" {
		t.Errorf("mount type should be overlay but received %q", m.Type)
	}
	if m.Source != "overlay" {
		t.Errorf("expected source %q but received %q", "overlay", m.Source)
	}
	var (
		bp    = getBasePath(ctx, o, root, "/tmp/layer2")
		work  = "workdir=" + filepath.Join(bp, "work")
		upper = "upperdir=" + filepath.Join(bp, "fs")
		lower = "lowerdir=" + getParents(ctx, o, root, "/tmp/layer2")[0]
	)
	for i, v := range []string{
		work,
		upper,
		lower,
	} {
		if m.Options[i] != v {
			t.Errorf("expected %q but received %q", v, m.Options[i])
		}
	}
}

func getBasePath(ctx context.Context, sn snapshot.Snapshotter, root, key string) string {
	o := sn.(*snapshotter)
	ctx, t, err := o.ms.TransactionContext(ctx, false)
	if err != nil {
		panic(err)
	}
	defer t.Rollback()

	s, err := storage.GetSnapshot(ctx, key)
	if err != nil {
		panic(err)
	}

	return filepath.Join(root, "snapshots", s.ID)
}

func getParents(ctx context.Context, sn snapshot.Snapshotter, root, key string) []string {
	o := sn.(*snapshotter)
	ctx, t, err := o.ms.TransactionContext(ctx, false)
	if err != nil {
		panic(err)
	}
	defer t.Rollback()
	s, err := storage.GetSnapshot(ctx, key)
	if err != nil {
		panic(err)
	}
	parents := make([]string, len(s.ParentIDs))
	for i := range s.ParentIDs {
		parents[i] = filepath.Join(root, "snapshots", s.ParentIDs[i], "fs")
	}
	return parents
}

func TestOverlayOverlayRead(t *testing.T) {
	testutil.RequiresRoot(t)
	ctx := context.TODO()
	root, err := ioutil.TempDir("", "overlay")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	o, _, err := newSnapshotter(ctx, root)
	if err != nil {
		t.Error(err)
		return
	}
	key := "/tmp/test"
	mounts, err := o.Prepare(ctx, key, "")
	if err != nil {
		t.Error(err)
		return
	}
	m := mounts[0]
	if err := ioutil.WriteFile(filepath.Join(m.Source, "foo"), []byte("hi"), 0660); err != nil {
		t.Error(err)
		return
	}
	if err := o.Commit(ctx, "base", key); err != nil {
		t.Error(err)
		return
	}
	if mounts, err = o.Prepare(ctx, "/tmp/layer2", "base"); err != nil {
		t.Error(err)
		return
	}
	dest := filepath.Join(root, "dest")
	if err := os.Mkdir(dest, 0700); err != nil {
		t.Error(err)
		return
	}
	if err := mount.All(mounts, dest); err != nil {
		t.Error(err)
		return
	}
	defer syscall.Unmount(dest, 0)
	data, err := ioutil.ReadFile(filepath.Join(dest, "foo"))
	if err != nil {
		t.Error(err)
		return
	}
	if e := string(data); e != "hi" {
		t.Errorf("expected file contents hi but got %q", e)
		return
	}
}

func TestOverlayView(t *testing.T) {
	ctx := context.TODO()
	root, err := ioutil.TempDir("", "overlay")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	o, _, err := newSnapshotter(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	key := "/tmp/base"
	mounts, err := o.Prepare(ctx, key, "")
	if err != nil {
		t.Fatal(err)
	}
	m := mounts[0]
	if err := ioutil.WriteFile(filepath.Join(m.Source, "foo"), []byte("hi"), 0660); err != nil {
		t.Fatal(err)
	}
	if err := o.Commit(ctx, "base", key); err != nil {
		t.Fatal(err)
	}

	key = "/tmp/top"
	_, err = o.Prepare(ctx, key, "base")
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(getParents(ctx, o, root, "/tmp/top")[0], "foo"), []byte("hi, again"), 0660); err != nil {
		t.Fatal(err)
	}
	if err := o.Commit(ctx, "top", key); err != nil {
		t.Fatal(err)
	}

	mounts, err = o.View(ctx, "/tmp/view1", "base")
	if err != nil {
		t.Fatal(err)
	}
	if len(mounts) != 1 {
		t.Fatalf("should only have 1 mount but received %d", len(mounts))
	}
	m = mounts[0]
	if m.Type != "bind" {
		t.Errorf("mount type should be bind but received %q", m.Type)
	}
	expected := getParents(ctx, o, root, "/tmp/view1")[0]
	if m.Source != expected {
		t.Errorf("expected source %q but received %q", expected, m.Source)
	}
	if m.Options[0] != "ro" {
		t.Errorf("expected mount option ro but received %q", m.Options[0])
	}
	if m.Options[1] != "rbind" {
		t.Errorf("expected mount option rbind but received %q", m.Options[1])
	}

	mounts, err = o.View(ctx, "/tmp/view2", "top")
	if err != nil {
		t.Fatal(err)
	}
	if len(mounts) != 1 {
		t.Fatalf("should only have 1 mount but received %d", len(mounts))
	}
	m = mounts[0]
	if m.Type != "overlay" {
		t.Errorf("mount type should be overlay but received %q", m.Type)
	}
	if m.Source != "overlay" {
		t.Errorf("mount source should be overlay but received %q", m.Source)
	}
	if len(m.Options) != 1 {
		t.Errorf("expected 1 mount option but got %d", len(m.Options))
	}
	lowers := getParents(ctx, o, root, "/tmp/view2")
	expected = fmt.Sprintf("lowerdir=%s:%s", lowers[0], lowers[1])
	if m.Options[0] != expected {
		t.Errorf("expected option %q but received %q", expected, m.Options[0])
	}
}
