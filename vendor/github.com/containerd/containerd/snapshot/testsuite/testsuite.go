package testsuite

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/fs/fstest"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/snapshot"
	"github.com/containerd/containerd/testutil"
	"github.com/stretchr/testify/assert"
)

// SnapshotterSuite runs a test suite on the snapshotter given a factory function.
func SnapshotterSuite(t *testing.T, name string, snapshotterFn func(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error)) {
	restoreMask := clearMask()
	defer restoreMask()

	t.Run("Basic", makeTest(name, snapshotterFn, checkSnapshotterBasic))
	t.Run("StatActive", makeTest(name, snapshotterFn, checkSnapshotterStatActive))
	t.Run("StatComitted", makeTest(name, snapshotterFn, checkSnapshotterStatCommitted))
	t.Run("TransitivityTest", makeTest(name, snapshotterFn, checkSnapshotterTransitivity))
	t.Run("PreareViewFailingtest", makeTest(name, snapshotterFn, checkSnapshotterPrepareView))
	t.Run("Update", makeTest(name, snapshotterFn, checkUpdate))
	t.Run("Remove", makeTest(name, snapshotterFn, checkRemove))

	t.Run("LayerFileupdate", makeTest(name, snapshotterFn, checkLayerFileUpdate))
	t.Run("RemoveDirectoryInLowerLayer", makeTest(name, snapshotterFn, checkRemoveDirectoryInLowerLayer))
	t.Run("Chown", makeTest(name, snapshotterFn, checkChown))
	t.Run("DirectoryPermissionOnCommit", makeTest(name, snapshotterFn, checkDirectoryPermissionOnCommit))
	t.Run("RemoveIntermediateSnapshot", makeTest(name, snapshotterFn, checkRemoveIntermediateSnapshot))
	t.Run("DeletedFilesInChildSnapshot", makeTest(name, snapshotterFn, checkDeletedFilesInChildSnapshot))
	t.Run("MoveFileFromLowerLayer", makeTest(name, snapshotterFn, checkFileFromLowerLayer))
	// Rename test still fails on some kernels with overlay
	//t.Run("Rename", makeTest(name, snapshotterFn, checkRename))

	t.Run("ViewReadonly", makeTest(name, snapshotterFn, checkSnapshotterViewReadonly))

	t.Run("StatInWalk", makeTest(name, snapshotterFn, checkStatInWalk))
}

func makeTest(name string, snapshotterFn func(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error), fn func(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string)) func(t *testing.T) {
	return func(t *testing.T) {
		t.Parallel()

		ctx := context.Background()
		ctx = namespaces.WithNamespace(ctx, "testsuite")
		// Make two directories: a snapshotter root and a play area for the tests:
		//
		// 	/tmp
		// 		work/ -> passed to test functions
		// 		root/ -> passed to snapshotter
		//
		tmpDir, err := ioutil.TempDir("", "snapshot-suite-"+name+"-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(tmpDir)

		root := filepath.Join(tmpDir, "root")
		if err := os.MkdirAll(root, 0777); err != nil {
			t.Fatal(err)
		}

		snapshotter, cleanup, err := snapshotterFn(ctx, root)
		if err != nil {
			t.Fatalf("Failed to initialize snapshotter: %+v", err)
		}
		defer func() {
			if cleanup != nil {
				if err := cleanup(); err != nil {
					t.Errorf("Cleanup failed: %v", err)
				}
			}
		}()

		work := filepath.Join(tmpDir, "work")
		if err := os.MkdirAll(work, 0777); err != nil {
			t.Fatal(err)
		}

		defer testutil.DumpDir(t, tmpDir)
		fn(ctx, t, snapshotter, work)
	}
}

// checkSnapshotterBasic tests the basic workflow of a snapshot snapshotter.
func checkSnapshotterBasic(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	initialApplier := fstest.Apply(
		fstest.CreateFile("/foo", []byte("foo\n"), 0777),
		fstest.CreateDir("/a", 0755),
		fstest.CreateDir("/a/b", 0755),
		fstest.CreateDir("/a/b/c", 0755),
	)

	diffApplier := fstest.Apply(
		fstest.CreateFile("/bar", []byte("bar\n"), 0777),
		// also, change content of foo to bar
		fstest.CreateFile("/foo", []byte("bar\n"), 0777),
		fstest.RemoveAll("/a/b"),
	)

	preparing := filepath.Join(work, "preparing")
	if err := os.MkdirAll(preparing, 0777); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	mounts, err := snapshotter.Prepare(ctx, preparing, "")
	if err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	if len(mounts) < 1 {
		t.Fatal("expected mounts to have entries")
	}

	if err := mount.All(mounts, preparing); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}
	defer testutil.Unmount(t, preparing)

	if err := initialApplier.Apply(preparing); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	committed := filepath.Join(work, "committed")
	if err := snapshotter.Commit(ctx, committed, preparing); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	si, err := snapshotter.Stat(ctx, committed)
	if err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	assert.Equal(t, "", si.Parent)
	assert.Equal(t, snapshot.KindCommitted, si.Kind)

	next := filepath.Join(work, "nextlayer")
	if err := os.MkdirAll(next, 0777); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	mounts, err = snapshotter.Prepare(ctx, next, committed)
	if err != nil {
		t.Fatalf("failure reason: %+v", err)
	}
	if err := mount.All(mounts, next); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}
	defer testutil.Unmount(t, next)

	if err := fstest.CheckDirectoryEqualWithApplier(next, initialApplier); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	if err := diffApplier.Apply(next); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	ni, err := snapshotter.Stat(ctx, next)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, committed, ni.Parent)
	assert.Equal(t, snapshot.KindActive, ni.Kind)

	nextCommitted := filepath.Join(work, "committed-next")
	if err := snapshotter.Commit(ctx, nextCommitted, next); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	si2, err := snapshotter.Stat(ctx, nextCommitted)
	if err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	assert.Equal(t, committed, si2.Parent)
	assert.Equal(t, snapshot.KindCommitted, si2.Kind)

	expected := map[string]snapshot.Info{
		si.Name:  si,
		si2.Name: si2,
	}
	walked := map[string]snapshot.Info{} // walk is not ordered
	assert.NoError(t, snapshotter.Walk(ctx, func(ctx context.Context, si snapshot.Info) error {
		walked[si.Name] = si
		return nil
	}))

	for ek, ev := range expected {
		av, ok := walked[ek]
		if !ok {
			t.Errorf("Missing stat for %v", ek)
			continue
		}
		assert.Equal(t, ev, av)
	}

	nextnext := filepath.Join(work, "nextnextlayer")
	if err := os.MkdirAll(nextnext, 0777); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	mounts, err = snapshotter.View(ctx, nextnext, nextCommitted)
	if err != nil {
		t.Fatalf("failure reason: %+v", err)
	}
	if err := mount.All(mounts, nextnext); err != nil {
		t.Fatalf("failure reason: %+v", err)
	}

	if err := fstest.CheckDirectoryEqualWithApplier(nextnext,
		fstest.Apply(initialApplier, diffApplier)); err != nil {
		testutil.Unmount(t, nextnext)
		t.Fatalf("failure reason: %+v", err)
	}

	testutil.Unmount(t, nextnext)
	assert.NoError(t, snapshotter.Remove(ctx, nextnext))
	assert.Error(t, snapshotter.Remove(ctx, committed))
	assert.NoError(t, snapshotter.Remove(ctx, nextCommitted))
	assert.NoError(t, snapshotter.Remove(ctx, committed))
}

// Create a New Layer on top of base layer with Prepare, Stat on new layer, should return Active layer.
func checkSnapshotterStatActive(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	preparing := filepath.Join(work, "preparing")
	if err := os.MkdirAll(preparing, 0777); err != nil {
		t.Fatal(err)
	}

	mounts, err := snapshotter.Prepare(ctx, preparing, "")
	if err != nil {
		t.Fatal(err)
	}

	if len(mounts) < 1 {
		t.Fatal("expected mounts to have entries")
	}

	if err = mount.All(mounts, preparing); err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, preparing)

	if err = ioutil.WriteFile(filepath.Join(preparing, "foo"), []byte("foo\n"), 0777); err != nil {
		t.Fatal(err)
	}

	si, err := snapshotter.Stat(ctx, preparing)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, si.Name, preparing)
	assert.Equal(t, snapshot.KindActive, si.Kind)
	assert.Equal(t, "", si.Parent)
}

// Commit a New Layer on top of base layer with Prepare & Commit , Stat on new layer, should return Committed layer.
func checkSnapshotterStatCommitted(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	preparing := filepath.Join(work, "preparing")
	if err := os.MkdirAll(preparing, 0777); err != nil {
		t.Fatal(err)
	}

	mounts, err := snapshotter.Prepare(ctx, preparing, "")
	if err != nil {
		t.Fatal(err)
	}

	if len(mounts) < 1 {
		t.Fatal("expected mounts to have entries")
	}

	if err = mount.All(mounts, preparing); err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, preparing)

	if err = ioutil.WriteFile(filepath.Join(preparing, "foo"), []byte("foo\n"), 0777); err != nil {
		t.Fatal(err)
	}

	committed := filepath.Join(work, "committed")
	if err = snapshotter.Commit(ctx, committed, preparing); err != nil {
		t.Fatal(err)
	}

	si, err := snapshotter.Stat(ctx, committed)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, si.Name, committed)
	assert.Equal(t, snapshot.KindCommitted, si.Kind)
	assert.Equal(t, "", si.Parent)

}

func snapshotterPrepareMount(ctx context.Context, snapshotter snapshot.Snapshotter, diffPathName string, parent string, work string) (string, error) {
	preparing := filepath.Join(work, diffPathName)
	if err := os.MkdirAll(preparing, 0777); err != nil {
		return "", err
	}

	mounts, err := snapshotter.Prepare(ctx, preparing, parent)
	if err != nil {
		return "", err
	}

	if len(mounts) < 1 {
		return "", fmt.Errorf("expected mounts to have entries")
	}

	if err = mount.All(mounts, preparing); err != nil {
		return "", err
	}
	return preparing, nil
}

// Given A <- B <- C, B is the parent of C and A is a transitive parent of C (in this case, a "grandparent")
func checkSnapshotterTransitivity(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	preparing, err := snapshotterPrepareMount(ctx, snapshotter, "preparing", "", work)
	if err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, preparing)

	if err = ioutil.WriteFile(filepath.Join(preparing, "foo"), []byte("foo\n"), 0777); err != nil {
		t.Fatal(err)
	}

	snapA := filepath.Join(work, "snapA")
	if err = snapshotter.Commit(ctx, snapA, preparing); err != nil {
		t.Fatal(err)
	}

	next, err := snapshotterPrepareMount(ctx, snapshotter, "next", snapA, work)
	if err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, next)

	if err = ioutil.WriteFile(filepath.Join(next, "foo"), []byte("foo bar\n"), 0777); err != nil {
		t.Fatal(err)
	}

	snapB := filepath.Join(work, "snapB")
	if err = snapshotter.Commit(ctx, snapB, next); err != nil {
		t.Fatal(err)
	}

	siA, err := snapshotter.Stat(ctx, snapA)
	if err != nil {
		t.Fatal(err)
	}

	siB, err := snapshotter.Stat(ctx, snapB)
	if err != nil {
		t.Fatal(err)
	}

	siParentB, err := snapshotter.Stat(ctx, siB.Parent)
	if err != nil {
		t.Fatal(err)
	}

	// Test the transivity
	assert.Equal(t, "", siA.Parent)
	assert.Equal(t, snapA, siB.Parent)
	assert.Equal(t, "", siParentB.Parent)

}

// Creating two layers with Prepare or View with same key must fail.
func checkSnapshotterPrepareView(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	preparing, err := snapshotterPrepareMount(ctx, snapshotter, "preparing", "", work)
	if err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, preparing)

	snapA := filepath.Join(work, "snapA")
	if err = snapshotter.Commit(ctx, snapA, preparing); err != nil {
		t.Fatal(err)
	}

	// Prepare & View with same key
	newLayer := filepath.Join(work, "newlayer")
	if err = os.MkdirAll(preparing, 0777); err != nil {
		t.Fatal(err)
	}

	// Prepare & View with same key
	_, err = snapshotter.Prepare(ctx, newLayer, snapA)
	if err != nil {
		t.Fatal(err)
	}

	_, err = snapshotter.View(ctx, newLayer, snapA)
	//must be err != nil
	assert.NotNil(t, err)

	// Two Prepare with same key
	prepLayer := filepath.Join(work, "prepLayer")
	if err = os.MkdirAll(preparing, 0777); err != nil {
		t.Fatal(err)
	}

	_, err = snapshotter.Prepare(ctx, prepLayer, snapA)
	if err != nil {
		t.Fatal(err)
	}

	_, err = snapshotter.Prepare(ctx, prepLayer, snapA)
	//must be err != nil
	assert.NotNil(t, err)

	// Two View with same key
	viewLayer := filepath.Join(work, "viewLayer")
	if err = os.MkdirAll(preparing, 0777); err != nil {
		t.Fatal(err)
	}

	_, err = snapshotter.View(ctx, viewLayer, snapA)
	if err != nil {
		t.Fatal(err)
	}

	_, err = snapshotter.View(ctx, viewLayer, snapA)
	//must be err != nil
	assert.NotNil(t, err)

}

// Deletion of files/folder of base layer in new layer, On Commit, those files should not be visible.
func checkDeletedFilesInChildSnapshot(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {

	l1Init := fstest.Apply(
		fstest.CreateFile("/foo", []byte("foo\n"), 0777),
		fstest.CreateFile("/foobar", []byte("foobar\n"), 0777),
	)
	l2Init := fstest.Apply(
		fstest.RemoveAll("/foobar"),
	)
	l3Init := fstest.Apply()

	if err := checkSnapshots(ctx, snapshotter, work, l1Init, l2Init, l3Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}

}

//Create three layers. Deleting intermediate layer must fail.
func checkRemoveIntermediateSnapshot(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {

	base, err := snapshotterPrepareMount(ctx, snapshotter, "base", "", work)
	if err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, base)

	committedBase := filepath.Join(work, "committed-base")
	if err = snapshotter.Commit(ctx, committedBase, base); err != nil {
		t.Fatal(err)
	}

	// Create intermediate layer
	intermediate := filepath.Join(work, "intermediate")
	if _, err = snapshotter.Prepare(ctx, intermediate, committedBase); err != nil {
		t.Fatal(err)
	}

	committedInter := filepath.Join(work, "committed-inter")
	if err = snapshotter.Commit(ctx, committedInter, intermediate); err != nil {
		t.Fatal(err)
	}

	// Create top layer
	topLayer := filepath.Join(work, "toplayer")
	if _, err = snapshotter.Prepare(ctx, topLayer, committedInter); err != nil {
		t.Fatal(err)
	}

	// Deletion of intermediate layer must fail.
	err = snapshotter.Remove(ctx, committedInter)
	if err == nil {
		t.Fatal("intermediate layer removal should fail.")
	}

	//Removal from toplayer to base should not fail.
	err = snapshotter.Remove(ctx, topLayer)
	if err != nil {
		t.Fatal(err)
	}
	err = snapshotter.Remove(ctx, committedInter)
	if err != nil {
		t.Fatal(err)
	}
	err = snapshotter.Remove(ctx, committedBase)
	if err != nil {
		t.Fatal(err)
	}
}

// baseTestSnapshots creates a base set of snapshots for tests, each snapshot is empty
// Tests snapshots:
//  c1 - committed snapshot, no parent
//  c2 - commited snapshot, c1 is parent
//  a1 - active snapshot, c2 is parent
//  a1 - active snapshot, no parent
//  v1 - view snapshot, v1 is parent
//  v2 - view snapshot, no parent
func baseTestSnapshots(ctx context.Context, snapshotter snapshot.Snapshotter) error {
	if _, err := snapshotter.Prepare(ctx, "c1-a", ""); err != nil {
		return err
	}
	if err := snapshotter.Commit(ctx, "c1", "c1-a"); err != nil {
		return err
	}
	if _, err := snapshotter.Prepare(ctx, "c2-a", "c1"); err != nil {
		return err
	}
	if err := snapshotter.Commit(ctx, "c2", "c2-a"); err != nil {
		return err
	}
	if _, err := snapshotter.Prepare(ctx, "a1", "c2"); err != nil {
		return err
	}
	if _, err := snapshotter.Prepare(ctx, "a2", ""); err != nil {
		return err
	}
	if _, err := snapshotter.View(ctx, "v1", "c2"); err != nil {
		return err
	}
	if _, err := snapshotter.View(ctx, "v2", ""); err != nil {
		return err
	}
	return nil
}

func checkUpdate(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	t1 := time.Now().UTC()
	if err := baseTestSnapshots(ctx, snapshotter); err != nil {
		t.Fatalf("Failed to create base snapshots: %v", err)
	}
	t2 := time.Now().UTC()
	testcases := []struct {
		name   string
		kind   snapshot.Kind
		parent string
	}{
		{
			name: "c1",
			kind: snapshot.KindCommitted,
		},
		{
			name:   "c2",
			kind:   snapshot.KindCommitted,
			parent: "c1",
		},
		{
			name:   "a1",
			kind:   snapshot.KindActive,
			parent: "c2",
		},
		{
			name: "a2",
			kind: snapshot.KindActive,
		},
		{
			name:   "v1",
			kind:   snapshot.KindView,
			parent: "c2",
		},
		{
			name: "v2",
			kind: snapshot.KindView,
		},
	}
	for _, tc := range testcases {
		st, err := snapshotter.Stat(ctx, tc.name)
		if err != nil {
			t.Fatalf("Failed to stat %s: %v", tc.name, err)
		}
		if st.Created.Before(t1) || st.Created.After(t2) {
			t.Errorf("(%s) wrong created time %s: expected between %s and %s", tc.name, st.Created, t1, t2)
			continue
		}
		if st.Created != st.Updated {
			t.Errorf("(%s) unexpected updated time %s: expected %s", tc.name, st.Updated, st.Created)
			continue
		}
		if st.Kind != tc.kind {
			t.Errorf("(%s) unexpected kind %s, expected %s", tc.name, st.Kind, tc.kind)
			continue
		}
		if st.Parent != tc.parent {
			t.Errorf("(%s) unexpected parent %q, expected %q", tc.name, st.Parent, tc.parent)
			continue
		}
		if st.Name != tc.name {
			t.Errorf("(%s) unexpected name %q, expected %q", tc.name, st.Name, tc.name)
			continue
		}

		createdAt := st.Created
		expected := map[string]string{
			"l1": "v1",
			"l2": "v2",
			"l3": "v3",
		}
		st.Parent = "doesnotexist"
		st.Labels = expected
		u1 := time.Now().UTC()
		st, err = snapshotter.Update(ctx, st)
		if err != nil {
			t.Fatalf("Failed to update %s: %v", tc.name, err)
		}
		u2 := time.Now().UTC()

		if st.Created != createdAt {
			t.Errorf("(%s) wrong created time %s: expected %s", tc.name, st.Created, createdAt)
			continue
		}
		if st.Updated.Before(u1) || st.Updated.After(u2) {
			t.Errorf("(%s) wrong updated time %s: expected between %s and %s", tc.name, st.Updated, u1, u2)
			continue
		}
		if st.Kind != tc.kind {
			t.Errorf("(%s) unexpected kind %s, expected %s", tc.name, st.Kind, tc.kind)
			continue
		}
		if st.Parent != tc.parent {
			t.Errorf("(%s) unexpected parent %q, expected %q", tc.name, st.Parent, tc.parent)
			continue
		}
		if st.Name != tc.name {
			t.Errorf("(%s) unexpected name %q, expected %q", tc.name, st.Name, tc.name)
			continue
		}
		assertLabels(t, st.Labels, expected)

		expected = map[string]string{
			"l1": "updated",
			"l3": "v3",
		}
		st.Labels = map[string]string{
			"l1": "updated",
			"l4": "v4",
		}
		st, err = snapshotter.Update(ctx, st, "labels.l1", "labels.l2")
		if err != nil {
			t.Fatalf("Failed to update %s: %v", tc.name, err)
		}
		assertLabels(t, st.Labels, expected)

		expected = map[string]string{
			"l4": "v4",
		}
		st.Labels = expected
		st, err = snapshotter.Update(ctx, st, "labels")
		if err != nil {
			t.Fatalf("Failed to update %s: %v", tc.name, err)
		}
		assertLabels(t, st.Labels, expected)

		// Test failure received when providing immutable field path
		st.Parent = "doesnotexist"
		st, err = snapshotter.Update(ctx, st, "parent")
		if err == nil {
			t.Errorf("Expected error updating with immutable field path")
		} else if !errdefs.IsInvalidArgument(err) {
			t.Fatalf("Unexpected error updating %s: %+v", tc.name, err)
		}
	}
}

func assertLabels(t *testing.T, actual, expected map[string]string) {
	if len(actual) != len(expected) {
		t.Fatalf("Label size mismatch: %d vs %d\n\tActual: %#v\n\tExpected: %#v", len(actual), len(expected), actual, expected)
	}
	for k, v := range expected {
		if a := actual[k]; v != a {
			t.Errorf("Wrong label value for %s, got %q, expected %q", k, a, v)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}

func checkRemove(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	if _, err := snapshotter.Prepare(ctx, "committed-a", ""); err != nil {
		t.Fatal(err)
	}
	if err := snapshotter.Commit(ctx, "committed-1", "committed-a"); err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotter.Prepare(ctx, "reuse-1", "committed-1"); err != nil {
		t.Fatal(err)
	}
	if err := snapshotter.Remove(ctx, "reuse-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotter.View(ctx, "reuse-1", "committed-1"); err != nil {
		t.Fatal(err)
	}
	if err := snapshotter.Remove(ctx, "reuse-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotter.Prepare(ctx, "reuse-1", ""); err != nil {
		t.Fatal(err)
	}
	if err := snapshotter.Remove(ctx, "committed-1"); err != nil {
		t.Fatal(err)
	}
	if err := snapshotter.Commit(ctx, "commited-1", "reuse-1"); err != nil {
		t.Fatal(err)
	}
}

// checkSnapshotterViewReadonly ensures a KindView snapshot to be mounted as a read-only filesystem.
// This function is called only when WithTestViewReadonly is true.
func checkSnapshotterViewReadonly(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	preparing := filepath.Join(work, "preparing")
	if _, err := snapshotter.Prepare(ctx, preparing, ""); err != nil {
		t.Fatal(err)
	}
	committed := filepath.Join(work, "commited")
	if err := snapshotter.Commit(ctx, committed, preparing); err != nil {
		t.Fatal(err)
	}
	view := filepath.Join(work, "view")
	m, err := snapshotter.View(ctx, view, committed)
	if err != nil {
		t.Fatal(err)
	}
	viewMountPoint := filepath.Join(work, "view-mount")
	if err := os.MkdirAll(viewMountPoint, 0777); err != nil {
		t.Fatal(err)
	}

	// Just checking the option string of m is not enough, we need to test real mount. (#1368)
	if err := mount.All(m, viewMountPoint); err != nil {
		t.Fatal(err)
	}

	testfile := filepath.Join(viewMountPoint, "testfile")
	if err := ioutil.WriteFile(testfile, []byte("testcontent"), 0777); err != nil {
		t.Logf("write to %q failed with %v (EROFS is expected but can be other error code)", testfile, err)
	} else {
		t.Fatalf("write to %q should fail (EROFS) but did not fail", testfile)
	}
	testutil.Unmount(t, viewMountPoint)
	assert.NoError(t, snapshotter.Remove(ctx, view))
	assert.NoError(t, snapshotter.Remove(ctx, committed))
}

// Move files from base layer to new location in intermediate layer.
// Verify if the file at source is deleted and copied to new location.
func checkFileFromLowerLayer(ctx context.Context, t *testing.T, snapshotter snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/dir1", 0700),
		fstest.CreateFile("/dir1/f1", []byte("Hello"), 0644),
		fstest.CreateDir("dir2", 0700),
		fstest.CreateFile("dir2/f2", []byte("..."), 0644),
	)
	l2Init := fstest.Apply(
		fstest.CreateDir("/dir3", 0700),
		fstest.CreateFile("/dir3/f1", []byte("Hello"), 0644),
		fstest.RemoveAll("/dir1"),
		fstest.Link("dir2/f2", "dir3/f2"),
		fstest.RemoveAll("dir2/f2"),
	)

	if err := checkSnapshots(ctx, snapshotter, work, l1Init, l2Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}
}
