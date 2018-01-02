package storage

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/snapshot"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
)

type testFunc func(context.Context, *testing.T, *MetaStore)

type metaFactory func(string) (*MetaStore, error)

type populateFunc func(context.Context, *MetaStore) error

// MetaStoreSuite runs a test suite on the metastore given a factory function.
func MetaStoreSuite(t *testing.T, name string, meta func(root string) (*MetaStore, error)) {
	t.Run("GetInfo", makeTest(t, name, meta, inReadTransaction(testGetInfo, basePopulate)))
	t.Run("GetInfoNotExist", makeTest(t, name, meta, inReadTransaction(testGetInfoNotExist, basePopulate)))
	t.Run("GetInfoEmptyDB", makeTest(t, name, meta, inReadTransaction(testGetInfoNotExist, nil)))
	t.Run("Walk", makeTest(t, name, meta, inReadTransaction(testWalk, basePopulate)))
	t.Run("GetSnapshot", makeTest(t, name, meta, testGetSnapshot))
	t.Run("GetSnapshotNotExist", makeTest(t, name, meta, inReadTransaction(testGetSnapshotNotExist, basePopulate)))
	t.Run("GetSnapshotCommitted", makeTest(t, name, meta, inReadTransaction(testGetSnapshotCommitted, basePopulate)))
	t.Run("GetSnapshotEmptyDB", makeTest(t, name, meta, inReadTransaction(testGetSnapshotNotExist, basePopulate)))
	t.Run("CreateActive", makeTest(t, name, meta, inWriteTransaction(testCreateActive)))
	t.Run("CreateActiveNotExist", makeTest(t, name, meta, inWriteTransaction(testCreateActiveNotExist)))
	t.Run("CreateActiveExist", makeTest(t, name, meta, inWriteTransaction(testCreateActiveExist)))
	t.Run("CreateActiveFromActive", makeTest(t, name, meta, inWriteTransaction(testCreateActiveFromActive)))
	t.Run("Commit", makeTest(t, name, meta, inWriteTransaction(testCommit)))
	t.Run("CommitNotExist", makeTest(t, name, meta, inWriteTransaction(testCommitExist)))
	t.Run("CommitExist", makeTest(t, name, meta, inWriteTransaction(testCommitExist)))
	t.Run("CommitCommitted", makeTest(t, name, meta, inWriteTransaction(testCommitCommitted)))
	t.Run("CommitViewFails", makeTest(t, name, meta, inWriteTransaction(testCommitViewFails)))
	t.Run("Remove", makeTest(t, name, meta, inWriteTransaction(testRemove)))
	t.Run("RemoveNotExist", makeTest(t, name, meta, inWriteTransaction(testRemoveNotExist)))
	t.Run("RemoveWithChildren", makeTest(t, name, meta, inWriteTransaction(testRemoveWithChildren)))
	t.Run("ParentIDs", makeTest(t, name, meta, inWriteTransaction(testParents)))
}

// makeTest creates a testsuite with a writable transaction
func makeTest(t *testing.T, name string, metaFn metaFactory, fn testFunc) func(t *testing.T) {
	return func(t *testing.T) {
		ctx := context.Background()
		tmpDir, err := ioutil.TempDir("", "metastore-test-"+name+"-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(tmpDir)

		ms, err := metaFn(tmpDir)
		if err != nil {
			t.Fatal(err)
		}

		fn(ctx, t, ms)
	}
}

func inReadTransaction(fn testFunc, pf populateFunc) testFunc {
	return func(ctx context.Context, t *testing.T, ms *MetaStore) {
		if pf != nil {
			ctx, tx, err := ms.TransactionContext(ctx, true)
			if err != nil {
				t.Fatal(err)
			}
			if err := pf(ctx, ms); err != nil {
				if rerr := tx.Rollback(); rerr != nil {
					t.Logf("Rollback failed: %+v", rerr)
				}
				t.Fatalf("Populate failed: %+v", err)
			}
			if err := tx.Commit(); err != nil {
				t.Fatalf("Populate commit failed: %+v", err)
			}
		}

		ctx, tx, err := ms.TransactionContext(ctx, false)
		if err != nil {
			t.Fatalf("Failed start transaction: %+v", err)
		}
		defer func() {
			if err := tx.Rollback(); err != nil {
				t.Logf("Rollback failed: %+v", err)
				if !t.Failed() {
					t.FailNow()
				}
			}
		}()

		fn(ctx, t, ms)
	}
}

func inWriteTransaction(fn testFunc) testFunc {
	return func(ctx context.Context, t *testing.T, ms *MetaStore) {
		ctx, tx, err := ms.TransactionContext(ctx, true)
		if err != nil {
			t.Fatalf("Failed to start transaction: %+v", err)
		}
		defer func() {
			if t.Failed() {
				if err := tx.Rollback(); err != nil {
					t.Logf("Rollback failed: %+v", err)
				}
			} else {
				if err := tx.Commit(); err != nil {
					t.Fatalf("Commit failed: %+v", err)
				}
			}
		}()
		fn(ctx, t, ms)
	}
}

// basePopulate creates 7 snapshots
// - "committed-1": committed without parent
// - "committed-2":  committed with parent "committed-1"
// - "active-1": active without parent
// - "active-2": active with parent "committed-1"
// - "active-3": active with parent "committed-2"
// - "active-4": readonly active without parent"
// - "active-5": readonly active with parent "committed-2"
func basePopulate(ctx context.Context, ms *MetaStore) error {
	if _, err := CreateSnapshot(ctx, snapshot.KindActive, "committed-tmp-1", ""); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CommitActive(ctx, "committed-tmp-1", "committed-1", snapshot.Usage{Size: 1}); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindActive, "committed-tmp-2", "committed-1"); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CommitActive(ctx, "committed-tmp-2", "committed-2", snapshot.Usage{Size: 2}); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", ""); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindActive, "active-2", "committed-1"); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindActive, "active-3", "committed-2"); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindView, "view-1", ""); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	if _, err := CreateSnapshot(ctx, snapshot.KindView, "view-2", "committed-2"); err != nil {
		return errors.Wrap(err, "failed to create active")
	}
	return nil
}

var baseInfo = map[string]snapshot.Info{
	"committed-1": {
		Name:   "committed-1",
		Parent: "",
		Kind:   snapshot.KindCommitted,
	},
	"committed-2": {
		Name:   "committed-2",
		Parent: "committed-1",
		Kind:   snapshot.KindCommitted,
	},
	"active-1": {
		Name:   "active-1",
		Parent: "",
		Kind:   snapshot.KindActive,
	},
	"active-2": {
		Name:   "active-2",
		Parent: "committed-1",
		Kind:   snapshot.KindActive,
	},
	"active-3": {
		Name:   "active-3",
		Parent: "committed-2",
		Kind:   snapshot.KindActive,
	},
	"view-1": {
		Name:   "view-1",
		Parent: "",
		Kind:   snapshot.KindView,
	},
	"view-2": {
		Name:   "view-2",
		Parent: "committed-2",
		Kind:   snapshot.KindView,
	},
}

func assertNotExist(t *testing.T, err error) {
	if err == nil {
		t.Fatal("Expected not exist error")
	}
	if !errdefs.IsNotFound(err) {
		t.Fatalf("Expected not exist error, got %+v", err)
	}
}

func assertNotActive(t *testing.T, err error) {
	if err == nil {
		t.Fatal("Expected not active error")
	}
	if !errdefs.IsFailedPrecondition(err) {
		t.Fatalf("Expected not active error, got %+v", err)
	}
}

func assertNotCommitted(t *testing.T, err error) {
	if err == nil {
		t.Fatal("Expected active error")
	}
	if !errdefs.IsInvalidArgument(err) {
		t.Fatalf("Expected active error, got %+v", err)
	}
}

func assertExist(t *testing.T, err error) {
	if err == nil {
		t.Fatal("Expected exist error")
	}
	if !errdefs.IsAlreadyExists(err) {
		t.Fatalf("Expected exist error, got %+v", err)
	}
}

func testGetInfo(ctx context.Context, t *testing.T, ms *MetaStore) {
	for key, expected := range baseInfo {
		_, info, _, err := GetInfo(ctx, key)
		if err != nil {
			t.Fatalf("GetInfo on %v failed: %+v", key, err)
		}
		// TODO: Check timestamp range
		info.Created = time.Time{}
		info.Updated = time.Time{}
		assert.Equal(t, expected, info)
	}
}

func testGetInfoNotExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, _, _, err := GetInfo(ctx, "active-not-exist")
	assertNotExist(t, err)
}

func testWalk(ctx context.Context, t *testing.T, ms *MetaStore) {
	found := map[string]snapshot.Info{}
	err := WalkInfo(ctx, func(ctx context.Context, info snapshot.Info) error {
		if _, ok := found[info.Name]; ok {
			return errors.Errorf("entry already encountered")
		}
		// TODO: Check time range
		info.Created = time.Time{}
		info.Updated = time.Time{}
		found[info.Name] = info
		return nil
	})
	if err != nil {
		t.Fatalf("Walk failed: %+v", err)
	}
	assert.Equal(t, baseInfo, found)
}

func testGetSnapshot(ctx context.Context, t *testing.T, ms *MetaStore) {
	snapshotMap := map[string]Snapshot{}
	populate := func(ctx context.Context, ms *MetaStore) error {
		if _, err := CreateSnapshot(ctx, snapshot.KindActive, "committed-tmp-1", ""); err != nil {
			return errors.Wrap(err, "failed to create active")
		}
		if _, err := CommitActive(ctx, "committed-tmp-1", "committed-1", snapshot.Usage{}); err != nil {
			return errors.Wrap(err, "failed to create active")
		}

		for _, opts := range []struct {
			Kind   snapshot.Kind
			Name   string
			Parent string
		}{
			{
				Name: "active-1",
				Kind: snapshot.KindActive,
			},
			{
				Name:   "active-2",
				Parent: "committed-1",
				Kind:   snapshot.KindActive,
			},
			{
				Name: "view-1",
				Kind: snapshot.KindView,
			},
			{
				Name:   "view-2",
				Parent: "committed-1",
				Kind:   snapshot.KindView,
			},
		} {
			active, err := CreateSnapshot(ctx, opts.Kind, opts.Name, opts.Parent)
			if err != nil {
				return errors.Wrap(err, "failed to create active")
			}
			snapshotMap[opts.Name] = active
		}
		return nil
	}

	test := func(ctx context.Context, t *testing.T, ms *MetaStore) {
		for key, expected := range snapshotMap {
			s, err := GetSnapshot(ctx, key)
			if err != nil {
				t.Fatalf("Failed to get active: %+v", err)
			}
			assert.Equal(t, expected, s)
		}
	}

	inReadTransaction(test, populate)(ctx, t, ms)
}

func testGetSnapshotCommitted(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, err := GetSnapshot(ctx, "committed-1")
	assertNotActive(t, err)
}

func testGetSnapshotNotExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, err := GetSnapshot(ctx, "active-not-exist")
	assertNotExist(t, err)
}

func testCreateActive(ctx context.Context, t *testing.T, ms *MetaStore) {
	a1, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", "")
	if err != nil {
		t.Fatal(err)
	}
	if a1.Kind != snapshot.KindActive {
		t.Fatal("Expected writable active")
	}

	a2, err := CreateSnapshot(ctx, snapshot.KindView, "view-1", "")
	if err != nil {
		t.Fatal(err)
	}
	if a2.ID == a1.ID {
		t.Fatal("Returned active identifiers must be unique")
	}
	if a2.Kind != snapshot.KindView {
		t.Fatal("Expected a view")
	}

	commitID, err := CommitActive(ctx, "active-1", "committed-1", snapshot.Usage{})
	if err != nil {
		t.Fatal(err)
	}
	if commitID != a1.ID {
		t.Fatal("Snapshot identifier must not change on commit")
	}

	a3, err := CreateSnapshot(ctx, snapshot.KindActive, "active-3", "committed-1")
	if err != nil {
		t.Fatal(err)
	}
	if a3.ID == a1.ID {
		t.Fatal("Returned active identifiers must be unique")
	}
	if len(a3.ParentIDs) != 1 {
		t.Fatalf("Expected 1 parent, got %d", len(a3.ParentIDs))
	}
	if a3.ParentIDs[0] != commitID {
		t.Fatal("Expected active parent to be same as commit ID")
	}
	if a3.Kind != snapshot.KindActive {
		t.Fatal("Expected writable active")
	}

	a4, err := CreateSnapshot(ctx, snapshot.KindView, "view-2", "committed-1")
	if err != nil {
		t.Fatal(err)
	}
	if a4.ID == a1.ID {
		t.Fatal("Returned active identifiers must be unique")
	}
	if len(a3.ParentIDs) != 1 {
		t.Fatalf("Expected 1 parent, got %d", len(a3.ParentIDs))
	}
	if a3.ParentIDs[0] != commitID {
		t.Fatal("Expected active parent to be same as commit ID")
	}
	if a4.Kind != snapshot.KindView {
		t.Fatal("Expected a view")
	}
}

func testCreateActiveExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", "")
	assertExist(t, err)
	_, err = CreateSnapshot(ctx, snapshot.KindActive, "committed-1", "")
	assertExist(t, err)
}

func testCreateActiveNotExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", "does-not-exist")
	assertNotExist(t, err)
}

func testCreateActiveFromActive(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, err := CreateSnapshot(ctx, snapshot.KindActive, "active-new", "active-1")
	assertNotCommitted(t, err)
}

func testCommit(ctx context.Context, t *testing.T, ms *MetaStore) {
	a1, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", "")
	if err != nil {
		t.Fatal(err)
	}
	if a1.Kind != snapshot.KindActive {
		t.Fatal("Expected writable active")
	}

	commitID, err := CommitActive(ctx, "active-1", "committed-1", snapshot.Usage{})
	if err != nil {
		t.Fatal(err)
	}
	if commitID != a1.ID {
		t.Fatal("Snapshot identifier must not change on commit")
	}

	_, err = GetSnapshot(ctx, "active-1")
	assertNotExist(t, err)
	_, err = GetSnapshot(ctx, "committed-1")
	assertNotActive(t, err)
}

func testCommitNotExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, err := CommitActive(ctx, "active-not-exist", "committed-1", snapshot.Usage{})
	assertNotExist(t, err)
}

func testCommitExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, err := CommitActive(ctx, "active-1", "committed-1", snapshot.Usage{})
	assertExist(t, err)
}

func testCommitCommitted(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, err := CommitActive(ctx, "committed-1", "committed-3", snapshot.Usage{})
	assertNotActive(t, err)
}

func testCommitViewFails(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, err := CommitActive(ctx, "view-1", "committed-3", snapshot.Usage{})
	if err == nil {
		t.Fatal("Expected error committing readonly active")
	}
}

func testRemove(ctx context.Context, t *testing.T, ms *MetaStore) {
	a1, err := CreateSnapshot(ctx, snapshot.KindActive, "active-1", "")
	if err != nil {
		t.Fatal(err)
	}

	commitID, err := CommitActive(ctx, "active-1", "committed-1", snapshot.Usage{})
	if err != nil {
		t.Fatal(err)
	}
	if commitID != a1.ID {
		t.Fatal("Snapshot identifier must not change on commit")
	}

	a2, err := CreateSnapshot(ctx, snapshot.KindView, "view-1", "committed-1")
	if err != nil {
		t.Fatal(err)
	}

	a3, err := CreateSnapshot(ctx, snapshot.KindView, "view-2", "committed-1")
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = Remove(ctx, "active-1")
	assertNotExist(t, err)

	r3, k3, err := Remove(ctx, "view-2")
	if err != nil {
		t.Fatal(err)
	}
	if r3 != a3.ID {
		t.Fatal("Expected remove ID to match create ID")
	}
	if k3 != snapshot.KindView {
		t.Fatalf("Expected view kind, got %v", k3)
	}

	r2, k2, err := Remove(ctx, "view-1")
	if err != nil {
		t.Fatal(err)
	}
	if r2 != a2.ID {
		t.Fatal("Expected remove ID to match create ID")
	}
	if k2 != snapshot.KindView {
		t.Fatalf("Expected view kind, got %v", k2)
	}

	r1, k1, err := Remove(ctx, "committed-1")
	if err != nil {
		t.Fatal(err)
	}
	if r1 != commitID {
		t.Fatal("Expected remove ID to match commit ID")
	}
	if k1 != snapshot.KindCommitted {
		t.Fatalf("Expected committed kind, got %v", k1)
	}
}

func testRemoveWithChildren(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}
	_, _, err := Remove(ctx, "committed-1")
	if err == nil {
		t.Fatalf("Expected removal of snapshot with children to error")
	}
	_, _, err = Remove(ctx, "committed-1")
	if err == nil {
		t.Fatalf("Expected removal of snapshot with children to error")
	}
}

func testRemoveNotExist(ctx context.Context, t *testing.T, ms *MetaStore) {
	_, _, err := Remove(ctx, "does-not-exist")
	assertNotExist(t, err)
}

func testParents(ctx context.Context, t *testing.T, ms *MetaStore) {
	if err := basePopulate(ctx, ms); err != nil {
		t.Fatalf("Populate failed: %+v", err)
	}

	testcases := []struct {
		Name    string
		Parents int
	}{
		{"committed-1", 0},
		{"committed-2", 1},
		{"active-1", 0},
		{"active-2", 1},
		{"active-3", 2},
		{"view-1", 0},
		{"view-2", 2},
	}

	for _, tc := range testcases {
		name := tc.Name
		expectedID := ""
		expectedParents := []string{}
		for i := tc.Parents; i >= 0; i-- {
			sid, info, _, err := GetInfo(ctx, name)
			if err != nil {
				t.Fatalf("Failed to get snapshot %s: %v", tc.Name, err)
			}
			var (
				id      string
				parents []string
			)
			if info.Kind == snapshot.KindCommitted {
				// When commited, create view and resolve from view
				nid := fmt.Sprintf("test-%s-%d", tc.Name, i)
				s, err := CreateSnapshot(ctx, snapshot.KindView, nid, name)
				if err != nil {
					t.Fatalf("Failed to get snapshot %s: %v", tc.Name, err)
				}
				if len(s.ParentIDs) != i+1 {
					t.Fatalf("Unexpected number of parents for view of %s: %d, expected %d", name, len(s.ParentIDs), i+1)
				}
				id = s.ParentIDs[0]
				parents = s.ParentIDs[1:]
			} else {
				s, err := GetSnapshot(ctx, name)
				if err != nil {
					t.Fatalf("Failed to get snapshot %s: %v", tc.Name, err)
				}
				if len(s.ParentIDs) != i {
					t.Fatalf("Unexpected number of parents for %s: %d, expected %d", name, len(s.ParentIDs), i)
				}

				id = s.ID
				parents = s.ParentIDs
			}
			if sid != id {
				t.Fatalf("Info ID mismatched resolved snapshot ID for %s, %s vs %s", name, sid, id)
			}

			if expectedID != "" {
				if id != expectedID {
					t.Errorf("Unexpected ID of parent: %s, expected %s", id, expectedID)
				}
			}

			if len(expectedParents) > 0 {
				for j := range expectedParents {
					if parents[j] != expectedParents[j] {
						t.Errorf("Unexpected ID in parent array at %d: %s, expected %s", j, parents[j], expectedParents[j])
					}
				}
			}

			if i > 0 {
				name = info.Parent
				expectedID = parents[0]
				expectedParents = parents[1:]
			}

		}
	}
}
