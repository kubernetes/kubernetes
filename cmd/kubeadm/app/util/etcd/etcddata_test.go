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

package etcd

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	bolt "go.etcd.io/bbolt"
	"go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/pkg/v3/pbutil"
	"go.etcd.io/etcd/server/v3/etcdserver/api/snap"
	"go.etcd.io/etcd/server/v3/storage/wal"
	"go.etcd.io/etcd/server/v3/storage/wal/walpb"
	"go.etcd.io/raft/v3/raftpb"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"
)

// makeSnapshotDB creates a minimal bbolt database file that looks like an etcd
// snapshot (as returned by Maintenance.Snapshot): it contains a "meta" bucket
// with a consistentIndex and a term entry, matching what SnapshotRestore reads.
func makeSnapshotDB(t *testing.T, dir string, index, term uint64) string {
	t.Helper()
	path := filepath.Join(dir, "etcd.db")

	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		t.Fatalf("bolt.Open: %v", err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("meta"))
		if err != nil {
			return err
		}
		// UnsafeReadConsistentIndex reads two separate keys in the "meta" bucket:
		//   "consistent_index" → 8-byte big-endian index
		//   "term"             → 8-byte big-endian term
		idx := make([]byte, 8)
		binary.BigEndian.PutUint64(idx, index)
		if err := b.Put([]byte("consistent_index"), idx); err != nil {
			return err
		}
		tm := make([]byte, 8)
		binary.BigEndian.PutUint64(tm, term)
		return b.Put([]byte("term"), tm)
	})
	if err != nil {
		t.Fatalf("db.Update: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("db.Close: %v", err)
	}
	return path
}

func TestSnapshotRestore(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := makeSnapshotDB(t, tmpDir, 42, 3)

	dataDir := filepath.Join(tmpDir, "etcd")
	err := snapshotRestore(snapshotPath, "", dataDir, "default", "https://127.0.0.1:2380", "etcd-cluster")
	if err != nil {
		t.Fatalf("snapshotRestore() error = %v", err)
	}

	// Verify the expected directory structure was created.
	assertFileExists(t, filepath.Join(dataDir, "member", "snap", "db"))
	assertDirExists(t, filepath.Join(dataDir, "member", "wal"))
	assertDirExists(t, filepath.Join(dataDir, "member", "snap"))

	// A raft snap file should exist (named <term>-<index>.snap).
	entries, err := os.ReadDir(filepath.Join(dataDir, "member", "snap"))
	if err != nil {
		t.Fatalf("ReadDir snap: %v", err)
	}
	snapFiles := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".snap" {
			snapFiles++
		}
	}
	if snapFiles == 0 {
		t.Error("expected at least one .snap file in member/snap")
	}

	// A WAL file should exist.
	walEntries, err := os.ReadDir(filepath.Join(dataDir, "member", "wal"))
	if err != nil {
		t.Fatalf("ReadDir wal: %v", err)
	}
	walFiles := 0
	for _, e := range walEntries {
		if filepath.Ext(e.Name()) == ".wal" {
			walFiles++
		}
	}
	if walFiles == 0 {
		t.Error("expected at least one .wal file in member/wal")
	}
}

func TestSnapshotRestoreOverwritesExistingDir(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := makeSnapshotDB(t, tmpDir, 10, 1)

	dataDir := filepath.Join(tmpDir, "etcd")

	// Pre-create the data dir with some stale contents.
	staleFile := filepath.Join(dataDir, "stale")
	if err := os.MkdirAll(dataDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(staleFile, []byte("stale"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := snapshotRestore(snapshotPath, "", dataDir, "default", "https://127.0.0.1:2380", "etcd-cluster"); err != nil {
		t.Fatalf("snapshotRestore() error = %v", err)
	}

	// Stale file should be gone.
	assertFileNotExists(t, staleFile)
	// Fresh structure should be present.
	assertFileExists(t, filepath.Join(dataDir, "member", "snap", "db"))
}

func TestSnapshotRestorePreservesExistingIDsAndConfState(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := makeSnapshotDB(t, tmpDir, 77, 5)

	dataDir := filepath.Join(tmpDir, "etcd")
	wantNodeID := uint64(0x22)
	wantClusterID := uint64(0x99)
	wantConfState := &raftpb.ConfState{Voters: []uint64{0x11, 0x22, 0x33}}
	writeExistingMemberState(t, dataDir, wantNodeID, wantClusterID, 60, 4, wantConfState)

	// Different inputs here should not alter restored identity when existing WAL metadata is available.
	if err := snapshotRestore(snapshotPath, "", dataDir, "different-member", "https://10.10.10.10:2380", "different-token"); err != nil {
		t.Fatalf("snapshotRestore() error = %v", err)
	}

	gotNodeID, gotClusterID, gotConfState := readWalMetadataAndConfState(t, filepath.Join(dataDir, "member", "wal"))
	if gotNodeID != wantNodeID {
		t.Fatalf("nodeID changed after restore, got %d want %d", gotNodeID, wantNodeID)
	}
	if gotClusterID != wantClusterID {
		t.Fatalf("clusterID changed after restore, got %d want %d", gotClusterID, wantClusterID)
	}
	if !reflect.DeepEqual(gotConfState.GetVoters(), wantConfState.GetVoters()) {
		t.Fatalf("confState voters changed after restore, got %v want %v", gotConfState.GetVoters(), wantConfState.GetVoters())
	}
}

func TestSnapshotRestoreUsesBackupMetadataFile(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := makeSnapshotDB(t, tmpDir, 88, 6)
	metadataPath := filepath.Join(tmpDir, "restore-metadata.json")

	backupStateDir := filepath.Join(tmpDir, "backup-state")
	wantNodeID := uint64(0xabc)
	wantClusterID := uint64(0xdef)
	wantConfState := &raftpb.ConfState{Voters: []uint64{0x1, 0x2, 0x3}}
	writeExistingMemberState(t, backupStateDir, wantNodeID, wantClusterID, 70, 5, wantConfState)
	if err := snapshotSaveRestoreMetadata(backupStateDir, metadataPath, "custom-token"); err != nil {
		t.Fatalf("snapshotSaveRestoreMetadata() error = %v", err)
	}

	// Simulate runtime corruption/drift before rollback.
	runtimeStateDir := filepath.Join(tmpDir, "runtime-state")
	writeExistingMemberState(t, runtimeStateDir, uint64(0x999), uint64(0x777), 71, 5, &raftpb.ConfState{Voters: []uint64{0x999}})

	if err := snapshotRestore(snapshotPath, metadataPath, runtimeStateDir, "runtime-member", "https://20.20.20.20:2380", "runtime-token"); err != nil {
		t.Fatalf("snapshotRestore() error = %v", err)
	}

	gotNodeID, gotClusterID, gotConfState := readWalMetadataAndConfState(t, filepath.Join(runtimeStateDir, "member", "wal"))
	if gotNodeID != wantNodeID {
		t.Fatalf("nodeID mismatch after metadata-driven restore, got %d want %d", gotNodeID, wantNodeID)
	}
	if gotClusterID != wantClusterID {
		t.Fatalf("clusterID mismatch after metadata-driven restore, got %d want %d", gotClusterID, wantClusterID)
	}
	if !reflect.DeepEqual(gotConfState.GetVoters(), wantConfState.GetVoters()) {
		t.Fatalf("confState voters mismatch after metadata-driven restore, got %v want %v", gotConfState.GetVoters(), wantConfState.GetVoters())
	}
}

func TestSnapshotRestoreRequiresTokenWithoutMetadata(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := makeSnapshotDB(t, tmpDir, 11, 2)

	dataDir := filepath.Join(tmpDir, "empty-etcd")
	err := snapshotRestore(snapshotPath, "", dataDir, "default", "https://127.0.0.1:2380", "")
	if err == nil {
		t.Fatal("expected snapshotRestore to fail when restore metadata and initialClusterToken are both missing")
	}
}

func writeExistingMemberState(t *testing.T, dataDir string, nodeID, clusterID, index, term uint64, confState *raftpb.ConfState) {
	t.Helper()
	lg := zap.NewNop()
	walDir := filepath.Join(dataDir, "member", "wal")
	snapDir := filepath.Join(dataDir, "member", "snap")
	if err := os.MkdirAll(walDir, 0700); err != nil {
		t.Fatalf("MkdirAll wal: %v", err)
	}
	if err := os.MkdirAll(snapDir, 0700); err != nil {
		t.Fatalf("MkdirAll snap: %v", err)
	}

	walMetadata := pbutil.MustMarshalMessage(&etcdserverpb.Metadata{
		NodeID:    &nodeID,
		ClusterID: &clusterID,
	})
	w, err := wal.Create(lg, walDir, walMetadata)
	if err != nil {
		t.Fatalf("wal.Create: %v", err)
	}
	if err := w.SaveSnapshot(&walpb.Snapshot{Index: &index, Term: &term, ConfState: confState}); err != nil {
		_ = w.Close()
		t.Fatalf("w.SaveSnapshot: %v", err)
	}
	commit := index
	if err := w.Save(&raftpb.HardState{Term: &term, Vote: &nodeID, Commit: &commit}, nil); err != nil {
		_ = w.Close()
		t.Fatalf("w.Save: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("w.Close: %v", err)
	}

	snapshotter := snap.New(lg, snapDir)
	if err := snapshotter.SaveSnap(&raftpb.Snapshot{Metadata: &raftpb.SnapshotMetadata{Index: &index, Term: &term, ConfState: confState}}); err != nil {
		t.Fatalf("SaveSnap: %v", err)
	}
}

func readWalMetadataAndConfState(t *testing.T, walDir string) (uint64, uint64, *raftpb.ConfState) {
	t.Helper()
	lg := zap.NewNop()
	walSnaps, err := wal.ValidSnapshotEntries(lg, walDir)
	if err != nil {
		t.Fatalf("ValidSnapshotEntries: %v", err)
	}
	if len(walSnaps) == 0 {
		t.Fatal("expected WAL to contain at least one snapshot entry")
	}

	startSnap := walSnaps[len(walSnaps)-1]
	w, err := wal.OpenForRead(lg, walDir, startSnap)
	if err != nil {
		t.Fatalf("OpenForRead: %v", err)
	}
	defer func() { _ = w.Close() }()

	mdBytes, _, _, err := w.ReadAll()
	if err != nil && err != wal.ErrSnapshotNotFound {
		t.Fatalf("ReadAll: %v", err)
	}
	md := &etcdserverpb.Metadata{}
	if err := proto.Unmarshal(mdBytes, md); err != nil {
		t.Fatalf("unmarshal metadata: %v", err)
	}
	return md.GetNodeID(), md.GetClusterID(), startSnap.GetConfState()
}

func assertFileExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected %q to exist: %v", path, err)
	}
}

func assertDirExists(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("expected %q to exist: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("expected %q to be a directory", path)
	}
}

func assertFileNotExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected %q not to exist, got err=%v", path, err)
	}
}
