package raft

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"testing"
)

func FileSnapTest(t *testing.T) (string, *FileSnapshotStore) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}

	snap, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return dir, snap
}

func TestFileSnapshotStoreImpl(t *testing.T) {
	var impl interface{} = &FileSnapshotStore{}
	if _, ok := impl.(SnapshotStore); !ok {
		t.Fatalf("FileSnapshotStore not a SnapshotStore")
	}
}

func TestFileSnapshotSinkImpl(t *testing.T) {
	var impl interface{} = &FileSnapshotSink{}
	if _, ok := impl.(SnapshotSink); !ok {
		t.Fatalf("FileSnapshotSink not a SnapshotSink")
	}
}

func TestFileSS_CreateSnapshotMissingParentDir(t *testing.T) {
	parent, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(parent)

	dir, err := ioutil.TempDir(parent, "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}

	snap, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	os.RemoveAll(parent)
	peers := []byte("all my lovely friends")
	_, err = snap.Create(10, 3, peers)
	if err != nil {
		t.Fatalf("should not fail when using non existing parent")
	}

}
func TestFileSS_CreateSnapshot(t *testing.T) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(dir)

	snap, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check no snapshots
	snaps, err := snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 0 {
		t.Fatalf("did not expect any snapshots: %v", snaps)
	}

	// Create a new sink
	peers := []byte("all my lovely friends")
	sink, err := snap.Create(10, 3, peers)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// The sink is not done, should not be in a list!
	snaps, err = snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 0 {
		t.Fatalf("did not expect any snapshots: %v", snaps)
	}

	// Write to the sink
	_, err = sink.Write([]byte("first\n"))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	_, err = sink.Write([]byte("second\n"))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Done!
	err = sink.Close()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should have a snapshot!
	snaps, err = snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 1 {
		t.Fatalf("expect a snapshots: %v", snaps)
	}

	// Check the latest
	latest := snaps[0]
	if latest.Index != 10 {
		t.Fatalf("bad snapshot: %v", *latest)
	}
	if latest.Term != 3 {
		t.Fatalf("bad snapshot: %v", *latest)
	}
	if bytes.Compare(latest.Peers, peers) != 0 {
		t.Fatalf("bad snapshot: %v", *latest)
	}
	if latest.Size != 13 {
		t.Fatalf("bad snapshot: %v", *latest)
	}

	// Read the snapshot
	_, r, err := snap.Open(latest.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Read out everything
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Ensure a match
	if bytes.Compare(buf.Bytes(), []byte("first\nsecond\n")) != 0 {
		t.Fatalf("content mismatch")
	}
}

func TestFileSS_CancelSnapshot(t *testing.T) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(dir)

	snap, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Create a new sink
	peers := []byte("all my lovely friends")
	sink, err := snap.Create(10, 3, peers)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Cancel the snapshot! Should delete
	err = sink.Cancel()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// The sink is canceled, should not be in a list!
	snaps, err := snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 0 {
		t.Fatalf("did not expect any snapshots: %v", snaps)
	}
}

func TestFileSS_Retention(t *testing.T) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(dir)

	snap, err := NewFileSnapshotStoreWithLogger(dir, 2, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Create a new sink
	peers := []byte("all my lovely friends")

	// Create a few snapshots
	for i := 10; i < 15; i++ {
		sink, err := snap.Create(uint64(i), 3, peers)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		err = sink.Close()
		if err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Should only have 2 listed!
	snaps, err := snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 2 {
		t.Fatalf("expect 2 snapshots: %v", snaps)
	}

	// Check they are the latest
	if snaps[0].Index != 14 {
		t.Fatalf("bad snap: %#v", *snaps[0])
	}
	if snaps[1].Index != 13 {
		t.Fatalf("bad snap: %#v", *snaps[1])
	}
}

func TestFileSS_BadPerm(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping file permission test on windows")
	}

	// Create a temp dir
	dir1, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(dir1)

	// Create a sub dir and remove all permissions
	dir2, err := ioutil.TempDir(dir1, "badperm")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if err := os.Chmod(dir2, 000); err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.Chmod(dir2, 777) // Set perms back for delete

	// Should fail
	if _, err := NewFileSnapshotStore(dir2, 3, nil); err == nil {
		t.Fatalf("should fail to use dir with bad perms")
	}
}

func TestFileSS_MissingParentDir(t *testing.T) {
	parent, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(parent)

	dir, err := ioutil.TempDir(parent, "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}

	os.RemoveAll(parent)
	_, err = NewFileSnapshotStore(dir, 3, nil)
	if err != nil {
		t.Fatalf("should not fail when using non existing parent")
	}
}

func TestFileSS_Ordering(t *testing.T) {
	// Create a test dir
	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	defer os.RemoveAll(dir)

	snap, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Create a new sink
	peers := []byte("all my lovely friends")

	sink, err := snap.Create(130350, 5, peers)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	err = sink.Close()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	sink, err = snap.Create(204917, 36, peers)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	err = sink.Close()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should only have 2 listed!
	snaps, err := snap.List()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(snaps) != 2 {
		t.Fatalf("expect 2 snapshots: %v", snaps)
	}

	// Check they are ordered
	if snaps[0].Term != 36 {
		t.Fatalf("bad snap: %#v", *snaps[0])
	}
	if snaps[1].Term != 5 {
		t.Fatalf("bad snap: %#v", *snaps[1])
	}
}
