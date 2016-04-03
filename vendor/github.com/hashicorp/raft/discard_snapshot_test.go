package raft

import "testing"

func TestDiscardSnapshotStoreImpl(t *testing.T) {
	var impl interface{} = &DiscardSnapshotStore{}
	if _, ok := impl.(SnapshotStore); !ok {
		t.Fatalf("DiscardSnapshotStore not a SnapshotStore")
	}
}

func TestDiscardSnapshotSinkImpl(t *testing.T) {
	var impl interface{} = &DiscardSnapshotSink{}
	if _, ok := impl.(SnapshotSink); !ok {
		t.Fatalf("DiscardSnapshotSink not a SnapshotSink")
	}
}
