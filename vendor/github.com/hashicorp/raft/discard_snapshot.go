package raft

import (
	"fmt"
	"io"
)

// DiscardSnapshotStore is used to successfully snapshot while
// always discarding the snapshot. This is useful for when the
// log should be truncated but no snapshot should be retained.
// This should never be used for production use, and is only
// suitable for testing.
type DiscardSnapshotStore struct{}

type DiscardSnapshotSink struct{}

// NewDiscardSnapshotStore is used to create a new DiscardSnapshotStore.
func NewDiscardSnapshotStore() *DiscardSnapshotStore {
	return &DiscardSnapshotStore{}
}

func (d *DiscardSnapshotStore) Create(index, term uint64, peers []byte) (SnapshotSink, error) {
	return &DiscardSnapshotSink{}, nil
}

func (d *DiscardSnapshotStore) List() ([]*SnapshotMeta, error) {
	return nil, nil
}

func (d *DiscardSnapshotStore) Open(id string) (*SnapshotMeta, io.ReadCloser, error) {
	return nil, nil, fmt.Errorf("open is not supported")
}

func (d *DiscardSnapshotSink) Write(b []byte) (int, error) {
	return len(b), nil
}

func (d *DiscardSnapshotSink) Close() error {
	return nil
}

func (d *DiscardSnapshotSink) ID() string {
	return "discard"
}

func (d *DiscardSnapshotSink) Cancel() error {
	return nil
}
