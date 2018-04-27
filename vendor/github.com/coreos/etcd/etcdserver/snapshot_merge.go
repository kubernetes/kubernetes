// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package etcdserver

import (
	"io"

	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/snap"
)

// createMergedSnapshotMessage creates a snapshot message that contains: raft status (term, conf),
// a snapshot of v2 store inside raft.Snapshot as []byte, a snapshot of v3 KV in the top level message
// as ReadCloser.
func (s *EtcdServer) createMergedSnapshotMessage(m raftpb.Message, snapt, snapi uint64, confState raftpb.ConfState) snap.Message {
	// get a snapshot of v2 store as []byte
	clone := s.store.Clone()
	d, err := clone.SaveNoCopy()
	if err != nil {
		plog.Panicf("store save should never fail: %v", err)
	}

	// commit kv to write metadata(for example: consistent index).
	s.KV().Commit()
	dbsnap := s.be.Snapshot()
	// get a snapshot of v3 KV as readCloser
	rc := newSnapshotReaderCloser(dbsnap)

	// put the []byte snapshot of store into raft snapshot and return the merged snapshot with
	// KV readCloser snapshot.
	snapshot := raftpb.Snapshot{
		Metadata: raftpb.SnapshotMetadata{
			Index:     snapi,
			Term:      snapt,
			ConfState: confState,
		},
		Data: d,
	}
	m.Snapshot = snapshot

	return *snap.NewMessage(m, rc, dbsnap.Size())
}

func newSnapshotReaderCloser(snapshot backend.Snapshot) io.ReadCloser {
	pr, pw := io.Pipe()
	go func() {
		n, err := snapshot.WriteTo(pw)
		if err == nil {
			plog.Infof("wrote database snapshot out [total bytes: %d]", n)
		} else {
			plog.Warningf("failed to write database snapshot out [written bytes: %d]: %v", n, err)
		}
		pw.CloseWithError(err)
		err = snapshot.Close()
		if err != nil {
			plog.Panicf("failed to close database snapshot: %v", err)
		}
	}()
	return pr
}
