// Copyright 2015 CoreOS, Inc.
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
	"log"

	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/snap"
	"github.com/coreos/etcd/storage/backend"
)

// createMergedSnapshotMessage creates a snapshot message that contains: raft status (term, conf),
// a snapshot of v2 store inside raft.Snapshot as []byte, a snapshot of v3 KV in the top level message
// as ReadCloser.
func (s *EtcdServer) createMergedSnapshotMessage(m raftpb.Message, snapi uint64, confState raftpb.ConfState) snap.Message {
	snapt, err := s.r.raftStorage.Term(snapi)
	if err != nil {
		log.Panicf("get term should never fail: %v", err)
	}

	// get a snapshot of v2 store as []byte
	clone := s.store.Clone()
	d, err := clone.SaveNoCopy()
	if err != nil {
		plog.Panicf("store save should never fail: %v", err)
	}

	// get a snapshot of v3 KV as readCloser
	rc := newSnapshotReaderCloser(s.be.Snapshot())

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

	return *snap.NewMessage(m, rc)
}

func newSnapshotReaderCloser(snapshot backend.Snapshot) io.ReadCloser {
	pr, pw := io.Pipe()
	go func() {
		n, err := snapshot.WriteTo(pw)
		if err == nil {
			plog.Infof("wrote database snapshot out [total bytes: %d]", n)
		}
		pw.CloseWithError(err)
		snapshot.Close()
	}()
	return pr
}
