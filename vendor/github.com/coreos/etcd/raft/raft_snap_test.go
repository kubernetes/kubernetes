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

package raft

import (
	"testing"

	pb "github.com/coreos/etcd/raft/raftpb"
)

var (
	testingSnap = pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     11, // magic number
			Term:      11, // magic number
			ConfState: pb.ConfState{Nodes: []uint64{1, 2}},
		},
	}
)

func TestSendingSnapshotSetPendingSnapshot(t *testing.T) {
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1}, 10, 1, storage)
	sm.restore(testingSnap)

	sm.becomeCandidate()
	sm.becomeLeader()

	// force set the next of node 1, so that
	// node 1 needs a snapshot
	sm.prs[2].Next = sm.raftLog.firstIndex()

	sm.Step(pb.Message{From: 2, To: 1, Type: pb.MsgAppResp, Index: sm.prs[2].Next - 1, Reject: true})
	if sm.prs[2].PendingSnapshot != 11 {
		t.Fatalf("PendingSnapshot = %d, want 11", sm.prs[2].PendingSnapshot)
	}
}

func TestPendingSnapshotPauseReplication(t *testing.T) {
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	sm.restore(testingSnap)

	sm.becomeCandidate()
	sm.becomeLeader()

	sm.prs[2].becomeSnapshot(11)

	sm.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})
	msgs := sm.readMessages()
	if len(msgs) != 0 {
		t.Fatalf("len(msgs) = %d, want 0", len(msgs))
	}
}

func TestSnapshotFailure(t *testing.T) {
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	sm.restore(testingSnap)

	sm.becomeCandidate()
	sm.becomeLeader()

	sm.prs[2].Next = 1
	sm.prs[2].becomeSnapshot(11)

	sm.Step(pb.Message{From: 2, To: 1, Type: pb.MsgSnapStatus, Reject: true})
	if sm.prs[2].PendingSnapshot != 0 {
		t.Fatalf("PendingSnapshot = %d, want 0", sm.prs[2].PendingSnapshot)
	}
	if sm.prs[2].Next != 1 {
		t.Fatalf("Next = %d, want 1", sm.prs[2].Next)
	}
	if sm.prs[2].Paused != true {
		t.Errorf("Paused = %v, want true", sm.prs[2].Paused)
	}
}

func TestSnapshotSucceed(t *testing.T) {
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	sm.restore(testingSnap)

	sm.becomeCandidate()
	sm.becomeLeader()

	sm.prs[2].Next = 1
	sm.prs[2].becomeSnapshot(11)

	sm.Step(pb.Message{From: 2, To: 1, Type: pb.MsgSnapStatus, Reject: false})
	if sm.prs[2].PendingSnapshot != 0 {
		t.Fatalf("PendingSnapshot = %d, want 0", sm.prs[2].PendingSnapshot)
	}
	if sm.prs[2].Next != 12 {
		t.Fatalf("Next = %d, want 12", sm.prs[2].Next)
	}
	if sm.prs[2].Paused != true {
		t.Errorf("Paused = %v, want true", sm.prs[2].Paused)
	}
}

func TestSnapshotAbort(t *testing.T) {
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	sm.restore(testingSnap)

	sm.becomeCandidate()
	sm.becomeLeader()

	sm.prs[2].Next = 1
	sm.prs[2].becomeSnapshot(11)

	// A successful msgAppResp that has a higher/equal index than the
	// pending snapshot should abort the pending snapshot.
	sm.Step(pb.Message{From: 2, To: 1, Type: pb.MsgAppResp, Index: 11})
	if sm.prs[2].PendingSnapshot != 0 {
		t.Fatalf("PendingSnapshot = %d, want 0", sm.prs[2].PendingSnapshot)
	}
	if sm.prs[2].Next != 12 {
		t.Fatalf("Next = %d, want 12", sm.prs[2].Next)
	}
}
