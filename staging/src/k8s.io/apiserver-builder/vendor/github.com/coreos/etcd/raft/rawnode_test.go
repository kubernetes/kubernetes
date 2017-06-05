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

package raft

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/coreos/etcd/raft/raftpb"
)

// TestRawNodeStep ensures that RawNode.Step ignore local message.
func TestRawNodeStep(t *testing.T) {
	for i, msgn := range raftpb.MessageType_name {
		s := NewMemoryStorage()
		rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, s), []Peer{{ID: 1}})
		if err != nil {
			t.Fatal(err)
		}
		msgt := raftpb.MessageType(i)
		err = rawNode.Step(raftpb.Message{Type: msgt})
		// LocalMsg should be ignored.
		if IsLocalMsg(msgt) {
			if err != ErrStepLocalMsg {
				t.Errorf("%d: step should ignore %s", msgt, msgn)
			}
		}
	}
}

// TestNodeStepUnblock from node_test.go has no equivalent in rawNode because there is
// no goroutine in RawNode.

// TestRawNodeProposeAndConfChange ensures that RawNode.Propose and RawNode.ProposeConfChange
// send the given proposal and ConfChange to the underlying raft.
func TestRawNodeProposeAndConfChange(t *testing.T) {
	s := NewMemoryStorage()
	var err error
	rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, s), []Peer{{ID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	rawNode.Campaign()
	proposed := false
	var lastIndex uint64
	var ccdata []byte
	for {
		rd := rawNode.Ready()
		s.Append(rd.Entries)
		// Once we are the leader, propose a command and a ConfChange.
		if !proposed && rd.SoftState.Lead == rawNode.raft.id {
			rawNode.Propose([]byte("somedata"))

			cc := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1}
			ccdata, err = cc.Marshal()
			if err != nil {
				t.Fatal(err)
			}
			rawNode.ProposeConfChange(cc)

			proposed = true
		}
		rawNode.Advance(rd)

		// Exit when we have four entries: one ConfChange, one no-op for the election,
		// our proposed command and proposed ConfChange.
		lastIndex, err = s.LastIndex()
		if err != nil {
			t.Fatal(err)
		}
		if lastIndex >= 4 {
			break
		}
	}

	entries, err := s.Entries(lastIndex-1, lastIndex+1, noLimit)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("len(entries) = %d, want %d", len(entries), 2)
	}
	if !bytes.Equal(entries[0].Data, []byte("somedata")) {
		t.Errorf("entries[0].Data = %v, want %v", entries[0].Data, []byte("somedata"))
	}
	if entries[1].Type != raftpb.EntryConfChange {
		t.Fatalf("type = %v, want %v", entries[1].Type, raftpb.EntryConfChange)
	}
	if !bytes.Equal(entries[1].Data, ccdata) {
		t.Errorf("data = %v, want %v", entries[1].Data, ccdata)
	}
}

// TestBlockProposal from node_test.go has no equivalent in rawNode because there is
// no leader check in RawNode.

// TestNodeTick from node_test.go has no equivalent in rawNode because
// it reaches into the raft object which is not exposed.

// TestNodeStop from node_test.go has no equivalent in rawNode because there is
// no goroutine in RawNode.

// TestRawNodeStart ensures that a node can be started correctly. The node should
// start with correct configuration change entries, and can accept and commit
// proposals.
func TestRawNodeStart(t *testing.T) {
	cc := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1}
	ccdata, err := cc.Marshal()
	if err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	wants := []Ready{
		{
			SoftState: &SoftState{Lead: 1, RaftState: StateLeader},
			HardState: raftpb.HardState{Term: 2, Commit: 2, Vote: 1},
			Entries: []raftpb.Entry{
				{Type: raftpb.EntryConfChange, Term: 1, Index: 1, Data: ccdata},
				{Term: 2, Index: 2},
			},
			CommittedEntries: []raftpb.Entry{
				{Type: raftpb.EntryConfChange, Term: 1, Index: 1, Data: ccdata},
				{Term: 2, Index: 2},
			},
		},
		{
			HardState:        raftpb.HardState{Term: 2, Commit: 3, Vote: 1},
			Entries:          []raftpb.Entry{{Term: 2, Index: 3, Data: []byte("foo")}},
			CommittedEntries: []raftpb.Entry{{Term: 2, Index: 3, Data: []byte("foo")}},
		},
	}

	storage := NewMemoryStorage()
	rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, storage), []Peer{{ID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	rawNode.Campaign()
	rd := rawNode.Ready()
	t.Logf("rd %v", rd)
	if !reflect.DeepEqual(rd, wants[0]) {
		t.Fatalf("#%d: g = %+v,\n             w   %+v", 1, rd, wants[0])
	} else {
		storage.Append(rd.Entries)
		rawNode.Advance(rd)
	}

	rawNode.Propose([]byte("foo"))
	if rd = rawNode.Ready(); !reflect.DeepEqual(rd, wants[1]) {
		t.Errorf("#%d: g = %+v,\n             w   %+v", 2, rd, wants[1])
	} else {
		storage.Append(rd.Entries)
		rawNode.Advance(rd)
	}

	if rawNode.HasReady() {
		t.Errorf("unexpected Ready: %+v", rawNode.Ready())
	}
}

func TestRawNodeRestart(t *testing.T) {
	entries := []raftpb.Entry{
		{Term: 1, Index: 1},
		{Term: 1, Index: 2, Data: []byte("foo")},
	}
	st := raftpb.HardState{Term: 1, Commit: 1}

	want := Ready{
		HardState: emptyState,
		// commit up to commit index in st
		CommittedEntries: entries[:st.Commit],
	}

	storage := NewMemoryStorage()
	storage.SetHardState(st)
	storage.Append(entries)
	rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, storage), nil)
	if err != nil {
		t.Fatal(err)
	}
	rd := rawNode.Ready()
	if !reflect.DeepEqual(rd, want) {
		t.Errorf("g = %+v,\n             w   %+v", rd, want)
	}
	rawNode.Advance(rd)
	if rawNode.HasReady() {
		t.Errorf("unexpected Ready: %+v", rawNode.Ready())
	}
}

func TestRawNodeRestartFromSnapshot(t *testing.T) {
	snap := raftpb.Snapshot{
		Metadata: raftpb.SnapshotMetadata{
			ConfState: raftpb.ConfState{Nodes: []uint64{1, 2}},
			Index:     2,
			Term:      1,
		},
	}
	entries := []raftpb.Entry{
		{Term: 1, Index: 3, Data: []byte("foo")},
	}
	st := raftpb.HardState{Term: 1, Commit: 3}

	want := Ready{
		HardState: emptyState,
		// commit up to commit index in st
		CommittedEntries: entries,
	}

	s := NewMemoryStorage()
	s.SetHardState(st)
	s.ApplySnapshot(snap)
	s.Append(entries)
	rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, s), nil)
	if err != nil {
		t.Fatal(err)
	}
	if rd := rawNode.Ready(); !reflect.DeepEqual(rd, want) {
		t.Errorf("g = %+v,\n             w   %+v", rd, want)
	} else {
		rawNode.Advance(rd)
	}
	if rawNode.HasReady() {
		t.Errorf("unexpected Ready: %+v", rawNode.HasReady())
	}
}

// TestNodeAdvance from node_test.go has no equivalent in rawNode because there is
// no dependency check between Ready() and Advance()

func TestRawNodeStatus(t *testing.T) {
	storage := NewMemoryStorage()
	rawNode, err := NewRawNode(newTestConfig(1, nil, 10, 1, storage), []Peer{{ID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	status := rawNode.Status()
	if status == nil {
		t.Errorf("expected status struct, got nil")
	}
}
