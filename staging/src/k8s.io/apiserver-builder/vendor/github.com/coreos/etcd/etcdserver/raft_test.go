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
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/pkg/mock/mockstorage"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/rafthttp"
)

func TestGetIDs(t *testing.T) {
	addcc := &raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 2}
	addEntry := raftpb.Entry{Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(addcc)}
	removecc := &raftpb.ConfChange{Type: raftpb.ConfChangeRemoveNode, NodeID: 2}
	removeEntry := raftpb.Entry{Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc)}
	normalEntry := raftpb.Entry{Type: raftpb.EntryNormal}
	updatecc := &raftpb.ConfChange{Type: raftpb.ConfChangeUpdateNode, NodeID: 2}
	updateEntry := raftpb.Entry{Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(updatecc)}

	tests := []struct {
		confState *raftpb.ConfState
		ents      []raftpb.Entry

		widSet []uint64
	}{
		{nil, []raftpb.Entry{}, []uint64{}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{}, []uint64{1}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{addEntry}, []uint64{1, 2}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{addEntry, removeEntry}, []uint64{1}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{addEntry, normalEntry}, []uint64{1, 2}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{addEntry, normalEntry, updateEntry}, []uint64{1, 2}},
		{&raftpb.ConfState{Nodes: []uint64{1}},
			[]raftpb.Entry{addEntry, removeEntry, normalEntry}, []uint64{1}},
	}

	for i, tt := range tests {
		var snap raftpb.Snapshot
		if tt.confState != nil {
			snap.Metadata.ConfState = *tt.confState
		}
		idSet := getIDs(&snap, tt.ents)
		if !reflect.DeepEqual(idSet, tt.widSet) {
			t.Errorf("#%d: idset = %#v, want %#v", i, idSet, tt.widSet)
		}
	}
}

func TestCreateConfigChangeEnts(t *testing.T) {
	m := membership.Member{
		ID:             types.ID(1),
		RaftAttributes: membership.RaftAttributes{PeerURLs: []string{"http://localhost:2380"}},
	}
	ctx, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	addcc1 := &raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1, Context: ctx}
	removecc2 := &raftpb.ConfChange{Type: raftpb.ConfChangeRemoveNode, NodeID: 2}
	removecc3 := &raftpb.ConfChange{Type: raftpb.ConfChangeRemoveNode, NodeID: 3}
	tests := []struct {
		ids         []uint64
		self        uint64
		term, index uint64

		wents []raftpb.Entry
	}{
		{
			[]uint64{1},
			1,
			1, 1,

			[]raftpb.Entry{},
		},
		{
			[]uint64{1, 2},
			1,
			1, 1,

			[]raftpb.Entry{{Term: 1, Index: 2, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc2)}},
		},
		{
			[]uint64{1, 2},
			1,
			2, 2,

			[]raftpb.Entry{{Term: 2, Index: 3, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc2)}},
		},
		{
			[]uint64{1, 2, 3},
			1,
			2, 2,

			[]raftpb.Entry{
				{Term: 2, Index: 3, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc2)},
				{Term: 2, Index: 4, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc3)},
			},
		},
		{
			[]uint64{2, 3},
			2,
			2, 2,

			[]raftpb.Entry{
				{Term: 2, Index: 3, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc3)},
			},
		},
		{
			[]uint64{2, 3},
			1,
			2, 2,

			[]raftpb.Entry{
				{Term: 2, Index: 3, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc2)},
				{Term: 2, Index: 4, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(removecc3)},
				{Term: 2, Index: 5, Type: raftpb.EntryConfChange, Data: pbutil.MustMarshal(addcc1)},
			},
		},
	}

	for i, tt := range tests {
		gents := createConfigChangeEnts(tt.ids, tt.self, tt.term, tt.index)
		if !reflect.DeepEqual(gents, tt.wents) {
			t.Errorf("#%d: ents = %v, want %v", i, gents, tt.wents)
		}
	}
}

func TestStopRaftWhenWaitingForApplyDone(t *testing.T) {
	n := newNopReadyNode()
	srv := &EtcdServer{r: raftNode{
		Node:        n,
		storage:     mockstorage.NewStorageRecorder(""),
		raftStorage: raft.NewMemoryStorage(),
		transport:   rafthttp.NewNopTransporter(),
	}}
	srv.r.start(srv)
	n.readyc <- raft.Ready{}
	select {
	case <-srv.r.applyc:
	case <-time.After(time.Second):
		t.Fatalf("failed to receive apply struct")
	}

	srv.r.stopped <- struct{}{}
	select {
	case <-srv.r.done:
	case <-time.After(time.Second):
		t.Fatalf("failed to stop raft loop")
	}
}
