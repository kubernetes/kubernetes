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
	"reflect"
	"testing"

	pb "github.com/coreos/etcd/raft/raftpb"
)

func TestUnstableMaybeFirstIndex(t *testing.T) {
	tests := []struct {
		entries []pb.Entry
		offset  uint64
		snap    *pb.Snapshot

		wok    bool
		windex uint64
	}{
		// no snapshot
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			false, 0,
		},
		{
			[]pb.Entry{}, 0, nil,
			false, 0,
		},
		// has snapshot
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			true, 5,
		},
		{
			[]pb.Entry{}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			true, 5,
		},
	}

	for i, tt := range tests {
		u := unstable{
			entries:  tt.entries,
			offset:   tt.offset,
			snapshot: tt.snap,
			logger:   raftLogger,
		}
		index, ok := u.maybeFirstIndex()
		if ok != tt.wok {
			t.Errorf("#%d: ok = %t, want %t", i, ok, tt.wok)
		}
		if index != tt.windex {
			t.Errorf("#%d: index = %d, want %d", i, index, tt.windex)
		}
	}
}

func TestMaybeLastIndex(t *testing.T) {
	tests := []struct {
		entries []pb.Entry
		offset  uint64
		snap    *pb.Snapshot

		wok    bool
		windex uint64
	}{
		// last in entries
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			true, 5,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			true, 5,
		},
		// last in snapshot
		{
			[]pb.Entry{}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			true, 4,
		},
		// empty unstable
		{
			[]pb.Entry{}, 0, nil,
			false, 0,
		},
	}

	for i, tt := range tests {
		u := unstable{
			entries:  tt.entries,
			offset:   tt.offset,
			snapshot: tt.snap,
			logger:   raftLogger,
		}
		index, ok := u.maybeLastIndex()
		if ok != tt.wok {
			t.Errorf("#%d: ok = %t, want %t", i, ok, tt.wok)
		}
		if index != tt.windex {
			t.Errorf("#%d: index = %d, want %d", i, index, tt.windex)
		}
	}
}

func TestUnstableMaybeTerm(t *testing.T) {
	tests := []struct {
		entries []pb.Entry
		offset  uint64
		snap    *pb.Snapshot
		index   uint64

		wok   bool
		wterm uint64
	}{
		// term from entries
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			5,
			true, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			6,
			false, 0,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			4,
			false, 0,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			5,
			true, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			6,
			false, 0,
		},
		// term from snapshot
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			4,
			true, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			3,
			false, 0,
		},
		{
			[]pb.Entry{}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			5,
			false, 0,
		},
		{
			[]pb.Entry{}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			4,
			true, 1,
		},
		{
			[]pb.Entry{}, 0, nil,
			5,
			false, 0,
		},
	}

	for i, tt := range tests {
		u := unstable{
			entries:  tt.entries,
			offset:   tt.offset,
			snapshot: tt.snap,
			logger:   raftLogger,
		}
		term, ok := u.maybeTerm(tt.index)
		if ok != tt.wok {
			t.Errorf("#%d: ok = %t, want %t", i, ok, tt.wok)
		}
		if term != tt.wterm {
			t.Errorf("#%d: term = %d, want %d", i, term, tt.wterm)
		}
	}
}

func TestUnstableRestore(t *testing.T) {
	u := unstable{
		entries:  []pb.Entry{{Index: 5, Term: 1}},
		offset:   5,
		snapshot: &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
		logger:   raftLogger,
	}
	s := pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 6, Term: 2}}
	u.restore(s)

	if u.offset != s.Metadata.Index+1 {
		t.Errorf("offset = %d, want %d", u.offset, s.Metadata.Index+1)
	}
	if len(u.entries) != 0 {
		t.Errorf("len = %d, want 0", len(u.entries))
	}
	if !reflect.DeepEqual(u.snapshot, &s) {
		t.Errorf("snap = %v, want %v", u.snapshot, &s)
	}
}

func TestUnstableStableTo(t *testing.T) {
	tests := []struct {
		entries     []pb.Entry
		offset      uint64
		snap        *pb.Snapshot
		index, term uint64

		woffset uint64
		wlen    int
	}{
		{
			[]pb.Entry{}, 0, nil,
			5, 1,
			0, 0,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			5, 1, // stable to the first entry
			6, 0,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}}, 5, nil,
			5, 1, // stable to the first entry
			6, 1,
		},
		{
			[]pb.Entry{{Index: 6, Term: 2}}, 6, nil,
			6, 1, // stable to the first entry and term mismatch
			6, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			4, 1, // stable to old entry
			5, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			4, 2, // stable to old entry
			5, 1,
		},
		// with snapshot
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			5, 1, // stable to the first entry
			6, 0,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			5, 1, // stable to the first entry
			6, 1,
		},
		{
			[]pb.Entry{{Index: 6, Term: 2}}, 6, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 5, Term: 1}},
			6, 1, // stable to the first entry and term mismatch
			6, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 1}},
			4, 1, // stable to snapshot
			5, 1,
		},
		{
			[]pb.Entry{{Index: 5, Term: 2}}, 5, &pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: 4, Term: 2}},
			4, 1, // stable to old entry
			5, 1,
		},
	}

	for i, tt := range tests {
		u := unstable{
			entries:  tt.entries,
			offset:   tt.offset,
			snapshot: tt.snap,
			logger:   raftLogger,
		}
		u.stableTo(tt.index, tt.term)
		if u.offset != tt.woffset {
			t.Errorf("#%d: offset = %d, want %d", i, u.offset, tt.woffset)
		}
		if len(u.entries) != tt.wlen {
			t.Errorf("#%d: len = %d, want %d", i, len(u.entries), tt.wlen)
		}
	}
}

func TestUnstableTruncateAndAppend(t *testing.T) {
	tests := []struct {
		entries  []pb.Entry
		offset   uint64
		snap     *pb.Snapshot
		toappend []pb.Entry

		woffset  uint64
		wentries []pb.Entry
	}{
		// append to the end
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			[]pb.Entry{{Index: 6, Term: 1}, {Index: 7, Term: 1}},
			5, []pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}, {Index: 7, Term: 1}},
		},
		// replace the unstable entries
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			[]pb.Entry{{Index: 5, Term: 2}, {Index: 6, Term: 2}},
			5, []pb.Entry{{Index: 5, Term: 2}, {Index: 6, Term: 2}},
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}}, 5, nil,
			[]pb.Entry{{Index: 4, Term: 2}, {Index: 5, Term: 2}, {Index: 6, Term: 2}},
			4, []pb.Entry{{Index: 4, Term: 2}, {Index: 5, Term: 2}, {Index: 6, Term: 2}},
		},
		// truncate the existing entries and append
		{
			[]pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}, {Index: 7, Term: 1}}, 5, nil,
			[]pb.Entry{{Index: 6, Term: 2}},
			5, []pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 2}},
		},
		{
			[]pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}, {Index: 7, Term: 1}}, 5, nil,
			[]pb.Entry{{Index: 7, Term: 2}, {Index: 8, Term: 2}},
			5, []pb.Entry{{Index: 5, Term: 1}, {Index: 6, Term: 1}, {Index: 7, Term: 2}, {Index: 8, Term: 2}},
		},
	}

	for i, tt := range tests {
		u := unstable{
			entries:  tt.entries,
			offset:   tt.offset,
			snapshot: tt.snap,
			logger:   raftLogger,
		}
		u.truncateAndAppend(tt.toappend)
		if u.offset != tt.woffset {
			t.Errorf("#%d: offset = %d, want %d", i, u.offset, tt.woffset)
		}
		if !reflect.DeepEqual(u.entries, tt.wentries) {
			t.Errorf("#%d: entries = %v, want %v", i, u.entries, tt.wentries)
		}
	}
}
