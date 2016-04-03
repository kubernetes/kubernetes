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
	"reflect"
	"testing"

	pb "github.com/coreos/etcd/raft/raftpb"
)

func TestFindConflict(t *testing.T) {
	previousEnts := []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}}
	tests := []struct {
		ents      []pb.Entry
		wconflict uint64
	}{
		// no conflict, empty ent
		{[]pb.Entry{}, 0},
		{[]pb.Entry{}, 0},
		// no conflict
		{[]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}}, 0},
		{[]pb.Entry{{Index: 2, Term: 2}, {Index: 3, Term: 3}}, 0},
		{[]pb.Entry{{Index: 3, Term: 3}}, 0},
		// no conflict, but has new entries
		{[]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}, {Index: 4, Term: 4}, {Index: 5, Term: 4}}, 4},
		{[]pb.Entry{{Index: 2, Term: 2}, {Index: 3, Term: 3}, {Index: 4, Term: 4}, {Index: 5, Term: 4}}, 4},
		{[]pb.Entry{{Index: 3, Term: 3}, {Index: 4, Term: 4}, {Index: 5, Term: 4}}, 4},
		{[]pb.Entry{{Index: 4, Term: 4}, {Index: 5, Term: 4}}, 4},
		// conflicts with existing entries
		{[]pb.Entry{{Index: 1, Term: 4}, {Index: 2, Term: 4}}, 1},
		{[]pb.Entry{{Index: 2, Term: 1}, {Index: 3, Term: 4}, {Index: 4, Term: 4}}, 2},
		{[]pb.Entry{{Index: 3, Term: 1}, {Index: 4, Term: 2}, {Index: 5, Term: 4}, {Index: 6, Term: 4}}, 3},
	}

	for i, tt := range tests {
		raftLog := newLog(NewMemoryStorage(), raftLogger)
		raftLog.append(previousEnts...)

		gconflict := raftLog.findConflict(tt.ents)
		if gconflict != tt.wconflict {
			t.Errorf("#%d: conflict = %d, want %d", i, gconflict, tt.wconflict)
		}
	}
}

func TestIsUpToDate(t *testing.T) {
	previousEnts := []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}}
	raftLog := newLog(NewMemoryStorage(), raftLogger)
	raftLog.append(previousEnts...)
	tests := []struct {
		lastIndex uint64
		term      uint64
		wUpToDate bool
	}{
		// greater term, ignore lastIndex
		{raftLog.lastIndex() - 1, 4, true},
		{raftLog.lastIndex(), 4, true},
		{raftLog.lastIndex() + 1, 4, true},
		// smaller term, ignore lastIndex
		{raftLog.lastIndex() - 1, 2, false},
		{raftLog.lastIndex(), 2, false},
		{raftLog.lastIndex() + 1, 2, false},
		// equal term, lager lastIndex wins
		{raftLog.lastIndex() - 1, 3, false},
		{raftLog.lastIndex(), 3, true},
		{raftLog.lastIndex() + 1, 3, true},
	}

	for i, tt := range tests {
		gUpToDate := raftLog.isUpToDate(tt.lastIndex, tt.term)
		if gUpToDate != tt.wUpToDate {
			t.Errorf("#%d: uptodate = %v, want %v", i, gUpToDate, tt.wUpToDate)
		}
	}
}

func TestAppend(t *testing.T) {
	previousEnts := []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}
	tests := []struct {
		ents      []pb.Entry
		windex    uint64
		wents     []pb.Entry
		wunstable uint64
	}{
		{
			[]pb.Entry{},
			2,
			[]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}},
			3,
		},
		{
			[]pb.Entry{{Index: 3, Term: 2}},
			3,
			[]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 2}},
			3,
		},
		// conflicts with index 1
		{
			[]pb.Entry{{Index: 1, Term: 2}},
			1,
			[]pb.Entry{{Index: 1, Term: 2}},
			1,
		},
		// conflicts with index 2
		{
			[]pb.Entry{{Index: 2, Term: 3}, {Index: 3, Term: 3}},
			3,
			[]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 3}, {Index: 3, Term: 3}},
			2,
		},
	}

	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.Append(previousEnts)
		raftLog := newLog(storage, raftLogger)

		index := raftLog.append(tt.ents...)
		if index != tt.windex {
			t.Errorf("#%d: lastIndex = %d, want %d", i, index, tt.windex)
		}
		g, err := raftLog.entries(1, noLimit)
		if err != nil {
			t.Fatalf("#%d: unexpected error %v", i, err)
		}
		if !reflect.DeepEqual(g, tt.wents) {
			t.Errorf("#%d: logEnts = %+v, want %+v", i, g, tt.wents)
		}
		if goff := raftLog.unstable.offset; goff != tt.wunstable {
			t.Errorf("#%d: unstable = %d, want %d", i, goff, tt.wunstable)
		}
	}
}

// TestLogMaybeAppend ensures:
// If the given (index, term) matches with the existing log:
// 	1. If an existing entry conflicts with a new one (same index
// 	but different terms), delete the existing entry and all that
// 	follow it
// 	2.Append any new entries not already in the log
// If the given (index, term) does not match with the existing log:
// 	return false
func TestLogMaybeAppend(t *testing.T) {
	previousEnts := []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}}
	lastindex := uint64(3)
	lastterm := uint64(3)
	commit := uint64(1)

	tests := []struct {
		logTerm   uint64
		index     uint64
		committed uint64
		ents      []pb.Entry

		wlasti  uint64
		wappend bool
		wcommit uint64
		wpanic  bool
	}{
		// not match: term is different
		{
			lastterm - 1, lastindex, lastindex, []pb.Entry{{Index: lastindex + 1, Term: 4}},
			0, false, commit, false,
		},
		// not match: index out of bound
		{
			lastterm, lastindex + 1, lastindex, []pb.Entry{{Index: lastindex + 2, Term: 4}},
			0, false, commit, false,
		},
		// match with the last existing entry
		{
			lastterm, lastindex, lastindex, nil,
			lastindex, true, lastindex, false,
		},
		{
			lastterm, lastindex, lastindex + 1, nil,
			lastindex, true, lastindex, false, // do not increase commit higher than lastnewi
		},
		{
			lastterm, lastindex, lastindex - 1, nil,
			lastindex, true, lastindex - 1, false, // commit up to the commit in the message
		},
		{
			lastterm, lastindex, 0, nil,
			lastindex, true, commit, false, // commit do not decrease
		},
		{
			0, 0, lastindex, nil,
			0, true, commit, false, // commit do not decrease
		},
		{
			lastterm, lastindex, lastindex, []pb.Entry{{Index: lastindex + 1, Term: 4}},
			lastindex + 1, true, lastindex, false,
		},
		{
			lastterm, lastindex, lastindex + 1, []pb.Entry{{Index: lastindex + 1, Term: 4}},
			lastindex + 1, true, lastindex + 1, false,
		},
		{
			lastterm, lastindex, lastindex + 2, []pb.Entry{{Index: lastindex + 1, Term: 4}},
			lastindex + 1, true, lastindex + 1, false, // do not increase commit higher than lastnewi
		},
		{
			lastterm, lastindex, lastindex + 2, []pb.Entry{{Index: lastindex + 1, Term: 4}, {Index: lastindex + 2, Term: 4}},
			lastindex + 2, true, lastindex + 2, false,
		},
		// match with the the entry in the middle
		{
			lastterm - 1, lastindex - 1, lastindex, []pb.Entry{{Index: lastindex, Term: 4}},
			lastindex, true, lastindex, false,
		},
		{
			lastterm - 2, lastindex - 2, lastindex, []pb.Entry{{Index: lastindex - 1, Term: 4}},
			lastindex - 1, true, lastindex - 1, false,
		},
		{
			lastterm - 3, lastindex - 3, lastindex, []pb.Entry{{Index: lastindex - 2, Term: 4}},
			lastindex - 2, true, lastindex - 2, true, // conflict with existing committed entry
		},
		{
			lastterm - 2, lastindex - 2, lastindex, []pb.Entry{{Index: lastindex - 1, Term: 4}, {Index: lastindex, Term: 4}},
			lastindex, true, lastindex, false,
		},
	}

	for i, tt := range tests {
		raftLog := newLog(NewMemoryStorage(), raftLogger)
		raftLog.append(previousEnts...)
		raftLog.committed = commit
		func() {
			defer func() {
				if r := recover(); r != nil {
					if tt.wpanic != true {
						t.Errorf("%d: panic = %v, want %v", i, true, tt.wpanic)
					}
				}
			}()
			glasti, gappend := raftLog.maybeAppend(tt.index, tt.logTerm, tt.committed, tt.ents...)
			gcommit := raftLog.committed

			if glasti != tt.wlasti {
				t.Errorf("#%d: lastindex = %d, want %d", i, glasti, tt.wlasti)
			}
			if gappend != tt.wappend {
				t.Errorf("#%d: append = %v, want %v", i, gappend, tt.wappend)
			}
			if gcommit != tt.wcommit {
				t.Errorf("#%d: committed = %d, want %d", i, gcommit, tt.wcommit)
			}
			if gappend && len(tt.ents) != 0 {
				gents, err := raftLog.slice(raftLog.lastIndex()-uint64(len(tt.ents))+1, raftLog.lastIndex()+1, noLimit)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if !reflect.DeepEqual(tt.ents, gents) {
					t.Errorf("%d: appended entries = %v, want %v", i, gents, tt.ents)
				}
			}
		}()
	}
}

// TestCompactionSideEffects ensures that all the log related functionality works correctly after
// a compaction.
func TestCompactionSideEffects(t *testing.T) {
	var i uint64
	// Populate the log with 1000 entries; 750 in stable storage and 250 in unstable.
	lastIndex := uint64(1000)
	unstableIndex := uint64(750)
	lastTerm := lastIndex
	storage := NewMemoryStorage()
	for i = 1; i <= unstableIndex; i++ {
		storage.Append([]pb.Entry{{Term: uint64(i), Index: uint64(i)}})
	}
	raftLog := newLog(storage, raftLogger)
	for i = unstableIndex; i < lastIndex; i++ {
		raftLog.append(pb.Entry{Term: uint64(i + 1), Index: uint64(i + 1)})
	}

	ok := raftLog.maybeCommit(lastIndex, lastTerm)
	if !ok {
		t.Fatalf("maybeCommit returned false")
	}
	raftLog.appliedTo(raftLog.committed)

	offset := uint64(500)
	storage.Compact(offset)

	if raftLog.lastIndex() != lastIndex {
		t.Errorf("lastIndex = %d, want %d", raftLog.lastIndex(), lastIndex)
	}

	for j := offset; j <= raftLog.lastIndex(); j++ {
		if mustTerm(raftLog.term(j)) != j {
			t.Errorf("term(%d) = %d, want %d", j, mustTerm(raftLog.term(j)), j)
		}
	}

	for j := offset; j <= raftLog.lastIndex(); j++ {
		if !raftLog.matchTerm(j, j) {
			t.Errorf("matchTerm(%d) = false, want true", j)
		}
	}

	unstableEnts := raftLog.unstableEntries()
	if g := len(unstableEnts); g != 250 {
		t.Errorf("len(unstableEntries) = %d, want = %d", g, 250)
	}
	if unstableEnts[0].Index != 751 {
		t.Errorf("Index = %d, want = %d", unstableEnts[0].Index, 751)
	}

	prev := raftLog.lastIndex()
	raftLog.append(pb.Entry{Index: raftLog.lastIndex() + 1, Term: raftLog.lastIndex() + 1})
	if raftLog.lastIndex() != prev+1 {
		t.Errorf("lastIndex = %d, want = %d", raftLog.lastIndex(), prev+1)
	}

	ents, err := raftLog.entries(raftLog.lastIndex(), noLimit)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(ents) != 1 {
		t.Errorf("len(entries) = %d, want = %d", len(ents), 1)
	}
}

func TestHasNextEnts(t *testing.T) {
	snap := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{Term: 1, Index: 3},
	}
	ents := []pb.Entry{
		{Term: 1, Index: 4},
		{Term: 1, Index: 5},
		{Term: 1, Index: 6},
	}
	tests := []struct {
		applied uint64
		hasNext bool
	}{
		{0, true},
		{3, true},
		{4, true},
		{5, false},
	}
	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.ApplySnapshot(snap)
		raftLog := newLog(storage, raftLogger)
		raftLog.append(ents...)
		raftLog.maybeCommit(5, 1)
		raftLog.appliedTo(tt.applied)

		hasNext := raftLog.hasNextEnts()
		if hasNext != tt.hasNext {
			t.Errorf("#%d: hasNext = %v, want %v", i, hasNext, tt.hasNext)
		}
	}
}

func TestNextEnts(t *testing.T) {
	snap := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{Term: 1, Index: 3},
	}
	ents := []pb.Entry{
		{Term: 1, Index: 4},
		{Term: 1, Index: 5},
		{Term: 1, Index: 6},
	}
	tests := []struct {
		applied uint64
		wents   []pb.Entry
	}{
		{0, ents[:2]},
		{3, ents[:2]},
		{4, ents[1:2]},
		{5, nil},
	}
	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.ApplySnapshot(snap)
		raftLog := newLog(storage, raftLogger)
		raftLog.append(ents...)
		raftLog.maybeCommit(5, 1)
		raftLog.appliedTo(tt.applied)

		nents := raftLog.nextEnts()
		if !reflect.DeepEqual(nents, tt.wents) {
			t.Errorf("#%d: nents = %+v, want %+v", i, nents, tt.wents)
		}
	}
}

// TestUnstableEnts ensures unstableEntries returns the unstable part of the
// entries correctly.
func TestUnstableEnts(t *testing.T) {
	previousEnts := []pb.Entry{{Term: 1, Index: 1}, {Term: 2, Index: 2}}
	tests := []struct {
		unstable uint64
		wents    []pb.Entry
	}{
		{3, nil},
		{1, previousEnts},
	}

	for i, tt := range tests {
		// append stable entries to storage
		storage := NewMemoryStorage()
		storage.Append(previousEnts[:tt.unstable-1])

		// append unstable entries to raftlog
		raftLog := newLog(storage, raftLogger)
		raftLog.append(previousEnts[tt.unstable-1:]...)

		ents := raftLog.unstableEntries()
		if l := len(ents); l > 0 {
			raftLog.stableTo(ents[l-1].Index, ents[l-i].Term)
		}
		if !reflect.DeepEqual(ents, tt.wents) {
			t.Errorf("#%d: unstableEnts = %+v, want %+v", i, ents, tt.wents)
		}
		w := previousEnts[len(previousEnts)-1].Index + 1
		if g := raftLog.unstable.offset; g != w {
			t.Errorf("#%d: unstable = %d, want %d", i, g, w)
		}
	}
}

func TestCommitTo(t *testing.T) {
	previousEnts := []pb.Entry{{Term: 1, Index: 1}, {Term: 2, Index: 2}, {Term: 3, Index: 3}}
	commit := uint64(2)
	tests := []struct {
		commit  uint64
		wcommit uint64
		wpanic  bool
	}{
		{3, 3, false},
		{1, 2, false}, // never decrease
		{4, 0, true},  // commit out of range -> panic
	}
	for i, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if tt.wpanic != true {
						t.Errorf("%d: panic = %v, want %v", i, true, tt.wpanic)
					}
				}
			}()
			raftLog := newLog(NewMemoryStorage(), raftLogger)
			raftLog.append(previousEnts...)
			raftLog.committed = commit
			raftLog.commitTo(tt.commit)
			if raftLog.committed != tt.wcommit {
				t.Errorf("#%d: committed = %d, want %d", i, raftLog.committed, tt.wcommit)
			}
		}()
	}
}

func TestStableTo(t *testing.T) {
	tests := []struct {
		stablei   uint64
		stablet   uint64
		wunstable uint64
	}{
		{1, 1, 2},
		{2, 2, 3},
		{2, 1, 1}, // bad term
		{3, 1, 1}, // bad index
	}
	for i, tt := range tests {
		raftLog := newLog(NewMemoryStorage(), raftLogger)
		raftLog.append([]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}...)
		raftLog.stableTo(tt.stablei, tt.stablet)
		if raftLog.unstable.offset != tt.wunstable {
			t.Errorf("#%d: unstable = %d, want %d", i, raftLog.unstable.offset, tt.wunstable)
		}
	}
}

func TestStableToWithSnap(t *testing.T) {
	snapi, snapt := uint64(5), uint64(2)
	tests := []struct {
		stablei uint64
		stablet uint64
		newEnts []pb.Entry

		wunstable uint64
	}{
		{snapi + 1, snapt, nil, snapi + 1},
		{snapi, snapt, nil, snapi + 1},
		{snapi - 1, snapt, nil, snapi + 1},

		{snapi + 1, snapt + 1, nil, snapi + 1},
		{snapi, snapt + 1, nil, snapi + 1},
		{snapi - 1, snapt + 1, nil, snapi + 1},

		{snapi + 1, snapt, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 2},
		{snapi, snapt, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 1},
		{snapi - 1, snapt, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 1},

		{snapi + 1, snapt + 1, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 1},
		{snapi, snapt + 1, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 1},
		{snapi - 1, snapt + 1, []pb.Entry{{Index: snapi + 1, Term: snapt}}, snapi + 1},
	}
	for i, tt := range tests {
		s := NewMemoryStorage()
		s.ApplySnapshot(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: snapi, Term: snapt}})
		raftLog := newLog(s, raftLogger)
		raftLog.append(tt.newEnts...)
		raftLog.stableTo(tt.stablei, tt.stablet)
		if raftLog.unstable.offset != tt.wunstable {
			t.Errorf("#%d: unstable = %d, want %d", i, raftLog.unstable.offset, tt.wunstable)
		}
	}
}

//TestCompaction ensures that the number of log entries is correct after compactions.
func TestCompaction(t *testing.T) {
	tests := []struct {
		lastIndex uint64
		compact   []uint64
		wleft     []int
		wallow    bool
	}{
		// out of upper bound
		{1000, []uint64{1001}, []int{-1}, false},
		{1000, []uint64{300, 500, 800, 900}, []int{700, 500, 200, 100}, true},
		// out of lower bound
		{1000, []uint64{300, 299}, []int{700, -1}, false},
	}

	for i, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if tt.wallow == true {
						t.Errorf("%d: allow = %v, want %v: %v", i, false, true, r)
					}
				}
			}()

			storage := NewMemoryStorage()
			for i := uint64(1); i <= tt.lastIndex; i++ {
				storage.Append([]pb.Entry{{Index: i}})
			}
			raftLog := newLog(storage, raftLogger)
			raftLog.maybeCommit(tt.lastIndex, 0)
			raftLog.appliedTo(raftLog.committed)

			for j := 0; j < len(tt.compact); j++ {
				err := storage.Compact(tt.compact[j])
				if err != nil {
					if tt.wallow {
						t.Errorf("#%d.%d allow = %t, want %t", i, j, false, tt.wallow)
					}
					continue
				}
				if len(raftLog.allEntries()) != tt.wleft[j] {
					t.Errorf("#%d.%d len = %d, want %d", i, j, len(raftLog.allEntries()), tt.wleft[j])
				}
			}
		}()
	}
}

func TestLogRestore(t *testing.T) {
	index := uint64(1000)
	term := uint64(1000)
	snap := pb.SnapshotMetadata{Index: index, Term: term}
	storage := NewMemoryStorage()
	storage.ApplySnapshot(pb.Snapshot{Metadata: snap})
	raftLog := newLog(storage, raftLogger)

	if len(raftLog.allEntries()) != 0 {
		t.Errorf("len = %d, want 0", len(raftLog.allEntries()))
	}
	if raftLog.firstIndex() != index+1 {
		t.Errorf("firstIndex = %d, want %d", raftLog.firstIndex(), index+1)
	}
	if raftLog.committed != index {
		t.Errorf("committed = %d, want %d", raftLog.committed, index)
	}
	if raftLog.unstable.offset != index+1 {
		t.Errorf("unstable = %d, want %d", raftLog.unstable.offset, index+1)
	}
	if mustTerm(raftLog.term(index)) != term {
		t.Errorf("term = %d, want %d", mustTerm(raftLog.term(index)), term)
	}
}

func TestIsOutOfBounds(t *testing.T) {
	offset := uint64(100)
	num := uint64(100)
	storage := NewMemoryStorage()
	storage.ApplySnapshot(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: offset}})
	l := newLog(storage, raftLogger)
	for i := uint64(1); i <= num; i++ {
		l.append(pb.Entry{Index: i + offset})
	}

	first := offset + 1
	tests := []struct {
		lo, hi        uint64
		wpanic        bool
		wErrCompacted bool
	}{
		{
			first - 2, first + 1,
			false,
			true,
		},
		{
			first - 1, first + 1,
			false,
			true,
		},
		{
			first, first,
			false,
			false,
		},
		{
			first + num/2, first + num/2,
			false,
			false,
		},
		{
			first + num - 1, first + num - 1,
			false,
			false,
		},
		{
			first + num, first + num,
			false,
			false,
		},
		{
			first + num, first + num + 1,
			true,
			false,
		},
		{
			first + num + 1, first + num + 1,
			true,
			false,
		},
	}

	for i, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if !tt.wpanic {
						t.Errorf("%d: panic = %v, want %v: %v", i, true, false, r)
					}
				}
			}()
			err := l.mustCheckOutOfBounds(tt.lo, tt.hi)
			if tt.wpanic {
				t.Errorf("%d: panic = %v, want %v", i, false, true)
			}
			if tt.wErrCompacted && err != ErrCompacted {
				t.Errorf("%d: err = %v, want %v", i, err, ErrCompacted)
			}
			if !tt.wErrCompacted && err != nil {
				t.Errorf("%d: unexpected err %v", i, err)
			}
		}()
	}
}

func TestTerm(t *testing.T) {
	var i uint64
	offset := uint64(100)
	num := uint64(100)

	storage := NewMemoryStorage()
	storage.ApplySnapshot(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: offset, Term: 1}})
	l := newLog(storage, raftLogger)
	for i = 1; i < num; i++ {
		l.append(pb.Entry{Index: offset + i, Term: i})
	}

	tests := []struct {
		index uint64
		w     uint64
	}{
		{offset - 1, 0},
		{offset, 1},
		{offset + num/2, num / 2},
		{offset + num - 1, num - 1},
		{offset + num, 0},
	}

	for j, tt := range tests {
		term := mustTerm(l.term(tt.index))
		if !reflect.DeepEqual(term, tt.w) {
			t.Errorf("#%d: at = %d, want %d", j, term, tt.w)
		}
	}
}

func TestTermWithUnstableSnapshot(t *testing.T) {
	storagesnapi := uint64(100)
	unstablesnapi := storagesnapi + 5

	storage := NewMemoryStorage()
	storage.ApplySnapshot(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: storagesnapi, Term: 1}})
	l := newLog(storage, raftLogger)
	l.restore(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: unstablesnapi, Term: 1}})

	tests := []struct {
		index uint64
		w     uint64
	}{
		// cannot get term from storage
		{storagesnapi, 0},
		// cannot get term from the gap between storage ents and unstable snapshot
		{storagesnapi + 1, 0},
		{unstablesnapi - 1, 0},
		// get term from unstable snapshot index
		{unstablesnapi, 1},
	}

	for i, tt := range tests {
		term := mustTerm(l.term(tt.index))
		if !reflect.DeepEqual(term, tt.w) {
			t.Errorf("#%d: at = %d, want %d", i, term, tt.w)
		}
	}
}

func TestSlice(t *testing.T) {
	var i uint64
	offset := uint64(100)
	num := uint64(100)
	last := offset + num
	half := offset + num/2
	halfe := pb.Entry{Index: half, Term: half}

	storage := NewMemoryStorage()
	storage.ApplySnapshot(pb.Snapshot{Metadata: pb.SnapshotMetadata{Index: offset}})
	for i = 1; i < num/2; i++ {
		storage.Append([]pb.Entry{{Index: offset + i, Term: offset + i}})
	}
	l := newLog(storage, raftLogger)
	for i = num / 2; i < num; i++ {
		l.append(pb.Entry{Index: offset + i, Term: offset + i})
	}

	tests := []struct {
		from  uint64
		to    uint64
		limit uint64

		w      []pb.Entry
		wpanic bool
	}{
		// test no limit
		{offset - 1, offset + 1, noLimit, nil, false},
		{offset, offset + 1, noLimit, nil, false},
		{half - 1, half + 1, noLimit, []pb.Entry{{Index: half - 1, Term: half - 1}, {Index: half, Term: half}}, false},
		{half, half + 1, noLimit, []pb.Entry{{Index: half, Term: half}}, false},
		{last - 1, last, noLimit, []pb.Entry{{Index: last - 1, Term: last - 1}}, false},
		{last, last + 1, noLimit, nil, true},

		// test limit
		{half - 1, half + 1, 0, []pb.Entry{{Index: half - 1, Term: half - 1}}, false},
		{half - 1, half + 1, uint64(halfe.Size() + 1), []pb.Entry{{Index: half - 1, Term: half - 1}}, false},
		{half - 1, half + 1, uint64(halfe.Size() * 2), []pb.Entry{{Index: half - 1, Term: half - 1}, {Index: half, Term: half}}, false},
		{half - 1, half + 2, uint64(halfe.Size() * 3), []pb.Entry{{Index: half - 1, Term: half - 1}, {Index: half, Term: half}, {Index: half + 1, Term: half + 1}}, false},
		{half, half + 2, uint64(halfe.Size()), []pb.Entry{{Index: half, Term: half}}, false},
		{half, half + 2, uint64(halfe.Size() * 2), []pb.Entry{{Index: half, Term: half}, {Index: half + 1, Term: half + 1}}, false},
	}

	for j, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if !tt.wpanic {
						t.Errorf("%d: panic = %v, want %v: %v", j, true, false, r)
					}
				}
			}()
			g, err := l.slice(tt.from, tt.to, tt.limit)
			if tt.from <= offset && err != ErrCompacted {
				t.Fatalf("#%d: err = %v, want %v", j, err, ErrCompacted)
			}
			if tt.from > offset && err != nil {
				t.Fatalf("#%d: unexpected error %v", j, err)
			}
			if !reflect.DeepEqual(g, tt.w) {
				t.Errorf("#%d: from %d to %d = %v, want %v", j, tt.from, tt.to, g, tt.w)
			}
		}()
	}
}

func mustTerm(term uint64, err error) uint64 {
	if err != nil {
		panic(err)
	}
	return term
}
