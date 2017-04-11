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
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"

	pb "github.com/coreos/etcd/raft/raftpb"
)

// nextEnts returns the appliable entries and updates the applied index
func nextEnts(r *raft, s *MemoryStorage) (ents []pb.Entry) {
	// Transfer all unstable entries to "stable" storage.
	s.Append(r.raftLog.unstableEntries())
	r.raftLog.stableTo(r.raftLog.lastIndex(), r.raftLog.lastTerm())

	ents = r.raftLog.nextEnts()
	r.raftLog.appliedTo(r.raftLog.committed)
	return ents
}

type stateMachine interface {
	Step(m pb.Message) error
	readMessages() []pb.Message
}

func (r *raft) readMessages() []pb.Message {
	msgs := r.msgs
	r.msgs = make([]pb.Message, 0)

	return msgs
}

func TestProgressBecomeProbe(t *testing.T) {
	match := uint64(1)
	tests := []struct {
		p     *Progress
		wnext uint64
	}{
		{
			&Progress{State: ProgressStateReplicate, Match: match, Next: 5, ins: newInflights(256)},
			2,
		},
		{
			// snapshot finish
			&Progress{State: ProgressStateSnapshot, Match: match, Next: 5, PendingSnapshot: 10, ins: newInflights(256)},
			11,
		},
		{
			// snapshot failure
			&Progress{State: ProgressStateSnapshot, Match: match, Next: 5, PendingSnapshot: 0, ins: newInflights(256)},
			2,
		},
	}
	for i, tt := range tests {
		tt.p.becomeProbe()
		if tt.p.State != ProgressStateProbe {
			t.Errorf("#%d: state = %s, want %s", i, tt.p.State, ProgressStateProbe)
		}
		if tt.p.Match != match {
			t.Errorf("#%d: match = %d, want %d", i, tt.p.Match, match)
		}
		if tt.p.Next != tt.wnext {
			t.Errorf("#%d: next = %d, want %d", i, tt.p.Next, tt.wnext)
		}
	}
}

func TestProgressBecomeReplicate(t *testing.T) {
	p := &Progress{State: ProgressStateProbe, Match: 1, Next: 5, ins: newInflights(256)}
	p.becomeReplicate()

	if p.State != ProgressStateReplicate {
		t.Errorf("state = %s, want %s", p.State, ProgressStateReplicate)
	}
	if p.Match != 1 {
		t.Errorf("match = %d, want 1", p.Match)
	}
	if w := p.Match + 1; p.Next != w {
		t.Errorf("next = %d, want %d", p.Next, w)
	}
}

func TestProgressBecomeSnapshot(t *testing.T) {
	p := &Progress{State: ProgressStateProbe, Match: 1, Next: 5, ins: newInflights(256)}
	p.becomeSnapshot(10)

	if p.State != ProgressStateSnapshot {
		t.Errorf("state = %s, want %s", p.State, ProgressStateSnapshot)
	}
	if p.Match != 1 {
		t.Errorf("match = %d, want 1", p.Match)
	}
	if p.PendingSnapshot != 10 {
		t.Errorf("pendingSnapshot = %d, want 10", p.PendingSnapshot)
	}
}

func TestProgressUpdate(t *testing.T) {
	prevM, prevN := uint64(3), uint64(5)
	tests := []struct {
		update uint64

		wm  uint64
		wn  uint64
		wok bool
	}{
		{prevM - 1, prevM, prevN, false},        // do not decrease match, next
		{prevM, prevM, prevN, false},            // do not decrease next
		{prevM + 1, prevM + 1, prevN, true},     // increase match, do not decrease next
		{prevM + 2, prevM + 2, prevN + 1, true}, // increase match, next
	}
	for i, tt := range tests {
		p := &Progress{
			Match: prevM,
			Next:  prevN,
		}
		ok := p.maybeUpdate(tt.update)
		if ok != tt.wok {
			t.Errorf("#%d: ok= %v, want %v", i, ok, tt.wok)
		}
		if p.Match != tt.wm {
			t.Errorf("#%d: match= %d, want %d", i, p.Match, tt.wm)
		}
		if p.Next != tt.wn {
			t.Errorf("#%d: next= %d, want %d", i, p.Next, tt.wn)
		}
	}
}

func TestProgressMaybeDecr(t *testing.T) {
	tests := []struct {
		state    ProgressStateType
		m        uint64
		n        uint64
		rejected uint64
		last     uint64

		w  bool
		wn uint64
	}{
		{
			// state replicate and rejected is not greater than match
			ProgressStateReplicate, 5, 10, 5, 5, false, 10,
		},
		{
			// state replicate and rejected is not greater than match
			ProgressStateReplicate, 5, 10, 4, 4, false, 10,
		},
		{
			// state replicate and rejected is greater than match
			// directly decrease to match+1
			ProgressStateReplicate, 5, 10, 9, 9, true, 6,
		},
		{
			// next-1 != rejected is always false
			ProgressStateProbe, 0, 0, 0, 0, false, 0,
		},
		{
			// next-1 != rejected is always false
			ProgressStateProbe, 0, 10, 5, 5, false, 10,
		},
		{
			// next>1 = decremented by 1
			ProgressStateProbe, 0, 10, 9, 9, true, 9,
		},
		{
			// next>1 = decremented by 1
			ProgressStateProbe, 0, 2, 1, 1, true, 1,
		},
		{
			// next<=1 = reset to 1
			ProgressStateProbe, 0, 1, 0, 0, true, 1,
		},
		{
			// decrease to min(rejected, last+1)
			ProgressStateProbe, 0, 10, 9, 2, true, 3,
		},
		{
			// rejected < 1, reset to 1
			ProgressStateProbe, 0, 10, 9, 0, true, 1,
		},
	}
	for i, tt := range tests {
		p := &Progress{
			State: tt.state,
			Match: tt.m,
			Next:  tt.n,
		}
		if g := p.maybeDecrTo(tt.rejected, tt.last); g != tt.w {
			t.Errorf("#%d: maybeDecrTo= %t, want %t", i, g, tt.w)
		}
		if gm := p.Match; gm != tt.m {
			t.Errorf("#%d: match= %d, want %d", i, gm, tt.m)
		}
		if gn := p.Next; gn != tt.wn {
			t.Errorf("#%d: next= %d, want %d", i, gn, tt.wn)
		}
	}
}

func TestProgressIsPaused(t *testing.T) {
	tests := []struct {
		state  ProgressStateType
		paused bool

		w bool
	}{
		{ProgressStateProbe, false, false},
		{ProgressStateProbe, true, true},
		{ProgressStateReplicate, false, false},
		{ProgressStateReplicate, true, false},
		{ProgressStateSnapshot, false, true},
		{ProgressStateSnapshot, true, true},
	}
	for i, tt := range tests {
		p := &Progress{
			State:  tt.state,
			Paused: tt.paused,
			ins:    newInflights(256),
		}
		if g := p.isPaused(); g != tt.w {
			t.Errorf("#%d: paused= %t, want %t", i, g, tt.w)
		}
	}
}

// TestProgressResume ensures that progress.maybeUpdate and progress.maybeDecrTo
// will reset progress.paused.
func TestProgressResume(t *testing.T) {
	p := &Progress{
		Next:   2,
		Paused: true,
	}
	p.maybeDecrTo(1, 1)
	if p.Paused {
		t.Errorf("paused= %v, want false", p.Paused)
	}
	p.Paused = true
	p.maybeUpdate(2)
	if p.Paused {
		t.Errorf("paused= %v, want false", p.Paused)
	}
}

// TestProgressResumeByHeartbeat ensures raft.heartbeat reset progress.paused by heartbeat.
func TestProgressResumeByHeartbeat(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 5, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.prs[2].Paused = true

	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgBeat})
	if r.prs[2].Paused {
		t.Errorf("paused = %v, want false", r.prs[2].Paused)
	}
}

func TestProgressPaused(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 5, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})

	ms := r.readMessages()
	if len(ms) != 1 {
		t.Errorf("len(ms) = %d, want 1", len(ms))
	}
}

func TestLeaderElection(t *testing.T) {
	tests := []struct {
		*network
		state StateType
	}{
		{newNetwork(nil, nil, nil), StateLeader},
		{newNetwork(nil, nil, nopStepper), StateLeader},
		{newNetwork(nil, nopStepper, nopStepper), StateCandidate},
		{newNetwork(nil, nopStepper, nopStepper, nil), StateCandidate},
		{newNetwork(nil, nopStepper, nopStepper, nil, nil), StateLeader},

		// three logs further along than 0
		{newNetwork(nil, ents(1), ents(2), ents(1, 3), nil), StateFollower},

		// logs converge
		{newNetwork(ents(1), nil, ents(2), ents(1), nil), StateLeader},
	}

	for i, tt := range tests {
		tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
		sm := tt.network.peers[1].(*raft)
		if sm.state != tt.state {
			t.Errorf("#%d: state = %s, want %s", i, sm.state, tt.state)
		}
		if g := sm.Term; g != 1 {
			t.Errorf("#%d: term = %d, want %d", i, g, 1)
		}
	}
}

func TestLogReplication(t *testing.T) {
	tests := []struct {
		*network
		msgs       []pb.Message
		wcommitted uint64
	}{
		{
			newNetwork(nil, nil, nil),
			[]pb.Message{
				{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}},
			},
			2,
		},
		{
			newNetwork(nil, nil, nil),
			[]pb.Message{
				{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}},
				{From: 1, To: 2, Type: pb.MsgHup},
				{From: 1, To: 2, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}},
			},
			4,
		},
	}

	for i, tt := range tests {
		tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

		for _, m := range tt.msgs {
			tt.send(m)
		}

		for j, x := range tt.network.peers {
			sm := x.(*raft)

			if sm.raftLog.committed != tt.wcommitted {
				t.Errorf("#%d.%d: committed = %d, want %d", i, j, sm.raftLog.committed, tt.wcommitted)
			}

			ents := []pb.Entry{}
			for _, e := range nextEnts(sm, tt.network.storage[j]) {
				if e.Data != nil {
					ents = append(ents, e)
				}
			}
			props := []pb.Message{}
			for _, m := range tt.msgs {
				if m.Type == pb.MsgProp {
					props = append(props, m)
				}
			}
			for k, m := range props {
				if !bytes.Equal(ents[k].Data, m.Entries[0].Data) {
					t.Errorf("#%d.%d: data = %d, want %d", i, j, ents[k].Data, m.Entries[0].Data)
				}
			}
		}
	}
}

func TestSingleNodeCommit(t *testing.T) {
	tt := newNetwork(nil)
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})

	sm := tt.peers[1].(*raft)
	if sm.raftLog.committed != 3 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 3)
	}
}

// TestCannotCommitWithoutNewTermEntry tests the entries cannot be committed
// when leader changes, no new proposal comes in and ChangeTerm proposal is
// filtered.
func TestCannotCommitWithoutNewTermEntry(t *testing.T) {
	tt := newNetwork(nil, nil, nil, nil, nil)
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	// 0 cannot reach 2,3,4
	tt.cut(1, 3)
	tt.cut(1, 4)
	tt.cut(1, 5)

	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})

	sm := tt.peers[1].(*raft)
	if sm.raftLog.committed != 1 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 1)
	}

	// network recovery
	tt.recover()
	// avoid committing ChangeTerm proposal
	tt.ignore(pb.MsgApp)

	// elect 2 as the new leader with term 2
	tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgHup})

	// no log entries from previous term should be committed
	sm = tt.peers[2].(*raft)
	if sm.raftLog.committed != 1 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 1)
	}

	tt.recover()
	// send heartbeat; reset wait
	tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgBeat})
	// append an entry at current term
	tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})
	// expect the committed to be advanced
	if sm.raftLog.committed != 5 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 5)
	}
}

// TestCommitWithoutNewTermEntry tests the entries could be committed
// when leader changes, no new proposal comes in.
func TestCommitWithoutNewTermEntry(t *testing.T) {
	tt := newNetwork(nil, nil, nil, nil, nil)
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	// 0 cannot reach 2,3,4
	tt.cut(1, 3)
	tt.cut(1, 4)
	tt.cut(1, 5)

	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("some data")}}})

	sm := tt.peers[1].(*raft)
	if sm.raftLog.committed != 1 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 1)
	}

	// network recovery
	tt.recover()

	// elect 1 as the new leader with term 2
	// after append a ChangeTerm entry from the current term, all entries
	// should be committed
	tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgHup})

	if sm.raftLog.committed != 4 {
		t.Errorf("committed = %d, want %d", sm.raftLog.committed, 4)
	}
}

func TestDuelingCandidates(t *testing.T) {
	a := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	b := newTestRaft(2, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	c := newTestRaft(3, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())

	nt := newNetwork(a, b, c)
	nt.cut(1, 3)

	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	// 1 becomes leader since it receives votes from 1 and 2
	sm := nt.peers[1].(*raft)
	if sm.state != StateLeader {
		t.Errorf("state = %s, want %s", sm.state, StateLeader)
	}

	// 3 stays as candidate since it receives a vote from 3 and a rejection from 2
	sm = nt.peers[3].(*raft)
	if sm.state != StateCandidate {
		t.Errorf("state = %s, want %s", sm.state, StateCandidate)
	}

	nt.recover()

	// candidate 3 now increases its term and tries to vote again
	// we expect it to disrupt the leader 1 since it has a higher term
	// 3 will be follower again since both 1 and 2 rejects its vote request since 3 does not have a long enough log
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	wlog := &raftLog{
		storage:   &MemoryStorage{ents: []pb.Entry{{}, {Data: nil, Term: 1, Index: 1}}},
		committed: 1,
		unstable:  unstable{offset: 2},
	}
	tests := []struct {
		sm      *raft
		state   StateType
		term    uint64
		raftLog *raftLog
	}{
		{a, StateFollower, 2, wlog},
		{b, StateFollower, 2, wlog},
		{c, StateFollower, 2, newLog(NewMemoryStorage(), raftLogger)},
	}

	for i, tt := range tests {
		if g := tt.sm.state; g != tt.state {
			t.Errorf("#%d: state = %s, want %s", i, g, tt.state)
		}
		if g := tt.sm.Term; g != tt.term {
			t.Errorf("#%d: term = %d, want %d", i, g, tt.term)
		}
		base := ltoa(tt.raftLog)
		if sm, ok := nt.peers[1+uint64(i)].(*raft); ok {
			l := ltoa(sm.raftLog)
			if g := diffu(base, l); g != "" {
				t.Errorf("#%d: diff:\n%s", i, g)
			}
		} else {
			t.Logf("#%d: empty log", i)
		}
	}
}

func TestCandidateConcede(t *testing.T) {
	tt := newNetwork(nil, nil, nil)
	tt.isolate(1)

	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	tt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	// heal the partition
	tt.recover()
	// send heartbeat; reset wait
	tt.send(pb.Message{From: 3, To: 3, Type: pb.MsgBeat})

	data := []byte("force follower")
	// send a proposal to 3 to flush out a MsgApp to 1
	tt.send(pb.Message{From: 3, To: 3, Type: pb.MsgProp, Entries: []pb.Entry{{Data: data}}})
	// send heartbeat; flush out commit
	tt.send(pb.Message{From: 3, To: 3, Type: pb.MsgBeat})

	a := tt.peers[1].(*raft)
	if g := a.state; g != StateFollower {
		t.Errorf("state = %s, want %s", g, StateFollower)
	}
	if g := a.Term; g != 1 {
		t.Errorf("term = %d, want %d", g, 1)
	}
	wantLog := ltoa(&raftLog{
		storage: &MemoryStorage{
			ents: []pb.Entry{{}, {Data: nil, Term: 1, Index: 1}, {Term: 1, Index: 2, Data: data}},
		},
		unstable:  unstable{offset: 3},
		committed: 2,
	})
	for i, p := range tt.peers {
		if sm, ok := p.(*raft); ok {
			l := ltoa(sm.raftLog)
			if g := diffu(wantLog, l); g != "" {
				t.Errorf("#%d: diff:\n%s", i, g)
			}
		} else {
			t.Logf("#%d: empty log", i)
		}
	}
}

func TestSingleNodeCandidate(t *testing.T) {
	tt := newNetwork(nil)
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	sm := tt.peers[1].(*raft)
	if sm.state != StateLeader {
		t.Errorf("state = %d, want %d", sm.state, StateLeader)
	}
}

func TestOldMessages(t *testing.T) {
	tt := newNetwork(nil, nil, nil)
	// make 0 leader @ term 3
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgHup})
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	// pretend we're an old leader trying to make progress; this entry is expected to be ignored.
	tt.send(pb.Message{From: 2, To: 1, Type: pb.MsgApp, Term: 2, Entries: []pb.Entry{{Index: 3, Term: 2}}})
	// commit a new entry
	tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})

	ilog := &raftLog{
		storage: &MemoryStorage{
			ents: []pb.Entry{
				{}, {Data: nil, Term: 1, Index: 1},
				{Data: nil, Term: 2, Index: 2}, {Data: nil, Term: 3, Index: 3},
				{Data: []byte("somedata"), Term: 3, Index: 4},
			},
		},
		unstable:  unstable{offset: 5},
		committed: 4,
	}
	base := ltoa(ilog)
	for i, p := range tt.peers {
		if sm, ok := p.(*raft); ok {
			l := ltoa(sm.raftLog)
			if g := diffu(base, l); g != "" {
				t.Errorf("#%d: diff:\n%s", i, g)
			}
		} else {
			t.Logf("#%d: empty log", i)
		}
	}
}

// TestOldMessagesReply - optimization - reply with new term.

func TestProposal(t *testing.T) {
	tests := []struct {
		*network
		success bool
	}{
		{newNetwork(nil, nil, nil), true},
		{newNetwork(nil, nil, nopStepper), true},
		{newNetwork(nil, nopStepper, nopStepper), false},
		{newNetwork(nil, nopStepper, nopStepper, nil), false},
		{newNetwork(nil, nopStepper, nopStepper, nil, nil), true},
	}

	for j, tt := range tests {
		send := func(m pb.Message) {
			defer func() {
				// only recover is we expect it to panic so
				// panics we don't expect go up.
				if !tt.success {
					e := recover()
					if e != nil {
						t.Logf("#%d: err: %s", j, e)
					}
				}
			}()
			tt.send(m)
		}

		data := []byte("somedata")

		// promote 0 the leader
		send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
		send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: data}}})

		wantLog := newLog(NewMemoryStorage(), raftLogger)
		if tt.success {
			wantLog = &raftLog{
				storage: &MemoryStorage{
					ents: []pb.Entry{{}, {Data: nil, Term: 1, Index: 1}, {Term: 1, Index: 2, Data: data}},
				},
				unstable:  unstable{offset: 3},
				committed: 2}
		}
		base := ltoa(wantLog)
		for i, p := range tt.peers {
			if sm, ok := p.(*raft); ok {
				l := ltoa(sm.raftLog)
				if g := diffu(base, l); g != "" {
					t.Errorf("#%d: diff:\n%s", i, g)
				}
			} else {
				t.Logf("#%d: empty log", i)
			}
		}
		sm := tt.network.peers[1].(*raft)
		if g := sm.Term; g != 1 {
			t.Errorf("#%d: term = %d, want %d", j, g, 1)
		}
	}
}

func TestProposalByProxy(t *testing.T) {
	data := []byte("somedata")
	tests := []*network{
		newNetwork(nil, nil, nil),
		newNetwork(nil, nil, nopStepper),
	}

	for j, tt := range tests {
		// promote 0 the leader
		tt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

		// propose via follower
		tt.send(pb.Message{From: 2, To: 2, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})

		wantLog := &raftLog{
			storage: &MemoryStorage{
				ents: []pb.Entry{{}, {Data: nil, Term: 1, Index: 1}, {Term: 1, Data: data, Index: 2}},
			},
			unstable:  unstable{offset: 3},
			committed: 2}
		base := ltoa(wantLog)
		for i, p := range tt.peers {
			if sm, ok := p.(*raft); ok {
				l := ltoa(sm.raftLog)
				if g := diffu(base, l); g != "" {
					t.Errorf("#%d: diff:\n%s", i, g)
				}
			} else {
				t.Logf("#%d: empty log", i)
			}
		}
		sm := tt.peers[1].(*raft)
		if g := sm.Term; g != 1 {
			t.Errorf("#%d: term = %d, want %d", j, g, 1)
		}
	}
}

func TestCommit(t *testing.T) {
	tests := []struct {
		matches []uint64
		logs    []pb.Entry
		smTerm  uint64
		w       uint64
	}{
		// single
		{[]uint64{1}, []pb.Entry{{Index: 1, Term: 1}}, 1, 1},
		{[]uint64{1}, []pb.Entry{{Index: 1, Term: 1}}, 2, 0},
		{[]uint64{2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 2, 2},
		{[]uint64{1}, []pb.Entry{{Index: 1, Term: 2}}, 2, 1},

		// odd
		{[]uint64{2, 1, 1}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 1, 1},
		{[]uint64{2, 1, 1}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 1}}, 2, 0},
		{[]uint64{2, 1, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 2, 2},
		{[]uint64{2, 1, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 1}}, 2, 0},

		// even
		{[]uint64{2, 1, 1, 1}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 1, 1},
		{[]uint64{2, 1, 1, 1}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 1}}, 2, 0},
		{[]uint64{2, 1, 1, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 1, 1},
		{[]uint64{2, 1, 1, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 1}}, 2, 0},
		{[]uint64{2, 1, 2, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}}, 2, 2},
		{[]uint64{2, 1, 2, 2}, []pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 1}}, 2, 0},
	}

	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.Append(tt.logs)
		storage.hardState = pb.HardState{Term: tt.smTerm}

		sm := newTestRaft(1, []uint64{1}, 5, 1, storage)
		for j := 0; j < len(tt.matches); j++ {
			sm.setProgress(uint64(j)+1, tt.matches[j], tt.matches[j]+1)
		}
		sm.maybeCommit()
		if g := sm.raftLog.committed; g != tt.w {
			t.Errorf("#%d: committed = %d, want %d", i, g, tt.w)
		}
	}
}

func TestPastElectionTimeout(t *testing.T) {
	tests := []struct {
		elapse       int
		wprobability float64
		round        bool
	}{
		{5, 0, false},
		{10, 0.1, true},
		{13, 0.4, true},
		{15, 0.6, true},
		{18, 0.9, true},
		{20, 1, false},
	}

	for i, tt := range tests {
		sm := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
		sm.electionElapsed = tt.elapse
		c := 0
		for j := 0; j < 10000; j++ {
			sm.resetRandomizedElectionTimeout()
			if sm.pastElectionTimeout() {
				c++
			}
		}
		got := float64(c) / 10000.0
		if tt.round {
			got = math.Floor(got*10+0.5) / 10.0
		}
		if got != tt.wprobability {
			t.Errorf("#%d: probability = %v, want %v", i, got, tt.wprobability)
		}
	}
}

// ensure that the Step function ignores the message from old term and does not pass it to the
// actual stepX function.
func TestStepIgnoreOldTermMsg(t *testing.T) {
	called := false
	fakeStep := func(r *raft, m pb.Message) {
		called = true
	}
	sm := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
	sm.step = fakeStep
	sm.Term = 2
	sm.Step(pb.Message{Type: pb.MsgApp, Term: sm.Term - 1})
	if called {
		t.Errorf("stepFunc called = %v , want %v", called, false)
	}
}

// TestHandleMsgApp ensures:
// 1. Reply false if log doesn’t contain an entry at prevLogIndex whose term matches prevLogTerm.
// 2. If an existing entry conflicts with a new one (same index but different terms),
//    delete the existing entry and all that follow it; append any new entries not already in the log.
// 3. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry).
func TestHandleMsgApp(t *testing.T) {
	tests := []struct {
		m       pb.Message
		wIndex  uint64
		wCommit uint64
		wReject bool
	}{
		// Ensure 1
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 3, Index: 2, Commit: 3}, 2, 0, true}, // previous log mismatch
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 3, Index: 3, Commit: 3}, 2, 0, true}, // previous log non-exist

		// Ensure 2
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 1, Index: 1, Commit: 1}, 2, 1, false},
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 0, Index: 0, Commit: 1, Entries: []pb.Entry{{Index: 1, Term: 2}}}, 1, 1, false},
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 2, Index: 2, Commit: 3, Entries: []pb.Entry{{Index: 3, Term: 2}, {Index: 4, Term: 2}}}, 4, 3, false},
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 2, Index: 2, Commit: 4, Entries: []pb.Entry{{Index: 3, Term: 2}}}, 3, 3, false},
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 1, Index: 1, Commit: 4, Entries: []pb.Entry{{Index: 2, Term: 2}}}, 2, 2, false},

		// Ensure 3
		{pb.Message{Type: pb.MsgApp, Term: 1, LogTerm: 1, Index: 1, Commit: 3}, 2, 1, false},                                           // match entry 1, commit up to last new entry 1
		{pb.Message{Type: pb.MsgApp, Term: 1, LogTerm: 1, Index: 1, Commit: 3, Entries: []pb.Entry{{Index: 2, Term: 2}}}, 2, 2, false}, // match entry 1, commit up to last new entry 2
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 2, Index: 2, Commit: 3}, 2, 2, false},                                           // match entry 2, commit up to last new entry 2
		{pb.Message{Type: pb.MsgApp, Term: 2, LogTerm: 2, Index: 2, Commit: 4}, 2, 2, false},                                           // commit up to log.last()
	}

	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.Append([]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}})
		sm := newTestRaft(1, []uint64{1}, 10, 1, storage)
		sm.becomeFollower(2, None)

		sm.handleAppendEntries(tt.m)
		if sm.raftLog.lastIndex() != tt.wIndex {
			t.Errorf("#%d: lastIndex = %d, want %d", i, sm.raftLog.lastIndex(), tt.wIndex)
		}
		if sm.raftLog.committed != tt.wCommit {
			t.Errorf("#%d: committed = %d, want %d", i, sm.raftLog.committed, tt.wCommit)
		}
		m := sm.readMessages()
		if len(m) != 1 {
			t.Fatalf("#%d: msg = nil, want 1", i)
		}
		if m[0].Reject != tt.wReject {
			t.Errorf("#%d: reject = %v, want %v", i, m[0].Reject, tt.wReject)
		}
	}
}

// TestHandleHeartbeat ensures that the follower commits to the commit in the message.
func TestHandleHeartbeat(t *testing.T) {
	commit := uint64(2)
	tests := []struct {
		m       pb.Message
		wCommit uint64
	}{
		{pb.Message{From: 2, To: 1, Type: pb.MsgHeartbeat, Term: 2, Commit: commit + 1}, commit + 1},
		{pb.Message{From: 2, To: 1, Type: pb.MsgHeartbeat, Term: 2, Commit: commit - 1}, commit}, // do not decrease commit
	}

	for i, tt := range tests {
		storage := NewMemoryStorage()
		storage.Append([]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}})
		sm := newTestRaft(1, []uint64{1, 2}, 5, 1, storage)
		sm.becomeFollower(2, 2)
		sm.raftLog.commitTo(commit)
		sm.handleHeartbeat(tt.m)
		if sm.raftLog.committed != tt.wCommit {
			t.Errorf("#%d: committed = %d, want %d", i, sm.raftLog.committed, tt.wCommit)
		}
		m := sm.readMessages()
		if len(m) != 1 {
			t.Fatalf("#%d: msg = nil, want 1", i)
		}
		if m[0].Type != pb.MsgHeartbeatResp {
			t.Errorf("#%d: type = %v, want MsgHeartbeatResp", i, m[0].Type)
		}
	}
}

// TestHandleHeartbeatResp ensures that we re-send log entries when we get a heartbeat response.
func TestHandleHeartbeatResp(t *testing.T) {
	storage := NewMemoryStorage()
	storage.Append([]pb.Entry{{Index: 1, Term: 1}, {Index: 2, Term: 2}, {Index: 3, Term: 3}})
	sm := newTestRaft(1, []uint64{1, 2}, 5, 1, storage)
	sm.becomeCandidate()
	sm.becomeLeader()
	sm.raftLog.commitTo(sm.raftLog.lastIndex())

	// A heartbeat response from a node that is behind; re-send MsgApp
	sm.Step(pb.Message{From: 2, Type: pb.MsgHeartbeatResp})
	msgs := sm.readMessages()
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1", len(msgs))
	}
	if msgs[0].Type != pb.MsgApp {
		t.Errorf("type = %v, want MsgApp", msgs[0].Type)
	}

	// A second heartbeat response with no AppResp does not re-send because we are in the wait state.
	sm.Step(pb.Message{From: 2, Type: pb.MsgHeartbeatResp})
	msgs = sm.readMessages()
	if len(msgs) != 0 {
		t.Fatalf("len(msgs) = %d, want 0", len(msgs))
	}

	// Send a heartbeat to reset the wait state; next heartbeat will re-send MsgApp.
	sm.bcastHeartbeat()
	sm.Step(pb.Message{From: 2, Type: pb.MsgHeartbeatResp})
	msgs = sm.readMessages()
	if len(msgs) != 2 {
		t.Fatalf("len(msgs) = %d, want 2", len(msgs))
	}
	if msgs[0].Type != pb.MsgHeartbeat {
		t.Errorf("type = %v, want MsgHeartbeat", msgs[0].Type)
	}
	if msgs[1].Type != pb.MsgApp {
		t.Errorf("type = %v, want MsgApp", msgs[1].Type)
	}

	// Once we have an MsgAppResp, heartbeats no longer send MsgApp.
	sm.Step(pb.Message{
		From:  2,
		Type:  pb.MsgAppResp,
		Index: msgs[1].Index + uint64(len(msgs[1].Entries)),
	})
	// Consume the message sent in response to MsgAppResp
	sm.readMessages()

	sm.bcastHeartbeat() // reset wait state
	sm.Step(pb.Message{From: 2, Type: pb.MsgHeartbeatResp})
	msgs = sm.readMessages()
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1: %+v", len(msgs), msgs)
	}
	if msgs[0].Type != pb.MsgHeartbeat {
		t.Errorf("type = %v, want MsgHeartbeat", msgs[0].Type)
	}
}

// TestMsgAppRespWaitReset verifies the resume behavior of a leader
// MsgAppResp.
func TestMsgAppRespWaitReset(t *testing.T) {
	sm := newTestRaft(1, []uint64{1, 2, 3}, 5, 1, NewMemoryStorage())
	sm.becomeCandidate()
	sm.becomeLeader()

	// The new leader has just emitted a new Term 4 entry; consume those messages
	// from the outgoing queue.
	sm.bcastAppend()
	sm.readMessages()

	// Node 2 acks the first entry, making it committed.
	sm.Step(pb.Message{
		From:  2,
		Type:  pb.MsgAppResp,
		Index: 1,
	})
	if sm.raftLog.committed != 1 {
		t.Fatalf("expected committed to be 1, got %d", sm.raftLog.committed)
	}
	// Also consume the MsgApp messages that update Commit on the followers.
	sm.readMessages()

	// A new command is now proposed on node 1.
	sm.Step(pb.Message{
		From:    1,
		Type:    pb.MsgProp,
		Entries: []pb.Entry{{}},
	})

	// The command is broadcast to all nodes not in the wait state.
	// Node 2 left the wait state due to its MsgAppResp, but node 3 is still waiting.
	msgs := sm.readMessages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Type != pb.MsgApp || msgs[0].To != 2 {
		t.Errorf("expected MsgApp to node 2, got %v to %d", msgs[0].Type, msgs[0].To)
	}
	if len(msgs[0].Entries) != 1 || msgs[0].Entries[0].Index != 2 {
		t.Errorf("expected to send entry 2, but got %v", msgs[0].Entries)
	}

	// Now Node 3 acks the first entry. This releases the wait and entry 2 is sent.
	sm.Step(pb.Message{
		From:  3,
		Type:  pb.MsgAppResp,
		Index: 1,
	})
	msgs = sm.readMessages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Type != pb.MsgApp || msgs[0].To != 3 {
		t.Errorf("expected MsgApp to node 3, got %v to %d", msgs[0].Type, msgs[0].To)
	}
	if len(msgs[0].Entries) != 1 || msgs[0].Entries[0].Index != 2 {
		t.Errorf("expected to send entry 2, but got %v", msgs[0].Entries)
	}
}

func TestRecvMsgVote(t *testing.T) {
	tests := []struct {
		state   StateType
		i, term uint64
		voteFor uint64
		wreject bool
	}{
		{StateFollower, 0, 0, None, true},
		{StateFollower, 0, 1, None, true},
		{StateFollower, 0, 2, None, true},
		{StateFollower, 0, 3, None, false},

		{StateFollower, 1, 0, None, true},
		{StateFollower, 1, 1, None, true},
		{StateFollower, 1, 2, None, true},
		{StateFollower, 1, 3, None, false},

		{StateFollower, 2, 0, None, true},
		{StateFollower, 2, 1, None, true},
		{StateFollower, 2, 2, None, false},
		{StateFollower, 2, 3, None, false},

		{StateFollower, 3, 0, None, true},
		{StateFollower, 3, 1, None, true},
		{StateFollower, 3, 2, None, false},
		{StateFollower, 3, 3, None, false},

		{StateFollower, 3, 2, 2, false},
		{StateFollower, 3, 2, 1, true},

		{StateLeader, 3, 3, 1, true},
		{StateCandidate, 3, 3, 1, true},
	}

	for i, tt := range tests {
		sm := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
		sm.state = tt.state
		switch tt.state {
		case StateFollower:
			sm.step = stepFollower
		case StateCandidate:
			sm.step = stepCandidate
		case StateLeader:
			sm.step = stepLeader
		}
		sm.Vote = tt.voteFor
		sm.raftLog = &raftLog{
			storage:  &MemoryStorage{ents: []pb.Entry{{}, {Index: 1, Term: 2}, {Index: 2, Term: 2}}},
			unstable: unstable{offset: 3},
		}

		sm.Step(pb.Message{Type: pb.MsgVote, From: 2, Index: tt.i, LogTerm: tt.term})

		msgs := sm.readMessages()
		if g := len(msgs); g != 1 {
			t.Fatalf("#%d: len(msgs) = %d, want 1", i, g)
			continue
		}
		if g := msgs[0].Reject; g != tt.wreject {
			t.Errorf("#%d, m.Reject = %v, want %v", i, g, tt.wreject)
		}
	}
}

func TestStateTransition(t *testing.T) {
	tests := []struct {
		from   StateType
		to     StateType
		wallow bool
		wterm  uint64
		wlead  uint64
	}{
		{StateFollower, StateFollower, true, 1, None},
		{StateFollower, StateCandidate, true, 1, None},
		{StateFollower, StateLeader, false, 0, None},

		{StateCandidate, StateFollower, true, 0, None},
		{StateCandidate, StateCandidate, true, 1, None},
		{StateCandidate, StateLeader, true, 0, 1},

		{StateLeader, StateFollower, true, 1, None},
		{StateLeader, StateCandidate, false, 1, None},
		{StateLeader, StateLeader, true, 0, 1},
	}

	for i, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if tt.wallow {
						t.Errorf("%d: allow = %v, want %v", i, false, true)
					}
				}
			}()

			sm := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
			sm.state = tt.from

			switch tt.to {
			case StateFollower:
				sm.becomeFollower(tt.wterm, tt.wlead)
			case StateCandidate:
				sm.becomeCandidate()
			case StateLeader:
				sm.becomeLeader()
			}

			if sm.Term != tt.wterm {
				t.Errorf("%d: term = %d, want %d", i, sm.Term, tt.wterm)
			}
			if sm.lead != tt.wlead {
				t.Errorf("%d: lead = %d, want %d", i, sm.lead, tt.wlead)
			}
		}()
	}
}

func TestAllServerStepdown(t *testing.T) {
	tests := []struct {
		state StateType

		wstate StateType
		wterm  uint64
		windex uint64
	}{
		{StateFollower, StateFollower, 3, 0},
		{StateCandidate, StateFollower, 3, 0},
		{StateLeader, StateFollower, 3, 1},
	}

	tmsgTypes := [...]pb.MessageType{pb.MsgVote, pb.MsgApp}
	tterm := uint64(3)

	for i, tt := range tests {
		sm := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
		switch tt.state {
		case StateFollower:
			sm.becomeFollower(1, None)
		case StateCandidate:
			sm.becomeCandidate()
		case StateLeader:
			sm.becomeCandidate()
			sm.becomeLeader()
		}

		for j, msgType := range tmsgTypes {
			sm.Step(pb.Message{From: 2, Type: msgType, Term: tterm, LogTerm: tterm})

			if sm.state != tt.wstate {
				t.Errorf("#%d.%d state = %v , want %v", i, j, sm.state, tt.wstate)
			}
			if sm.Term != tt.wterm {
				t.Errorf("#%d.%d term = %v , want %v", i, j, sm.Term, tt.wterm)
			}
			if uint64(sm.raftLog.lastIndex()) != tt.windex {
				t.Errorf("#%d.%d index = %v , want %v", i, j, sm.raftLog.lastIndex(), tt.windex)
			}
			if uint64(len(sm.raftLog.allEntries())) != tt.windex {
				t.Errorf("#%d.%d len(ents) = %v , want %v", i, j, len(sm.raftLog.allEntries()), tt.windex)
			}
			wlead := uint64(2)
			if msgType == pb.MsgVote {
				wlead = None
			}
			if sm.lead != wlead {
				t.Errorf("#%d, sm.lead = %d, want %d", i, sm.lead, None)
			}
		}
	}
}

func TestLeaderStepdownWhenQuorumActive(t *testing.T) {
	sm := newTestRaft(1, []uint64{1, 2, 3}, 5, 1, NewMemoryStorage())

	sm.checkQuorum = true

	sm.becomeCandidate()
	sm.becomeLeader()

	for i := 0; i < sm.electionTimeout+1; i++ {
		sm.Step(pb.Message{From: 2, Type: pb.MsgHeartbeatResp, Term: sm.Term})
		sm.tick()
	}

	if sm.state != StateLeader {
		t.Errorf("state = %v, want %v", sm.state, StateLeader)
	}
}

func TestLeaderStepdownWhenQuorumLost(t *testing.T) {
	sm := newTestRaft(1, []uint64{1, 2, 3}, 5, 1, NewMemoryStorage())

	sm.checkQuorum = true

	sm.becomeCandidate()
	sm.becomeLeader()

	for i := 0; i < sm.electionTimeout+1; i++ {
		sm.tick()
	}

	if sm.state != StateFollower {
		t.Errorf("state = %v, want %v", sm.state, StateFollower)
	}
}

func TestLeaderSupersedingWithCheckQuorum(t *testing.T) {
	a := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	b := newTestRaft(2, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	c := newTestRaft(3, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())

	a.checkQuorum = true
	b.checkQuorum = true
	c.checkQuorum = true

	nt := newNetwork(a, b, c)

	// Prevent campaigning from b
	b.randomizedElectionTimeout = b.electionTimeout + 1
	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	if a.state != StateLeader {
		t.Errorf("state = %s, want %s", a.state, StateLeader)
	}

	if c.state != StateFollower {
		t.Errorf("state = %s, want %s", c.state, StateFollower)
	}

	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	// Peer b rejected c's vote since its electionElapsed had not reached to electionTimeout
	if c.state != StateCandidate {
		t.Errorf("state = %s, want %s", c.state, StateCandidate)
	}

	// Letting b's electionElapsed reach to electionTimeout
	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	if c.state != StateLeader {
		t.Errorf("state = %s, want %s", c.state, StateLeader)
	}
}

func TestLeaderElectionWithCheckQuorum(t *testing.T) {
	a := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	b := newTestRaft(2, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	c := newTestRaft(3, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())

	a.checkQuorum = true
	b.checkQuorum = true
	c.checkQuorum = true

	nt := newNetwork(a, b, c)

	// Letting b's electionElapsed reach to timeout so that it can vote for a
	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	if a.state != StateLeader {
		t.Errorf("state = %s, want %s", a.state, StateLeader)
	}

	if c.state != StateFollower {
		t.Errorf("state = %s, want %s", c.state, StateFollower)
	}

	for i := 0; i < a.electionTimeout; i++ {
		a.tick()
	}
	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	if a.state != StateFollower {
		t.Errorf("state = %s, want %s", a.state, StateFollower)
	}

	if c.state != StateLeader {
		t.Errorf("state = %s, want %s", c.state, StateLeader)
	}
}

// TestFreeStuckCandidateWithCheckQuorum ensures that a candidate with a higher term
// can disrupt the leader even if the leader still "officially" holds the lease, The
// leader is expected to step down and adopt the candidate's term
func TestFreeStuckCandidateWithCheckQuorum(t *testing.T) {
	a := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	b := newTestRaft(2, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	c := newTestRaft(3, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())

	a.checkQuorum = true
	b.checkQuorum = true
	c.checkQuorum = true

	nt := newNetwork(a, b, c)
	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(1)
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	if b.state != StateFollower {
		t.Errorf("state = %s, want %s", b.state, StateFollower)
	}

	if c.state != StateCandidate {
		t.Errorf("state = %s, want %s", c.state, StateCandidate)
	}

	if c.Term != b.Term+1 {
		t.Errorf("term = %d, want %d", c.Term, b.Term+1)
	}

	// Vote again for safety
	nt.send(pb.Message{From: 3, To: 3, Type: pb.MsgHup})

	if b.state != StateFollower {
		t.Errorf("state = %s, want %s", b.state, StateFollower)
	}

	if c.state != StateCandidate {
		t.Errorf("state = %s, want %s", c.state, StateCandidate)
	}

	if c.Term != b.Term+2 {
		t.Errorf("term = %d, want %d", c.Term, b.Term+2)
	}

	nt.recover()
	nt.send(pb.Message{From: 1, To: 3, Type: pb.MsgHeartbeat, Term: a.Term})

	// Disrupt the leader so that the stuck peer is freed
	if a.state != StateFollower {
		t.Errorf("state = %s, want %s", a.state, StateFollower)
	}

	if c.Term != a.Term {
		t.Errorf("term = %d, want %d", c.Term, a.Term)
	}
}

func TestNonPromotableVoterWithCheckQuorum(t *testing.T) {
	a := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	b := newTestRaft(2, []uint64{1}, 10, 1, NewMemoryStorage())

	a.checkQuorum = true
	b.checkQuorum = true

	nt := newNetwork(a, b)
	// Need to remove 2 again to make it a non-promotable node since newNetwork overwritten some internal states
	b.delProgress(2)

	if b.promotable() {
		t.Fatalf("promotable = %v, want false", b.promotable())
	}

	for i := 0; i < b.electionTimeout; i++ {
		b.tick()
	}
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	if a.state != StateLeader {
		t.Errorf("state = %s, want %s", a.state, StateLeader)
	}

	if b.state != StateFollower {
		t.Errorf("state = %s, want %s", b.state, StateFollower)
	}

	if b.lead != 1 {
		t.Errorf("lead = %d, want 1", b.lead)
	}
}

func TestLeaderAppResp(t *testing.T) {
	// initial progress: match = 0; next = 3
	tests := []struct {
		index  uint64
		reject bool
		// progress
		wmatch uint64
		wnext  uint64
		// message
		wmsgNum    int
		windex     uint64
		wcommitted uint64
	}{
		{3, true, 0, 3, 0, 0, 0},  // stale resp; no replies
		{2, true, 0, 2, 1, 1, 0},  // denied resp; leader does not commit; decrease next and send probing msg
		{2, false, 2, 4, 2, 2, 2}, // accept resp; leader commits; broadcast with commit index
		{0, false, 0, 3, 0, 0, 0}, // ignore heartbeat replies
	}

	for i, tt := range tests {
		// sm term is 1 after it becomes the leader.
		// thus the last log term must be 1 to be committed.
		sm := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
		sm.raftLog = &raftLog{
			storage:  &MemoryStorage{ents: []pb.Entry{{}, {Index: 1, Term: 0}, {Index: 2, Term: 1}}},
			unstable: unstable{offset: 3},
		}
		sm.becomeCandidate()
		sm.becomeLeader()
		sm.readMessages()
		sm.Step(pb.Message{From: 2, Type: pb.MsgAppResp, Index: tt.index, Term: sm.Term, Reject: tt.reject, RejectHint: tt.index})

		p := sm.prs[2]
		if p.Match != tt.wmatch {
			t.Errorf("#%d match = %d, want %d", i, p.Match, tt.wmatch)
		}
		if p.Next != tt.wnext {
			t.Errorf("#%d next = %d, want %d", i, p.Next, tt.wnext)
		}

		msgs := sm.readMessages()

		if len(msgs) != tt.wmsgNum {
			t.Errorf("#%d msgNum = %d, want %d", i, len(msgs), tt.wmsgNum)
		}
		for j, msg := range msgs {
			if msg.Index != tt.windex {
				t.Errorf("#%d.%d index = %d, want %d", i, j, msg.Index, tt.windex)
			}
			if msg.Commit != tt.wcommitted {
				t.Errorf("#%d.%d commit = %d, want %d", i, j, msg.Commit, tt.wcommitted)
			}
		}
	}
}

// When the leader receives a heartbeat tick, it should
// send a MsgApp with m.Index = 0, m.LogTerm=0 and empty entries.
func TestBcastBeat(t *testing.T) {
	offset := uint64(1000)
	// make a state machine with log.offset = 1000
	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     offset,
			Term:      1,
			ConfState: pb.ConfState{Nodes: []uint64{1, 2, 3}},
		},
	}
	storage := NewMemoryStorage()
	storage.ApplySnapshot(s)
	sm := newTestRaft(1, nil, 10, 1, storage)
	sm.Term = 1

	sm.becomeCandidate()
	sm.becomeLeader()
	for i := 0; i < 10; i++ {
		sm.appendEntry(pb.Entry{Index: uint64(i) + 1})
	}
	// slow follower
	sm.prs[2].Match, sm.prs[2].Next = 5, 6
	// normal follower
	sm.prs[3].Match, sm.prs[3].Next = sm.raftLog.lastIndex(), sm.raftLog.lastIndex()+1

	sm.Step(pb.Message{Type: pb.MsgBeat})
	msgs := sm.readMessages()
	if len(msgs) != 2 {
		t.Fatalf("len(msgs) = %v, want 2", len(msgs))
	}
	wantCommitMap := map[uint64]uint64{
		2: min(sm.raftLog.committed, sm.prs[2].Match),
		3: min(sm.raftLog.committed, sm.prs[3].Match),
	}
	for i, m := range msgs {
		if m.Type != pb.MsgHeartbeat {
			t.Fatalf("#%d: type = %v, want = %v", i, m.Type, pb.MsgHeartbeat)
		}
		if m.Index != 0 {
			t.Fatalf("#%d: prevIndex = %d, want %d", i, m.Index, 0)
		}
		if m.LogTerm != 0 {
			t.Fatalf("#%d: prevTerm = %d, want %d", i, m.LogTerm, 0)
		}
		if wantCommitMap[m.To] == 0 {
			t.Fatalf("#%d: unexpected to %d", i, m.To)
		} else {
			if m.Commit != wantCommitMap[m.To] {
				t.Fatalf("#%d: commit = %d, want %d", i, m.Commit, wantCommitMap[m.To])
			}
			delete(wantCommitMap, m.To)
		}
		if len(m.Entries) != 0 {
			t.Fatalf("#%d: len(entries) = %d, want 0", i, len(m.Entries))
		}
	}
}

// tests the output of the state machine when receiving MsgBeat
func TestRecvMsgBeat(t *testing.T) {
	tests := []struct {
		state StateType
		wMsg  int
	}{
		{StateLeader, 2},
		// candidate and follower should ignore MsgBeat
		{StateCandidate, 0},
		{StateFollower, 0},
	}

	for i, tt := range tests {
		sm := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
		sm.raftLog = &raftLog{storage: &MemoryStorage{ents: []pb.Entry{{}, {Index: 1, Term: 0}, {Index: 2, Term: 1}}}}
		sm.Term = 1
		sm.state = tt.state
		switch tt.state {
		case StateFollower:
			sm.step = stepFollower
		case StateCandidate:
			sm.step = stepCandidate
		case StateLeader:
			sm.step = stepLeader
		}
		sm.Step(pb.Message{From: 1, To: 1, Type: pb.MsgBeat})

		msgs := sm.readMessages()
		if len(msgs) != tt.wMsg {
			t.Errorf("%d: len(msgs) = %d, want %d", i, len(msgs), tt.wMsg)
		}
		for _, m := range msgs {
			if m.Type != pb.MsgHeartbeat {
				t.Errorf("%d: msg.type = %v, want %v", i, m.Type, pb.MsgHeartbeat)
			}
		}
	}
}

func TestLeaderIncreaseNext(t *testing.T) {
	previousEnts := []pb.Entry{{Term: 1, Index: 1}, {Term: 1, Index: 2}, {Term: 1, Index: 3}}
	tests := []struct {
		// progress
		state ProgressStateType
		next  uint64

		wnext uint64
	}{
		// state replicate, optimistically increase next
		// previous entries + noop entry + propose + 1
		{ProgressStateReplicate, 2, uint64(len(previousEnts) + 1 + 1 + 1)},
		// state probe, not optimistically increase next
		{ProgressStateProbe, 2, 2},
	}

	for i, tt := range tests {
		sm := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
		sm.raftLog.append(previousEnts...)
		sm.becomeCandidate()
		sm.becomeLeader()
		sm.prs[2].State = tt.state
		sm.prs[2].Next = tt.next
		sm.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})

		p := sm.prs[2]
		if p.Next != tt.wnext {
			t.Errorf("#%d next = %d, want %d", i, p.Next, tt.wnext)
		}
	}
}

func TestSendAppendForProgressProbe(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.readMessages()
	r.prs[2].becomeProbe()

	// each round is a heartbeat
	for i := 0; i < 3; i++ {
		// we expect that raft will only send out one msgAPP per heartbeat timeout
		r.appendEntry(pb.Entry{Data: []byte("somedata")})
		r.sendAppend(2)
		msg := r.readMessages()
		if len(msg) != 1 {
			t.Errorf("len(msg) = %d, want %d", len(msg), 1)
		}
		if msg[0].Index != 0 {
			t.Errorf("index = %d, want %d", msg[0].Index, 0)
		}

		if !r.prs[2].Paused {
			t.Errorf("paused = %v, want true", r.prs[2].Paused)
		}
		for j := 0; j < 10; j++ {
			r.appendEntry(pb.Entry{Data: []byte("somedata")})
			r.sendAppend(2)
			if l := len(r.readMessages()); l != 0 {
				t.Errorf("len(msg) = %d, want %d", l, 0)
			}
		}

		// do a heartbeat
		for j := 0; j < r.heartbeatTimeout; j++ {
			r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgBeat})
		}
		// consume the heartbeat
		msg = r.readMessages()
		if len(msg) != 1 {
			t.Errorf("len(msg) = %d, want %d", len(msg), 1)
		}
		if msg[0].Type != pb.MsgHeartbeat {
			t.Errorf("type = %v, want %v", msg[0].Type, pb.MsgHeartbeat)
		}
	}
}

func TestSendAppendForProgressReplicate(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.readMessages()
	r.prs[2].becomeReplicate()

	for i := 0; i < 10; i++ {
		r.appendEntry(pb.Entry{Data: []byte("somedata")})
		r.sendAppend(2)
		msgs := r.readMessages()
		if len(msgs) != 1 {
			t.Errorf("len(msg) = %d, want %d", len(msgs), 1)
		}
	}
}

func TestSendAppendForProgressSnapshot(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.readMessages()
	r.prs[2].becomeSnapshot(10)

	for i := 0; i < 10; i++ {
		r.appendEntry(pb.Entry{Data: []byte("somedata")})
		r.sendAppend(2)
		msgs := r.readMessages()
		if len(msgs) != 0 {
			t.Errorf("len(msg) = %d, want %d", len(msgs), 0)
		}
	}
}

func TestRecvMsgUnreachable(t *testing.T) {
	previousEnts := []pb.Entry{{Term: 1, Index: 1}, {Term: 1, Index: 2}, {Term: 1, Index: 3}}
	s := NewMemoryStorage()
	s.Append(previousEnts)
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, s)
	r.becomeCandidate()
	r.becomeLeader()
	r.readMessages()
	// set node 2 to state replicate
	r.prs[2].Match = 3
	r.prs[2].becomeReplicate()
	r.prs[2].optimisticUpdate(5)

	r.Step(pb.Message{From: 2, To: 1, Type: pb.MsgUnreachable})

	if r.prs[2].State != ProgressStateProbe {
		t.Errorf("state = %s, want %s", r.prs[2].State, ProgressStateProbe)
	}
	if wnext := r.prs[2].Match + 1; r.prs[2].Next != wnext {
		t.Errorf("next = %d, want %d", r.prs[2].Next, wnext)
	}
}

func TestRestore(t *testing.T) {
	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     11, // magic number
			Term:      11, // magic number
			ConfState: pb.ConfState{Nodes: []uint64{1, 2, 3}},
		},
	}

	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	if ok := sm.restore(s); !ok {
		t.Fatal("restore fail, want succeed")
	}

	if sm.raftLog.lastIndex() != s.Metadata.Index {
		t.Errorf("log.lastIndex = %d, want %d", sm.raftLog.lastIndex(), s.Metadata.Index)
	}
	if mustTerm(sm.raftLog.term(s.Metadata.Index)) != s.Metadata.Term {
		t.Errorf("log.lastTerm = %d, want %d", mustTerm(sm.raftLog.term(s.Metadata.Index)), s.Metadata.Term)
	}
	sg := sm.nodes()
	if !reflect.DeepEqual(sg, s.Metadata.ConfState.Nodes) {
		t.Errorf("sm.Nodes = %+v, want %+v", sg, s.Metadata.ConfState.Nodes)
	}

	if ok := sm.restore(s); ok {
		t.Fatal("restore succeed, want fail")
	}
}

func TestRestoreIgnoreSnapshot(t *testing.T) {
	previousEnts := []pb.Entry{{Term: 1, Index: 1}, {Term: 1, Index: 2}, {Term: 1, Index: 3}}
	commit := uint64(1)
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1, 2}, 10, 1, storage)
	sm.raftLog.append(previousEnts...)
	sm.raftLog.commitTo(commit)

	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     commit,
			Term:      1,
			ConfState: pb.ConfState{Nodes: []uint64{1, 2}},
		},
	}

	// ignore snapshot
	if ok := sm.restore(s); ok {
		t.Errorf("restore = %t, want %t", ok, false)
	}
	if sm.raftLog.committed != commit {
		t.Errorf("commit = %d, want %d", sm.raftLog.committed, commit)
	}

	// ignore snapshot and fast forward commit
	s.Metadata.Index = commit + 1
	if ok := sm.restore(s); ok {
		t.Errorf("restore = %t, want %t", ok, false)
	}
	if sm.raftLog.committed != commit+1 {
		t.Errorf("commit = %d, want %d", sm.raftLog.committed, commit+1)
	}
}

func TestProvideSnap(t *testing.T) {
	// restore the state machine from a snapshot so it has a compacted log and a snapshot
	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     11, // magic number
			Term:      11, // magic number
			ConfState: pb.ConfState{Nodes: []uint64{1, 2}},
		},
	}
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1}, 10, 1, storage)
	sm.restore(s)

	sm.becomeCandidate()
	sm.becomeLeader()

	// force set the next of node 2, so that node 2 needs a snapshot
	sm.prs[2].Next = sm.raftLog.firstIndex()
	sm.Step(pb.Message{From: 2, To: 1, Type: pb.MsgAppResp, Index: sm.prs[2].Next - 1, Reject: true})

	msgs := sm.readMessages()
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1", len(msgs))
	}
	m := msgs[0]
	if m.Type != pb.MsgSnap {
		t.Errorf("m.Type = %v, want %v", m.Type, pb.MsgSnap)
	}
}

func TestIgnoreProvidingSnap(t *testing.T) {
	// restore the state machine from a snapshot so it has a compacted log and a snapshot
	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     11, // magic number
			Term:      11, // magic number
			ConfState: pb.ConfState{Nodes: []uint64{1, 2}},
		},
	}
	storage := NewMemoryStorage()
	sm := newTestRaft(1, []uint64{1}, 10, 1, storage)
	sm.restore(s)

	sm.becomeCandidate()
	sm.becomeLeader()

	// force set the next of node 2, so that node 2 needs a snapshot
	// change node 2 to be inactive, expect node 1 ignore sending snapshot to 2
	sm.prs[2].Next = sm.raftLog.firstIndex() - 1
	sm.prs[2].RecentActive = false

	sm.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Data: []byte("somedata")}}})

	msgs := sm.readMessages()
	if len(msgs) != 0 {
		t.Errorf("len(msgs) = %d, want 0", len(msgs))
	}
}

func TestRestoreFromSnapMsg(t *testing.T) {
	s := pb.Snapshot{
		Metadata: pb.SnapshotMetadata{
			Index:     11, // magic number
			Term:      11, // magic number
			ConfState: pb.ConfState{Nodes: []uint64{1, 2}},
		},
	}
	m := pb.Message{Type: pb.MsgSnap, From: 1, Term: 2, Snapshot: s}

	sm := newTestRaft(2, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	sm.Step(m)

	// TODO(bdarnell): what should this test?
}

func TestSlowNodeRestore(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)
	for j := 0; j <= 100; j++ {
		nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})
	}
	lead := nt.peers[1].(*raft)
	nextEnts(lead, nt.storage[1])
	nt.storage[1].CreateSnapshot(lead.raftLog.applied, &pb.ConfState{Nodes: lead.nodes()}, nil)
	nt.storage[1].Compact(lead.raftLog.applied)

	nt.recover()
	// send heartbeats so that the leader can learn everyone is active.
	// node 3 will only be considered as active when node 1 receives a reply from it.
	for {
		nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgBeat})
		if lead.prs[3].RecentActive {
			break
		}
	}

	// trigger a snapshot
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})

	follower := nt.peers[3].(*raft)

	// trigger a commit
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})
	if follower.raftLog.committed != lead.raftLog.committed {
		t.Errorf("follower.committed = %d, want %d", follower.raftLog.committed, lead.raftLog.committed)
	}
}

// TestStepConfig tests that when raft step msgProp in EntryConfChange type,
// it appends the entry to log and sets pendingConf to be true.
func TestStepConfig(t *testing.T) {
	// a raft that cannot make progress
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	index := r.raftLog.lastIndex()
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Type: pb.EntryConfChange}}})
	if g := r.raftLog.lastIndex(); g != index+1 {
		t.Errorf("index = %d, want %d", g, index+1)
	}
	if !r.pendingConf {
		t.Errorf("pendingConf = %v, want true", r.pendingConf)
	}
}

// TestStepIgnoreConfig tests that if raft step the second msgProp in
// EntryConfChange type when the first one is uncommitted, the node will set
// the proposal to noop and keep its original state.
func TestStepIgnoreConfig(t *testing.T) {
	// a raft that cannot make progress
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.becomeCandidate()
	r.becomeLeader()
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Type: pb.EntryConfChange}}})
	index := r.raftLog.lastIndex()
	pendingConf := r.pendingConf
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{Type: pb.EntryConfChange}}})
	wents := []pb.Entry{{Type: pb.EntryNormal, Term: 1, Index: 3, Data: nil}}
	ents, err := r.raftLog.entries(index+1, noLimit)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(ents, wents) {
		t.Errorf("ents = %+v, want %+v", ents, wents)
	}
	if r.pendingConf != pendingConf {
		t.Errorf("pendingConf = %v, want %v", r.pendingConf, pendingConf)
	}
}

// TestRecoverPendingConfig tests that new leader recovers its pendingConf flag
// based on uncommitted entries.
func TestRecoverPendingConfig(t *testing.T) {
	tests := []struct {
		entType  pb.EntryType
		wpending bool
	}{
		{pb.EntryNormal, false},
		{pb.EntryConfChange, true},
	}
	for i, tt := range tests {
		r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
		r.appendEntry(pb.Entry{Type: tt.entType})
		r.becomeCandidate()
		r.becomeLeader()
		if r.pendingConf != tt.wpending {
			t.Errorf("#%d: pendingConf = %v, want %v", i, r.pendingConf, tt.wpending)
		}
	}
}

// TestRecoverDoublePendingConfig tests that new leader will panic if
// there exist two uncommitted config entries.
func TestRecoverDoublePendingConfig(t *testing.T) {
	func() {
		defer func() {
			if err := recover(); err == nil {
				t.Errorf("expect panic, but nothing happens")
			}
		}()
		r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
		r.appendEntry(pb.Entry{Type: pb.EntryConfChange})
		r.appendEntry(pb.Entry{Type: pb.EntryConfChange})
		r.becomeCandidate()
		r.becomeLeader()
	}()
}

// TestAddNode tests that addNode could update pendingConf and nodes correctly.
func TestAddNode(t *testing.T) {
	r := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
	r.pendingConf = true
	r.addNode(2)
	if r.pendingConf {
		t.Errorf("pendingConf = %v, want false", r.pendingConf)
	}
	nodes := r.nodes()
	wnodes := []uint64{1, 2}
	if !reflect.DeepEqual(nodes, wnodes) {
		t.Errorf("nodes = %v, want %v", nodes, wnodes)
	}
}

// TestRemoveNode tests that removeNode could update pendingConf, nodes and
// and removed list correctly.
func TestRemoveNode(t *testing.T) {
	r := newTestRaft(1, []uint64{1, 2}, 10, 1, NewMemoryStorage())
	r.pendingConf = true
	r.removeNode(2)
	if r.pendingConf {
		t.Errorf("pendingConf = %v, want false", r.pendingConf)
	}
	w := []uint64{1}
	if g := r.nodes(); !reflect.DeepEqual(g, w) {
		t.Errorf("nodes = %v, want %v", g, w)
	}

	// remove all nodes from cluster
	r.removeNode(1)
	w = []uint64{}
	if g := r.nodes(); !reflect.DeepEqual(g, w) {
		t.Errorf("nodes = %v, want %v", g, w)
	}
}

func TestPromotable(t *testing.T) {
	id := uint64(1)
	tests := []struct {
		peers []uint64
		wp    bool
	}{
		{[]uint64{1}, true},
		{[]uint64{1, 2, 3}, true},
		{[]uint64{}, false},
		{[]uint64{2, 3}, false},
	}
	for i, tt := range tests {
		r := newTestRaft(id, tt.peers, 5, 1, NewMemoryStorage())
		if g := r.promotable(); g != tt.wp {
			t.Errorf("#%d: promotable = %v, want %v", i, g, tt.wp)
		}
	}
}

func TestRaftNodes(t *testing.T) {
	tests := []struct {
		ids  []uint64
		wids []uint64
	}{
		{
			[]uint64{1, 2, 3},
			[]uint64{1, 2, 3},
		},
		{
			[]uint64{3, 2, 1},
			[]uint64{1, 2, 3},
		},
	}
	for i, tt := range tests {
		r := newTestRaft(1, tt.ids, 10, 1, NewMemoryStorage())
		if !reflect.DeepEqual(r.nodes(), tt.wids) {
			t.Errorf("#%d: nodes = %+v, want %+v", i, r.nodes(), tt.wids)
		}
	}
}

func TestCampaignWhileLeader(t *testing.T) {
	r := newTestRaft(1, []uint64{1}, 5, 1, NewMemoryStorage())
	if r.state != StateFollower {
		t.Errorf("expected new node to be follower but got %s", r.state)
	}
	// We don't call campaign() directly because it comes after the check
	// for our current state.
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	if r.state != StateLeader {
		t.Errorf("expected single-node election to become leader but got %s", r.state)
	}
	term := r.Term
	r.Step(pb.Message{From: 1, To: 1, Type: pb.MsgHup})
	if r.state != StateLeader {
		t.Errorf("expected to remain leader but got %s", r.state)
	}
	if r.Term != term {
		t.Errorf("expected to remain in term %v but got %v", term, r.Term)
	}
}

// TestCommitAfterRemoveNode verifies that pending commands can become
// committed when a config change reduces the quorum requirements.
func TestCommitAfterRemoveNode(t *testing.T) {
	// Create a cluster with two nodes.
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1, 2}, 5, 1, s)
	r.becomeCandidate()
	r.becomeLeader()

	// Begin to remove the second node.
	cc := pb.ConfChange{
		Type:   pb.ConfChangeRemoveNode,
		NodeID: 2,
	}
	ccData, err := cc.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	r.Step(pb.Message{
		Type: pb.MsgProp,
		Entries: []pb.Entry{
			{Type: pb.EntryConfChange, Data: ccData},
		},
	})
	// Stabilize the log and make sure nothing is committed yet.
	if ents := nextEnts(r, s); len(ents) > 0 {
		t.Fatalf("unexpected committed entries: %v", ents)
	}
	ccIndex := r.raftLog.lastIndex()

	// While the config change is pending, make another proposal.
	r.Step(pb.Message{
		Type: pb.MsgProp,
		Entries: []pb.Entry{
			{Type: pb.EntryNormal, Data: []byte("hello")},
		},
	})

	// Node 2 acknowledges the config change, committing it.
	r.Step(pb.Message{
		Type:  pb.MsgAppResp,
		From:  2,
		Index: ccIndex,
	})
	ents := nextEnts(r, s)
	if len(ents) != 2 {
		t.Fatalf("expected two committed entries, got %v", ents)
	}
	if ents[0].Type != pb.EntryNormal || ents[0].Data != nil {
		t.Fatalf("expected ents[0] to be empty, but got %v", ents[0])
	}
	if ents[1].Type != pb.EntryConfChange {
		t.Fatalf("expected ents[1] to be EntryConfChange, got %v", ents[1])
	}

	// Apply the config change. This reduces quorum requirements so the
	// pending command can now commit.
	r.removeNode(2)
	ents = nextEnts(r, s)
	if len(ents) != 1 || ents[0].Type != pb.EntryNormal ||
		string(ents[0].Data) != "hello" {
		t.Fatalf("expected one committed EntryNormal, got %v", ents)
	}
}

// TestLeaderTransferToUpToDateNode verifies transferring should succeed
// if the transferee has the most up-to-date log entires when transfer starts.
func TestLeaderTransferToUpToDateNode(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	lead := nt.peers[1].(*raft)

	if lead.lead != 1 {
		t.Fatalf("after election leader is %x, want 1", lead.lead)
	}

	// Transfer leadership to 2.
	nt.send(pb.Message{From: 2, To: 1, Type: pb.MsgTransferLeader})

	checkLeaderTransferState(t, lead, StateFollower, 2)

	// After some log replication, transfer leadership back to 1.
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})

	nt.send(pb.Message{From: 1, To: 2, Type: pb.MsgTransferLeader})

	checkLeaderTransferState(t, lead, StateLeader, 1)
}

func TestLeaderTransferToSlowFollower(t *testing.T) {
	defaultLogger.EnableDebug()
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})

	nt.recover()
	lead := nt.peers[1].(*raft)
	if lead.prs[3].Match != 1 {
		t.Fatalf("node 1 has match %x for node 3, want %x", lead.prs[3].Match, 1)
	}

	// Transfer leadership to 3 when node 3 is lack of log.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})

	checkLeaderTransferState(t, lead, StateFollower, 3)
}

func TestLeaderTransferAfterSnapshot(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})
	lead := nt.peers[1].(*raft)
	nextEnts(lead, nt.storage[1])
	nt.storage[1].CreateSnapshot(lead.raftLog.applied, &pb.ConfState{Nodes: lead.nodes()}, nil)
	nt.storage[1].Compact(lead.raftLog.applied)

	nt.recover()
	if lead.prs[3].Match != 1 {
		t.Fatalf("node 1 has match %x for node 3, want %x", lead.prs[3].Match, 1)
	}

	// Transfer leadership to 3 when node 3 is lack of snapshot.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	// Send pb.MsgHeartbeatResp to leader to trigger a snapshot for node 3.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgHeartbeatResp})

	checkLeaderTransferState(t, lead, StateFollower, 3)
}

func TestLeaderTransferToSelf(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	lead := nt.peers[1].(*raft)

	// Transfer leadership to self, there will be noop.
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgTransferLeader})
	checkLeaderTransferState(t, lead, StateLeader, 1)
}

func TestLeaderTransferToNonExistingNode(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	lead := nt.peers[1].(*raft)
	// Transfer leadership to non-existing node, there will be noop.
	nt.send(pb.Message{From: 4, To: 1, Type: pb.MsgTransferLeader})
	checkLeaderTransferState(t, lead, StateLeader, 1)
}

func TestLeaderTransferTimeout(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	// Transfer leadership to isolated node, wait for timeout.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}
	for i := 0; i < lead.heartbeatTimeout; i++ {
		lead.tick()
	}
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	for i := 0; i < lead.electionTimeout-lead.heartbeatTimeout; i++ {
		lead.tick()
	}

	checkLeaderTransferState(t, lead, StateLeader, 1)
}

func TestLeaderTransferIgnoreProposal(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	// Transfer leadership to isolated node to let transfer pending, then send proposal.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgProp, Entries: []pb.Entry{{}}})

	if lead.prs[1].Match != 1 {
		t.Fatalf("node 1 has match %x, want %x", lead.prs[1].Match, 1)
	}
}

func TestLeaderTransferReceiveHigherTermVote(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	// Transfer leadership to isolated node to let transfer pending.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	nt.send(pb.Message{From: 2, To: 2, Type: pb.MsgHup, Index: 1, Term: 2})

	checkLeaderTransferState(t, lead, StateFollower, 2)
}

func TestLeaderTransferRemoveNode(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.ignore(pb.MsgTimeoutNow)

	lead := nt.peers[1].(*raft)

	// The leadTransferee is removed when leadship transferring.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	lead.removeNode(3)

	checkLeaderTransferState(t, lead, StateLeader, 1)
}

// TestLeaderTransferBack verifies leadership can transfer back to self when last transfer is pending.
func TestLeaderTransferBack(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	// Transfer leadership back to self.
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgTransferLeader})

	checkLeaderTransferState(t, lead, StateLeader, 1)
}

// TestLeaderTransferSecondTransferToAnotherNode verifies leader can transfer to another node
// when last transfer is pending.
func TestLeaderTransferSecondTransferToAnotherNode(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	// Transfer leadership to another node.
	nt.send(pb.Message{From: 2, To: 1, Type: pb.MsgTransferLeader})

	checkLeaderTransferState(t, lead, StateFollower, 2)
}

// TestLeaderTransferSecondTransferToSameNode verifies second transfer leader request
// to the same node should not extend the timeout while the first one is pending.
func TestLeaderTransferSecondTransferToSameNode(t *testing.T) {
	nt := newNetwork(nil, nil, nil)
	nt.send(pb.Message{From: 1, To: 1, Type: pb.MsgHup})

	nt.isolate(3)

	lead := nt.peers[1].(*raft)

	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})
	if lead.leadTransferee != 3 {
		t.Fatalf("wait transferring, leadTransferee = %v, want %v", lead.leadTransferee, 3)
	}

	for i := 0; i < lead.heartbeatTimeout; i++ {
		lead.tick()
	}
	// Second transfer leadership request to the same node.
	nt.send(pb.Message{From: 3, To: 1, Type: pb.MsgTransferLeader})

	for i := 0; i < lead.electionTimeout-lead.heartbeatTimeout; i++ {
		lead.tick()
	}

	checkLeaderTransferState(t, lead, StateLeader, 1)
}

func checkLeaderTransferState(t *testing.T, r *raft, state StateType, lead uint64) {
	if r.state != state || r.lead != lead {
		t.Fatalf("after transferring, node has state %v lead %v, want state %v lead %v", r.state, r.lead, state, lead)
	}
	if r.leadTransferee != None {
		t.Fatalf("after transferring, node has leadTransferee %v, want leadTransferee %v", r.leadTransferee, None)
	}
}

func ents(terms ...uint64) *raft {
	storage := NewMemoryStorage()
	for i, term := range terms {
		storage.Append([]pb.Entry{{Index: uint64(i + 1), Term: term}})
	}
	sm := newTestRaft(1, []uint64{}, 5, 1, storage)
	sm.reset(0)
	return sm
}

type network struct {
	peers   map[uint64]stateMachine
	storage map[uint64]*MemoryStorage
	dropm   map[connem]float64
	ignorem map[pb.MessageType]bool
}

// newNetwork initializes a network from peers.
// A nil node will be replaced with a new *stateMachine.
// A *stateMachine will get its k, id.
// When using stateMachine, the address list is always [1, n].
func newNetwork(peers ...stateMachine) *network {
	size := len(peers)
	peerAddrs := idsBySize(size)

	npeers := make(map[uint64]stateMachine, size)
	nstorage := make(map[uint64]*MemoryStorage, size)

	for j, p := range peers {
		id := peerAddrs[j]
		switch v := p.(type) {
		case nil:
			nstorage[id] = NewMemoryStorage()
			sm := newTestRaft(id, peerAddrs, 10, 1, nstorage[id])
			npeers[id] = sm
		case *raft:
			v.id = id
			v.prs = make(map[uint64]*Progress)
			for i := 0; i < size; i++ {
				v.prs[peerAddrs[i]] = &Progress{}
			}
			v.reset(0)
			npeers[id] = v
		case *blackHole:
			npeers[id] = v
		default:
			panic(fmt.Sprintf("unexpected state machine type: %T", p))
		}
	}
	return &network{
		peers:   npeers,
		storage: nstorage,
		dropm:   make(map[connem]float64),
		ignorem: make(map[pb.MessageType]bool),
	}
}

func (nw *network) send(msgs ...pb.Message) {
	for len(msgs) > 0 {
		m := msgs[0]
		p := nw.peers[m.To]
		p.Step(m)
		msgs = append(msgs[1:], nw.filter(p.readMessages())...)
	}
}

func (nw *network) drop(from, to uint64, perc float64) {
	nw.dropm[connem{from, to}] = perc
}

func (nw *network) cut(one, other uint64) {
	nw.drop(one, other, 1)
	nw.drop(other, one, 1)
}

func (nw *network) isolate(id uint64) {
	for i := 0; i < len(nw.peers); i++ {
		nid := uint64(i) + 1
		if nid != id {
			nw.drop(id, nid, 1.0)
			nw.drop(nid, id, 1.0)
		}
	}
}

func (nw *network) ignore(t pb.MessageType) {
	nw.ignorem[t] = true
}

func (nw *network) recover() {
	nw.dropm = make(map[connem]float64)
	nw.ignorem = make(map[pb.MessageType]bool)
}

func (nw *network) filter(msgs []pb.Message) []pb.Message {
	mm := []pb.Message{}
	for _, m := range msgs {
		if nw.ignorem[m.Type] {
			continue
		}
		switch m.Type {
		case pb.MsgHup:
			// hups never go over the network, so don't drop them but panic
			panic("unexpected msgHup")
		default:
			perc := nw.dropm[connem{m.From, m.To}]
			if n := rand.Float64(); n < perc {
				continue
			}
		}
		mm = append(mm, m)
	}
	return mm
}

type connem struct {
	from, to uint64
}

type blackHole struct{}

func (blackHole) Step(pb.Message) error      { return nil }
func (blackHole) readMessages() []pb.Message { return nil }

var nopStepper = &blackHole{}

func idsBySize(size int) []uint64 {
	ids := make([]uint64, size)
	for i := 0; i < size; i++ {
		ids[i] = 1 + uint64(i)
	}
	return ids
}

func newTestConfig(id uint64, peers []uint64, election, heartbeat int, storage Storage) *Config {
	return &Config{
		ID:              id,
		peers:           peers,
		ElectionTick:    election,
		HeartbeatTick:   heartbeat,
		Storage:         storage,
		MaxSizePerMsg:   noLimit,
		MaxInflightMsgs: 256,
	}
}

func newTestRaft(id uint64, peers []uint64, election, heartbeat int, storage Storage) *raft {
	return newRaft(newTestConfig(id, peers, election, heartbeat, storage))
}
