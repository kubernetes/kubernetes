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
	"time"

	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/raft/raftpb"
	"golang.org/x/net/context"
)

// TestNodeStep ensures that node.Step sends msgProp to propc chan
// and other kinds of messages to recvc chan.
func TestNodeStep(t *testing.T) {
	for i, msgn := range raftpb.MessageType_name {
		n := &node{
			propc: make(chan raftpb.Message, 1),
			recvc: make(chan raftpb.Message, 1),
		}
		msgt := raftpb.MessageType(i)
		n.Step(context.TODO(), raftpb.Message{Type: msgt})
		// Proposal goes to proc chan. Others go to recvc chan.
		if msgt == raftpb.MsgProp {
			select {
			case <-n.propc:
			default:
				t.Errorf("%d: cannot receive %s on propc chan", msgt, msgn)
			}
		} else {
			if IsLocalMsg(msgt) {
				select {
				case <-n.recvc:
					t.Errorf("%d: step should ignore %s", msgt, msgn)
				default:
				}
			} else {
				select {
				case <-n.recvc:
				default:
					t.Errorf("%d: cannot receive %s on recvc chan", msgt, msgn)
				}
			}
		}
	}
}

// Cancel and Stop should unblock Step()
func TestNodeStepUnblock(t *testing.T) {
	// a node without buffer to block step
	n := &node{
		propc: make(chan raftpb.Message),
		done:  make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())
	stopFunc := func() { close(n.done) }

	tests := []struct {
		unblock func()
		werr    error
	}{
		{stopFunc, ErrStopped},
		{cancel, context.Canceled},
	}

	for i, tt := range tests {
		errc := make(chan error, 1)
		go func() {
			err := n.Step(ctx, raftpb.Message{Type: raftpb.MsgProp})
			errc <- err
		}()
		tt.unblock()
		select {
		case err := <-errc:
			if err != tt.werr {
				t.Errorf("#%d: err = %v, want %v", i, err, tt.werr)
			}
			//clean up side-effect
			if ctx.Err() != nil {
				ctx = context.TODO()
			}
			select {
			case <-n.done:
				n.done = make(chan struct{})
			default:
			}
		case <-time.After(1 * time.Second):
			t.Fatalf("#%d: failed to unblock step", i)
		}
	}
}

// TestNodePropose ensures that node.Propose sends the given proposal to the underlying raft.
func TestNodePropose(t *testing.T) {
	msgs := []raftpb.Message{}
	appendStep := func(r *raft, m raftpb.Message) {
		msgs = append(msgs, m)
	}

	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	go n.run(r)
	n.Campaign(context.TODO())
	for {
		rd := <-n.Ready()
		s.Append(rd.Entries)
		// change the step function to appendStep until this raft becomes leader
		if rd.SoftState.Lead == r.id {
			r.step = appendStep
			n.Advance()
			break
		}
		n.Advance()
	}
	n.Propose(context.TODO(), []byte("somedata"))
	n.Stop()

	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want %d", len(msgs), 1)
	}
	if msgs[0].Type != raftpb.MsgProp {
		t.Errorf("msg type = %d, want %d", msgs[0].Type, raftpb.MsgProp)
	}
	if !bytes.Equal(msgs[0].Entries[0].Data, []byte("somedata")) {
		t.Errorf("data = %v, want %v", msgs[0].Entries[0].Data, []byte("somedata"))
	}
}

// TestNodeReadIndex ensures that node.ReadIndex sends the MsgReadIndex message to the underlying raft.
// It also ensures that ReadState can be read out through ready chan.
func TestNodeReadIndex(t *testing.T) {
	msgs := []raftpb.Message{}
	appendStep := func(r *raft, m raftpb.Message) {
		msgs = append(msgs, m)
	}
	wrs := []ReadState{{Index: uint64(1), RequestCtx: []byte("somedata")}}

	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	r.readStates = wrs

	go n.run(r)
	n.Campaign(context.TODO())
	for {
		rd := <-n.Ready()
		if !reflect.DeepEqual(rd.ReadStates, wrs) {
			t.Errorf("ReadStates = %v, want %v", rd.ReadStates, wrs)
		}

		s.Append(rd.Entries)

		if rd.SoftState.Lead == r.id {
			n.Advance()
			break
		}
		n.Advance()
	}

	r.step = appendStep
	wrequestCtx := []byte("somedata2")
	n.ReadIndex(context.TODO(), wrequestCtx)
	n.Stop()

	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want %d", len(msgs), 1)
	}
	if msgs[0].Type != raftpb.MsgReadIndex {
		t.Errorf("msg type = %d, want %d", msgs[0].Type, raftpb.MsgReadIndex)
	}
	if !bytes.Equal(msgs[0].Entries[0].Data, wrequestCtx) {
		t.Errorf("data = %v, want %v", msgs[0].Entries[0].Data, wrequestCtx)
	}
}

// TestNodeReadIndexToOldLeader ensures that raftpb.MsgReadIndex to old leader
// gets forwarded to the new leader and 'send' method does not attach its term.
func TestNodeReadIndexToOldLeader(t *testing.T) {
	r1 := newTestRaft(1, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	r2 := newTestRaft(2, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())
	r3 := newTestRaft(3, []uint64{1, 2, 3}, 10, 1, NewMemoryStorage())

	nt := newNetwork(r1, r2, r3)

	// elect r1 as leader
	nt.send(raftpb.Message{From: 1, To: 1, Type: raftpb.MsgHup})

	var testEntries = []raftpb.Entry{{Data: []byte("testdata")}}

	// send readindex request to r2(follower)
	r2.Step(raftpb.Message{From: 2, To: 2, Type: raftpb.MsgReadIndex, Entries: testEntries})

	// verify r2(follower) forwards this message to r1(leader) with term not set
	if len(r2.msgs) != 1 {
		t.Fatalf("len(r2.msgs) expected 1, got %d", len(r2.msgs))
	}
	readIndxMsg1 := raftpb.Message{From: 2, To: 1, Type: raftpb.MsgReadIndex, Entries: testEntries}
	if !reflect.DeepEqual(r2.msgs[0], readIndxMsg1) {
		t.Fatalf("r2.msgs[0] expected %+v, got %+v", readIndxMsg1, r2.msgs[0])
	}

	// send readindex request to r3(follower)
	r3.Step(raftpb.Message{From: 3, To: 3, Type: raftpb.MsgReadIndex, Entries: testEntries})

	// verify r3(follower) forwards this message to r1(leader) with term not set as well.
	if len(r3.msgs) != 1 {
		t.Fatalf("len(r3.msgs) expected 1, got %d", len(r3.msgs))
	}
	readIndxMsg2 := raftpb.Message{From: 3, To: 1, Type: raftpb.MsgReadIndex, Entries: testEntries}
	if !reflect.DeepEqual(r3.msgs[0], readIndxMsg2) {
		t.Fatalf("r3.msgs[0] expected %+v, got %+v", readIndxMsg2, r3.msgs[0])
	}

	// now elect r3 as leader
	nt.send(raftpb.Message{From: 3, To: 3, Type: raftpb.MsgHup})

	// let r1 steps the two messages previously we got from r2, r3
	r1.Step(readIndxMsg1)
	r1.Step(readIndxMsg2)

	// verify r1(follower) forwards these messages again to r3(new leader)
	if len(r1.msgs) != 2 {
		t.Fatalf("len(r1.msgs) expected 1, got %d", len(r1.msgs))
	}
	readIndxMsg3 := raftpb.Message{From: 1, To: 3, Type: raftpb.MsgReadIndex, Entries: testEntries}
	if !reflect.DeepEqual(r1.msgs[0], readIndxMsg3) {
		t.Fatalf("r1.msgs[0] expected %+v, got %+v", readIndxMsg3, r1.msgs[0])
	}
	if !reflect.DeepEqual(r1.msgs[1], readIndxMsg3) {
		t.Fatalf("r1.msgs[1] expected %+v, got %+v", readIndxMsg3, r1.msgs[1])
	}
}

// TestNodeProposeConfig ensures that node.ProposeConfChange sends the given configuration proposal
// to the underlying raft.
func TestNodeProposeConfig(t *testing.T) {
	msgs := []raftpb.Message{}
	appendStep := func(r *raft, m raftpb.Message) {
		msgs = append(msgs, m)
	}

	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	go n.run(r)
	n.Campaign(context.TODO())
	for {
		rd := <-n.Ready()
		s.Append(rd.Entries)
		// change the step function to appendStep until this raft becomes leader
		if rd.SoftState.Lead == r.id {
			r.step = appendStep
			n.Advance()
			break
		}
		n.Advance()
	}
	cc := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1}
	ccdata, err := cc.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	n.ProposeConfChange(context.TODO(), cc)
	n.Stop()

	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want %d", len(msgs), 1)
	}
	if msgs[0].Type != raftpb.MsgProp {
		t.Errorf("msg type = %d, want %d", msgs[0].Type, raftpb.MsgProp)
	}
	if !bytes.Equal(msgs[0].Entries[0].Data, ccdata) {
		t.Errorf("data = %v, want %v", msgs[0].Entries[0].Data, ccdata)
	}
}

// TestNodeProposeAddDuplicateNode ensures that two proposes to add the same node should
// not affect the later propose to add new node.
func TestNodeProposeAddDuplicateNode(t *testing.T) {
	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	go n.run(r)
	n.Campaign(context.TODO())
	rdyEntries := make([]raftpb.Entry, 0)
	ticker := time.NewTicker(time.Millisecond * 100)
	done := make(chan struct{})
	stop := make(chan struct{})
	applyConfChan := make(chan struct{})

	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				n.Tick()
			case rd := <-n.Ready():
				s.Append(rd.Entries)
				for _, e := range rd.Entries {
					rdyEntries = append(rdyEntries, e)
					switch e.Type {
					case raftpb.EntryNormal:
					case raftpb.EntryConfChange:
						var cc raftpb.ConfChange
						cc.Unmarshal(e.Data)
						n.ApplyConfChange(cc)
						applyConfChan <- struct{}{}
					}
				}
				n.Advance()
			}
		}
	}()

	cc1 := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1}
	ccdata1, _ := cc1.Marshal()
	n.ProposeConfChange(context.TODO(), cc1)
	<-applyConfChan

	// try add the same node again
	n.ProposeConfChange(context.TODO(), cc1)
	<-applyConfChan

	// the new node join should be ok
	cc2 := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 2}
	ccdata2, _ := cc2.Marshal()
	n.ProposeConfChange(context.TODO(), cc2)
	<-applyConfChan

	close(stop)
	<-done

	if len(rdyEntries) != 4 {
		t.Errorf("len(entry) = %d, want %d, %v\n", len(rdyEntries), 4, rdyEntries)
	}
	if !bytes.Equal(rdyEntries[1].Data, ccdata1) {
		t.Errorf("data = %v, want %v", rdyEntries[1].Data, ccdata1)
	}
	if !bytes.Equal(rdyEntries[3].Data, ccdata2) {
		t.Errorf("data = %v, want %v", rdyEntries[3].Data, ccdata2)
	}
	n.Stop()
}

// TestBlockProposal ensures that node will block proposal when it does not
// know who is the current leader; node will accept proposal when it knows
// who is the current leader.
func TestBlockProposal(t *testing.T) {
	n := newNode()
	r := newTestRaft(1, []uint64{1}, 10, 1, NewMemoryStorage())
	go n.run(r)
	defer n.Stop()

	errc := make(chan error, 1)
	go func() {
		errc <- n.Propose(context.TODO(), []byte("somedata"))
	}()

	testutil.WaitSchedule()
	select {
	case err := <-errc:
		t.Errorf("err = %v, want blocking", err)
	default:
	}

	n.Campaign(context.TODO())
	select {
	case err := <-errc:
		if err != nil {
			t.Errorf("err = %v, want %v", err, nil)
		}
	case <-time.After(10 * time.Second):
		t.Errorf("blocking proposal, want unblocking")
	}
}

// TestNodeTick ensures that node.Tick() will increase the
// elapsed of the underlying raft state machine.
func TestNodeTick(t *testing.T) {
	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	go n.run(r)
	elapsed := r.electionElapsed
	n.Tick()
	testutil.WaitSchedule()
	n.Stop()
	if r.electionElapsed != elapsed+1 {
		t.Errorf("elapsed = %d, want %d", r.electionElapsed, elapsed+1)
	}
}

// TestNodeStop ensures that node.Stop() blocks until the node has stopped
// processing, and that it is idempotent
func TestNodeStop(t *testing.T) {
	n := newNode()
	s := NewMemoryStorage()
	r := newTestRaft(1, []uint64{1}, 10, 1, s)
	donec := make(chan struct{})

	go func() {
		n.run(r)
		close(donec)
	}()

	status := n.Status()
	n.Stop()

	select {
	case <-donec:
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for node to stop!")
	}

	emptyStatus := Status{}

	if reflect.DeepEqual(status, emptyStatus) {
		t.Errorf("status = %v, want not empty", status)
	}
	// Further status should return be empty, the node is stopped.
	status = n.Status()
	if !reflect.DeepEqual(status, emptyStatus) {
		t.Errorf("status = %v, want empty", status)
	}
	// Subsequent Stops should have no effect.
	n.Stop()
}

func TestReadyContainUpdates(t *testing.T) {
	tests := []struct {
		rd       Ready
		wcontain bool
	}{
		{Ready{}, false},
		{Ready{SoftState: &SoftState{Lead: 1}}, true},
		{Ready{HardState: raftpb.HardState{Vote: 1}}, true},
		{Ready{Entries: make([]raftpb.Entry, 1, 1)}, true},
		{Ready{CommittedEntries: make([]raftpb.Entry, 1, 1)}, true},
		{Ready{Messages: make([]raftpb.Message, 1, 1)}, true},
		{Ready{Snapshot: raftpb.Snapshot{Metadata: raftpb.SnapshotMetadata{Index: 1}}}, true},
	}

	for i, tt := range tests {
		if g := tt.rd.containsUpdates(); g != tt.wcontain {
			t.Errorf("#%d: containUpdates = %v, want %v", i, g, tt.wcontain)
		}
	}
}

// TestNodeStart ensures that a node can be started correctly. The node should
// start with correct configuration change entries, and can accept and commit
// proposals.
func TestNodeStart(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cc := raftpb.ConfChange{Type: raftpb.ConfChangeAddNode, NodeID: 1}
	ccdata, err := cc.Marshal()
	if err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	wants := []Ready{
		{
			HardState: raftpb.HardState{Term: 1, Commit: 1, Vote: 0},
			Entries: []raftpb.Entry{
				{Type: raftpb.EntryConfChange, Term: 1, Index: 1, Data: ccdata},
			},
			CommittedEntries: []raftpb.Entry{
				{Type: raftpb.EntryConfChange, Term: 1, Index: 1, Data: ccdata},
			},
		},
		{
			HardState:        raftpb.HardState{Term: 2, Commit: 3, Vote: 1},
			Entries:          []raftpb.Entry{{Term: 2, Index: 3, Data: []byte("foo")}},
			CommittedEntries: []raftpb.Entry{{Term: 2, Index: 3, Data: []byte("foo")}},
		},
	}
	storage := NewMemoryStorage()
	c := &Config{
		ID:              1,
		ElectionTick:    10,
		HeartbeatTick:   1,
		Storage:         storage,
		MaxSizePerMsg:   noLimit,
		MaxInflightMsgs: 256,
	}
	n := StartNode(c, []Peer{{ID: 1}})
	defer n.Stop()
	g := <-n.Ready()
	if !reflect.DeepEqual(g, wants[0]) {
		t.Fatalf("#%d: g = %+v,\n             w   %+v", 1, g, wants[0])
	} else {
		storage.Append(g.Entries)
		n.Advance()
	}

	n.Campaign(ctx)
	rd := <-n.Ready()
	storage.Append(rd.Entries)
	n.Advance()

	n.Propose(ctx, []byte("foo"))
	if g2 := <-n.Ready(); !reflect.DeepEqual(g2, wants[1]) {
		t.Errorf("#%d: g = %+v,\n             w   %+v", 2, g2, wants[1])
	} else {
		storage.Append(g2.Entries)
		n.Advance()
	}

	select {
	case rd := <-n.Ready():
		t.Errorf("unexpected Ready: %+v", rd)
	case <-time.After(time.Millisecond):
	}
}

func TestNodeRestart(t *testing.T) {
	entries := []raftpb.Entry{
		{Term: 1, Index: 1},
		{Term: 1, Index: 2, Data: []byte("foo")},
	}
	st := raftpb.HardState{Term: 1, Commit: 1}

	want := Ready{
		HardState: st,
		// commit up to index commit index in st
		CommittedEntries: entries[:st.Commit],
	}

	storage := NewMemoryStorage()
	storage.SetHardState(st)
	storage.Append(entries)
	c := &Config{
		ID:              1,
		ElectionTick:    10,
		HeartbeatTick:   1,
		Storage:         storage,
		MaxSizePerMsg:   noLimit,
		MaxInflightMsgs: 256,
	}
	n := RestartNode(c)
	defer n.Stop()
	if g := <-n.Ready(); !reflect.DeepEqual(g, want) {
		t.Errorf("g = %+v,\n             w   %+v", g, want)
	}
	n.Advance()

	select {
	case rd := <-n.Ready():
		t.Errorf("unexpected Ready: %+v", rd)
	case <-time.After(time.Millisecond):
	}
}

func TestNodeRestartFromSnapshot(t *testing.T) {
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
		HardState: st,
		// commit up to index commit index in st
		CommittedEntries: entries,
	}

	s := NewMemoryStorage()
	s.SetHardState(st)
	s.ApplySnapshot(snap)
	s.Append(entries)
	c := &Config{
		ID:              1,
		ElectionTick:    10,
		HeartbeatTick:   1,
		Storage:         s,
		MaxSizePerMsg:   noLimit,
		MaxInflightMsgs: 256,
	}
	n := RestartNode(c)
	defer n.Stop()
	if g := <-n.Ready(); !reflect.DeepEqual(g, want) {
		t.Errorf("g = %+v,\n             w   %+v", g, want)
	} else {
		n.Advance()
	}

	select {
	case rd := <-n.Ready():
		t.Errorf("unexpected Ready: %+v", rd)
	case <-time.After(time.Millisecond):
	}
}

func TestNodeAdvance(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	storage := NewMemoryStorage()
	c := &Config{
		ID:              1,
		ElectionTick:    10,
		HeartbeatTick:   1,
		Storage:         storage,
		MaxSizePerMsg:   noLimit,
		MaxInflightMsgs: 256,
	}
	n := StartNode(c, []Peer{{ID: 1}})
	defer n.Stop()
	rd := <-n.Ready()
	storage.Append(rd.Entries)
	n.Advance()

	n.Campaign(ctx)
	<-n.Ready()

	n.Propose(ctx, []byte("foo"))
	select {
	case rd = <-n.Ready():
		t.Fatalf("unexpected Ready before Advance: %+v", rd)
	case <-time.After(time.Millisecond):
	}
	storage.Append(rd.Entries)
	n.Advance()
	select {
	case <-n.Ready():
	case <-time.After(100 * time.Millisecond):
		t.Errorf("expect Ready after Advance, but there is no Ready available")
	}
}

func TestSoftStateEqual(t *testing.T) {
	tests := []struct {
		st *SoftState
		we bool
	}{
		{&SoftState{}, true},
		{&SoftState{Lead: 1}, false},
		{&SoftState{RaftState: StateLeader}, false},
	}
	for i, tt := range tests {
		if g := tt.st.equal(&SoftState{}); g != tt.we {
			t.Errorf("#%d, equal = %v, want %v", i, g, tt.we)
		}
	}
}

func TestIsHardStateEqual(t *testing.T) {
	tests := []struct {
		st raftpb.HardState
		we bool
	}{
		{emptyState, true},
		{raftpb.HardState{Vote: 1}, false},
		{raftpb.HardState{Commit: 1}, false},
		{raftpb.HardState{Term: 1}, false},
	}

	for i, tt := range tests {
		if isHardStateEqual(tt.st, emptyState) != tt.we {
			t.Errorf("#%d, equal = %v, want %v", i, isHardStateEqual(tt.st, emptyState), tt.we)
		}
	}
}
