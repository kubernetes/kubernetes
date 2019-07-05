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
	"errors"

	pb "github.com/coreos/etcd/raft/raftpb"
)

// ErrStepLocalMsg is returned when try to step a local raft message
var ErrStepLocalMsg = errors.New("raft: cannot step raft local message")

// ErrStepPeerNotFound is returned when try to step a response message
// but there is no peer found in raft.prs for that node.
var ErrStepPeerNotFound = errors.New("raft: cannot step as peer not found")

// RawNode is a thread-unsafe Node.
// The methods of this struct correspond to the methods of Node and are described
// more fully there.
type RawNode struct {
	raft       *raft
	prevSoftSt *SoftState
	prevHardSt pb.HardState
}

func (rn *RawNode) newReady() Ready {
	return newReady(rn.raft, rn.prevSoftSt, rn.prevHardSt)
}

func (rn *RawNode) commitReady(rd Ready) {
	if rd.SoftState != nil {
		rn.prevSoftSt = rd.SoftState
	}
	if !IsEmptyHardState(rd.HardState) {
		rn.prevHardSt = rd.HardState
	}
	if rn.prevHardSt.Commit != 0 {
		// In most cases, prevHardSt and rd.HardState will be the same
		// because when there are new entries to apply we just sent a
		// HardState with an updated Commit value. However, on initial
		// startup the two are different because we don't send a HardState
		// until something changes, but we do send any un-applied but
		// committed entries (and previously-committed entries may be
		// incorporated into the snapshot, even if rd.CommittedEntries is
		// empty). Therefore we mark all committed entries as applied
		// whether they were included in rd.HardState or not.
		rn.raft.raftLog.appliedTo(rn.prevHardSt.Commit)
	}
	if len(rd.Entries) > 0 {
		e := rd.Entries[len(rd.Entries)-1]
		rn.raft.raftLog.stableTo(e.Index, e.Term)
	}
	if !IsEmptySnap(rd.Snapshot) {
		rn.raft.raftLog.stableSnapTo(rd.Snapshot.Metadata.Index)
	}
	if len(rd.ReadStates) != 0 {
		rn.raft.readStates = nil
	}
}

// NewRawNode returns a new RawNode given configuration and a list of raft peers.
func NewRawNode(config *Config, peers []Peer) (*RawNode, error) {
	if config.ID == 0 {
		panic("config.ID must not be zero")
	}
	r := newRaft(config)
	rn := &RawNode{
		raft: r,
	}
	lastIndex, err := config.Storage.LastIndex()
	if err != nil {
		panic(err) // TODO(bdarnell)
	}
	// If the log is empty, this is a new RawNode (like StartNode); otherwise it's
	// restoring an existing RawNode (like RestartNode).
	// TODO(bdarnell): rethink RawNode initialization and whether the application needs
	// to be able to tell us when it expects the RawNode to exist.
	if lastIndex == 0 {
		r.becomeFollower(1, None)
		ents := make([]pb.Entry, len(peers))
		for i, peer := range peers {
			cc := pb.ConfChange{Type: pb.ConfChangeAddNode, NodeID: peer.ID, Context: peer.Context}
			data, err := cc.Marshal()
			if err != nil {
				panic("unexpected marshal error")
			}

			ents[i] = pb.Entry{Type: pb.EntryConfChange, Term: 1, Index: uint64(i + 1), Data: data}
		}
		r.raftLog.append(ents...)
		r.raftLog.committed = uint64(len(ents))
		for _, peer := range peers {
			r.addNode(peer.ID)
		}
	}

	// Set the initial hard and soft states after performing all initialization.
	rn.prevSoftSt = r.softState()
	if lastIndex == 0 {
		rn.prevHardSt = emptyState
	} else {
		rn.prevHardSt = r.hardState()
	}

	return rn, nil
}

// Tick advances the internal logical clock by a single tick.
func (rn *RawNode) Tick() {
	rn.raft.tick()
}

// TickQuiesced advances the internal logical clock by a single tick without
// performing any other state machine processing. It allows the caller to avoid
// periodic heartbeats and elections when all of the peers in a Raft group are
// known to be at the same state. Expected usage is to periodically invoke Tick
// or TickQuiesced depending on whether the group is "active" or "quiesced".
//
// WARNING: Be very careful about using this method as it subverts the Raft
// state machine. You should probably be using Tick instead.
func (rn *RawNode) TickQuiesced() {
	rn.raft.electionElapsed++
}

// Campaign causes this RawNode to transition to candidate state.
func (rn *RawNode) Campaign() error {
	return rn.raft.Step(pb.Message{
		Type: pb.MsgHup,
	})
}

// Propose proposes data be appended to the raft log.
func (rn *RawNode) Propose(data []byte) error {
	return rn.raft.Step(pb.Message{
		Type: pb.MsgProp,
		From: rn.raft.id,
		Entries: []pb.Entry{
			{Data: data},
		}})
}

// ProposeConfChange proposes a config change.
func (rn *RawNode) ProposeConfChange(cc pb.ConfChange) error {
	data, err := cc.Marshal()
	if err != nil {
		return err
	}
	return rn.raft.Step(pb.Message{
		Type: pb.MsgProp,
		Entries: []pb.Entry{
			{Type: pb.EntryConfChange, Data: data},
		},
	})
}

// ApplyConfChange applies a config change to the local node.
func (rn *RawNode) ApplyConfChange(cc pb.ConfChange) *pb.ConfState {
	if cc.NodeID == None {
		rn.raft.resetPendingConf()
		return &pb.ConfState{Nodes: rn.raft.nodes()}
	}
	switch cc.Type {
	case pb.ConfChangeAddNode:
		rn.raft.addNode(cc.NodeID)
	case pb.ConfChangeAddLearnerNode:
		rn.raft.addLearner(cc.NodeID)
	case pb.ConfChangeRemoveNode:
		rn.raft.removeNode(cc.NodeID)
	case pb.ConfChangeUpdateNode:
		rn.raft.resetPendingConf()
	default:
		panic("unexpected conf type")
	}
	return &pb.ConfState{Nodes: rn.raft.nodes()}
}

// Step advances the state machine using the given message.
func (rn *RawNode) Step(m pb.Message) error {
	// ignore unexpected local messages receiving over network
	if IsLocalMsg(m.Type) {
		return ErrStepLocalMsg
	}
	if pr := rn.raft.getProgress(m.From); pr != nil || !IsResponseMsg(m.Type) {
		return rn.raft.Step(m)
	}
	return ErrStepPeerNotFound
}

// Ready returns the current point-in-time state of this RawNode.
func (rn *RawNode) Ready() Ready {
	rd := rn.newReady()
	rn.raft.msgs = nil
	return rd
}

// HasReady called when RawNode user need to check if any Ready pending.
// Checking logic in this method should be consistent with Ready.containsUpdates().
func (rn *RawNode) HasReady() bool {
	r := rn.raft
	if !r.softState().equal(rn.prevSoftSt) {
		return true
	}
	if hardSt := r.hardState(); !IsEmptyHardState(hardSt) && !isHardStateEqual(hardSt, rn.prevHardSt) {
		return true
	}
	if r.raftLog.unstable.snapshot != nil && !IsEmptySnap(*r.raftLog.unstable.snapshot) {
		return true
	}
	if len(r.msgs) > 0 || len(r.raftLog.unstableEntries()) > 0 || r.raftLog.hasNextEnts() {
		return true
	}
	if len(r.readStates) != 0 {
		return true
	}
	return false
}

// Advance notifies the RawNode that the application has applied and saved progress in the
// last Ready results.
func (rn *RawNode) Advance(rd Ready) {
	rn.commitReady(rd)
}

// Status returns the current status of the given group.
func (rn *RawNode) Status() *Status {
	status := getStatus(rn.raft)
	return &status
}

// ReportUnreachable reports the given node is not reachable for the last send.
func (rn *RawNode) ReportUnreachable(id uint64) {
	_ = rn.raft.Step(pb.Message{Type: pb.MsgUnreachable, From: id})
}

// ReportSnapshot reports the status of the sent snapshot.
func (rn *RawNode) ReportSnapshot(id uint64, status SnapshotStatus) {
	rej := status == SnapshotFailure

	_ = rn.raft.Step(pb.Message{Type: pb.MsgSnapStatus, From: id, Reject: rej})
}

// TransferLeader tries to transfer leadership to the given transferee.
func (rn *RawNode) TransferLeader(transferee uint64) {
	_ = rn.raft.Step(pb.Message{Type: pb.MsgTransferLeader, From: transferee})
}

// ReadIndex requests a read state. The read state will be set in ready.
// Read State has a read index. Once the application advances further than the read
// index, any linearizable read requests issued before the read request can be
// processed safely. The read state will have the same rctx attached.
func (rn *RawNode) ReadIndex(rctx []byte) {
	_ = rn.raft.Step(pb.Message{Type: pb.MsgReadIndex, Entries: []pb.Entry{{Data: rctx}}})
}
