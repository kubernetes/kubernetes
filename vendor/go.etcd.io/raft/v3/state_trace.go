// Copyright 2024 The etcd Authors
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

//go:build with_tla

package raft

import (
	"strconv"
	"time"

	"go.etcd.io/raft/v3/raftpb"
	"go.etcd.io/raft/v3/tracker"
)

const StateTraceDeployed = true

type stateMachineEventType int

const (
	rsmInitState stateMachineEventType = iota
	rsmBecomeCandidate
	rsmBecomeFollower
	rsmBecomeLeader
	rsmCommit
	rsmReplicate
	rsmChangeConf
	rsmApplyConfChange
	rsmReady
	rsmSendAppendEntriesRequest
	rsmReceiveAppendEntriesRequest
	rsmSendAppendEntriesResponse
	rsmReceiveAppendEntriesResponse
	rsmSendRequestVoteRequest
	rsmReceiveRequestVoteRequest
	rsmSendRequestVoteResponse
	rsmReceiveRequestVoteResponse
	rsmSendSnapshot
	rsmReceiveSnapshot
)

func (e stateMachineEventType) String() string {
	return []string{
		"InitState",
		"BecomeCandidate",
		"BecomeFollower",
		"BecomeLeader",
		"Commit",
		"Replicate",
		"ChangeConf",
		"ApplyConfChange",
		"Ready",
		"SendAppendEntriesRequest",
		"ReceiveAppendEntriesRequest",
		"SendAppendEntriesResponse",
		"ReceiveAppendEntriesResponse",
		"SendRequestVoteRequest",
		"ReceiveRequestVoteRequest",
		"SendRequestVoteResponse",
		"ReceiveRequestVoteResponse",
		"SendSnapshot",
		"ReceiveSnapshot",
	}[e]
}

const (
	ConfChangeAddNewServer string = "AddNewServer"
	ConfChangeRemoveServer string = "RemoveServer"
	ConfChangeAddLearner   string = "AddLearner"
)

type TracingEvent struct {
	Name       string             `json:"name"`
	NodeID     string             `json:"nid"`
	State      TracingState       `json:"state"`
	Role       string             `json:"role"`
	LogSize    uint64             `json:"log"`
	Conf       [2][]string        `json:"conf"`
	Message    *TracingMessage    `json:"msg,omitempty"`
	ConfChange *TracingConfChange `json:"cc,omitempty"`
	Properties map[string]any     `json:"prop,omitempty"`
}

type TracingState struct {
	Term   uint64 `json:"term"`
	Vote   string `json:"vote"`
	Commit uint64 `json:"commit"`
}

type TracingMessage struct {
	Type        string `json:"type"`
	Term        uint64 `json:"term"`
	From        string `json:"from"`
	To          string `json:"to"`
	EntryLength int    `json:"entries"`
	LogTerm     uint64 `json:"logTerm"`
	Index       uint64 `json:"index"`
	Commit      uint64 `json:"commit"`
	Vote        string `json:"vote"`
	Reject      bool   `json:"reject"`
	RejectHint  uint64 `json:"rejectHint"`
}

type SingleConfChange struct {
	NodeID string `json:"nid"`
	Action string `json:"action"`
}

type TracingConfChange struct {
	Changes []SingleConfChange `json:"changes,omitempty"`
	NewConf []string           `json:"newconf,omitempty"`
}

func makeTracingState(r *raft) TracingState {
	hs := r.hardState()
	return TracingState{
		Term:   hs.Term,
		Vote:   strconv.FormatUint(hs.Vote, 10),
		Commit: hs.Commit,
	}
}

func makeTracingMessage(m *raftpb.Message) *TracingMessage {
	if m == nil {
		return nil
	}

	logTerm := m.LogTerm
	entries := len(m.Entries)
	index := m.Index
	if m.Type == raftpb.MsgSnap {
		index = 0
		logTerm = 0
		entries = int(m.Snapshot.Metadata.Index)
	}
	return &TracingMessage{
		Type:        m.Type.String(),
		Term:        m.Term,
		From:        strconv.FormatUint(m.From, 10),
		To:          strconv.FormatUint(m.To, 10),
		EntryLength: entries,
		LogTerm:     logTerm,
		Index:       index,
		Commit:      m.Commit,
		Vote:        strconv.FormatUint(m.Vote, 10),
		Reject:      m.Reject,
		RejectHint:  m.RejectHint,
	}
}

type TraceLogger interface {
	TraceEvent(*TracingEvent)
}

func traceEvent(evt stateMachineEventType, r *raft, m *raftpb.Message, prop map[string]any) {
	if r.traceLogger == nil {
		return
	}

	r.traceLogger.TraceEvent(&TracingEvent{
		Name:       evt.String(),
		NodeID:     strconv.FormatUint(r.id, 10),
		State:      makeTracingState(r),
		LogSize:    r.raftLog.lastIndex(),
		Conf:       [2][]string{formatConf(r.trk.Voters[0].Slice()), formatConf(r.trk.Voters[1].Slice())},
		Role:       r.state.String(),
		Message:    makeTracingMessage(m),
		Properties: prop,
	})
}

func traceNodeEvent(evt stateMachineEventType, r *raft) {
	traceEvent(evt, r, nil, nil)
}

func formatConf(s []uint64) []string {
	if s == nil {
		return []string{}
	}

	r := make([]string, len(s))
	for i, v := range s {
		r[i] = strconv.FormatUint(v, 10)
	}
	return r
}

// Use following helper functions to trace specific state and/or
// transition at corresponding code lines
func traceInitState(r *raft) {
	if r.traceLogger == nil {
		return
	}

	traceNodeEvent(rsmInitState, r)
}

func traceReady(r *raft) {
	traceNodeEvent(rsmReady, r)
}

func traceCommit(r *raft) {
	traceNodeEvent(rsmCommit, r)
}

func traceReplicate(r *raft, es ...raftpb.Entry) {
	for i := range es {
		if es[i].Type == raftpb.EntryNormal {
			traceNodeEvent(rsmReplicate, r)
		}
	}
}

func traceBecomeFollower(r *raft) {
	traceNodeEvent(rsmBecomeFollower, r)
}

func traceBecomeCandidate(r *raft) {
	traceNodeEvent(rsmBecomeCandidate, r)
}

func traceBecomeLeader(r *raft) {
	traceNodeEvent(rsmBecomeLeader, r)
}

func traceChangeConfEvent(cci raftpb.ConfChangeI, r *raft) {
	cc2 := cci.AsV2()
	cc := &TracingConfChange{
		Changes: []SingleConfChange{},
		NewConf: []string{},
	}
	for _, c := range cc2.Changes {
		switch c.Type {
		case raftpb.ConfChangeAddNode:
			cc.Changes = append(cc.Changes, SingleConfChange{
				NodeID: strconv.FormatUint(c.NodeID, 10),
				Action: ConfChangeAddNewServer,
			})
		case raftpb.ConfChangeRemoveNode:
			cc.Changes = append(cc.Changes, SingleConfChange{
				NodeID: strconv.FormatUint(c.NodeID, 10),
				Action: ConfChangeRemoveServer,
			})
		case raftpb.ConfChangeAddLearnerNode:
			cc.Changes = append(cc.Changes, SingleConfChange{
				NodeID: strconv.FormatUint(c.NodeID, 10),
				Action: ConfChangeAddLearner,
			})
		}
	}

	if len(cc.Changes) == 0 {
		return
	}

	p := map[string]any{}
	p["cc"] = cc
	traceEvent(rsmChangeConf, r, nil, p)
}

func traceConfChangeEvent(cfg tracker.Config, r *raft) {
	if r.traceLogger == nil {
		return
	}

	cc := &TracingConfChange{
		Changes: []SingleConfChange{},
		NewConf: formatConf(cfg.Voters[0].Slice()),
	}

	p := map[string]any{}
	p["cc"] = cc
	traceEvent(rsmApplyConfChange, r, nil, p)
}

func traceSendMessage(r *raft, m *raftpb.Message) {
	if r.traceLogger == nil {
		return
	}

	prop := map[string]any{}

	var evt stateMachineEventType
	switch m.Type {
	case raftpb.MsgApp:
		evt = rsmSendAppendEntriesRequest
		if p, exist := r.trk.Progress[m.From]; exist {
			prop["match"] = p.Match
			prop["next"] = p.Next
		}

	case raftpb.MsgHeartbeat, raftpb.MsgSnap:
		evt = rsmSendAppendEntriesRequest
	case raftpb.MsgAppResp, raftpb.MsgHeartbeatResp:
		evt = rsmSendAppendEntriesResponse
	case raftpb.MsgVote:
		evt = rsmSendRequestVoteRequest
	case raftpb.MsgVoteResp:
		evt = rsmSendRequestVoteResponse
	default:
		return
	}

	traceEvent(evt, r, m, prop)
}

func traceReceiveMessage(r *raft, m *raftpb.Message) {
	if r.traceLogger == nil {
		return
	}

	var evt stateMachineEventType
	switch m.Type {
	case raftpb.MsgApp, raftpb.MsgHeartbeat, raftpb.MsgSnap:
		evt = rsmReceiveAppendEntriesRequest
	case raftpb.MsgAppResp, raftpb.MsgHeartbeatResp:
		evt = rsmReceiveAppendEntriesResponse
	case raftpb.MsgVote:
		evt = rsmReceiveRequestVoteRequest
	case raftpb.MsgVoteResp:
		evt = rsmReceiveRequestVoteResponse
	default:
		return
	}

	time.Sleep(time.Millisecond) // sleep 1ms to reduce time shift impact accross node
	traceEvent(evt, r, m, nil)
}
