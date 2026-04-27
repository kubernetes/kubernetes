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

//go:build !with_tla

package raft

import (
	"go.etcd.io/raft/v3/raftpb"
	"go.etcd.io/raft/v3/tracker"
)

const StateTraceDeployed = false

type TraceLogger interface{}

type TracingEvent struct{}

func traceInitState(*raft) {}

func traceReady(*raft) {}

func traceCommit(*raft) {}

func traceReplicate(*raft, ...raftpb.Entry) {}

func traceBecomeFollower(*raft) {}

func traceBecomeCandidate(*raft) {}

func traceBecomeLeader(*raft) {}

func traceChangeConfEvent(raftpb.ConfChangeI, *raft) {}

func traceConfChangeEvent(tracker.Config, *raft) {}

func traceSendMessage(*raft, *raftpb.Message) {}

func traceReceiveMessage(*raft, *raftpb.Message) {}
