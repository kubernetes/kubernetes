// Copyright 2026 The etcd Authors
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

package raftpb

const (
	EntryNormal                 = EntryType_EntryNormal
	EntryConfChange             = EntryType_EntryConfChange
	EntryConfChangeV2 EntryType = EntryType_EntryConfChangeV2

	MsgHup               = MessageType_MsgHup
	MsgBeat              = MessageType_MsgBeat
	MsgProp              = MessageType_MsgProp
	MsgApp               = MessageType_MsgApp
	MsgAppResp           = MessageType_MsgAppResp
	MsgVote              = MessageType_MsgVote
	MsgVoteResp          = MessageType_MsgVoteResp
	MsgSnap              = MessageType_MsgSnap
	MsgHeartbeat         = MessageType_MsgHeartbeat
	MsgHeartbeatResp     = MessageType_MsgHeartbeatResp
	MsgUnreachable       = MessageType_MsgUnreachable
	MsgSnapStatus        = MessageType_MsgSnapStatus
	MsgCheckQuorum       = MessageType_MsgCheckQuorum
	MsgTransferLeader    = MessageType_MsgTransferLeader
	MsgTimeoutNow        = MessageType_MsgTimeoutNow
	MsgReadIndex         = MessageType_MsgReadIndex
	MsgReadIndexResp     = MessageType_MsgReadIndexResp
	MsgPreVote           = MessageType_MsgPreVote
	MsgPreVoteResp       = MessageType_MsgPreVoteResp
	MsgStorageAppend     = MessageType_MsgStorageAppend
	MsgStorageAppendResp = MessageType_MsgStorageAppendResp
	MsgStorageApply      = MessageType_MsgStorageApply
	MsgStorageApplyResp  = MessageType_MsgStorageApplyResp
	MsgForgetLeader      = MessageType_MsgForgetLeader

	ConfChangeTransitionAuto          = ConfChangeTransition_ConfChangeTransitionAuto
	ConfChangeTransitionJointImplicit = ConfChangeTransition_ConfChangeTransitionJointImplicit
	ConfChangeTransitionJointExplicit = ConfChangeTransition_ConfChangeTransitionJointExplicit

	ConfChangeAddNode        = ConfChangeType_ConfChangeAddNode
	ConfChangeRemoveNode     = ConfChangeType_ConfChangeRemoveNode
	ConfChangeUpdateNode     = ConfChangeType_ConfChangeUpdateNode
	ConfChangeAddLearnerNode = ConfChangeType_ConfChangeAddLearnerNode
)
