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

	pb "github.com/coreos/etcd/raft/raftpb"
)

func (st StateType) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf("%q", st.String())), nil
}

// uint64Slice implements sort interface
type uint64Slice []uint64

func (p uint64Slice) Len() int           { return len(p) }
func (p uint64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p uint64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func min(a, b uint64) uint64 {
	if a > b {
		return b
	}
	return a
}

func max(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func IsLocalMsg(msgt pb.MessageType) bool {
	return msgt == pb.MsgHup || msgt == pb.MsgBeat || msgt == pb.MsgUnreachable ||
		msgt == pb.MsgSnapStatus || msgt == pb.MsgCheckQuorum
}

func IsResponseMsg(msgt pb.MessageType) bool {
	return msgt == pb.MsgAppResp || msgt == pb.MsgVoteResp || msgt == pb.MsgHeartbeatResp || msgt == pb.MsgUnreachable || msgt == pb.MsgPreVoteResp
}

// voteResponseType maps vote and prevote message types to their corresponding responses.
func voteRespMsgType(msgt pb.MessageType) pb.MessageType {
	switch msgt {
	case pb.MsgVote:
		return pb.MsgVoteResp
	case pb.MsgPreVote:
		return pb.MsgPreVoteResp
	default:
		panic(fmt.Sprintf("not a vote message: %s", msgt))
	}
}

// EntryFormatter can be implemented by the application to provide human-readable formatting
// of entry data. Nil is a valid EntryFormatter and will use a default format.
type EntryFormatter func([]byte) string

// DescribeMessage returns a concise human-readable description of a
// Message for debugging.
func DescribeMessage(m pb.Message, f EntryFormatter) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%x->%x %v Term:%d Log:%d/%d", m.From, m.To, m.Type, m.Term, m.LogTerm, m.Index)
	if m.Reject {
		fmt.Fprintf(&buf, " Rejected")
		if m.RejectHint != 0 {
			fmt.Fprintf(&buf, "(Hint:%d)", m.RejectHint)
		}
	}
	if m.Commit != 0 {
		fmt.Fprintf(&buf, " Commit:%d", m.Commit)
	}
	if len(m.Entries) > 0 {
		fmt.Fprintf(&buf, " Entries:[")
		for i, e := range m.Entries {
			if i != 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(DescribeEntry(e, f))
		}
		fmt.Fprintf(&buf, "]")
	}
	if !IsEmptySnap(m.Snapshot) {
		fmt.Fprintf(&buf, " Snapshot:%v", m.Snapshot)
	}
	return buf.String()
}

// DescribeEntry returns a concise human-readable description of an
// Entry for debugging.
func DescribeEntry(e pb.Entry, f EntryFormatter) string {
	var formatted string
	if e.Type == pb.EntryNormal && f != nil {
		formatted = f(e.Data)
	} else {
		formatted = fmt.Sprintf("%q", e.Data)
	}
	return fmt.Sprintf("%d/%d %s %s", e.Term, e.Index, e.Type, formatted)
}

func limitSize(ents []pb.Entry, maxSize uint64) []pb.Entry {
	if len(ents) == 0 {
		return ents
	}
	size := ents[0].Size()
	var limit int
	for limit = 1; limit < len(ents); limit++ {
		size += ents[limit].Size()
		if uint64(size) > maxSize {
			break
		}
	}
	return ents[:limit]
}
