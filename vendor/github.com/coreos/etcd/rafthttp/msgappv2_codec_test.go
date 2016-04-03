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

package rafthttp

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
)

func TestMsgAppV2(t *testing.T) {
	tests := []raftpb.Message{
		linkHeartbeatMessage,
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    1,
			LogTerm: 1,
			Index:   0,
			Entries: []raftpb.Entry{
				{Term: 1, Index: 1, Data: []byte("some data")},
				{Term: 1, Index: 2, Data: []byte("some data")},
				{Term: 1, Index: 3, Data: []byte("some data")},
			},
		},
		// consecutive MsgApp
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    1,
			LogTerm: 1,
			Index:   3,
			Entries: []raftpb.Entry{
				{Term: 1, Index: 4, Data: []byte("some data")},
			},
		},
		linkHeartbeatMessage,
		// consecutive MsgApp after linkHeartbeatMessage
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    1,
			LogTerm: 1,
			Index:   4,
			Entries: []raftpb.Entry{
				{Term: 1, Index: 5, Data: []byte("some data")},
			},
		},
		// MsgApp with higher term
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    3,
			LogTerm: 1,
			Index:   5,
			Entries: []raftpb.Entry{
				{Term: 3, Index: 6, Data: []byte("some data")},
			},
		},
		linkHeartbeatMessage,
		// consecutive MsgApp
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    3,
			LogTerm: 2,
			Index:   6,
			Entries: []raftpb.Entry{
				{Term: 3, Index: 7, Data: []byte("some data")},
			},
		},
		// consecutive empty MsgApp
		{
			Type:    raftpb.MsgApp,
			From:    1,
			To:      2,
			Term:    3,
			LogTerm: 2,
			Index:   7,
			Entries: nil,
		},
		linkHeartbeatMessage,
	}
	b := &bytes.Buffer{}
	enc := newMsgAppV2Encoder(b, &stats.FollowerStats{})
	dec := newMsgAppV2Decoder(b, types.ID(2), types.ID(1))

	for i, tt := range tests {
		if err := enc.encode(tt); err != nil {
			t.Errorf("#%d: unexpected encode message error: %v", i, err)
			continue
		}
		m, err := dec.decode()
		if err != nil {
			t.Errorf("#%d: unexpected decode message error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(m, tt) {
			t.Errorf("#%d: message = %+v, want %+v", i, m, tt)
		}
	}
}
