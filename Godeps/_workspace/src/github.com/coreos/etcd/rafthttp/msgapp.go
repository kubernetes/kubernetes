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
	"encoding/binary"
	"io"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
)

// msgAppEncoder is a optimized encoder for append messages. It assumes
// that the decoder has enough information to recover the fields except
// Entries, and it writes only Entries into the Writer.
// It MUST be used with a paired msgAppDecoder.
type msgAppEncoder struct {
	w io.Writer
	// TODO: move the fs stats and use new metrics
	fs *stats.FollowerStats
}

func (enc *msgAppEncoder) encode(m raftpb.Message) error {
	if isLinkHeartbeatMessage(m) {
		return binary.Write(enc.w, binary.BigEndian, uint64(0))
	}

	start := time.Now()
	ents := m.Entries
	l := len(ents)
	// There is no need to send empty ents, and it avoids confusion with
	// heartbeat.
	if l == 0 {
		return nil
	}
	if err := binary.Write(enc.w, binary.BigEndian, uint64(l)); err != nil {
		return err
	}
	for i := 0; i < l; i++ {
		ent := &ents[i]
		if err := writeEntryTo(enc.w, ent); err != nil {
			return err
		}
	}
	enc.fs.Succ(time.Since(start))
	return nil
}

// msgAppDecoder is a optimized decoder for append messages. It reads data
// from the Reader and parses it into Entries, then builds messages.
type msgAppDecoder struct {
	r             io.Reader
	local, remote types.ID
	term          uint64
}

func (dec *msgAppDecoder) decode() (raftpb.Message, error) {
	var m raftpb.Message
	var l uint64
	if err := binary.Read(dec.r, binary.BigEndian, &l); err != nil {
		return m, err
	}
	if l == 0 {
		return linkHeartbeatMessage, nil
	}
	ents := make([]raftpb.Entry, int(l))
	for i := 0; i < int(l); i++ {
		ent := &ents[i]
		if err := readEntryFrom(dec.r, ent); err != nil {
			return m, err
		}
	}

	m = raftpb.Message{
		Type:    raftpb.MsgApp,
		From:    uint64(dec.remote),
		To:      uint64(dec.local),
		Term:    dec.term,
		LogTerm: dec.term,
		Index:   ents[0].Index - 1,
		Entries: ents,
	}
	return m, nil
}
