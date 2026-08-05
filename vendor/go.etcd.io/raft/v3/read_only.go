// Copyright 2016 The etcd Authors
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
	"encoding/binary"

	"go.etcd.io/raft/v3/quorum"
	pb "go.etcd.io/raft/v3/raftpb"
)

// ReadState provides state for read only query.
// It's caller's responsibility to call ReadIndex first before getting
// this state from ready, it's also caller's duty to differentiate if this
// state is what it requests through RequestCtx, eg. given a unique id as
// RequestCtx
type ReadState struct {
	Index      uint64
	RequestCtx []byte
}

type readIndexRequest struct {
	req   *pb.Message
	index uint64
}

type readOnly struct {
	option ReadOnlyOption
	acks   map[uint64]uint64

	unconfirmedReads []*readIndexRequest
	// Number of readIndexRequests that were confirmed in the past by this
	// readOnly, which were removed from the beginning of `unconfirmedReads`.
	confirmedReads uint64
}

func newReadOnly(option ReadOnlyOption) *readOnly {
	return &readOnly{
		option: option,
		acks:   make(map[uint64]uint64),
	}
}

// addRequest adds a read only request into the `readOnly`.
// `commitIndex` is the commit index of the raft state machine when it received
// the read only request.
// `req` is the original read only request message from the local or remote node.
func (ro *readOnly) addRequest(commitIndex uint64, req *pb.Message) {
	ro.unconfirmedReads = append(ro.unconfirmedReads, &readIndexRequest{req: req, index: commitIndex})
}

// recvAck notifies the `readOnly` of an acknowledgment of a heartbeat response.
func (ro *readOnly) recvAck(from uint64, ctx []byte) {
	if len(ctx) != 0 {
		ro.acks[from] = max(ro.acks[from], binary.LittleEndian.Uint64(ctx))
	}
}

// AckedIndex allows for using `CommittedIndex` in `maybeAdvance`.
func (ro *readOnly) AckedIndex(voterID uint64) (quorum.Index, bool) {
	idx, found := ro.acks[voterID]
	return quorum.Index(idx), found
}

// maybeAdvance uses the existing acknowledgements and current raft
// configuration to confirm and return as many unconfirmed reads as possible.
func (ro *readOnly) maybeAdvance(c quorum.JointConfig) []*readIndexRequest {
	// Use `CommittedIndex` to figure out how many reads are now confirmed.
	newConfirmedReads := uint64(c.CommittedIndex(ro))
	if newConfirmedReads <= ro.confirmedReads {
		return nil
	}
	readStates := ro.unconfirmedReads[:newConfirmedReads-ro.confirmedReads]
	ro.unconfirmedReads = ro.unconfirmedReads[newConfirmedReads-ro.confirmedReads:]
	ro.confirmedReads = newConfirmedReads
	return readStates
}

// heartbeatCtx returns the `Context` that should be sent in order to confirm
// all currently unconfirmed reads.
func (ro *readOnly) heartbeatCtx() []byte {
	if len(ro.unconfirmedReads) == 0 {
		return nil
	}
	unconfirmedReadPosition := ro.confirmedReads + uint64(len(ro.unconfirmedReads))
	encLastIndex := make([]byte, 8)
	binary.LittleEndian.PutUint64(encLastIndex, unconfirmedReadPosition)
	return encLastIndex
}
