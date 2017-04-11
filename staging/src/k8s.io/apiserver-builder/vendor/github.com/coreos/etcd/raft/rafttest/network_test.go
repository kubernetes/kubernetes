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

package rafttest

import (
	"testing"
	"time"

	"github.com/coreos/etcd/raft/raftpb"
)

func TestNetworkDrop(t *testing.T) {
	// drop around 10% messages
	sent := 1000
	droprate := 0.1
	nt := newRaftNetwork(1, 2)
	nt.drop(1, 2, droprate)
	for i := 0; i < sent; i++ {
		nt.send(raftpb.Message{From: 1, To: 2})
	}

	c := nt.recvFrom(2)

	received := 0
	done := false
	for !done {
		select {
		case <-c:
			received++
		default:
			done = true
		}
	}

	drop := sent - received
	if drop > int((droprate+0.1)*float64(sent)) || drop < int((droprate-0.1)*float64(sent)) {
		t.Errorf("drop = %d, want around %.2f", drop, droprate*float64(sent))
	}
}

func TestNetworkDelay(t *testing.T) {
	sent := 1000
	delay := time.Millisecond
	delayrate := 0.1
	nt := newRaftNetwork(1, 2)

	nt.delay(1, 2, delay, delayrate)
	var total time.Duration
	for i := 0; i < sent; i++ {
		s := time.Now()
		nt.send(raftpb.Message{From: 1, To: 2})
		total += time.Since(s)
	}

	w := time.Duration(float64(sent)*delayrate/2) * delay
	// there is some overhead in the send call since it generates random numbers.
	if total < w {
		t.Errorf("total = %v, want > %v", total, w)
	}
}
