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
	"math/rand"
	"sync"
	"time"

	"github.com/coreos/etcd/raft/raftpb"
)

// a network interface
type iface interface {
	send(m raftpb.Message)
	recv() chan raftpb.Message
	disconnect()
	connect()
}

// a network
type network interface {
	// drop message at given rate (1.0 drops all messages)
	drop(from, to uint64, rate float64)
	// delay message for (0, d] randomly at given rate (1.0 delay all messages)
	// do we need rate here?
	delay(from, to uint64, d time.Duration, rate float64)
	disconnect(id uint64)
	connect(id uint64)
	// heal heals the network
	heal()
}

type raftNetwork struct {
	mu           sync.Mutex
	disconnected map[uint64]bool
	dropmap      map[conn]float64
	delaymap     map[conn]delay
	recvQueues   map[uint64]chan raftpb.Message
}

type conn struct {
	from, to uint64
}

type delay struct {
	d    time.Duration
	rate float64
}

func newRaftNetwork(nodes ...uint64) *raftNetwork {
	pn := &raftNetwork{
		recvQueues:   make(map[uint64]chan raftpb.Message),
		dropmap:      make(map[conn]float64),
		delaymap:     make(map[conn]delay),
		disconnected: make(map[uint64]bool),
	}

	for _, n := range nodes {
		pn.recvQueues[n] = make(chan raftpb.Message, 1024)
	}
	return pn
}

func (rn *raftNetwork) nodeNetwork(id uint64) iface {
	return &nodeNetwork{id: id, raftNetwork: rn}
}

func (rn *raftNetwork) send(m raftpb.Message) {
	rn.mu.Lock()
	to := rn.recvQueues[m.To]
	if rn.disconnected[m.To] {
		to = nil
	}
	drop := rn.dropmap[conn{m.From, m.To}]
	dl := rn.delaymap[conn{m.From, m.To}]
	rn.mu.Unlock()

	if to == nil {
		return
	}
	if drop != 0 && rand.Float64() < drop {
		return
	}
	// TODO: shall we dl without blocking the send call?
	if dl.d != 0 && rand.Float64() < dl.rate {
		rd := rand.Int63n(int64(dl.d))
		time.Sleep(time.Duration(rd))
	}

	select {
	case to <- m:
	default:
		// drop messages when the receiver queue is full.
	}
}

func (rn *raftNetwork) recvFrom(from uint64) chan raftpb.Message {
	rn.mu.Lock()
	fromc := rn.recvQueues[from]
	if rn.disconnected[from] {
		fromc = nil
	}
	rn.mu.Unlock()

	return fromc
}

func (rn *raftNetwork) drop(from, to uint64, rate float64) {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	rn.dropmap[conn{from, to}] = rate
}

func (rn *raftNetwork) delay(from, to uint64, d time.Duration, rate float64) {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	rn.delaymap[conn{from, to}] = delay{d, rate}
}

func (rn *raftNetwork) heal() {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	rn.dropmap = make(map[conn]float64)
	rn.delaymap = make(map[conn]delay)
}

func (rn *raftNetwork) disconnect(id uint64) {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	rn.disconnected[id] = true
}

func (rn *raftNetwork) connect(id uint64) {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	rn.disconnected[id] = false
}

type nodeNetwork struct {
	id uint64
	*raftNetwork
}

func (nt *nodeNetwork) connect() {
	nt.raftNetwork.connect(nt.id)
}

func (nt *nodeNetwork) disconnect() {
	nt.raftNetwork.disconnect(nt.id)
}

func (nt *nodeNetwork) send(m raftpb.Message) {
	nt.raftNetwork.send(m)
}

func (nt *nodeNetwork) recv() chan raftpb.Message {
	return nt.recvFrom(nt.id)
}
