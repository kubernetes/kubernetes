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

	"github.com/coreos/etcd/raft"
	"golang.org/x/net/context"
)

func TestBasicProgress(t *testing.T) {
	peers := []raft.Peer{{1, nil}, {2, nil}, {3, nil}, {4, nil}, {5, nil}}
	nt := newRaftNetwork(1, 2, 3, 4, 5)

	nodes := make([]*node, 0)

	for i := 1; i <= 5; i++ {
		n := startNode(uint64(i), peers, nt.nodeNetwork(uint64(i)))
		nodes = append(nodes, n)
	}

	time.Sleep(10 * time.Millisecond)

	for i := 0; i < 10000; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}

	time.Sleep(500 * time.Millisecond)
	for _, n := range nodes {
		n.stop()
		if n.state.Commit != 10006 {
			t.Errorf("commit = %d, want = 10006", n.state.Commit)
		}
	}
}

func TestRestart(t *testing.T) {
	peers := []raft.Peer{{1, nil}, {2, nil}, {3, nil}, {4, nil}, {5, nil}}
	nt := newRaftNetwork(1, 2, 3, 4, 5)

	nodes := make([]*node, 0)

	for i := 1; i <= 5; i++ {
		n := startNode(uint64(i), peers, nt.nodeNetwork(uint64(i)))
		nodes = append(nodes, n)
	}

	time.Sleep(50 * time.Millisecond)
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].stop()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].stop()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].restart()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].restart()

	// give some time for nodes to catch up with the raft leader
	time.Sleep(500 * time.Millisecond)
	for _, n := range nodes {
		n.stop()
		if n.state.Commit != 1206 {
			t.Errorf("commit = %d, want = 1206", n.state.Commit)
		}
	}
}

func TestPause(t *testing.T) {
	peers := []raft.Peer{{1, nil}, {2, nil}, {3, nil}, {4, nil}, {5, nil}}
	nt := newRaftNetwork(1, 2, 3, 4, 5)

	nodes := make([]*node, 0)

	for i := 1; i <= 5; i++ {
		n := startNode(uint64(i), peers, nt.nodeNetwork(uint64(i)))
		nodes = append(nodes, n)
	}

	time.Sleep(50 * time.Millisecond)
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].pause()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].pause()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].resume()
	for i := 0; i < 300; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].resume()

	// give some time for nodes to catch up with the raft leader
	time.Sleep(300 * time.Millisecond)
	for _, n := range nodes {
		n.stop()
		if n.state.Commit != 1206 {
			t.Errorf("commit = %d, want = 1206", n.state.Commit)
		}
	}
}
