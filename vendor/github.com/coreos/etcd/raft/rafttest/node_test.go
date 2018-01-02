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

	waitLeader(nodes)

	for i := 0; i < 100; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}

	if !waitCommitConverge(nodes, 100) {
		t.Errorf("commits failed to converge!")
	}

	for _, n := range nodes {
		n.stop()
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

	l := waitLeader(nodes)
	k1, k2 := (l+1)%5, (l+2)%5

	for i := 0; i < 30; i++ {
		nodes[l].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[k1].stop()
	for i := 0; i < 30; i++ {
		nodes[(l+3)%5].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[k2].stop()
	for i := 0; i < 30; i++ {
		nodes[(l+4)%5].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[k2].restart()
	for i := 0; i < 30; i++ {
		nodes[l].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[k1].restart()

	if !waitCommitConverge(nodes, 120) {
		t.Errorf("commits failed to converge!")
	}

	for _, n := range nodes {
		n.stop()
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

	waitLeader(nodes)

	for i := 0; i < 30; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].pause()
	for i := 0; i < 30; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].pause()
	for i := 0; i < 30; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[2].resume()
	for i := 0; i < 30; i++ {
		nodes[0].Propose(context.TODO(), []byte("somedata"))
	}
	nodes[1].resume()

	if !waitCommitConverge(nodes, 120) {
		t.Errorf("commits failed to converge!")
	}

	for _, n := range nodes {
		n.stop()
	}
}

func waitLeader(ns []*node) int {
	var l map[uint64]struct{}
	var lindex int

	for {
		l = make(map[uint64]struct{})

		for i, n := range ns {
			lead := n.Status().SoftState.Lead
			if lead != 0 {
				l[lead] = struct{}{}
				if n.id == lead {
					lindex = i
				}
			}
		}

		if len(l) == 1 {
			return lindex
		}
	}
}

func waitCommitConverge(ns []*node, target uint64) bool {
	var c map[uint64]struct{}

	for i := 0; i < 50; i++ {
		c = make(map[uint64]struct{})
		var good int

		for _, n := range ns {
			commit := n.Node.Status().HardState.Commit
			c[commit] = struct{}{}
			if commit > target {
				good++
			}
		}

		if len(c) == 1 && good == len(ns) {
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}

	return false
}
