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

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/etcd/raft/raftpb"
)

type cluster struct {
	peers       []string
	commitC     []<-chan *string
	errorC      []<-chan error
	proposeC    []chan string
	confChangeC []chan raftpb.ConfChange
}

// newCluster creates a cluster of n nodes
func newCluster(n int) *cluster {
	peers := make([]string, n)
	for i := range peers {
		peers[i] = fmt.Sprintf("http://127.0.0.1:%d", 10000+i)
	}

	clus := &cluster{
		peers:       peers,
		commitC:     make([]<-chan *string, len(peers)),
		errorC:      make([]<-chan error, len(peers)),
		proposeC:    make([]chan string, len(peers)),
		confChangeC: make([]chan raftpb.ConfChange, len(peers)),
	}

	for i := range clus.peers {
		os.RemoveAll(fmt.Sprintf("raftexample-%d", i+1))
		os.RemoveAll(fmt.Sprintf("raftexample-%d-snap", i+1))
		clus.proposeC[i] = make(chan string, 1)
		clus.confChangeC[i] = make(chan raftpb.ConfChange, 1)
		clus.commitC[i], clus.errorC[i], _ = newRaftNode(i+1, clus.peers, false, nil, clus.proposeC[i], clus.confChangeC[i])
	}

	return clus
}

// sinkReplay reads all commits in each node's local log.
func (clus *cluster) sinkReplay() {
	for i := range clus.peers {
		for s := range clus.commitC[i] {
			if s == nil {
				break
			}
		}
	}
}

// Close closes all cluster nodes and returns an error if any failed.
func (clus *cluster) Close() (err error) {
	for i := range clus.peers {
		close(clus.proposeC[i])
		for range clus.commitC[i] {
			// drain pending commits
		}
		// wait for channel to close
		if erri, _ := <-clus.errorC[i]; erri != nil {
			err = erri
		}
		// clean intermediates
		os.RemoveAll(fmt.Sprintf("raftexample-%d", i+1))
		os.RemoveAll(fmt.Sprintf("raftexample-%d-snap", i+1))
	}
	return err
}

func (clus *cluster) closeNoErrors(t *testing.T) {
	if err := clus.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestProposeOnCommit starts three nodes and feeds commits back into the proposal
// channel. The intent is to ensure blocking on a proposal won't block raft progress.
func TestProposeOnCommit(t *testing.T) {
	clus := newCluster(3)
	defer clus.closeNoErrors(t)

	clus.sinkReplay()

	donec := make(chan struct{})
	for i := range clus.peers {
		// feedback for "n" committed entries, then update donec
		go func(pC chan<- string, cC <-chan *string, eC <-chan error) {
			for n := 0; n < 100; n++ {
				s, ok := <-cC
				if !ok {
					pC = nil
				}
				select {
				case pC <- *s:
					continue
				case err, _ := <-eC:
					t.Fatalf("eC message (%v)", err)
				}
			}
			donec <- struct{}{}
			for range cC {
				// acknowledge the commits from other nodes so
				// raft continues to make progress
			}
		}(clus.proposeC[i], clus.commitC[i], clus.errorC[i])

		// one message feedback per node
		go func(i int) { clus.proposeC[i] <- "foo" }(i)
	}

	for range clus.peers {
		<-donec
	}
}

// TestCloseProposerBeforeReplay tests closing the producer before raft starts.
func TestCloseProposerBeforeReplay(t *testing.T) {
	clus := newCluster(1)
	// close before replay so raft never starts
	defer clus.closeNoErrors(t)
}

// TestCloseProposerInflight tests closing the producer while
// committed messages are being published to the client.
func TestCloseProposerInflight(t *testing.T) {
	clus := newCluster(1)
	defer clus.closeNoErrors(t)

	clus.sinkReplay()

	// some inflight ops
	go func() {
		clus.proposeC[0] <- "foo"
		clus.proposeC[0] <- "bar"
	}()

	// wait for one message
	if c, ok := <-clus.commitC[0]; *c != "foo" || !ok {
		t.Fatalf("Commit failed")
	}
}
