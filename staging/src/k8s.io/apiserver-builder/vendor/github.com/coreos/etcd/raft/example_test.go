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
	pb "github.com/coreos/etcd/raft/raftpb"
)

func applyToStore(ents []pb.Entry)    {}
func sendMessages(msgs []pb.Message)  {}
func saveStateToDisk(st pb.HardState) {}
func saveToDisk(ents []pb.Entry)      {}

func ExampleNode() {
	c := &Config{}
	n := StartNode(c, nil)
	defer n.Stop()

	// stuff to n happens in other goroutines

	// the last known state
	var prev pb.HardState
	for {
		// Ready blocks until there is new state ready.
		rd := <-n.Ready()
		if !isHardStateEqual(prev, rd.HardState) {
			saveStateToDisk(rd.HardState)
			prev = rd.HardState
		}

		saveToDisk(rd.Entries)
		go applyToStore(rd.CommittedEntries)
		sendMessages(rd.Messages)
	}
}
