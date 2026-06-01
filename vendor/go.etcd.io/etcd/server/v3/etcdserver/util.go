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

package etcdserver

import (
	"time"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/etcdserver/api/rafthttp"
)

// isConnectedToQuorumSince reports whether the local member has been connected
// to a quorum of the current cluster continuously since the given time.
func isConnectedToQuorumSince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) bool {
	return numConnectedSince(transport, since, self, members) >= quorum(len(members))
}

// isConnectedToQuorumAfterAddingNewMemberSince reports whether the local member
// has been connected to a quorum continuously since the given time, assuming a
// new member is being added to the cluster.
//
// For a single-member cluster, it always returns true to allow membership
// expansion.
func isConnectedToQuorumAfterAddingNewMemberSince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) bool {
	if len(members) == 1 {
		// If it's a single member cluster, we should allow adding a new member
		return true
	}
	return numConnectedSince(transport, since, self, members) >= quorum(len(members)+1)
}

func quorum(num int) int {
	return num/2 + 1
}

// isConnectedSince checks whether the local member is connected to the
// remote member since the given time.
func isConnectedSince(transport rafthttp.Transporter, since time.Time, remote types.ID) bool {
	t := transport.ActiveSince(remote)
	return !t.IsZero() && t.Before(since)
}

// exceedsRequestLimit checks if the committed index is too far ahead of the applied index.
// LeaseRevoke requests are prioritized to ensure timely lease expiration,
// which helps mitigate pressure on the cluster.
func exceedsRequestLimit(appliedIndex, committedIndex uint64, r *pb.InternalRaftRequest, enablePriority bool) bool {
	if committedIndex <= appliedIndex+maxNormalGap {
		return false
	}
	if enablePriority && isPriorityRequest(r) {
		if committedIndex <= appliedIndex+maxPriorityGap {
			return false
		}
	}
	return true
}

func isPriorityRequest(r *pb.InternalRaftRequest) bool {
	return r != nil && r.LeaseRevoke != nil
}

// numConnectedSince counts how many members are connected to the local member
// since the given time.
func numConnectedSince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) int {
	connectedNum := 0
	for _, m := range members {
		if m.ID == self || isConnectedSince(transport, since, m.ID) {
			connectedNum++
		}
	}
	return connectedNum
}

// longestConnected chooses the member with longest active-since-time.
// It returns false, if nothing is active.
func longestConnected(tp rafthttp.Transporter, membs []types.ID) (types.ID, bool) {
	var longest types.ID
	var oldest time.Time
	for _, id := range membs {
		tm := tp.ActiveSince(id)
		if tm.IsZero() { // inactive
			continue
		}

		if oldest.IsZero() { // first longest candidate
			oldest = tm
			longest = id
		}

		if tm.Before(oldest) {
			oldest = tm
			longest = id
		}
	}
	if uint64(longest) == 0 {
		return longest, false
	}
	return longest, true
}
