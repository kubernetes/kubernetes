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

package integration

import (
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/testutil"
)

func TestNetworkPartition5MembersLeaderInMinority(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := NewClusterV3(t, &ClusterConfig{Size: 5})
	defer clus.Terminate(t)

	leadIndex := clus.WaitLeader(t)

	// minority: leader, follower / majority: follower, follower, follower
	minority := []int{leadIndex, (leadIndex + 1) % 5}
	majority := []int{(leadIndex + 2) % 5, (leadIndex + 3) % 5, (leadIndex + 4) % 5}

	minorityMembers := getMembersByIndexSlice(clus.cluster, minority)
	majorityMembers := getMembersByIndexSlice(clus.cluster, majority)

	// network partition (bi-directional)
	injectPartition(t, minorityMembers, majorityMembers)

	// minority leader must be lost
	clus.waitNoLeader(t, minorityMembers)

	// wait extra election timeout
	time.Sleep(2 * majorityMembers[0].electionTimeout())

	// new leader must be from majority
	clus.waitLeader(t, majorityMembers)

	// recover network partition (bi-directional)
	recoverPartition(t, minorityMembers, majorityMembers)

	// write to majority first
	clusterMustProgress(t, append(majorityMembers, minorityMembers...))
}

func TestNetworkPartition5MembersLeaderInMajority(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := NewClusterV3(t, &ClusterConfig{Size: 5})
	defer clus.Terminate(t)

	leadIndex := clus.WaitLeader(t)

	// majority: leader, follower, follower / minority: follower, follower
	majority := []int{leadIndex, (leadIndex + 1) % 5, (leadIndex + 2) % 5}
	minority := []int{(leadIndex + 3) % 5, (leadIndex + 4) % 5}

	majorityMembers := getMembersByIndexSlice(clus.cluster, majority)
	minorityMembers := getMembersByIndexSlice(clus.cluster, minority)

	// network partition (bi-directional)
	injectPartition(t, majorityMembers, minorityMembers)

	// minority leader must be lost
	clus.waitNoLeader(t, minorityMembers)

	// wait extra election timeout
	time.Sleep(2 * majorityMembers[0].electionTimeout())

	// leader must be hold in majority
	leadIndex2 := clus.waitLeader(t, majorityMembers)
	leadID, leadID2 := clus.Members[leadIndex].s.ID(), majorityMembers[leadIndex2].s.ID()
	if leadID != leadID2 {
		t.Fatalf("unexpected leader change from %s, got %s", leadID, leadID2)
	}

	// recover network partition (bi-directional)
	recoverPartition(t, majorityMembers, minorityMembers)

	// write to majority first
	clusterMustProgress(t, append(majorityMembers, minorityMembers...))
}

func TestNetworkPartition4Members(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := NewClusterV3(t, &ClusterConfig{Size: 4})
	defer clus.Terminate(t)

	leadIndex := clus.WaitLeader(t)

	// groupA: leader, follower / groupB: follower, follower
	groupA := []int{leadIndex, (leadIndex + 1) % 4}
	groupB := []int{(leadIndex + 2) % 4, (leadIndex + 3) % 4}

	leaderPartition := getMembersByIndexSlice(clus.cluster, groupA)
	followerPartition := getMembersByIndexSlice(clus.cluster, groupB)

	// network partition (bi-directional)
	injectPartition(t, leaderPartition, followerPartition)

	// no group has quorum, so leader must be lost in all members
	clus.WaitNoLeader(t)

	// recover network partition (bi-directional)
	recoverPartition(t, leaderPartition, followerPartition)

	// need to wait since it recovered with no leader
	clus.WaitLeader(t)

	clusterMustProgress(t, clus.Members)
}

func getMembersByIndexSlice(clus *cluster, idxs []int) []*member {
	ms := make([]*member, len(idxs))
	for i, idx := range idxs {
		ms[i] = clus.Members[idx]
	}
	return ms
}

func injectPartition(t *testing.T, src, others []*member) {
	for _, m := range src {
		m.InjectPartition(t, others)
	}
}

func recoverPartition(t *testing.T, src, others []*member) {
	for _, m := range src {
		m.RecoverPartition(t, others)
	}
}
