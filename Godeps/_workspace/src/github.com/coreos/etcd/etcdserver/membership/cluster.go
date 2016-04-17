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

package membership

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
	"sync"

	"github.com/coreos/etcd/pkg/netutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/store"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

// RaftCluster is a list of Members that belong to the same raft cluster
type RaftCluster struct {
	id    types.ID
	token string

	store store.Store
	be    backend.Backend

	sync.Mutex // guards the fields below
	version    *semver.Version
	members    map[types.ID]*Member
	// removed contains the ids of removed members in the cluster.
	// removed id cannot be reused.
	removed map[types.ID]bool
}

func NewClusterFromURLsMap(token string, urlsmap types.URLsMap) (*RaftCluster, error) {
	c := NewCluster(token)
	for name, urls := range urlsmap {
		m := NewMember(name, urls, token, nil)
		if _, ok := c.members[m.ID]; ok {
			return nil, fmt.Errorf("member exists with identical ID %v", m)
		}
		if uint64(m.ID) == raft.None {
			return nil, fmt.Errorf("cannot use %x as member id", raft.None)
		}
		c.members[m.ID] = m
	}
	c.genID()
	return c, nil
}

func NewClusterFromMembers(token string, id types.ID, membs []*Member) *RaftCluster {
	c := NewCluster(token)
	c.id = id
	for _, m := range membs {
		c.members[m.ID] = m
	}
	return c
}

func NewCluster(token string) *RaftCluster {
	return &RaftCluster{
		token:   token,
		members: make(map[types.ID]*Member),
		removed: make(map[types.ID]bool),
	}
}

func (c *RaftCluster) ID() types.ID { return c.id }

func (c *RaftCluster) Members() []*Member {
	c.Lock()
	defer c.Unlock()
	var ms MembersByID
	for _, m := range c.members {
		ms = append(ms, m.Clone())
	}
	sort.Sort(ms)
	return []*Member(ms)
}

func (c *RaftCluster) Member(id types.ID) *Member {
	c.Lock()
	defer c.Unlock()
	return c.members[id].Clone()
}

// MemberByName returns a Member with the given name if exists.
// If more than one member has the given name, it will panic.
func (c *RaftCluster) MemberByName(name string) *Member {
	c.Lock()
	defer c.Unlock()
	var memb *Member
	for _, m := range c.members {
		if m.Name == name {
			if memb != nil {
				plog.Panicf("two members with the given name %q exist", name)
			}
			memb = m
		}
	}
	return memb.Clone()
}

func (c *RaftCluster) MemberIDs() []types.ID {
	c.Lock()
	defer c.Unlock()
	var ids []types.ID
	for _, m := range c.members {
		ids = append(ids, m.ID)
	}
	sort.Sort(types.IDSlice(ids))
	return ids
}

func (c *RaftCluster) IsIDRemoved(id types.ID) bool {
	c.Lock()
	defer c.Unlock()
	return c.removed[id]
}

// PeerURLs returns a list of all peer addresses.
// The returned list is sorted in ascending lexicographical order.
func (c *RaftCluster) PeerURLs() []string {
	c.Lock()
	defer c.Unlock()
	urls := make([]string, 0)
	for _, p := range c.members {
		for _, addr := range p.PeerURLs {
			urls = append(urls, addr)
		}
	}
	sort.Strings(urls)
	return urls
}

// ClientURLs returns a list of all client addresses.
// The returned list is sorted in ascending lexicographical order.
func (c *RaftCluster) ClientURLs() []string {
	c.Lock()
	defer c.Unlock()
	urls := make([]string, 0)
	for _, p := range c.members {
		for _, url := range p.ClientURLs {
			urls = append(urls, url)
		}
	}
	sort.Strings(urls)
	return urls
}

func (c *RaftCluster) String() string {
	c.Lock()
	defer c.Unlock()
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "{ClusterID:%s ", c.id)
	var ms []string
	for _, m := range c.members {
		ms = append(ms, fmt.Sprintf("%+v", m))
	}
	fmt.Fprintf(b, "Members:[%s] ", strings.Join(ms, " "))
	var ids []string
	for id := range c.removed {
		ids = append(ids, fmt.Sprintf("%s", id))
	}
	fmt.Fprintf(b, "RemovedMemberIDs:[%s]}", strings.Join(ids, " "))
	return b.String()
}

func (c *RaftCluster) genID() {
	mIDs := c.MemberIDs()
	b := make([]byte, 8*len(mIDs))
	for i, id := range mIDs {
		binary.BigEndian.PutUint64(b[8*i:], uint64(id))
	}
	hash := sha1.Sum(b)
	c.id = types.ID(binary.BigEndian.Uint64(hash[:8]))
}

func (c *RaftCluster) SetID(id types.ID) { c.id = id }

func (c *RaftCluster) SetStore(st store.Store) { c.store = st }

func (c *RaftCluster) Recover() {
	c.Lock()
	defer c.Unlock()

	c.members, c.removed = membersFromStore(c.store)
	c.version = clusterVersionFromStore(c.store)
	mustDetectDowngrade(c.version)

	for _, m := range c.members {
		plog.Infof("added member %s %v to cluster %s from store", m.ID, m.PeerURLs, c.id)
	}
	if c.version != nil {
		plog.Infof("set the cluster version to %v from store", version.Cluster(c.version.String()))
	}
}

// ValidateConfigurationChange takes a proposed ConfChange and
// ensures that it is still valid.
func (c *RaftCluster) ValidateConfigurationChange(cc raftpb.ConfChange) error {
	members, removed := membersFromStore(c.store)
	id := types.ID(cc.NodeID)
	if removed[id] {
		return ErrIDRemoved
	}
	switch cc.Type {
	case raftpb.ConfChangeAddNode:
		if members[id] != nil {
			return ErrIDExists
		}
		urls := make(map[string]bool)
		for _, m := range members {
			for _, u := range m.PeerURLs {
				urls[u] = true
			}
		}
		m := new(Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			plog.Panicf("unmarshal member should never fail: %v", err)
		}
		for _, u := range m.PeerURLs {
			if urls[u] {
				return ErrPeerURLexists
			}
		}
	case raftpb.ConfChangeRemoveNode:
		if members[id] == nil {
			return ErrIDNotFound
		}
	case raftpb.ConfChangeUpdateNode:
		if members[id] == nil {
			return ErrIDNotFound
		}
		urls := make(map[string]bool)
		for _, m := range members {
			if m.ID == id {
				continue
			}
			for _, u := range m.PeerURLs {
				urls[u] = true
			}
		}
		m := new(Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			plog.Panicf("unmarshal member should never fail: %v", err)
		}
		for _, u := range m.PeerURLs {
			if urls[u] {
				return ErrPeerURLexists
			}
		}
	default:
		plog.Panicf("ConfChange type should be either AddNode, RemoveNode or UpdateNode")
	}
	return nil
}

// AddMember adds a new Member into the cluster, and saves the given member's
// raftAttributes into the store. The given member should have empty attributes.
// A Member with a matching id must not exist.
func (c *RaftCluster) AddMember(m *Member) {
	c.Lock()
	defer c.Unlock()
	if c.store != nil {
		mustSaveMemberToStore(c.store, m)
	}
	if c.be != nil {
		mustSaveMemberToBackend(c.be, m)
	}

	c.members[m.ID] = m
}

// RemoveMember removes a member from the store.
// The given id MUST exist, or the function panics.
func (c *RaftCluster) RemoveMember(id types.ID) {
	c.Lock()
	defer c.Unlock()
	if c.store != nil {
		mustDeleteMemberFromStore(c.store, id)
	}
	if c.be != nil {
		mustDeleteMemberFromBackend(c.be, id)
	}

	delete(c.members, id)
	c.removed[id] = true
}

func (c *RaftCluster) UpdateAttributes(id types.ID, attr Attributes) bool {
	c.Lock()
	defer c.Unlock()
	if m, ok := c.members[id]; ok {
		m.Attributes = attr
		return true
	}
	_, ok := c.removed[id]
	if ok {
		plog.Warningf("skipped updating attributes of removed member %s", id)
	} else {
		plog.Panicf("error updating attributes of unknown member %s", id)
	}
	// TODO: update store in this function
	return false
}

func (c *RaftCluster) UpdateRaftAttributes(id types.ID, raftAttr RaftAttributes) {
	c.Lock()
	defer c.Unlock()

	c.members[id].RaftAttributes = raftAttr
	if c.store != nil {
		mustUpdateMemberInStore(c.store, c.members[id])
	}
	if c.be != nil {
		mustSaveMemberToBackend(c.be, c.members[id])
	}
}

func (c *RaftCluster) Version() *semver.Version {
	c.Lock()
	defer c.Unlock()
	if c.version == nil {
		return nil
	}
	return semver.Must(semver.NewVersion(c.version.String()))
}

func (c *RaftCluster) SetVersion(ver *semver.Version) {
	c.Lock()
	defer c.Unlock()
	if c.version != nil {
		plog.Noticef("updated the cluster version from %v to %v", version.Cluster(c.version.String()), version.Cluster(ver.String()))
	} else {
		plog.Noticef("set the initial cluster version to %v", version.Cluster(ver.String()))
	}
	c.version = ver
	mustDetectDowngrade(c.version)
}

func (c *RaftCluster) IsReadyToAddNewMember() bool {
	nmembers := 1
	nstarted := 0

	for _, member := range c.members {
		if member.IsStarted() {
			nstarted++
		}
		nmembers++
	}

	if nstarted == 1 && nmembers == 2 {
		// a case of adding a new node to 1-member cluster for restoring cluster data
		// https://github.com/coreos/etcd/blob/master/Documentation/admin_guide.md#restoring-the-cluster

		plog.Debugf("The number of started member is 1. This cluster can accept add member request.")
		return true
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		plog.Warningf("Reject add member request: the number of started member (%d) will be less than the quorum number of the cluster (%d)", nstarted, nquorum)
		return false
	}

	return true
}

func (c *RaftCluster) IsReadyToRemoveMember(id uint64) bool {
	nmembers := 0
	nstarted := 0

	for _, member := range c.members {
		if uint64(member.ID) == id {
			continue
		}

		if member.IsStarted() {
			nstarted++
		}
		nmembers++
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		plog.Warningf("Reject remove member request: the number of started member (%d) will be less than the quorum number of the cluster (%d)", nstarted, nquorum)
		return false
	}

	return true
}

func membersFromStore(st store.Store) (map[types.ID]*Member, map[types.ID]bool) {
	members := make(map[types.ID]*Member)
	removed := make(map[types.ID]bool)
	e, err := st.Get(StoreMembersPrefix, true, true)
	if err != nil {
		if isKeyNotFound(err) {
			return members, removed
		}
		plog.Panicf("get storeMembers should never fail: %v", err)
	}
	for _, n := range e.Node.Nodes {
		var m *Member
		m, err = nodeToMember(n)
		if err != nil {
			plog.Panicf("nodeToMember should never fail: %v", err)
		}
		members[m.ID] = m
	}

	e, err = st.Get(storeRemovedMembersPrefix, true, true)
	if err != nil {
		if isKeyNotFound(err) {
			return members, removed
		}
		plog.Panicf("get storeRemovedMembers should never fail: %v", err)
	}
	for _, n := range e.Node.Nodes {
		removed[MustParseMemberIDFromKey(n.Key)] = true
	}
	return members, removed
}

func clusterVersionFromStore(st store.Store) *semver.Version {
	e, err := st.Get(path.Join(storePrefix, "version"), false, false)
	if err != nil {
		if isKeyNotFound(err) {
			return nil
		}
		plog.Panicf("unexpected error (%v) when getting cluster version from store", err)
	}
	return semver.Must(semver.NewVersion(*e.Node.Value))
}

// ValidateClusterAndAssignIDs validates the local cluster by matching the PeerURLs
// with the existing cluster. If the validation succeeds, it assigns the IDs
// from the existing cluster to the local cluster.
// If the validation fails, an error will be returned.
func ValidateClusterAndAssignIDs(local *RaftCluster, existing *RaftCluster) error {
	ems := existing.Members()
	lms := local.Members()
	if len(ems) != len(lms) {
		return fmt.Errorf("member count is unequal")
	}
	sort.Sort(MembersByPeerURLs(ems))
	sort.Sort(MembersByPeerURLs(lms))

	for i := range ems {
		if !netutil.URLStringsEqual(ems[i].PeerURLs, lms[i].PeerURLs) {
			return fmt.Errorf("unmatched member while checking PeerURLs")
		}
		lms[i].ID = ems[i].ID
	}
	local.members = make(map[types.ID]*Member)
	for _, m := range lms {
		local.members[m.ID] = m
	}
	return nil
}

func mustDetectDowngrade(cv *semver.Version) {
	lv := semver.Must(semver.NewVersion(version.Version))
	// only keep major.minor version for comparison against cluster version
	lv = &semver.Version{Major: lv.Major, Minor: lv.Minor}
	if cv != nil && lv.LessThan(*cv) {
		plog.Fatalf("cluster cannot be downgraded (current version: %s is lower than determined cluster version: %s).", version.Version, version.Cluster(cv.String()))
	}
}
