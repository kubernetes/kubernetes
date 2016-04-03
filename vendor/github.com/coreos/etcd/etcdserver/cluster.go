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

package etcdserver

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
	"github.com/coreos/etcd/store"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

const (
	raftAttributesSuffix = "raftAttributes"
	attributesSuffix     = "attributes"
)

type Cluster interface {
	// ID returns the cluster ID
	ID() types.ID
	// ClientURLs returns an aggregate set of all URLs on which this
	// cluster is listening for client requests
	ClientURLs() []string
	// Members returns a slice of members sorted by their ID
	Members() []*Member
	// Member retrieves a particular member based on ID, or nil if the
	// member does not exist in the cluster
	Member(id types.ID) *Member
	// IsIDRemoved checks whether the given ID has been removed from this
	// cluster at some point in the past
	IsIDRemoved(id types.ID) bool
	// Version is the cluster-wide minimum major.minor version.
	Version() *semver.Version
}

// Cluster is a list of Members that belong to the same raft cluster
type cluster struct {
	id    types.ID
	token string
	store store.Store

	sync.Mutex // guards the fields below
	version    *semver.Version
	members    map[types.ID]*Member
	// removed contains the ids of removed members in the cluster.
	// removed id cannot be reused.
	removed map[types.ID]bool
}

func newClusterFromURLsMap(token string, urlsmap types.URLsMap) (*cluster, error) {
	c := newCluster(token)
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

func newClusterFromMembers(token string, id types.ID, membs []*Member) *cluster {
	c := newCluster(token)
	c.id = id
	for _, m := range membs {
		c.members[m.ID] = m
	}
	return c
}

func newCluster(token string) *cluster {
	return &cluster{
		token:   token,
		members: make(map[types.ID]*Member),
		removed: make(map[types.ID]bool),
	}
}

func (c *cluster) ID() types.ID { return c.id }

func (c *cluster) Members() []*Member {
	c.Lock()
	defer c.Unlock()
	var ms MembersByID
	for _, m := range c.members {
		ms = append(ms, m.Clone())
	}
	sort.Sort(ms)
	return []*Member(ms)
}

func (c *cluster) Member(id types.ID) *Member {
	c.Lock()
	defer c.Unlock()
	return c.members[id].Clone()
}

// MemberByName returns a Member with the given name if exists.
// If more than one member has the given name, it will panic.
func (c *cluster) MemberByName(name string) *Member {
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

func (c *cluster) MemberIDs() []types.ID {
	c.Lock()
	defer c.Unlock()
	var ids []types.ID
	for _, m := range c.members {
		ids = append(ids, m.ID)
	}
	sort.Sort(types.IDSlice(ids))
	return ids
}

func (c *cluster) IsIDRemoved(id types.ID) bool {
	c.Lock()
	defer c.Unlock()
	return c.removed[id]
}

// PeerURLs returns a list of all peer addresses.
// The returned list is sorted in ascending lexicographical order.
func (c *cluster) PeerURLs() []string {
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
func (c *cluster) ClientURLs() []string {
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

func (c *cluster) String() string {
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

func (c *cluster) genID() {
	mIDs := c.MemberIDs()
	b := make([]byte, 8*len(mIDs))
	for i, id := range mIDs {
		binary.BigEndian.PutUint64(b[8*i:], uint64(id))
	}
	hash := sha1.Sum(b)
	c.id = types.ID(binary.BigEndian.Uint64(hash[:8]))
}

func (c *cluster) SetID(id types.ID) { c.id = id }

func (c *cluster) SetStore(st store.Store) { c.store = st }

func (c *cluster) Recover() {
	c.Lock()
	defer c.Unlock()

	c.members, c.removed = membersFromStore(c.store)
	c.version = clusterVersionFromStore(c.store)
	MustDetectDowngrade(c.version)

	for _, m := range c.members {
		plog.Infof("added member %s %v to cluster %s from store", m.ID, m.PeerURLs, c.id)
	}
	if c.version != nil {
		plog.Infof("set the cluster version to %v from store", version.Cluster(c.version.String()))
	}
}

// ValidateConfigurationChange takes a proposed ConfChange and
// ensures that it is still valid.
func (c *cluster) ValidateConfigurationChange(cc raftpb.ConfChange) error {
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
func (c *cluster) AddMember(m *Member) {
	c.Lock()
	defer c.Unlock()
	b, err := json.Marshal(m.RaftAttributes)
	if err != nil {
		plog.Panicf("marshal raftAttributes should never fail: %v", err)
	}
	p := path.Join(memberStoreKey(m.ID), raftAttributesSuffix)
	if _, err := c.store.Create(p, false, string(b), false, store.TTLOptionSet{ExpireTime: store.Permanent}); err != nil {
		plog.Panicf("create raftAttributes should never fail: %v", err)
	}
	c.members[m.ID] = m
}

// RemoveMember removes a member from the store.
// The given id MUST exist, or the function panics.
func (c *cluster) RemoveMember(id types.ID) {
	c.Lock()
	defer c.Unlock()
	if _, err := c.store.Delete(memberStoreKey(id), true, true); err != nil {
		plog.Panicf("delete member should never fail: %v", err)
	}
	delete(c.members, id)
	if _, err := c.store.Create(removedMemberStoreKey(id), false, "", false, store.TTLOptionSet{ExpireTime: store.Permanent}); err != nil {
		plog.Panicf("create removedMember should never fail: %v", err)
	}
	c.removed[id] = true
}

func (c *cluster) UpdateAttributes(id types.ID, attr Attributes) bool {
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

func (c *cluster) UpdateRaftAttributes(id types.ID, raftAttr RaftAttributes) {
	c.Lock()
	defer c.Unlock()
	b, err := json.Marshal(raftAttr)
	if err != nil {
		plog.Panicf("marshal raftAttributes should never fail: %v", err)
	}
	p := path.Join(memberStoreKey(id), raftAttributesSuffix)
	if _, err := c.store.Update(p, string(b), store.TTLOptionSet{ExpireTime: store.Permanent}); err != nil {
		plog.Panicf("update raftAttributes should never fail: %v", err)
	}
	c.members[id].RaftAttributes = raftAttr
}

func (c *cluster) Version() *semver.Version {
	c.Lock()
	defer c.Unlock()
	if c.version == nil {
		return nil
	}
	return semver.Must(semver.NewVersion(c.version.String()))
}

func (c *cluster) SetVersion(ver *semver.Version) {
	c.Lock()
	defer c.Unlock()
	if c.version != nil {
		plog.Noticef("updated the cluster version from %v to %v", version.Cluster(c.version.String()), version.Cluster(ver.String()))
	} else {
		plog.Noticef("set the initial cluster version to %v", version.Cluster(ver.String()))
	}
	c.version = ver
	MustDetectDowngrade(c.version)
}

func (c *cluster) isReadyToAddNewMember() bool {
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

func (c *cluster) isReadyToRemoveMember(id uint64) bool {
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
	e, err := st.Get(storeMembersPrefix, true, true)
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
		removed[mustParseMemberIDFromKey(n.Key)] = true
	}
	return members, removed
}

func clusterVersionFromStore(st store.Store) *semver.Version {
	e, err := st.Get(path.Join(StoreClusterPrefix, "version"), false, false)
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
func ValidateClusterAndAssignIDs(local *cluster, existing *cluster) error {
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
