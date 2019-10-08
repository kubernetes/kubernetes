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

package membership

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"go.etcd.io/etcd/etcdserver/api/v2store"
	"go.etcd.io/etcd/mvcc/backend"
	"go.etcd.io/etcd/pkg/netutil"
	"go.etcd.io/etcd/pkg/types"
	"go.etcd.io/etcd/raft"
	"go.etcd.io/etcd/raft/raftpb"
	"go.etcd.io/etcd/version"

	"github.com/coreos/go-semver/semver"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

const maxLearners = 1

// RaftCluster is a list of Members that belong to the same raft cluster
type RaftCluster struct {
	lg *zap.Logger

	localID types.ID
	cid     types.ID
	token   string

	v2store v2store.Store
	be      backend.Backend

	sync.Mutex // guards the fields below
	version    *semver.Version
	members    map[types.ID]*Member
	// removed contains the ids of removed members in the cluster.
	// removed id cannot be reused.
	removed map[types.ID]bool
}

// ConfigChangeContext represents a context for confChange.
type ConfigChangeContext struct {
	Member
	// IsPromote indicates if the config change is for promoting a learner member.
	// This flag is needed because both adding a new member and promoting a learner member
	// uses the same config change type 'ConfChangeAddNode'.
	IsPromote bool `json:"isPromote"`
}

// NewClusterFromURLsMap creates a new raft cluster using provided urls map. Currently, it does not support creating
// cluster with raft learner member.
func NewClusterFromURLsMap(lg *zap.Logger, token string, urlsmap types.URLsMap) (*RaftCluster, error) {
	c := NewCluster(lg, token)
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

func NewClusterFromMembers(lg *zap.Logger, token string, id types.ID, membs []*Member) *RaftCluster {
	c := NewCluster(lg, token)
	c.cid = id
	for _, m := range membs {
		c.members[m.ID] = m
	}
	return c
}

func NewCluster(lg *zap.Logger, token string) *RaftCluster {
	return &RaftCluster{
		lg:      lg,
		token:   token,
		members: make(map[types.ID]*Member),
		removed: make(map[types.ID]bool),
	}
}

func (c *RaftCluster) ID() types.ID { return c.cid }

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

func (c *RaftCluster) VotingMembers() []*Member {
	c.Lock()
	defer c.Unlock()
	var ms MembersByID
	for _, m := range c.members {
		if !m.IsLearner {
			ms = append(ms, m.Clone())
		}
	}
	sort.Sort(ms)
	return []*Member(ms)
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
				if c.lg != nil {
					c.lg.Panic("two member with same name found", zap.String("name", name))
				} else {
					plog.Panicf("two members with the given name %q exist", name)
				}
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
		urls = append(urls, p.PeerURLs...)
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
		urls = append(urls, p.ClientURLs...)
	}
	sort.Strings(urls)
	return urls
}

func (c *RaftCluster) String() string {
	c.Lock()
	defer c.Unlock()
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "{ClusterID:%s ", c.cid)
	var ms []string
	for _, m := range c.members {
		ms = append(ms, fmt.Sprintf("%+v", m))
	}
	fmt.Fprintf(b, "Members:[%s] ", strings.Join(ms, " "))
	var ids []string
	for id := range c.removed {
		ids = append(ids, id.String())
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
	c.cid = types.ID(binary.BigEndian.Uint64(hash[:8]))
}

func (c *RaftCluster) SetID(localID, cid types.ID) {
	c.localID = localID
	c.cid = cid
}

func (c *RaftCluster) SetStore(st v2store.Store) { c.v2store = st }

func (c *RaftCluster) SetBackend(be backend.Backend) {
	c.be = be
	mustCreateBackendBuckets(c.be)
}

func (c *RaftCluster) Recover(onSet func(*zap.Logger, *semver.Version)) {
	c.Lock()
	defer c.Unlock()

	c.members, c.removed = membersFromStore(c.lg, c.v2store)
	c.version = clusterVersionFromStore(c.lg, c.v2store)
	mustDetectDowngrade(c.lg, c.version)
	onSet(c.lg, c.version)

	for _, m := range c.members {
		if c.lg != nil {
			c.lg.Info(
				"recovered/added member from store",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("recovered-remote-peer-id", m.ID.String()),
				zap.Strings("recovered-remote-peer-urls", m.PeerURLs),
			)
		} else {
			plog.Infof("added member %s %v to cluster %s from store", m.ID, m.PeerURLs, c.cid)
		}
	}
	if c.version != nil {
		if c.lg != nil {
			c.lg.Info(
				"set cluster version from store",
				zap.String("cluster-version", version.Cluster(c.version.String())),
			)
		} else {
			plog.Infof("set the cluster version to %v from store", version.Cluster(c.version.String()))
		}
	}
}

// ValidateConfigurationChange takes a proposed ConfChange and
// ensures that it is still valid.
func (c *RaftCluster) ValidateConfigurationChange(cc raftpb.ConfChange) error {
	members, removed := membersFromStore(c.lg, c.v2store)
	id := types.ID(cc.NodeID)
	if removed[id] {
		return ErrIDRemoved
	}
	switch cc.Type {
	case raftpb.ConfChangeAddNode, raftpb.ConfChangeAddLearnerNode:
		confChangeContext := new(ConfigChangeContext)
		if err := json.Unmarshal(cc.Context, confChangeContext); err != nil {
			if c.lg != nil {
				c.lg.Panic("failed to unmarshal confChangeContext", zap.Error(err))
			} else {
				plog.Panicf("unmarshal confChangeContext should never fail: %v", err)
			}
		}

		if confChangeContext.IsPromote { // promoting a learner member to voting member
			if members[id] == nil {
				return ErrIDNotFound
			}
			if !members[id].IsLearner {
				return ErrMemberNotLearner
			}
		} else { // adding a new member
			if members[id] != nil {
				return ErrIDExists
			}

			urls := make(map[string]bool)
			for _, m := range members {
				for _, u := range m.PeerURLs {
					urls[u] = true
				}
			}
			for _, u := range confChangeContext.Member.PeerURLs {
				if urls[u] {
					return ErrPeerURLexists
				}
			}

			if confChangeContext.Member.IsLearner { // the new member is a learner
				numLearners := 0
				for _, m := range members {
					if m.IsLearner {
						numLearners++
					}
				}
				if numLearners+1 > maxLearners {
					return ErrTooManyLearners
				}
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
			if c.lg != nil {
				c.lg.Panic("failed to unmarshal member", zap.Error(err))
			} else {
				plog.Panicf("unmarshal member should never fail: %v", err)
			}
		}
		for _, u := range m.PeerURLs {
			if urls[u] {
				return ErrPeerURLexists
			}
		}

	default:
		if c.lg != nil {
			c.lg.Panic("unknown ConfChange type", zap.String("type", cc.Type.String()))
		} else {
			plog.Panicf("ConfChange type should be either AddNode, RemoveNode or UpdateNode")
		}
	}
	return nil
}

// AddMember adds a new Member into the cluster, and saves the given member's
// raftAttributes into the store. The given member should have empty attributes.
// A Member with a matching id must not exist.
func (c *RaftCluster) AddMember(m *Member) {
	c.Lock()
	defer c.Unlock()
	if c.v2store != nil {
		mustSaveMemberToStore(c.v2store, m)
	}
	if c.be != nil {
		mustSaveMemberToBackend(c.be, m)
	}

	c.members[m.ID] = m

	if c.lg != nil {
		c.lg.Info(
			"added member",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("added-peer-id", m.ID.String()),
			zap.Strings("added-peer-peer-urls", m.PeerURLs),
		)
	} else {
		plog.Infof("added member %s %v to cluster %s", m.ID, m.PeerURLs, c.cid)
	}
}

// RemoveMember removes a member from the store.
// The given id MUST exist, or the function panics.
func (c *RaftCluster) RemoveMember(id types.ID) {
	c.Lock()
	defer c.Unlock()
	if c.v2store != nil {
		mustDeleteMemberFromStore(c.v2store, id)
	}
	if c.be != nil {
		mustDeleteMemberFromBackend(c.be, id)
	}

	m, ok := c.members[id]
	delete(c.members, id)
	c.removed[id] = true

	if c.lg != nil {
		if ok {
			c.lg.Info(
				"removed member",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("removed-remote-peer-id", id.String()),
				zap.Strings("removed-remote-peer-urls", m.PeerURLs),
			)
		} else {
			c.lg.Warn(
				"skipped removing already removed member",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("removed-remote-peer-id", id.String()),
			)
		}
	} else {
		plog.Infof("removed member %s from cluster %s", id, c.cid)
	}
}

func (c *RaftCluster) UpdateAttributes(id types.ID, attr Attributes) {
	c.Lock()
	defer c.Unlock()

	if m, ok := c.members[id]; ok {
		m.Attributes = attr
		if c.v2store != nil {
			mustUpdateMemberAttrInStore(c.v2store, m)
		}
		if c.be != nil {
			mustSaveMemberToBackend(c.be, m)
		}
		return
	}

	_, ok := c.removed[id]
	if !ok {
		if c.lg != nil {
			c.lg.Panic(
				"failed to update; member unknown",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("unknown-remote-peer-id", id.String()),
			)
		} else {
			plog.Panicf("error updating attributes of unknown member %s", id)
		}
	}

	if c.lg != nil {
		c.lg.Warn(
			"skipped attributes update of removed member",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("updated-peer-id", id.String()),
		)
	} else {
		plog.Warningf("skipped updating attributes of removed member %s", id)
	}
}

// PromoteMember marks the member's IsLearner RaftAttributes to false.
func (c *RaftCluster) PromoteMember(id types.ID) {
	c.Lock()
	defer c.Unlock()

	c.members[id].RaftAttributes.IsLearner = false
	if c.v2store != nil {
		mustUpdateMemberInStore(c.v2store, c.members[id])
	}
	if c.be != nil {
		mustSaveMemberToBackend(c.be, c.members[id])
	}

	if c.lg != nil {
		c.lg.Info(
			"promote member",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
		)
	} else {
		plog.Noticef("promote member %s in cluster %s", id, c.cid)
	}
}

func (c *RaftCluster) UpdateRaftAttributes(id types.ID, raftAttr RaftAttributes) {
	c.Lock()
	defer c.Unlock()

	c.members[id].RaftAttributes = raftAttr
	if c.v2store != nil {
		mustUpdateMemberInStore(c.v2store, c.members[id])
	}
	if c.be != nil {
		mustSaveMemberToBackend(c.be, c.members[id])
	}

	if c.lg != nil {
		c.lg.Info(
			"updated member",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("updated-remote-peer-id", id.String()),
			zap.Strings("updated-remote-peer-urls", raftAttr.PeerURLs),
		)
	} else {
		plog.Noticef("updated member %s %v in cluster %s", id, raftAttr.PeerURLs, c.cid)
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

func (c *RaftCluster) SetVersion(ver *semver.Version, onSet func(*zap.Logger, *semver.Version)) {
	c.Lock()
	defer c.Unlock()
	if c.version != nil {
		if c.lg != nil {
			c.lg.Info(
				"updated cluster version",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("from", version.Cluster(c.version.String())),
				zap.String("from", version.Cluster(ver.String())),
			)
		} else {
			plog.Noticef("updated the cluster version from %v to %v", version.Cluster(c.version.String()), version.Cluster(ver.String()))
		}
	} else {
		if c.lg != nil {
			c.lg.Info(
				"set initial cluster version",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
				zap.String("cluster-version", version.Cluster(ver.String())),
			)
		} else {
			plog.Noticef("set the initial cluster version to %v", version.Cluster(ver.String()))
		}
	}
	c.version = ver
	mustDetectDowngrade(c.lg, c.version)
	if c.v2store != nil {
		mustSaveClusterVersionToStore(c.v2store, ver)
	}
	if c.be != nil {
		mustSaveClusterVersionToBackend(c.be, ver)
	}
	ClusterVersionMetrics.With(prometheus.Labels{"cluster_version": ver.String()}).Set(1)
	onSet(c.lg, ver)
}

func (c *RaftCluster) IsReadyToAddVotingMember() bool {
	nmembers := 1
	nstarted := 0

	for _, member := range c.VotingMembers() {
		if member.IsStarted() {
			nstarted++
		}
		nmembers++
	}

	if nstarted == 1 && nmembers == 2 {
		// a case of adding a new node to 1-member cluster for restoring cluster data
		// https://github.com/etcd-io/etcd/blob/master/Documentation/v2/admin_guide.md#restoring-the-cluster
		if c.lg != nil {
			c.lg.Debug("number of started member is 1; can accept add member request")
		} else {
			plog.Debugf("The number of started member is 1. This cluster can accept add member request.")
		}
		return true
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		if c.lg != nil {
			c.lg.Warn(
				"rejecting member add; started member will be less than quorum",
				zap.Int("number-of-started-member", nstarted),
				zap.Int("quorum", nquorum),
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
			)
		} else {
			plog.Warningf("Reject add member request: the number of started member (%d) will be less than the quorum number of the cluster (%d)", nstarted, nquorum)
		}
		return false
	}

	return true
}

func (c *RaftCluster) IsReadyToRemoveVotingMember(id uint64) bool {
	nmembers := 0
	nstarted := 0

	for _, member := range c.VotingMembers() {
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
		if c.lg != nil {
			c.lg.Warn(
				"rejecting member remove; started member will be less than quorum",
				zap.Int("number-of-started-member", nstarted),
				zap.Int("quorum", nquorum),
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
			)
		} else {
			plog.Warningf("Reject remove member request: the number of started member (%d) will be less than the quorum number of the cluster (%d)", nstarted, nquorum)
		}
		return false
	}

	return true
}

func (c *RaftCluster) IsReadyToPromoteMember(id uint64) bool {
	nmembers := 1
	nstarted := 0

	for _, member := range c.VotingMembers() {
		if member.IsStarted() {
			nstarted++
		}
		nmembers++
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		if c.lg != nil {
			c.lg.Warn(
				"rejecting member promote; started member will be less than quorum",
				zap.Int("number-of-started-member", nstarted),
				zap.Int("quorum", nquorum),
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
			)
		} else {
			plog.Warningf("Reject promote member request: the number of started member (%d) will be less than the quorum number of the cluster (%d)", nstarted, nquorum)
		}
		return false
	}

	return true
}

func membersFromStore(lg *zap.Logger, st v2store.Store) (map[types.ID]*Member, map[types.ID]bool) {
	members := make(map[types.ID]*Member)
	removed := make(map[types.ID]bool)
	e, err := st.Get(StoreMembersPrefix, true, true)
	if err != nil {
		if isKeyNotFound(err) {
			return members, removed
		}
		if lg != nil {
			lg.Panic("failed to get members from store", zap.String("path", StoreMembersPrefix), zap.Error(err))
		} else {
			plog.Panicf("get storeMembers should never fail: %v", err)
		}
	}
	for _, n := range e.Node.Nodes {
		var m *Member
		m, err = nodeToMember(n)
		if err != nil {
			if lg != nil {
				lg.Panic("failed to nodeToMember", zap.Error(err))
			} else {
				plog.Panicf("nodeToMember should never fail: %v", err)
			}
		}
		members[m.ID] = m
	}

	e, err = st.Get(storeRemovedMembersPrefix, true, true)
	if err != nil {
		if isKeyNotFound(err) {
			return members, removed
		}
		if lg != nil {
			lg.Panic(
				"failed to get removed members from store",
				zap.String("path", storeRemovedMembersPrefix),
				zap.Error(err),
			)
		} else {
			plog.Panicf("get storeRemovedMembers should never fail: %v", err)
		}
	}
	for _, n := range e.Node.Nodes {
		removed[MustParseMemberIDFromKey(n.Key)] = true
	}
	return members, removed
}

func clusterVersionFromStore(lg *zap.Logger, st v2store.Store) *semver.Version {
	e, err := st.Get(path.Join(storePrefix, "version"), false, false)
	if err != nil {
		if isKeyNotFound(err) {
			return nil
		}
		if lg != nil {
			lg.Panic(
				"failed to get cluster version from store",
				zap.String("path", path.Join(storePrefix, "version")),
				zap.Error(err),
			)
		} else {
			plog.Panicf("unexpected error (%v) when getting cluster version from store", err)
		}
	}
	return semver.Must(semver.NewVersion(*e.Node.Value))
}

// ValidateClusterAndAssignIDs validates the local cluster by matching the PeerURLs
// with the existing cluster. If the validation succeeds, it assigns the IDs
// from the existing cluster to the local cluster.
// If the validation fails, an error will be returned.
func ValidateClusterAndAssignIDs(lg *zap.Logger, local *RaftCluster, existing *RaftCluster) error {
	ems := existing.Members()
	lms := local.Members()
	if len(ems) != len(lms) {
		return fmt.Errorf("member count is unequal")
	}
	sort.Sort(MembersByPeerURLs(ems))
	sort.Sort(MembersByPeerURLs(lms))

	ctx, cancel := context.WithTimeout(context.TODO(), 30*time.Second)
	defer cancel()
	for i := range ems {
		if ok, err := netutil.URLStringsEqual(ctx, lg, ems[i].PeerURLs, lms[i].PeerURLs); !ok {
			return fmt.Errorf("unmatched member while checking PeerURLs (%v)", err)
		}
		lms[i].ID = ems[i].ID
	}
	local.members = make(map[types.ID]*Member)
	for _, m := range lms {
		local.members[m.ID] = m
	}
	return nil
}

func mustDetectDowngrade(lg *zap.Logger, cv *semver.Version) {
	lv := semver.Must(semver.NewVersion(version.Version))
	// only keep major.minor version for comparison against cluster version
	lv = &semver.Version{Major: lv.Major, Minor: lv.Minor}
	if cv != nil && lv.LessThan(*cv) {
		if lg != nil {
			lg.Fatal(
				"invalid downgrade; server version is lower than determined cluster version",
				zap.String("current-server-version", version.Version),
				zap.String("determined-cluster-version", version.Cluster(cv.String())),
			)
		} else {
			plog.Fatalf("cluster cannot be downgraded (current version: %s is lower than determined cluster version: %s).", version.Version, version.Cluster(cv.String()))
		}
	}
}

// IsLocalMemberLearner returns if the local member is raft learner
func (c *RaftCluster) IsLocalMemberLearner() bool {
	c.Lock()
	defer c.Unlock()
	localMember, ok := c.members[c.localID]
	if !ok {
		if c.lg != nil {
			c.lg.Panic(
				"failed to find local ID in cluster members",
				zap.String("cluster-id", c.cid.String()),
				zap.String("local-member-id", c.localID.String()),
			)
		} else {
			plog.Panicf("failed to find local ID %s in cluster %s", c.localID.String(), c.cid.String())
		}
	}
	return localMember.IsLearner
}

// IsMemberExist returns if the member with the given id exists in cluster.
func (c *RaftCluster) IsMemberExist(id types.ID) bool {
	c.Lock()
	defer c.Unlock()
	_, ok := c.members[id]
	return ok
}

// VotingMemberIDs returns the ID of voting members in cluster.
func (c *RaftCluster) VotingMemberIDs() []types.ID {
	c.Lock()
	defer c.Unlock()
	var ids []types.ID
	for _, m := range c.members {
		if !m.IsLearner {
			ids = append(ids, m.ID)
		}
	}
	sort.Sort(types.IDSlice(ids))
	return ids
}
