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

	"go.etcd.io/etcd/api/v3/version"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/pkg/v3/netutil"
	"go.etcd.io/etcd/raft/v3"
	"go.etcd.io/etcd/raft/v3/raftpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"

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

	v2store v2store.Store
	be      backend.Backend

	sync.Mutex // guards the fields below
	version    *semver.Version
	members    map[types.ID]*Member
	// removed contains the ids of removed members in the cluster.
	// removed id cannot be reused.
	removed map[types.ID]bool

	downgradeInfo *DowngradeInfo
}

// ConfigChangeContext represents a context for confChange.
type ConfigChangeContext struct {
	Member
	// IsPromote indicates if the config change is for promoting a learner member.
	// This flag is needed because both adding a new member and promoting a learner member
	// uses the same config change type 'ConfChangeAddNode'.
	IsPromote bool `json:"isPromote"`
}

type ShouldApplyV3 bool

const (
	ApplyBoth        = ShouldApplyV3(true)
	ApplyV2storeOnly = ShouldApplyV3(false)
)

// NewClusterFromURLsMap creates a new raft cluster using provided urls map. Currently, it does not support creating
// cluster with raft learner member.
func NewClusterFromURLsMap(lg *zap.Logger, token string, urlsmap types.URLsMap) (*RaftCluster, error) {
	c := NewCluster(lg)
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

func NewClusterFromMembers(lg *zap.Logger, id types.ID, membs []*Member) *RaftCluster {
	c := NewCluster(lg)
	c.cid = id
	for _, m := range membs {
		c.members[m.ID] = m
	}
	return c
}

func NewCluster(lg *zap.Logger) *RaftCluster {
	if lg == nil {
		lg = zap.NewNop()
	}
	return &RaftCluster{
		lg:            lg,
		members:       make(map[types.ID]*Member),
		removed:       make(map[types.ID]bool),
		downgradeInfo: &DowngradeInfo{Enabled: false},
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
				c.lg.Panic("two member with same name found", zap.String("name", name))
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

	if c.be != nil {
		c.version = clusterVersionFromBackend(c.lg, c.be)
		c.members, c.removed = membersFromBackend(c.lg, c.be)
	} else {
		c.version = clusterVersionFromStore(c.lg, c.v2store)
		c.members, c.removed = membersFromStore(c.lg, c.v2store)
	}

	if c.be != nil {
		c.downgradeInfo = downgradeInfoFromBackend(c.lg, c.be)
	}
	d := &DowngradeInfo{Enabled: false}
	if c.downgradeInfo != nil {
		d = &DowngradeInfo{Enabled: c.downgradeInfo.Enabled, TargetVersion: c.downgradeInfo.TargetVersion}
	}
	mustDetectDowngrade(c.lg, c.version, d)
	onSet(c.lg, c.version)

	for _, m := range c.members {
		c.lg.Info(
			"recovered/added member from store",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("recovered-remote-peer-id", m.ID.String()),
			zap.Strings("recovered-remote-peer-urls", m.PeerURLs),
		)
	}
	if c.version != nil {
		c.lg.Info(
			"set cluster version from store",
			zap.String("cluster-version", version.Cluster(c.version.String())),
		)
	}
}

// ValidateConfigurationChange takes a proposed ConfChange and
// ensures that it is still valid.
func (c *RaftCluster) ValidateConfigurationChange(cc raftpb.ConfChange) error {
	// TODO: this must be switched to backend as well.
	members, removed := membersFromStore(c.lg, c.v2store)
	id := types.ID(cc.NodeID)
	if removed[id] {
		return ErrIDRemoved
	}
	switch cc.Type {
	case raftpb.ConfChangeAddNode, raftpb.ConfChangeAddLearnerNode:
		confChangeContext := new(ConfigChangeContext)
		if err := json.Unmarshal(cc.Context, confChangeContext); err != nil {
			c.lg.Panic("failed to unmarshal confChangeContext", zap.Error(err))
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
			c.lg.Panic("failed to unmarshal member", zap.Error(err))
		}
		for _, u := range m.PeerURLs {
			if urls[u] {
				return ErrPeerURLexists
			}
		}

	default:
		c.lg.Panic("unknown ConfChange type", zap.String("type", cc.Type.String()))
	}
	return nil
}

// AddMember adds a new Member into the cluster, and saves the given member's
// raftAttributes into the store. The given member should have empty attributes.
// A Member with a matching id must not exist.
func (c *RaftCluster) AddMember(m *Member, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()
	if c.v2store != nil {
		mustSaveMemberToStore(c.lg, c.v2store, m)
	}
	if c.be != nil && shouldApplyV3 {
		mustSaveMemberToBackend(c.lg, c.be, m)
	}

	c.members[m.ID] = m

	c.lg.Info(
		"added member",
		zap.String("cluster-id", c.cid.String()),
		zap.String("local-member-id", c.localID.String()),
		zap.String("added-peer-id", m.ID.String()),
		zap.Strings("added-peer-peer-urls", m.PeerURLs),
	)
}

// RemoveMember removes a member from the store.
// The given id MUST exist, or the function panics.
func (c *RaftCluster) RemoveMember(id types.ID, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()
	if c.v2store != nil {
		mustDeleteMemberFromStore(c.lg, c.v2store, id)
	}
	if c.be != nil && shouldApplyV3 {
		mustDeleteMemberFromBackend(c.be, id)
	}

	m, ok := c.members[id]
	delete(c.members, id)
	c.removed[id] = true

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
}

func (c *RaftCluster) UpdateAttributes(id types.ID, attr Attributes, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()

	if m, ok := c.members[id]; ok {
		m.Attributes = attr
		if c.v2store != nil {
			mustUpdateMemberAttrInStore(c.lg, c.v2store, m)
		}
		if c.be != nil && shouldApplyV3 {
			mustSaveMemberToBackend(c.lg, c.be, m)
		}
		return
	}

	_, ok := c.removed[id]
	if !ok {
		c.lg.Panic(
			"failed to update; member unknown",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("unknown-remote-peer-id", id.String()),
		)
	}

	c.lg.Warn(
		"skipped attributes update of removed member",
		zap.String("cluster-id", c.cid.String()),
		zap.String("local-member-id", c.localID.String()),
		zap.String("updated-peer-id", id.String()),
	)
}

// PromoteMember marks the member's IsLearner RaftAttributes to false.
func (c *RaftCluster) PromoteMember(id types.ID, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()

	c.members[id].RaftAttributes.IsLearner = false
	if c.v2store != nil {
		mustUpdateMemberInStore(c.lg, c.v2store, c.members[id])
	}
	if c.be != nil && shouldApplyV3 {
		mustSaveMemberToBackend(c.lg, c.be, c.members[id])
	}

	c.lg.Info(
		"promote member",
		zap.String("cluster-id", c.cid.String()),
		zap.String("local-member-id", c.localID.String()),
	)
}

func (c *RaftCluster) UpdateRaftAttributes(id types.ID, raftAttr RaftAttributes, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()

	c.members[id].RaftAttributes = raftAttr
	if c.v2store != nil {
		mustUpdateMemberInStore(c.lg, c.v2store, c.members[id])
	}
	if c.be != nil && shouldApplyV3 {
		mustSaveMemberToBackend(c.lg, c.be, c.members[id])
	}

	c.lg.Info(
		"updated member",
		zap.String("cluster-id", c.cid.String()),
		zap.String("local-member-id", c.localID.String()),
		zap.String("updated-remote-peer-id", id.String()),
		zap.Strings("updated-remote-peer-urls", raftAttr.PeerURLs),
	)
}

func (c *RaftCluster) Version() *semver.Version {
	c.Lock()
	defer c.Unlock()
	if c.version == nil {
		return nil
	}
	return semver.Must(semver.NewVersion(c.version.String()))
}

func (c *RaftCluster) SetVersion(ver *semver.Version, onSet func(*zap.Logger, *semver.Version), shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()
	if c.version != nil {
		c.lg.Info(
			"updated cluster version",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("from", version.Cluster(c.version.String())),
			zap.String("to", version.Cluster(ver.String())),
		)
	} else {
		c.lg.Info(
			"set initial cluster version",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
			zap.String("cluster-version", version.Cluster(ver.String())),
		)
	}
	oldVer := c.version
	c.version = ver
	mustDetectDowngrade(c.lg, c.version, c.downgradeInfo)
	if c.v2store != nil {
		mustSaveClusterVersionToStore(c.lg, c.v2store, ver)
	}
	if c.be != nil && shouldApplyV3 {
		mustSaveClusterVersionToBackend(c.be, ver)
	}
	if oldVer != nil {
		ClusterVersionMetrics.With(prometheus.Labels{"cluster_version": version.Cluster(oldVer.String())}).Set(0)
	}
	ClusterVersionMetrics.With(prometheus.Labels{"cluster_version": version.Cluster(ver.String())}).Set(1)
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
		// https://github.com/etcd-io/website/blob/main/content/docs/v2/admin_guide.md#restoring-the-cluster
		c.lg.Debug("number of started member is 1; can accept add member request")
		return true
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		c.lg.Warn(
			"rejecting member add; started member will be less than quorum",
			zap.Int("number-of-started-member", nstarted),
			zap.Int("quorum", nquorum),
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
		)
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
		c.lg.Warn(
			"rejecting member remove; started member will be less than quorum",
			zap.Int("number-of-started-member", nstarted),
			zap.Int("quorum", nquorum),
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
		)
		return false
	}

	return true
}

func (c *RaftCluster) IsReadyToPromoteMember(id uint64) bool {
	nmembers := 1 // We count the learner to be promoted for the future quorum
	nstarted := 1 // and we also count it as started.

	for _, member := range c.VotingMembers() {
		if member.IsStarted() {
			nstarted++
		}
		nmembers++
	}

	nquorum := nmembers/2 + 1
	if nstarted < nquorum {
		c.lg.Warn(
			"rejecting member promote; started member will be less than quorum",
			zap.Int("number-of-started-member", nstarted),
			zap.Int("quorum", nquorum),
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
		)
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
		lg.Panic("failed to get members from store", zap.String("path", StoreMembersPrefix), zap.Error(err))
	}
	for _, n := range e.Node.Nodes {
		var m *Member
		m, err = nodeToMember(lg, n)
		if err != nil {
			lg.Panic("failed to nodeToMember", zap.Error(err))
		}
		members[m.ID] = m
	}

	e, err = st.Get(storeRemovedMembersPrefix, true, true)
	if err != nil {
		if isKeyNotFound(err) {
			return members, removed
		}
		lg.Panic(
			"failed to get removed members from store",
			zap.String("path", storeRemovedMembersPrefix),
			zap.Error(err),
		)
	}
	for _, n := range e.Node.Nodes {
		removed[MustParseMemberIDFromKey(lg, n.Key)] = true
	}
	return members, removed
}

func membersFromBackend(lg *zap.Logger, be backend.Backend) (map[types.ID]*Member, map[types.ID]bool) {
	return mustReadMembersFromBackend(lg, be)
}

func clusterVersionFromStore(lg *zap.Logger, st v2store.Store) *semver.Version {
	e, err := st.Get(path.Join(storePrefix, "version"), false, false)
	if err != nil {
		if isKeyNotFound(err) {
			return nil
		}
		lg.Panic(
			"failed to get cluster version from store",
			zap.String("path", path.Join(storePrefix, "version")),
			zap.Error(err),
		)
	}
	return semver.Must(semver.NewVersion(*e.Node.Value))
}

// The field is populated since etcd v3.5.
func clusterVersionFromBackend(lg *zap.Logger, be backend.Backend) *semver.Version {
	ckey := backendClusterVersionKey()
	tx := be.ReadTx()
	tx.RLock()
	defer tx.RUnlock()
	keys, vals := tx.UnsafeRange(buckets.Cluster, ckey, nil, 0)
	if len(keys) == 0 {
		return nil
	}
	if len(keys) != 1 {
		lg.Panic(
			"unexpected number of keys when getting cluster version from backend",
			zap.Int("number-of-key", len(keys)),
		)
	}
	return semver.Must(semver.NewVersion(string(vals[0])))
}

// The field is populated since etcd v3.5.
func downgradeInfoFromBackend(lg *zap.Logger, be backend.Backend) *DowngradeInfo {
	dkey := backendDowngradeKey()
	tx := be.ReadTx()
	tx.Lock()
	defer tx.Unlock()
	keys, vals := tx.UnsafeRange(buckets.Cluster, dkey, nil, 0)
	if len(keys) == 0 {
		return nil
	}

	if len(keys) != 1 {
		lg.Panic(
			"unexpected number of keys when getting cluster version from backend",
			zap.Int("number-of-key", len(keys)),
		)
	}
	var d DowngradeInfo
	if err := json.Unmarshal(vals[0], &d); err != nil {
		lg.Panic("failed to unmarshal downgrade information", zap.Error(err))
	}

	// verify the downgrade info from backend
	if d.Enabled {
		if _, err := semver.NewVersion(d.TargetVersion); err != nil {
			lg.Panic(
				"unexpected version format of the downgrade target version from backend",
				zap.String("target-version", d.TargetVersion),
			)
		}
	}
	return &d
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

	ctx, cancel := context.WithTimeout(context.TODO(), 30*time.Second)
	defer cancel()
	for i := range ems {
		var err error
		ok := false
		for j := range lms {
			if ok, err = netutil.URLStringsEqual(ctx, lg, ems[i].PeerURLs, lms[j].PeerURLs); ok {
				lms[j].ID = ems[i].ID
				break
			}
		}
		if !ok {
			return fmt.Errorf("PeerURLs: no match found for existing member (%v, %v), last resolver error (%v)", ems[i].ID, ems[i].PeerURLs, err)
		}
	}
	local.members = make(map[types.ID]*Member)
	for _, m := range lms {
		local.members[m.ID] = m
	}
	return nil
}

// IsValidVersionChange checks the two scenario when version is valid to change:
// 1. Downgrade: cluster version is 1 minor version higher than local version,
// cluster version should change.
// 2. Cluster start: when not all members version are available, cluster version
// is set to MinVersion(3.0), when all members are at higher version, cluster version
// is lower than local version, cluster version should change
func IsValidVersionChange(cv *semver.Version, lv *semver.Version) bool {
	cv = &semver.Version{Major: cv.Major, Minor: cv.Minor}
	lv = &semver.Version{Major: lv.Major, Minor: lv.Minor}

	if isValidDowngrade(cv, lv) || (cv.Major == lv.Major && cv.LessThan(*lv)) {
		return true
	}
	return false
}

// IsLocalMemberLearner returns if the local member is raft learner
func (c *RaftCluster) IsLocalMemberLearner() bool {
	c.Lock()
	defer c.Unlock()
	localMember, ok := c.members[c.localID]
	if !ok {
		c.lg.Panic(
			"failed to find local ID in cluster members",
			zap.String("cluster-id", c.cid.String()),
			zap.String("local-member-id", c.localID.String()),
		)
	}
	return localMember.IsLearner
}

// DowngradeInfo returns the downgrade status of the cluster
func (c *RaftCluster) DowngradeInfo() *DowngradeInfo {
	c.Lock()
	defer c.Unlock()
	if c.downgradeInfo == nil {
		return &DowngradeInfo{Enabled: false}
	}
	d := &DowngradeInfo{Enabled: c.downgradeInfo.Enabled, TargetVersion: c.downgradeInfo.TargetVersion}
	return d
}

func (c *RaftCluster) SetDowngradeInfo(d *DowngradeInfo, shouldApplyV3 ShouldApplyV3) {
	c.Lock()
	defer c.Unlock()

	if c.be != nil && shouldApplyV3 {
		mustSaveDowngradeToBackend(c.lg, c.be, d)
	}

	c.downgradeInfo = d

	if d.Enabled {
		c.lg.Info(
			"The server is ready to downgrade",
			zap.String("target-version", d.TargetVersion),
			zap.String("server-version", version.Version),
		)
	}
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

// PushMembershipToStorage is overriding storage information about cluster's
// members, such that they fully reflect internal RaftCluster's storage.
func (c *RaftCluster) PushMembershipToStorage() {
	if c.be != nil {
		TrimMembershipFromBackend(c.lg, c.be)
		for _, m := range c.members {
			mustSaveMemberToBackend(c.lg, c.be, m)
		}
	}
	if c.v2store != nil {
		TrimMembershipFromV2Store(c.lg, c.v2store)
		for _, m := range c.members {
			mustSaveMemberToStore(c.lg, c.v2store, m)
		}
	}
}
