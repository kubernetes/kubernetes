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

package membership

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"

	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"

	"github.com/coreos/go-semver/semver"
	"go.uber.org/zap"
)

const (
	attributesSuffix     = "attributes"
	raftAttributesSuffix = "raftAttributes"

	// the prefix for storing membership related information in store provided by store pkg.
	storePrefix = "/0"
)

var (
	StoreMembersPrefix        = path.Join(storePrefix, "members")
	storeRemovedMembersPrefix = path.Join(storePrefix, "removed_members")
	errMemberAlreadyExist     = fmt.Errorf("member already exists")
	errMemberNotFound         = fmt.Errorf("member not found")
)

func unsafeSaveMemberToBackend(lg *zap.Logger, be backend.Backend, m *Member) error {
	mkey := backendMemberKey(m.ID)
	mvalue, err := json.Marshal(m)
	if err != nil {
		lg.Panic("failed to marshal member", zap.Error(err))
	}

	tx := be.BatchTx()
	tx.LockInsideApply()
	defer tx.Unlock()
	if unsafeMemberExists(tx, mkey) {
		return errMemberAlreadyExist
	}
	tx.UnsafePut(buckets.Members, mkey, mvalue)
	return nil
}

// TrimClusterFromBackend removes all information about cluster (versions)
// from the v3 backend.
func TrimClusterFromBackend(be backend.Backend) error {
	tx := be.BatchTx()
	tx.LockOutsideApply()
	defer tx.Unlock()
	tx.UnsafeDeleteBucket(buckets.Cluster)
	return nil
}

func unsafeDeleteMemberFromBackend(be backend.Backend, id types.ID) error {
	mkey := backendMemberKey(id)

	tx := be.BatchTx()
	tx.LockInsideApply()
	defer tx.Unlock()
	tx.UnsafePut(buckets.MembersRemoved, mkey, []byte("removed"))
	if !unsafeMemberExists(tx, mkey) {
		return errMemberNotFound
	}
	tx.UnsafeDelete(buckets.Members, mkey)
	return nil
}

func unsafeMemberExists(tx backend.ReadTx, mkey []byte) bool {
	var found bool
	tx.UnsafeForEach(buckets.Members, func(k, v []byte) error {
		if bytes.Equal(k, mkey) {
			found = true
		}
		return nil
	})
	return found
}

func readMembersFromBackend(lg *zap.Logger, be backend.Backend) (map[types.ID]*Member, map[types.ID]bool, error) {
	members := make(map[types.ID]*Member)
	removed := make(map[types.ID]bool)

	tx := be.ReadTx()
	tx.RLock()
	defer tx.RUnlock()
	err := tx.UnsafeForEach(buckets.Members, func(k, v []byte) error {
		memberId := mustParseMemberIDFromBytes(lg, k)
		m := &Member{ID: memberId}
		if err := json.Unmarshal(v, &m); err != nil {
			return err
		}
		members[memberId] = m
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("couldn't read members from backend: %w", err)
	}

	err = tx.UnsafeForEach(buckets.MembersRemoved, func(k, v []byte) error {
		memberId := mustParseMemberIDFromBytes(lg, k)
		removed[memberId] = true
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("couldn't read members_removed from backend: %w", err)
	}
	return members, removed, nil
}

func mustReadMembersFromBackend(lg *zap.Logger, be backend.Backend) (map[types.ID]*Member, map[types.ID]bool) {
	members, removed, err := readMembersFromBackend(lg, be)
	if err != nil {
		lg.Panic("couldn't read members from backend", zap.Error(err))
	}
	return members, removed
}

// TrimMembershipFromBackend removes all information about members &
// removed_members from the v3 backend.
func TrimMembershipFromBackend(lg *zap.Logger, be backend.Backend) error {
	lg.Info("Trimming membership information from the backend...")
	tx := be.BatchTx()
	tx.LockOutsideApply()
	defer tx.Unlock()
	err := tx.UnsafeForEach(buckets.Members, func(k, v []byte) error {
		tx.UnsafeDelete(buckets.Members, k)
		lg.Debug("Removed member from the backend",
			zap.Stringer("member", mustParseMemberIDFromBytes(lg, k)))
		return nil
	})
	if err != nil {
		return err
	}
	return tx.UnsafeForEach(buckets.MembersRemoved, func(k, v []byte) error {
		tx.UnsafeDelete(buckets.MembersRemoved, k)
		lg.Debug("Removed removed_member from the backend",
			zap.Stringer("member", mustParseMemberIDFromBytes(lg, k)))
		return nil
	})
}

// TrimMembershipFromV2Store removes all information about members &
// removed_members from the v2 store.
func TrimMembershipFromV2Store(lg *zap.Logger, s v2store.Store) error {
	members, removed := membersFromStore(lg, s)

	for mID := range members {
		_, err := s.Delete(MemberStoreKey(mID), true, true)
		if err != nil {
			return err
		}
	}
	for mID := range removed {
		_, err := s.Delete(RemovedMemberStoreKey(mID), true, true)
		if err != nil {
			return err
		}
	}

	return nil
}

// The field is populated since etcd v3.5.
func mustSaveClusterVersionToBackend(be backend.Backend, ver *semver.Version) {
	ckey := backendClusterVersionKey()

	tx := be.BatchTx()
	tx.LockInsideApply()
	defer tx.Unlock()
	tx.UnsafePut(buckets.Cluster, ckey, []byte(ver.String()))
}

// The field is populated since etcd v3.5.
func mustSaveDowngradeToBackend(lg *zap.Logger, be backend.Backend, downgrade *DowngradeInfo) {
	dkey := backendDowngradeKey()
	dvalue, err := json.Marshal(downgrade)
	if err != nil {
		lg.Panic("failed to marshal downgrade information", zap.Error(err))
	}
	tx := be.BatchTx()
	tx.LockInsideApply()
	defer tx.Unlock()
	tx.UnsafePut(buckets.Cluster, dkey, dvalue)
}

func mustSaveMemberToStore(lg *zap.Logger, s v2store.Store, m *Member) {
	err := unsafeSaveMemberToStore(lg, s, m)
	if err != nil {
		lg.Panic(
			"failed to save member to store",
			zap.String("member-id", m.ID.String()),
			zap.Error(err),
		)
	}
}

func unsafeSaveMemberToStore(lg *zap.Logger, s v2store.Store, m *Member) error {
	b, err := json.Marshal(m.RaftAttributes)
	if err != nil {
		lg.Panic("failed to marshal raftAttributes", zap.Error(err))
	}
	p := path.Join(MemberStoreKey(m.ID), raftAttributesSuffix)
	_, err = s.Create(p, false, string(b), false, v2store.TTLOptionSet{ExpireTime: v2store.Permanent})
	return err
}

func unsafeDeleteMemberFromStore(s v2store.Store, id types.ID) error {
	if _, err := s.Delete(MemberStoreKey(id), true, true); err != nil {
		return err
	}
	if _, err := s.Create(RemovedMemberStoreKey(id), false, "", false, v2store.TTLOptionSet{ExpireTime: v2store.Permanent}); err != nil {
		return err
	}
	return nil
}

func mustUpdateMemberInStore(lg *zap.Logger, s v2store.Store, m *Member) {
	b, err := json.Marshal(m.RaftAttributes)
	if err != nil {
		lg.Panic("failed to marshal raftAttributes", zap.Error(err))
	}
	p := path.Join(MemberStoreKey(m.ID), raftAttributesSuffix)
	if _, err := s.Update(p, string(b), v2store.TTLOptionSet{ExpireTime: v2store.Permanent}); err != nil {
		lg.Panic(
			"failed to update raftAttributes",
			zap.String("path", p),
			zap.Error(err),
		)
	}
}

func mustUpdateMemberAttrInStore(lg *zap.Logger, s v2store.Store, m *Member) {
	b, err := json.Marshal(m.Attributes)
	if err != nil {
		lg.Panic("failed to marshal attributes", zap.Error(err))
	}
	p := path.Join(MemberStoreKey(m.ID), attributesSuffix)
	if _, err := s.Set(p, false, string(b), v2store.TTLOptionSet{ExpireTime: v2store.Permanent}); err != nil {
		lg.Panic(
			"failed to update attributes",
			zap.String("path", p),
			zap.Error(err),
		)
	}
}

func mustSaveClusterVersionToStore(lg *zap.Logger, s v2store.Store, ver *semver.Version) {
	if _, err := s.Set(StoreClusterVersionKey(), false, ver.String(), v2store.TTLOptionSet{ExpireTime: v2store.Permanent}); err != nil {
		lg.Panic(
			"failed to save cluster version to store",
			zap.String("path", StoreClusterVersionKey()),
			zap.Error(err),
		)
	}
}

// nodeToMember builds member from a key value node.
// the child nodes of the given node MUST be sorted by key.
func nodeToMember(lg *zap.Logger, n *v2store.NodeExtern) (*Member, error) {
	m := &Member{ID: MustParseMemberIDFromKey(lg, n.Key)}
	attrs := make(map[string][]byte)
	raftAttrKey := path.Join(n.Key, raftAttributesSuffix)
	attrKey := path.Join(n.Key, attributesSuffix)
	for _, nn := range n.Nodes {
		if nn.Key != raftAttrKey && nn.Key != attrKey {
			return nil, fmt.Errorf("unknown key %q", nn.Key)
		}
		attrs[nn.Key] = []byte(*nn.Value)
	}
	if data := attrs[raftAttrKey]; data != nil {
		if err := json.Unmarshal(data, &m.RaftAttributes); err != nil {
			return nil, fmt.Errorf("unmarshal raftAttributes error: %v", err)
		}
	} else {
		return nil, fmt.Errorf("raftAttributes key doesn't exist")
	}
	if data := attrs[attrKey]; data != nil {
		if err := json.Unmarshal(data, &m.Attributes); err != nil {
			return m, fmt.Errorf("unmarshal attributes error: %v", err)
		}
	}
	return m, nil
}

func backendMemberKey(id types.ID) []byte {
	return []byte(id.String())
}

func backendClusterVersionKey() []byte {
	return []byte("clusterVersion")
}

func backendDowngradeKey() []byte {
	return []byte("downgrade")
}

func mustCreateBackendBuckets(be backend.Backend) {
	tx := be.BatchTx()
	tx.LockOutsideApply()
	defer tx.Unlock()
	tx.UnsafeCreateBucket(buckets.Members)
	tx.UnsafeCreateBucket(buckets.MembersRemoved)
	tx.UnsafeCreateBucket(buckets.Cluster)
}

func MemberStoreKey(id types.ID) string {
	return path.Join(StoreMembersPrefix, id.String())
}

func StoreClusterVersionKey() string {
	return path.Join(storePrefix, "version")
}

func MemberAttributesStorePath(id types.ID) string {
	return path.Join(MemberStoreKey(id), attributesSuffix)
}

func mustParseMemberIDFromBytes(lg *zap.Logger, key []byte) types.ID {
	id, err := types.IDFromString(string(key))
	if err != nil {
		lg.Panic("failed to parse member id from key", zap.Error(err))
	}
	return id
}

func MustParseMemberIDFromKey(lg *zap.Logger, key string) types.ID {
	id, err := types.IDFromString(path.Base(key))
	if err != nil {
		lg.Panic("failed to parse member id from key", zap.Error(err))
	}
	return id
}

func RemovedMemberStoreKey(id types.ID) string {
	return path.Join(storeRemovedMembersPrefix, id.String())
}
