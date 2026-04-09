/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package watchgroup

import (
	"context"
	"path"
	"strings"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
)

const (
	// DefaultLeaseTTL is the default TTL for watch group member leases in seconds.
	DefaultLeaseTTL int64 = 30
)

// MembershipChange represents a change in group membership detected via etcd watch.
type MembershipChange struct {
	GroupName string
	MemberID  MemberID
	Joined    bool // true = joined, false = left
}

// EtcdGroupStore manages watch group membership in etcd for a specific resource type.
// Each resource type (e.g., customers.piny940.com) has its own EtcdGroupStore instance,
// so membership is fully isolated per resource.
// Each member is stored as a key with a TTL-based lease. When a watcher disconnects,
// the lease expires and the member is automatically removed.
type EtcdGroupStore struct {
	client        *clientv3.Client
	prefix        string                 // base prefix, e.g., "/watchgroups/"
	groupResource schema.GroupResource   // the resource type this store manages
}

// NewEtcdGroupStore creates a new etcd-backed group membership store for the given resource type.
// The prefix is the base etcd key prefix (e.g., "/watchgroups/"). The actual keys include
// the group and resource names to ensure isolation between resource types.
func NewEtcdGroupStore(client *clientv3.Client, prefix string, groupResource schema.GroupResource) *EtcdGroupStore {
	if !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}
	return &EtcdGroupStore{
		client:        client,
		prefix:        prefix,
		groupResource: groupResource,
	}
}

// resourcePrefix returns the etcd key prefix for this resource type.
// Key structure: {prefix}{apiGroup}/{resource}/
func (s *EtcdGroupStore) resourcePrefix() string {
	return path.Join(s.prefix, s.groupResource.Group, s.groupResource.Resource) + "/"
}

func (s *EtcdGroupStore) memberKey(groupName string, memberID MemberID) string {
	return path.Join(s.resourcePrefix(), groupName, string(memberID))
}

func (s *EtcdGroupStore) groupPrefix(groupName string) string {
	return path.Join(s.resourcePrefix(), groupName) + "/"
}

// Join registers a member in the group with a TTL-based lease.
// Returns the lease ID for keep-alive management.
func (s *EtcdGroupStore) Join(ctx context.Context, groupName string, memberID MemberID, ttl int64) (clientv3.LeaseID, error) {
	lease, err := s.client.Grant(ctx, ttl)
	if err != nil {
		return 0, err
	}

	key := s.memberKey(groupName, memberID)
	_, err = s.client.Put(ctx, key, string(memberID), clientv3.WithLease(lease.ID))
	if err != nil {
		return 0, err
	}

	return lease.ID, nil
}

// KeepAlive starts a keep-alive loop for the given lease.
// The returned channel is closed when the keep-alive fails.
func (s *EtcdGroupStore) KeepAlive(ctx context.Context, leaseID clientv3.LeaseID) (<-chan *clientv3.LeaseKeepAliveResponse, error) {
	return s.client.KeepAlive(ctx, leaseID)
}

// Leave removes a member from the group and revokes its lease.
func (s *EtcdGroupStore) Leave(ctx context.Context, groupName string, memberID MemberID, leaseID clientv3.LeaseID) error {
	key := s.memberKey(groupName, memberID)
	_, err := s.client.Delete(ctx, key)
	if err != nil {
		klog.Errorf("Failed to delete watch group member key %s: %v", key, err)
	}

	_, err = s.client.Revoke(ctx, leaseID)
	if err != nil {
		klog.Errorf("Failed to revoke lease for watch group member %s/%s: %v", groupName, memberID, err)
	}
	return nil
}

// ListMembers returns all current members of a group.
func (s *EtcdGroupStore) ListMembers(ctx context.Context, groupName string) ([]MemberID, error) {
	prefix := s.groupPrefix(groupName)
	resp, err := s.client.Get(ctx, prefix, clientv3.WithPrefix())
	if err != nil {
		return nil, err
	}

	members := make([]MemberID, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		memberID := MemberID(kv.Value)
		members = append(members, memberID)
	}
	return members, nil
}

// WatchMembers watches for membership changes in a group.
// The returned channel emits MembershipChange events.
func (s *EtcdGroupStore) WatchMembers(ctx context.Context, groupName string) <-chan MembershipChange {
	prefix := s.groupPrefix(groupName)
	ch := make(chan MembershipChange, 16)

	go func() {
		defer close(ch)
		wch := s.client.Watch(ctx, prefix, clientv3.WithPrefix(), clientv3.WithPrevKV())
		for resp := range wch {
			for _, ev := range resp.Events {
				var change MembershipChange
				change.GroupName = groupName
				switch ev.Type {
				case clientv3.EventTypePut:
					change.MemberID = MemberID(ev.Kv.Value)
					change.Joined = true
				case clientv3.EventTypeDelete:
					if ev.PrevKv != nil {
						change.MemberID = MemberID(ev.PrevKv.Value)
					}
					change.Joined = false
				}
				if change.MemberID != "" {
					select {
					case ch <- change:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()

	return ch
}
