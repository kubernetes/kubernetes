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
	"sync"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/klog/v2"
)

const (
	// defaultReplicas is the number of virtual nodes per member on the hash ring.
	defaultReplicas = 256
)

// SendEventFunc is a callback to send a watch event to a local watcher.
type SendEventFunc func(eventType string, key string, obj interface{})

// LocalWatcherInfo holds information about a watcher connected to this API server replica.
type LocalWatcherInfo struct {
	MemberID  MemberID
	LeaseID   clientv3.LeaseID
	SendEvent SendEventFunc
}

// Group represents the local state of a watch group.
type Group struct {
	mu            sync.RWMutex
	ring          *HashRing
	members       map[MemberID]struct{}
	localWatchers map[MemberID]*LocalWatcherInfo
	cancelSync    context.CancelFunc
}

// Manager manages watch groups for a single Cacher instance.
// It coordinates with etcd for distributed group membership
// and maintains a local hash ring for routing events.
type Manager struct {
	mu     sync.RWMutex
	store  *EtcdGroupStore
	groups map[string]*Group

	// onRebalance is called when a group's membership changes.
	// The callback receives the group name, old ring, new ring.
	onRebalance func(groupName string, oldRing *HashRing, newRing *HashRing)
}

// NewManager creates a new watch group manager.
func NewManager(store *EtcdGroupStore, onRebalance func(groupName string, oldRing *HashRing, newRing *HashRing)) *Manager {
	return &Manager{
		store:       store,
		groups:      make(map[string]*Group),
		onRebalance: onRebalance,
	}
}

// JoinGroup registers a member in a group. It writes membership to etcd,
// synchronously updates the local hash ring, and registers the local watcher.
// This must be called AFTER addWatcher but BEFORE processInterval, so the
// hash ring is up-to-date when the filter function starts evaluating events.
func (m *Manager) JoinGroup(ctx context.Context, groupName string, memberID MemberID, sendEvent SendEventFunc) error {
	leaseID, err := m.store.Join(ctx, groupName, memberID, DefaultLeaseTTL)
	if err != nil {
		return err
	}

	// Start keep-alive for the lease
	kaCh, err := m.store.KeepAlive(ctx, leaseID)
	if err != nil {
		return err
	}
	go func() {
		for range kaCh {
			// consume keep-alive responses
		}
	}()

	m.mu.Lock()
	group, exists := m.groups[groupName]
	if !exists {
		group = &Group{
			ring:          NewHashRing(defaultReplicas),
			members:       make(map[MemberID]struct{}),
			localWatchers: make(map[MemberID]*LocalWatcherInfo),
		}
		m.groups[groupName] = group
	}
	m.mu.Unlock()

	// Register the local watcher
	group.mu.Lock()
	group.localWatchers[memberID] = &LocalWatcherInfo{
		MemberID:  memberID,
		LeaseID:   leaseID,
		SendEvent: sendEvent,
	}
	group.mu.Unlock()

	// Synchronously sync ring from etcd to ensure we have the latest membership.
	// This is critical: the ring must be up-to-date BEFORE processInterval starts
	// sending initial events through the filter function.
	members, err := m.store.ListMembers(ctx, groupName)
	if err != nil {
		klog.Errorf("Failed to list members for group %s: %v", groupName, err)
	} else {
		group.mu.Lock()
		group.ring = NewHashRing(defaultReplicas)
		group.members = make(map[MemberID]struct{}, len(members))
		for _, mid := range members {
			group.ring.AddNode(mid)
			group.members[mid] = struct{}{}
		}
		group.mu.Unlock()
	}

	// Start sync loop for this group if it's new
	if !exists {
		syncCtx, cancel := context.WithCancel(ctx)
		group.mu.Lock()
		group.cancelSync = cancel
		group.mu.Unlock()
		go m.syncLoop(syncCtx, groupName, group)
	}

	return nil
}

// LeaveGroup removes a watcher from a group.
func (m *Manager) LeaveGroup(ctx context.Context, groupName string, memberID MemberID) error {
	m.mu.RLock()
	group, exists := m.groups[groupName]
	m.mu.RUnlock()

	if !exists {
		return nil
	}

	group.mu.Lock()
	info, ok := group.localWatchers[memberID]
	if ok {
		delete(group.localWatchers, memberID)
	}
	hasLocalWatchers := len(group.localWatchers) > 0
	group.mu.Unlock()

	if ok && info.LeaseID != 0 {
		if err := m.store.Leave(ctx, groupName, memberID, info.LeaseID); err != nil {
			klog.Errorf("Failed to leave group %s for member %s: %v", groupName, memberID, err)
		}
	}

	// If no more local watchers, stop the sync loop and clean up the group
	if !hasLocalWatchers {
		group.mu.Lock()
		if group.cancelSync != nil {
			group.cancelSync()
		}
		group.mu.Unlock()

		m.mu.Lock()
		delete(m.groups, groupName)
		m.mu.Unlock()
	}

	return nil
}

// IsOwner checks if the given member is the owner of the resource key
// according to the current hash ring state.
func (m *Manager) IsOwner(groupName string, memberID MemberID, resourceKey string) bool {
	m.mu.RLock()
	group, exists := m.groups[groupName]
	m.mu.RUnlock()

	if !exists {
		return true // group not yet initialized, no filtering
	}

	group.mu.RLock()
	defer group.mu.RUnlock()

	owner := group.ring.GetNode(resourceKey)
	return owner == memberID
}

// SetOnRebalance sets the rebalance callback.
func (m *Manager) SetOnRebalance(callback func(groupName string, oldRing *HashRing, newRing *HashRing)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onRebalance = callback
}

// GetGroup returns the group state for the given group name.
func (m *Manager) GetGroup(groupName string) *Group {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.groups[groupName]
}

// GetLocalWatchers returns a snapshot of local watchers for a group.
func (g *Group) GetLocalWatchers() map[MemberID]*LocalWatcherInfo {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make(map[MemberID]*LocalWatcherInfo, len(g.localWatchers))
	for k, v := range g.localWatchers {
		result[k] = v
	}
	return result
}

// GetRing returns the current hash ring (under read lock).
func (g *Group) GetRing() *HashRing {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.ring
}

// syncLoop watches etcd for membership changes and updates the local hash ring.
func (m *Manager) syncLoop(ctx context.Context, groupName string, group *Group) {
	ch := m.store.WatchMembers(ctx, groupName)
	for {
		select {
		case change, ok := <-ch:
			if !ok {
				return
			}
			m.handleMembershipChange(groupName, group, change)
		case <-ctx.Done():
			return
		}
	}
}

func (m *Manager) handleMembershipChange(groupName string, group *Group, change MembershipChange) {
	group.mu.Lock()
	oldRing := group.ring.Clone()

	if change.Joined {
		if _, exists := group.members[change.MemberID]; !exists {
			group.ring.AddNode(change.MemberID)
			group.members[change.MemberID] = struct{}{}
		}
	} else {
		if _, exists := group.members[change.MemberID]; exists {
			group.ring.RemoveNode(change.MemberID)
			delete(group.members, change.MemberID)
		}
	}

	newRing := group.ring.Clone()
	group.mu.Unlock()

	if m.onRebalance != nil {
		m.onRebalance(groupName, oldRing, newRing)
	}
}
