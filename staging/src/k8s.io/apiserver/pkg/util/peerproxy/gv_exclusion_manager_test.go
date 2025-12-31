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

package peerproxy

import (
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetExclusionSet(t *testing.T) {
	tests := []struct {
		name       string
		activeGVs  map[schema.GroupVersion]struct{}
		deletedGVs map[schema.GroupVersion]time.Time
		wantGVs    []schema.GroupVersion
	}{
		{
			name:       "empty state",
			activeGVs:  nil,
			deletedGVs: nil,
			wantGVs:    []schema.GroupVersion{},
		},
		{
			name: "only active GVs",
			activeGVs: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}:         {},
				{Group: "batch", Version: "v1"}:        {},
				{Group: "custom", Version: "v1alpha1"}: {},
			},
			wantGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
				{Group: "batch", Version: "v1"},
				{Group: "custom", Version: "v1alpha1"},
			},
		},
		{
			name: "only deleted GVs",
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "deprecated", Version: "v1beta1"}: time.Now(),
				{Group: "legacy", Version: "v1alpha1"}:    time.Now(),
			},
			// Reaper hasnt removed deleted GVs yet.
			wantGVs: []schema.GroupVersion{
				{Group: "deprecated", Version: "v1beta1"},
				{Group: "legacy", Version: "v1alpha1"},
			},
		},
		{
			name: "different active and deleted GVs",
			activeGVs: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}:  {},
				{Group: "batch", Version: "v1"}: {},
			},
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "old", Version: "v1"}: time.Now(),
			},
			// Include both active GVs and recently deleted GVs.
			// Deleted GVs remain in the exclusion set until the reaper removes them after the grace period.
			wantGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
				{Group: "batch", Version: "v1"},
				{Group: "old", Version: "v1"},
			},
		},
		{
			// A GV can appear in both active and deleted sets if:
			// 1. CRD was deleted (moved from active to deleted)
			// 2. CRD was recreated (added back to active)
			// 3. Reaper hasn't cleaned up the deleted entry yet
			name: "same GV in both active and deleted",
			activeGVs: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "apps", Version: "v1"}: time.Now(),
			},
			wantGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewGVExclusionManager(5*time.Minute, 1*time.Minute, &atomic.Value{}, &atomic.Pointer[func()]{})

			if tt.activeGVs != nil {
				mgr.currentlyActiveGVs.Store(tt.activeGVs)
			}
			if tt.deletedGVs != nil {
				mgr.recentlyDeletedGVs.Store(tt.deletedGVs)
			}

			exclusionSet := mgr.getExclusionSet()
			if len(exclusionSet) != len(tt.wantGVs) {
				t.Errorf("Want exclusion set size %d, got %d", len(tt.wantGVs), len(exclusionSet))
			}

			for _, gv := range tt.wantGVs {
				if _, found := exclusionSet[gv]; !found {
					t.Errorf("Want GV %v in exclusion set, but not found", gv)
				}
			}
		})
	}
}

func TestReapExpiredGVs(t *testing.T) {
	tests := []struct {
		name        string
		gracePeriod time.Duration
		deletedGVs  map[schema.GroupVersion]time.Time
		wantReaped  []schema.GroupVersion
	}{
		{
			name:        "empty state",
			gracePeriod: 100 * time.Millisecond,
			deletedGVs:  map[schema.GroupVersion]time.Time{},
			wantReaped:  []schema.GroupVersion{},
		},
		{
			name:        "reap old GV keep recent",
			gracePeriod: 100 * time.Millisecond,
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "old", Version: "v1"}:    time.Now().Add(-200 * time.Millisecond),
				{Group: "recent", Version: "v1"}: time.Now().Add(-50 * time.Millisecond),
				{Group: "new", Version: "v1"}:    time.Now(),
			},
			wantReaped: []schema.GroupVersion{
				{Group: "old", Version: "v1"},
			},
		},
		{
			name:        "all expired",
			gracePeriod: 50 * time.Millisecond,
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "old1", Version: "v1"}: time.Now().Add(-100 * time.Millisecond),
				{Group: "old2", Version: "v1"}: time.Now().Add(-200 * time.Millisecond),
			},
			wantReaped: []schema.GroupVersion{
				{Group: "old1", Version: "v1"},
				{Group: "old2", Version: "v1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewGVExclusionManager(tt.gracePeriod, 50*time.Millisecond, &atomic.Value{}, &atomic.Pointer[func()]{})
			mgr.recentlyDeletedGVs.Store(tt.deletedGVs)
			mgr.reapExpiredGVs()
			result := mgr.loadRecentlyDeletedGVs()

			for _, gv := range tt.wantReaped {
				if _, found := result[gv]; found {
					t.Errorf("GV %v should have been reaped but still exists", gv)
				}
			}
		})
	}
}

func TestDetectDiff(t *testing.T) {
	mgr := NewGVExclusionManager(5*time.Minute, 1*time.Minute, &atomic.Value{}, &atomic.Pointer[func()]{})

	tests := []struct {
		name string
		old  map[schema.GroupVersion]struct{}
		new  map[schema.GroupVersion]struct{}
		want bool
	}{
		{
			name: "both empty",
			old:  map[schema.GroupVersion]struct{}{},
			new:  map[schema.GroupVersion]struct{}{},
			want: false,
		},
		{
			name: "identical",
			old: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			new: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			want: false,
		},
		{
			name: "added GV",
			old: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			new: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}:  {},
				{Group: "batch", Version: "v1"}: {},
			},
			want: true,
		},
		{
			name: "removed GV",
			old: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}:  {},
				{Group: "batch", Version: "v1"}: {},
			},
			new: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			want: true,
		},
		{
			name: "different GVs same size",
			old: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			new: map[schema.GroupVersion]struct{}{
				{Group: "batch", Version: "v1"}: {},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mgr.diffGVs(tt.old, tt.new)
			if result != tt.want {
				t.Errorf("detectDiff() = %v, want %v", result, tt.want)
			}
		})
	}
}

func TestFilterPeerDiscoveryCache(t *testing.T) {
	tests := []struct {
		name         string
		activeGVs    map[schema.GroupVersion]struct{}
		deletedGVs   map[schema.GroupVersion]time.Time
		cacheMap     map[string]PeerDiscoveryCacheEntry
		wantChanged  bool
		wantPeerGVRs map[string]int // peer name -> GVR count
	}{
		{
			name:      "empty exclusion no changes",
			activeGVs: nil,
			cacheMap: map[string]PeerDiscoveryCacheEntry{
				"peer1": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "apps", Version: "v1", Resource: "deployments"}: true,
					},
				},
				"peer2": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "batch", Version: "v1", Resource: "jobs"}: true,
					},
				},
			},
			wantChanged: false,
			wantPeerGVRs: map[string]int{
				"peer1": 1,
				"peer2": 1,
			},
		},
		{
			name: "filter active GVs",
			activeGVs: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			cacheMap: map[string]PeerDiscoveryCacheEntry{
				"peer1": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "apps", Version: "v1", Resource: "deployments"}: true,
					},
				},
				"peer2": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "batch", Version: "v1", Resource: "jobs"}: true,
					},
				},
			},
			wantChanged: true,
			wantPeerGVRs: map[string]int{
				"peer1": 0, // apps/v1 filtered out
				"peer2": 1, // unchanged
			},
		},
		{
			// Recently deleted GVs are still filtered from peer discovery during the
			// grace period (before the reaper cleans them up) to avoid routing requests
			// to peers for GVs that were just deleted locally.
			name: "filter deleted GVs",
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "custom", Version: "v1alpha1"}: time.Now(),
			},
			cacheMap: map[string]PeerDiscoveryCacheEntry{
				"peer1": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "custom", Version: "v1alpha1", Resource: "myresources"}: true,
					},
				},
				"peer2": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "apps", Version: "v1", Resource: "deployments"}: true,
					},
				},
			},
			wantChanged: true,
			wantPeerGVRs: map[string]int{
				"peer1": 0, // custom/v1alpha1 filtered out
				"peer2": 1, // unchanged
			},
		},
		{
			// Both active GVs and recently deleted GVs (within grace period, not yet
			// cleaned up by the reaper) are filtered from peer discovery.
			name: "filter both active and deleted GVs",
			activeGVs: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			deletedGVs: map[schema.GroupVersion]time.Time{
				{Group: "custom", Version: "v1alpha1"}: time.Now(),
			},
			cacheMap: map[string]PeerDiscoveryCacheEntry{
				"peer1": {
					GVRs: map[schema.GroupVersionResource]bool{
						{Group: "apps", Version: "v1", Resource: "deployments"}:         true,
						{Group: "custom", Version: "v1alpha1", Resource: "myresources"}: true,
						{Group: "batch", Version: "v1", Resource: "jobs"}:               true,
					},
				},
			},
			wantChanged: true,
			wantPeerGVRs: map[string]int{
				"peer1": 1, // apps/v1 and custom/v1alpha1 filtered out, batch/v1 remains
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewGVExclusionManager(5*time.Minute, 1*time.Minute, &atomic.Value{}, &atomic.Pointer[func()]{})

			if tt.activeGVs != nil {
				mgr.currentlyActiveGVs.Store(tt.activeGVs)
			}
			if tt.deletedGVs != nil {
				mgr.recentlyDeletedGVs.Store(tt.deletedGVs)
			}

			filtered, changed := mgr.FilterPeerDiscoveryCache(tt.cacheMap)
			if changed != tt.wantChanged {
				t.Errorf("Want changed=%v, got %v", tt.wantChanged, changed)
			}

			if len(filtered) != len(tt.cacheMap) {
				t.Errorf("Want %d peers in filtered cache, got %d", len(tt.cacheMap), len(filtered))
			}

			for peerName, wantCount := range tt.wantPeerGVRs {
				entry, found := filtered[peerName]
				if !found {
					t.Errorf("Peer %s not found in filtered cache", peerName)
					continue
				}
				if len(entry.GVRs) != wantCount {
					t.Errorf("Peer %s: want %d GVRs, got %d", peerName, wantCount, len(entry.GVRs))
				}
			}
		})
	}
}

func TestFilterPeerCacheEntry(t *testing.T) {
	tests := []struct {
		name         string
		entry        PeerDiscoveryCacheEntry
		exclusionSet map[schema.GroupVersion]struct{}
		wantChanged  bool
		wantGVRs     []schema.GroupVersionResource
	}{
		{
			name: "filter GVRs and groups",
			entry: PeerDiscoveryCacheEntry{
				GVRs: map[schema.GroupVersionResource]bool{
					{Group: "apps", Version: "v1", Resource: "deployments"}:   true,
					{Group: "batch", Version: "v1", Resource: "jobs"}:         true,
					{Group: "custom", Version: "v1", Resource: "myresources"}: true,
				},
				GroupDiscovery: []apidiscoveryv2.APIGroupDiscovery{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "apps"},
						Versions: []apidiscoveryv2.APIVersionDiscovery{
							{Version: "v1"},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "batch"},
						Versions: []apidiscoveryv2.APIVersionDiscovery{
							{Version: "v1"},
						},
					},
				},
			},
			exclusionSet: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			wantChanged: true,
			wantGVRs: []schema.GroupVersionResource{
				{Group: "batch", Version: "v1", Resource: "jobs"},
				{Group: "custom", Version: "v1", Resource: "myresources"},
			},
		},
		{
			name: "no changes when exclusion doesn't match",
			entry: PeerDiscoveryCacheEntry{
				GVRs: map[schema.GroupVersionResource]bool{
					{Group: "batch", Version: "v1", Resource: "jobs"}: true,
				},
			},
			exclusionSet: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			wantChanged: false,
			wantGVRs: []schema.GroupVersionResource{
				{Group: "batch", Version: "v1", Resource: "jobs"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewGVExclusionManager(5*time.Minute, 1*time.Minute, &atomic.Value{}, &atomic.Pointer[func()]{})

			filtered, changed := mgr.filterPeerCacheEntry(tt.entry, tt.exclusionSet)

			if changed != tt.wantChanged {
				t.Errorf("Want changed=%v, got %v", tt.wantChanged, changed)
			}

			for _, gvr := range tt.wantGVRs {
				if _, found := filtered.GVRs[gvr]; !found {
					t.Errorf("Want GVR %v to be present", gvr)
				}
			}
		})
	}
}

func TestFilterGroupDiscovery(t *testing.T) {
	tests := []struct {
		name             string
		groupDiscoveries []apidiscoveryv2.APIGroupDiscovery
		exclusionSet     map[schema.GroupVersion]struct{}
		wantGroups       map[string][]string // group name -> list of versions
	}{
		{
			name: "partial version exclusion",
			groupDiscoveries: []apidiscoveryv2.APIGroupDiscovery{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "apps"},
					Versions: []apidiscoveryv2.APIVersionDiscovery{
						{Version: "v1"},
						{Version: "v1beta1"},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "batch"},
					Versions: []apidiscoveryv2.APIVersionDiscovery{
						{Version: "v1"},
					},
				},
			},
			exclusionSet: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			wantGroups: map[string][]string{
				"apps":  {"v1beta1"},
				"batch": {"v1"},
			},
		},
		{
			name: "all versions excluded",
			groupDiscoveries: []apidiscoveryv2.APIGroupDiscovery{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "apps"},
					Versions: []apidiscoveryv2.APIVersionDiscovery{
						{Version: "v1"},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "batch"},
					Versions: []apidiscoveryv2.APIVersionDiscovery{
						{Version: "v1"},
					},
				},
			},
			exclusionSet: map[schema.GroupVersion]struct{}{
				{Group: "apps", Version: "v1"}: {},
			},
			wantGroups: map[string][]string{
				"batch": {"v1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewGVExclusionManager(5*time.Minute, 1*time.Minute, &atomic.Value{}, &atomic.Pointer[func()]{})

			filtered := mgr.filterGroupDiscovery(tt.groupDiscoveries, tt.exclusionSet)
			for groupName, wantVersionStrs := range tt.wantGroups {
				var group *apidiscoveryv2.APIGroupDiscovery
				for i := range filtered {
					if filtered[i].Name == groupName {
						group = &filtered[i]
						break
					}
				}

				if group == nil {
					t.Errorf("Want group %s not found in filtered results", groupName)
					continue
				}

				if len(group.Versions) != len(wantVersionStrs) {
					t.Errorf("Group %s: want %d versions, got %d", groupName, len(wantVersionStrs), len(group.Versions))
					continue
				}

				for _, wantVer := range wantVersionStrs {
					found := false
					for _, ver := range group.Versions {
						if ver.Version == wantVer {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("Group %s: want version %s not found", groupName, wantVer)
					}
				}
			}
		})
	}
}
