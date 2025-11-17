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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFilteringLogic(t *testing.T) {
	gvr := func(g, v, r string) schema.GroupVersionResource {
		return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
	}
	gv := func(g, v string) schema.GroupVersion {
		return schema.GroupVersion{Group: g, Version: v}
	}

	testCases := []struct {
		name               string
		initialPeerCache   PeerDiscoveryCacheEntry
		excludedGVs        map[schema.GroupVersion]struct{}
		wantFilteredGVRs   map[schema.GroupVersionResource]bool
		wantFilteredGroups []apidiscoveryv2.APIGroupDiscovery
		wantChange         bool
	}{
		{
			name: "no exclusions",
			initialPeerCache: PeerDiscoveryCacheEntry{
				GVRs: map[schema.GroupVersionResource]bool{gvr("apps", "v1", "deployments"): true},
				GroupDiscovery: []apidiscoveryv2.APIGroupDiscovery{{
					ObjectMeta: metav1.ObjectMeta{Name: "apps"},
					Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
				}},
			},
			excludedGVs:      map[schema.GroupVersion]struct{}{},
			wantFilteredGVRs: map[schema.GroupVersionResource]bool{gvr("apps", "v1", "deployments"): true},
			wantFilteredGroups: []apidiscoveryv2.APIGroupDiscovery{{
				ObjectMeta: metav1.ObjectMeta{Name: "apps"},
				Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
			}},
			wantChange: false,
		},
		{
			name: "exclude one GV that exists",
			initialPeerCache: PeerDiscoveryCacheEntry{
				GVRs: map[schema.GroupVersionResource]bool{
					gvr("apps", "v1", "deployments"):  true,
					gvr("apps", "v1", "statefulsets"): true,
					gvr("batch", "v1", "jobs"):        true,
				},
				GroupDiscovery: []apidiscoveryv2.APIGroupDiscovery{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "apps"},
						Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "batch"},
						Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
					},
				},
			},
			excludedGVs: map[schema.GroupVersion]struct{}{
				gv("apps", "v1"): {},
			},
			wantFilteredGVRs: map[schema.GroupVersionResource]bool{gvr("batch", "v1", "jobs"): true},
			wantFilteredGroups: []apidiscoveryv2.APIGroupDiscovery{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "batch"},
					Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
				},
			},
			wantChange: true,
		},
		{
			name: "exclude a GV that does not exist",
			initialPeerCache: PeerDiscoveryCacheEntry{
				GVRs: map[schema.GroupVersionResource]bool{gvr("apps", "v1", "deployments"): true},
				GroupDiscovery: []apidiscoveryv2.APIGroupDiscovery{{
					ObjectMeta: metav1.ObjectMeta{Name: "apps"},
					Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
				}},
			},
			excludedGVs: map[schema.GroupVersion]struct{}{
				gv("foo", "v1"): {},
			},
			wantFilteredGVRs: map[schema.GroupVersionResource]bool{gvr("apps", "v1", "deployments"): true},
			wantFilteredGroups: []apidiscoveryv2.APIGroupDiscovery{{
				ObjectMeta: metav1.ObjectMeta{Name: "apps"},
				Versions:   []apidiscoveryv2.APIVersionDiscovery{{Version: "v1"}},
			}},
			wantChange: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			h := &peerProxyHandler{}
			filteredEntry, changed := h.filterPeerCacheEntry(tc.initialPeerCache, tc.excludedGVs)

			if changed != tc.wantChange {
				t.Errorf("want change to be %v, got %v", tc.wantChange, changed)
			}

			if !reflect.DeepEqual(filteredEntry.GVRs, tc.wantFilteredGVRs) {
				t.Errorf("filtered GVRs mismatch: got %v, want %v", filteredEntry.GVRs, tc.wantFilteredGVRs)
			}

			if !reflect.DeepEqual(filteredEntry.GroupDiscovery, tc.wantFilteredGroups) {
				t.Errorf("filtered GroupDiscovery mismatch: got %v, want %v", filteredEntry.GroupDiscovery, tc.wantFilteredGroups)
			}
		})
	}
}
