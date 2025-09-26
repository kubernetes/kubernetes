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

package aggregated_test

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestMergedDiscovery(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager("apis")
	localGroup := newAPIGroup("local.example.com", "v1", "local-resource")
	manager.AddGroupVersion(localGroup.Name, localGroup.Versions[0])

	peerProvider := &mockPeerDiscoveryProvider{
		resources: map[string][]apidiscoveryv2.APIGroupDiscovery{
			"peer-server-1": {
				newAPIGroup("peer.example.com", "v2", "peer-resource"),
			},
		},
	}

	testCases := []struct {
		name                      string
		enablePeerMergedDiscovery bool
		acceptHeader              string
		peerProvider              discoveryendpoint.PeerDiscoveryProvider
		wantGroupNames            []string
	}{
		{
			name:                      "Peer merged discovery disabled (should get unmerged)",
			enablePeerMergedDiscovery: false,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList;profile=unmerged",
			wantGroupNames:            []string{"local.example.com"},
		},
		{
			name:                      "Merged request without profile (should default to merged)",
			enablePeerMergedDiscovery: true,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
			peerProvider:              peerProvider,
			wantGroupNames:            []string{"local.example.com", "peer.example.com"},
		},
		{
			name:                      "Merged request with unknown profile (should default to merged)",
			enablePeerMergedDiscovery: true,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList;profile=foo",
			peerProvider:              peerProvider,
			wantGroupNames:            []string{"local.example.com", "peer.example.com"},
		},
		{
			name:                      "Merged request with profile=unmerged (should get unmerged)",
			enablePeerMergedDiscovery: true,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList;profile=unmerged",
			peerProvider:              peerProvider,
			wantGroupNames:            []string{"local.example.com"},
		},
		{
			name:                      "Peer provider nil (should get unmerged)",
			enablePeerMergedDiscovery: true,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
			peerProvider:              nil,
			wantGroupNames:            []string{"local.example.com"},
		},
		{
			name:                      "Multiple peer resources (should return all)",
			enablePeerMergedDiscovery: true,
			acceptHeader:              "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
			peerProvider: &mockPeerDiscoveryProvider{
				resources: map[string][]apidiscoveryv2.APIGroupDiscovery{
					"peer-server-1": {
						newAPIGroup("peer.example.com", "v2", "peer-resource"),
					},
					"peer-server-2": {
						newAPIGroup("peer2.example.com", "v1", "peer2-resource"),
					},
				},
			},
			wantGroupNames: []string{
				"local.example.com", "peer.example.com", "peer2.example.com",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.UnknownVersionInteroperabilityProxy, tc.enablePeerMergedDiscovery)

			peerMergedDiscoveryManager := discoveryendpoint.NewPeerMergedDiscoveryHandler(manager, tc.peerProvider, "apis")
			wrapped := discoveryendpoint.WrapAggregatedDiscoveryToHandler(manager, manager, peerMergedDiscoveryManager)

			resp, _, decoded := fetchPath(wrapped, tc.acceptHeader, "/apis", "")
			require.Equal(t, http.StatusOK, resp.StatusCode)
			require.NotNil(t, decoded)
			gotGroupNames := make([]string, 0, len(decoded.Items))
			for _, group := range decoded.Items {
				gotGroupNames = append(gotGroupNames, group.Name)
			}
			assert.ElementsMatch(t, tc.wantGroupNames, gotGroupNames, "group names mismatch")
		})
	}
}

func TestMergedDiscoveryMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.UnknownVersionInteroperabilityProxy, true)
	manager := discoveryendpoint.NewResourceManager("apis")
	localGroup := newAPIGroup("local.example.com", "v1", "local-resource")
	manager.AddGroupVersion(localGroup.Name, localGroup.Versions[0])
	peerProvider := &mockPeerDiscoveryProvider{
		resources: map[string][]apidiscoveryv2.APIGroupDiscovery{
			"peer-server-1": {
				newAPIGroup("peer.example.com", "v2", "peer-resource"),
			},
		},
	}
	peerMergedDiscoveryManager := discoveryendpoint.NewPeerMergedDiscoveryHandler(manager, peerProvider, "apis")
	wrapped := discoveryendpoint.WrapAggregatedDiscoveryToHandler(manager, manager, peerMergedDiscoveryManager)

	legacyregistry.MustRegister(discoveryendpoint.MergedRequestCounter)
	legacyregistry.MustRegister(discoveryendpoint.UnmergedRequestCounter)

	// Make 4 merged requests.
	fetchPath(wrapped, "application/json", "/apis", "")
	peerMergedDiscoveryManager.InvalidateCache()
	fetchPath(wrapped, "application/json;profile=merged", "/apis", "")
	peerMergedDiscoveryManager.InvalidateCache()
	fetchPath(wrapped, "application/json;profile=foo", "/apis", "")
	peerMergedDiscoveryManager.InvalidateCache()
	fetchPath(wrapped, "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList", "/apis", "")

	// Make 2 unmerged requests.
	fetchPath(wrapped, "application/json;profile=unmerged", "/apis", "")
	peerMergedDiscoveryManager.InvalidateCache()
	fetchPath(wrapped, "application/json;profile=unmerged", "/apis", "")
	peerMergedDiscoveryManager.InvalidateCache()

	mergedCountTotalMetric := `
# HELP aggregator_discovery_merged_count_total [ALPHA] Counter of number of times discovery was merged across all API servers
# TYPE aggregator_discovery_merged_count_total counter
aggregator_discovery_merged_count_total 4
`

	unmergedCountTotalMetric := `
# HELP aggregator_discovery_unmerged_count_total [ALPHA] Counter of number of times unmerged discovery was requested
# TYPE aggregator_discovery_unmerged_count_total counter
aggregator_discovery_unmerged_count_total 2
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(mergedCountTotalMetric), "aggregator_discovery_merged_count_total"); err != nil {
		t.Errorf("unexpected metrics output: %v", err)
	}

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(unmergedCountTotalMetric), "aggregator_discovery_unmerged_count_total"); err != nil {
		t.Errorf("unexpected metrics output: %v", err)
	}
}

func TestMergedDiscovery_ETagHandling(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.UnknownVersionInteroperabilityProxy, true)
	manager := discoveryendpoint.NewResourceManager("apis")
	localGroup := newAPIGroup("local.example.com", "v1", "local-resource")
	manager.AddGroupVersion(localGroup.Name, localGroup.Versions[0])
	peerProvider := &mockPeerDiscoveryProvider{
		resources: map[string][]apidiscoveryv2.APIGroupDiscovery{
			"peer-server-1": {
				newAPIGroup("peer.example.com", "v2", "peer-resource"),
			},
		},
	}

	peerMergedDiscoveryManager := discoveryendpoint.NewPeerMergedDiscoveryHandler(manager, peerProvider, "apis")
	wrapped := discoveryendpoint.WrapAggregatedDiscoveryToHandler(manager, manager, peerMergedDiscoveryManager)

	// First request, get ETag
	rr1 := &responseRecorder{header: make(http.Header)}
	req1, _ := http.NewRequest(http.MethodGet, "/apis", nil)
	req1.Header.Set("Accept", "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList")
	wrapped.ServeHTTP(rr1, req1)
	etag1 := rr1.header.Get("ETag")
	assert.NotEmpty(t, etag1, "ETag should be set on first response")

	// Second request, ETag should be the same (cache hit)
	rr2 := &responseRecorder{header: make(http.Header)}
	req2, _ := http.NewRequest(http.MethodGet, "/apis", nil)
	req2.Header.Set("Accept", "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList")
	wrapped.ServeHTTP(rr2, req2)
	etag2 := rr2.header.Get("ETag")
	assert.Equal(t, etag1, etag2, "ETag should be the same for cached response")

	// Invalidate cache and add a new peer resource
	peerMergedDiscoveryManager.InvalidateCache()
	gvr := schema.GroupVersionResource{Group: "peer.example.com", Version: "v2", Resource: "peer-resource2"}
	peerProvider.addResource("peer-server-3", gvr, newAPIGroup(gvr.Group, gvr.Version, "peer-resource2"))

	// Third request, ETag should change
	rr3 := &responseRecorder{header: make(http.Header)}
	req3, _ := http.NewRequest(http.MethodGet, "/apis", nil)
	req3.Header.Set("Accept", "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList")
	wrapped.ServeHTTP(rr3, req3)
	etag3 := rr3.header.Get("ETag")
	assert.NotEmpty(t, etag3, "ETag should be set after resource change")
	assert.NotEqual(t, etag1, etag3, "ETag should change when resources change")
}

type mockPeerDiscoveryProvider struct {
	resources map[string][]apidiscoveryv2.APIGroupDiscovery
}

// responseRecorder is a minimal http.ResponseWriter for testing status codes and headers
type responseRecorder struct {
	header     http.Header
	statusCode int
	body       strings.Builder
}

func (r *responseRecorder) Header() http.Header         { return r.header }
func (r *responseRecorder) Write(b []byte) (int, error) { return r.body.Write(b) }
func (r *responseRecorder) WriteHeader(statusCode int)  { r.statusCode = statusCode }

func (m *mockPeerDiscoveryProvider) GetPeerResources() map[string][]apidiscoveryv2.APIGroupDiscovery {
	return m.resources
}

func (m *mockPeerDiscoveryProvider) addResource(serverID string, gvr schema.GroupVersionResource, groupDiscovery apidiscoveryv2.APIGroupDiscovery) {
	if m.resources == nil {
		m.resources = make(map[string][]apidiscoveryv2.APIGroupDiscovery)
	}

	if _, exists := m.resources[serverID]; !exists {
		m.resources[serverID] = []apidiscoveryv2.APIGroupDiscovery{}
	}
	m.resources[serverID] = append(m.resources[serverID], groupDiscovery)
}

func newAPIGroup(groupName, versionName, resourceName string) apidiscoveryv2.APIGroupDiscovery {
	return apidiscoveryv2.APIGroupDiscovery{
		ObjectMeta: metav1.ObjectMeta{Name: groupName},
		Versions: []apidiscoveryv2.APIVersionDiscovery{
			{
				Version: versionName,
				Resources: []apidiscoveryv2.APIResourceDiscovery{
					{Resource: resourceName},
				},
			},
		},
	}
}
