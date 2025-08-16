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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/features"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const mergedDiscoveryPath = "/apis/merged"

func TestMergedDiscoveryHandler_Success(t *testing.T) {
	tests := []struct {
		name             string
		localResources   []testResource
		peerResources    []testPeerResource
		includeServerIDs bool
		wantGroupCount   int
		wantServerIDs    map[schema.GroupVersionResource][]string
	}{
		{
			name: "no server IDs requested",
			localResources: []testResource{
				{"local.example.com", "v1", "test-resource-local"},
			},
			peerResources: []testPeerResource{
				{"server-peer-1", schema.GroupVersionResource{Group: "peer.example.com", Version: "v1", Resource: "test-resource-peer"}},
			},
			includeServerIDs: false,
			wantGroupCount:   2,
			wantServerIDs: map[schema.GroupVersionResource][]string{
				{Group: "local.example.com", Version: "v1", Resource: "test-resource-local"}: nil,
				{Group: "peer.example.com", Version: "v1", Resource: "test-resource-peer"}:   nil,
			},
		},
		{
			name: "with server IDs requested",
			localResources: []testResource{
				{"test.group", "v1", "localresource"},
			},
			peerResources: []testPeerResource{
				{"peer-server-1", schema.GroupVersionResource{Group: "test.group", Version: "v2", Resource: "peerresource-1"}},
				{"peer-server-2", schema.GroupVersionResource{Group: "test.group", Version: "v3", Resource: "peerresource-2"}},
				{"peer-server-3", schema.GroupVersionResource{Group: "test.group", Version: "v4", Resource: "sharedresource"}},
			},
			includeServerIDs: true,
			wantGroupCount:   1,
			wantServerIDs: map[schema.GroupVersionResource][]string{
				{Group: "test.group", Version: "v1", Resource: "localresource"}:  {"test-server"},
				{Group: "test.group", Version: "v2", Resource: "peerresource-1"}: {"peer-server-1"},
				{Group: "test.group", Version: "v3", Resource: "peerresource-2"}: {"peer-server-2"},
				{Group: "test.group", Version: "v4", Resource: "sharedresource"}: {"peer-server-3", "test-server"},
			},
		},
		{
			name: "local only resources",
			localResources: []testResource{
				{"local.example.com", "v1", "test-resource-local"},
			},
			peerResources:    []testPeerResource{},
			includeServerIDs: false,
			wantGroupCount:   1,
		},
		{
			name:           "peer only resources",
			localResources: []testResource{},
			peerResources: []testPeerResource{
				{"server-peer-1", schema.GroupVersionResource{Group: "peer.example.com", Version: "v1", Resource: "test-resource-peer"}},
			},
			includeServerIDs: false,
			wantGroupCount:   1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, true)

			// Setup local manager and peer provider.
			localManager := discoveryendpoint.NewResourceManager("apis")
			peerProvider := &mockPeerDiscoveryProvider{}
			mergedHandler := discoveryendpoint.NewMergedDiscoveryHandler(localManager)
			mergedHandler.SetPeerDiscoveryProvider(peerProvider)

			// Add local resources.
			for _, res := range tc.localResources {
				localGroup := newAPIGroup(res.group, res.version, res.resource)
				localManager.AddGroupVersion(localGroup.Name, localGroup.Versions[0])
			}

			for gvr, serverIDs := range tc.wantServerIDs {
				if contains(serverIDs, "test-server") {
					// Check if this resource is already added as a local resource.
					found := false
					for _, res := range tc.localResources {
						if res.group == gvr.Group && res.version == gvr.Version && res.resource == gvr.Resource {
							found = true
							break
						}
					}
					if !found {
						localManager.AddGroupVersion(gvr.Group, apidiscoveryv2.APIVersionDiscovery{
							Version: gvr.Version,
							Resources: []apidiscoveryv2.APIResourceDiscovery{
								{Resource: gvr.Resource},
							},
						})
					}
				}
			}

			// Add peer resources.
			for _, peerRes := range tc.peerResources {
				peerProvider.addResource(peerRes.serverID, peerRes.gvr, newAPIResource(peerRes.gvr))
			}

			// Build request path.
			path := mergedDiscoveryPath
			if tc.includeServerIDs {
				path += "?includeServerIds=true"
			}

			response, _, decoded := fetchPath(mergedHandler, "application/json", path, "")

			// Verify successful response.
			require.Equal(t, http.StatusOK, response.StatusCode, "unexpected status code")
			require.NotNil(t, decoded, "decoded response should not be nil")

			// Verify group count.
			if tc.wantGroupCount > 0 {
				assert.Len(t, decoded.Items, tc.wantGroupCount, "unexpected number of groups")
			}

			// Test ETag functionality.
			etag := response.Header.Get("ETag")
			require.NotEmpty(t, etag, "ETag should be present")

			// Request with same ETag should return 304.
			response2, body2, _ := fetchPath(mergedHandler, "application/json", path, etag)
			assert.Equal(t, http.StatusNotModified, response2.StatusCode, "should return 304 with same ETag")
			assert.Empty(t, body2, "304 response should have empty body")
		})
	}
}

func TestMergedDiscoveryHandler_ContentNegotiation(t *testing.T) {
	tests := []struct {
		name            string
		accept          string
		wantContentType string
	}{
		{
			name:            "JSON content type",
			accept:          "application/json",
			wantContentType: "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
		},
		{
			name:            "protobuf content type",
			accept:          "application/vnd.kubernetes.protobuf",
			wantContentType: "application/vnd.kubernetes.protobuf;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
		},
		{
			name:            "wildcard accept header",
			accept:          "*/*",
			wantContentType: "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, true)

			// Setup handler with test data.
			localManager := discoveryendpoint.NewResourceManager("apis")
			peerProvider := &mockPeerDiscoveryProvider{}
			mergedHandler := discoveryendpoint.NewMergedDiscoveryHandler(localManager)
			mergedHandler.SetPeerDiscoveryProvider(peerProvider)

			// Add a simple test resource.
			testGroup := newAPIGroup("test.example.com", "v1", "test-resource")
			localManager.AddGroupVersion(testGroup.Name, testGroup.Versions[0])

			// Make request with specified accept header.
			response, _, decoded := fetchPath(mergedHandler, tc.accept, mergedDiscoveryPath, "")

			// Verify successful response.
			require.Equal(t, http.StatusOK, response.StatusCode, "unexpected status code")
			require.NotNil(t, decoded, "decoded response should not be nil")

			// Verify content type.
			assert.Equal(t, tc.wantContentType, response.Header.Get("Content-Type"), "unexpected content type")

			// Verify we can decode the response properly.
			assert.Len(t, decoded.Items, 1, "should have one group")
			assert.Equal(t, "test.example.com", decoded.Items[0].Name, "unexpected group name")
		})
	}
}

func TestMergedDiscoveryHandler_Caching(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, true)

	localManager := discoveryendpoint.NewResourceManager("apis")
	peerProvider := &mockPeerDiscoveryProvider{}
	mergedHandler := discoveryendpoint.NewMergedDiscoveryHandler(localManager)
	mergedHandler.SetPeerDiscoveryProvider(peerProvider)

	// Add resources.
	localGroup := newAPIGroup("local.example.com", "v1", "test-resource-local")
	localManager.AddGroupVersion(localGroup.Name, localGroup.Versions[0])
	peerGVR := schema.GroupVersionResource{Group: "peer.example.com", Version: "v1", Resource: "test-resource-peer"}
	peerProvider.addResource("server-peer-1", peerGVR, newAPIResource(peerGVR))

	// First request without server IDs.
	response1, _, _ := fetchPath(mergedHandler, "application/json", mergedDiscoveryPath, "")
	require.Equal(t, http.StatusOK, response1.StatusCode)
	etag1 := response1.Header.Get("ETag")
	require.NotEmpty(t, etag1)

	// Second request with the same E-tag should return 304 Not Modified.
	response2, body2, _ := fetchPath(mergedHandler, "application/json", mergedDiscoveryPath, etag1)
	assert.Equal(t, http.StatusNotModified, response2.StatusCode)
	assert.Empty(t, body2)

	// First request WITH server IDs.
	response3, _, _ := fetchPath(mergedHandler, "application/json", mergedDiscoveryPath+"?includeServerIds=true", "")
	require.Equal(t, http.StatusOK, response3.StatusCode)
	etag3 := response3.Header.Get("ETag")
	require.NotEmpty(t, etag3)
	assert.NotEqual(t, etag1, etag3, "E-tags should differ when server IDs are included")

	// Second request WITH server IDs and same E-tag should return 304 Not Modified.
	response4, body4, _ := fetchPath(mergedHandler, "application/json", mergedDiscoveryPath+"?includeServerIds=true", etag3)
	assert.Equal(t, http.StatusNotModified, response4.StatusCode)
	assert.Empty(t, body4)
}

func TestMergedDiscoveryHandler_Failures(t *testing.T) {
	tests := []struct {
		name               string
		featureGateEnabled bool
		peerProvider       discoveryendpoint.PeerDiscoveryProvider
		wantStatusCode     int
	}{
		{
			name:               "nil peer provider",
			featureGateEnabled: true,
			peerProvider:       nil,
			wantStatusCode:     http.StatusServiceUnavailable,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			localManager := discoveryendpoint.NewResourceManager("apis")
			mergedHandler := discoveryendpoint.NewMergedDiscoveryHandler(localManager)
			mergedHandler.SetPeerDiscoveryProvider(tc.peerProvider)

			response, _, decoded := fetchPath(mergedHandler, "application/json", mergedDiscoveryPath, "")

			assert.Equal(t, tc.wantStatusCode, response.StatusCode, "unexpected status code")
			assert.Nil(t, decoded, "should not decode response for error status")
		})
	}
}

type testResource struct {
	group    string
	version  string
	resource string
}

type testPeerResource struct {
	serverID string
	gvr      schema.GroupVersionResource
}

type mockPeerDiscoveryProvider struct {
	sync.RWMutex
	resources map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery
}

func (m *mockPeerDiscoveryProvider) GetPeerResources() map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery {
	m.RLock()
	defer m.RUnlock()
	if m.resources == nil {
		return make(map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery)
	}
	return m.resources
}

func (m *mockPeerDiscoveryProvider) addResource(serverID string, gvr schema.GroupVersionResource, resource *apidiscoveryv2.APIResourceDiscovery) {
	m.Lock()
	defer m.Unlock()
	if m.resources == nil {
		m.resources = make(map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery)
	}
	if _, ok := m.resources[serverID]; !ok {
		m.resources[serverID] = make(map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery)
	}
	m.resources[serverID][gvr] = resource
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

func newAPIResource(gvr schema.GroupVersionResource) *apidiscoveryv2.APIResourceDiscovery {
	return &apidiscoveryv2.APIResourceDiscovery{
		Resource: gvr.Resource,
		Scope:    apidiscoveryv2.ScopeCluster,
		Verbs:    []string{"get", "list", "watch"},
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
