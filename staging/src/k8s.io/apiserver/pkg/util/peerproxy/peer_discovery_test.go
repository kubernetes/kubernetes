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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	peerproxymetrics "k8s.io/apiserver/pkg/util/peerproxy/metrics"
)

func TestRunPeerDiscoveryCacheSync(t *testing.T) {
	testCases := []struct {
		desc                string
		leases              []*v1.Lease
		labelSelectorString string
		updatedLease        *v1.Lease
		deletedLeaseNames   []string
		wantCache           map[string]PeerDiscoveryCacheEntry
	}{
		{
			desc:                "single remote server",
			labelSelectorString: "apiserver-identity=testserver",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-1",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
				},
			},
			wantCache: map[string]PeerDiscoveryCacheEntry{
				"remote-1": makePeerDiscoveryCacheEntry("testgroup", "v1", "testresources"),
			},
		},
		{
			desc:                "multiple remote servers",
			labelSelectorString: "apiserver-identity=testserver",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-1",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-2",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-2")},
				},
			},
			wantCache: map[string]PeerDiscoveryCacheEntry{
				"remote-1": makePeerDiscoveryCacheEntry("testgroup", "v1", "testresources"),
				"remote-2": makePeerDiscoveryCacheEntry("testgroup", "v1", "testresources"),
			},
		},
		{
			desc:                "lease update",
			labelSelectorString: "apiserver-identity=testserver",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-1",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
				},
			},
			updatedLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "remote-1",
					Labels: map[string]string{"apiserver-identity": "testserver"},
				},
				Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-2")},
			},
			wantCache: map[string]PeerDiscoveryCacheEntry{
				"remote-1": makePeerDiscoveryCacheEntry("testgroup", "v1", "testresources"),
			},
		},
		{
			desc:                "lease deletion",
			labelSelectorString: "apiserver-identity=testserver",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-1",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
				},
			},
			deletedLeaseNames: []string{"remote-1"},
			wantCache:         map[string]PeerDiscoveryCacheEntry{},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			h, fakeReconciler, leaseInformer, fakeClient, fakeInformerFactory := setupPeerProxyHandler(t, tt.labelSelectorString)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Add leases to the fake client and informer.
			for _, lease := range tt.leases {
				_, err := fakeClient.CoordinationV1().Leases("default").Create(ctx, lease, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create lease: %v", err)
				}
				if err = leaseInformer.GetIndexer().Add(lease); err != nil {
					t.Fatalf("failed to create lease: %v", err)
				}
			}

			go fakeInformerFactory.Start(ctx.Done())
			cache.WaitForCacheSync(ctx.Done(), leaseInformer.HasSynced)

			// Create test servers based on leases
			testServers := make(map[string]*httptest.Server)
			for _, lease := range tt.leases {
				testServer := newTestTLSServer(t)
				defer testServer.Close()
				testServers[lease.Name] = testServer
			}

			// Modify the reconciler to return the test server URLs
			for name, server := range testServers {
				fakeReconciler.setEndpoint(name, server.URL[8:])
			}

			go h.RunPeerDiscoveryCacheSync(ctx, 1)
			go h.RunPeerDiscoveryRefilter(ctx)

			// Wait for initial cache update.
			initialCache := map[string]PeerDiscoveryCacheEntry{}
			for _, lease := range tt.leases {
				initialCache[lease.Name] = makePeerDiscoveryCacheEntry("testgroup", "v1", "testresources")
			}
			err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
				select {
				case <-ctx.Done():
					return false, ctx.Err()
				default:
				}
				gotCache := h.gvExclusionManager.GetFilteredPeerDiscoveryCache()
				return assert.ObjectsAreEqual(initialCache, gotCache), nil
			})
			if err != nil {
				t.Errorf("initial cache update failed: %v", err)
			}

			// Update the lease if indicated.
			if tt.updatedLease != nil {
				updatedLease := tt.updatedLease.DeepCopy()
				_, err = fakeClient.CoordinationV1().Leases("default").Update(ctx, updatedLease, metav1.UpdateOptions{})
				if err != nil {
					t.Fatalf("failed to update lease: %v", err)
				}
				if err = leaseInformer.GetIndexer().Update(updatedLease); err != nil {
					t.Fatalf("failed to update lease: %v", err)
				}
			}

			// Delete leases if indicated.
			if len(tt.deletedLeaseNames) > 0 {
				for _, leaseName := range tt.deletedLeaseNames {
					lease, exists, err := leaseInformer.GetIndexer().GetByKey("default/" + leaseName)
					if err != nil {
						t.Fatalf("failed to get lease from indexer: %v", err)
					}
					if !exists {
						t.Fatalf("lease %s not found", leaseName)
					}
					deletedLease := lease.(*v1.Lease)
					err = fakeClient.CoordinationV1().Leases("default").Delete(ctx, deletedLease.Name, metav1.DeleteOptions{})
					if err != nil {
						t.Fatalf("failed to delete lease: %v", err)
					}
					if err = leaseInformer.GetIndexer().Delete(deletedLease); err != nil {
						t.Fatalf("failed to delete lease: %v", err)
					}

				}
			}

			err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
				select {
				case <-ctx.Done():
					return false, ctx.Err()
				default:
				}
				gotCache := h.gvExclusionManager.GetFilteredPeerDiscoveryCache()
				r := assert.ObjectsAreEqual(tt.wantCache, gotCache)
				return r, nil
			})
			if err != nil {
				t.Errorf("cache doesnt match expectation: %v", err)
			}

		})
	}
}

func TestPeerDiscoveryMetrics(t *testing.T) {
	testCases := []struct {
		desc             string
		leases           []*v1.Lease
		peerServerConfig map[string]http.HandlerFunc
		wantMetrics      string
	}{
		{
			desc: "hostport resolution error",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-resolution-error",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-error")},
				},
			},
			// No peer server configured means no endpoint will be registered in the reconciler.
			// This should cause GetEndpoint to fail, triggering the "hostport_resolution" error.
			peerServerConfig: nil,
			wantMetrics: `
				# HELP apiserver_peer_discovery_sync_errors_total [ALPHA] Total number of errors encountered while syncing discovery information from a peer kube-apiserver
				# TYPE apiserver_peer_discovery_sync_errors_total counter
				apiserver_peer_discovery_sync_errors_total{type="hostport_resolution"} 1
			`,
		},
		{
			desc: "fetch discovery error",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "remote-fetch-error",
						Labels: map[string]string{"apiserver-identity": "testserver"},
					},
					Spec: v1.LeaseSpec{HolderIdentity: proto.String("holder-fetch-error")},
				},
			},
			peerServerConfig: map[string]http.HandlerFunc{
				"remote-fetch-error": func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				},
			},
			wantMetrics: `
				# HELP apiserver_peer_discovery_sync_errors_total [ALPHA] Total number of errors encountered while syncing discovery information from a peer kube-apiserver
				# TYPE apiserver_peer_discovery_sync_errors_total counter
				apiserver_peer_discovery_sync_errors_total{type="fetch_discovery"} 2
			`,
		},
	}

	peerproxymetrics.Register()
	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer peerproxymetrics.Reset()
			h, fakeReconciler, leaseInformer, _, _ := setupPeerProxyHandler(t, "apiserver-identity=testserver")

			for _, lease := range tt.leases {
				if err := leaseInformer.GetIndexer().Add(lease); err != nil {
					t.Fatalf("failed to create lease: %v", err)
				}
			}

			for leaseName, handler := range tt.peerServerConfig {
				if handler == nil {
					handler = func(w http.ResponseWriter, r *http.Request) {}
				}
				ts := httptest.NewServer(handler)
				defer ts.Close()
				fakeReconciler.setEndpoint(leaseName, ts.URL[7:])
			}

			// Directly call syncPeerDiscoveryCache
			// We don't care about the return error of syncPeerDiscoveryCache for this test,
			// we only care that metrics are incremented.
			_ = h.syncPeerDiscoveryCache(context.Background())
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.wantMetrics), "apiserver_peer_discovery_sync_errors_total"); err != nil {
				t.Error(err)
			}
		})
	}
}

func setupPeerProxyHandler(t *testing.T, labelSelector string) (*peerProxyHandler, *fakeReconciler, cache.SharedIndexInformer, *fake.Clientset, informers.SharedInformerFactory) {
	localServerID := "local-server"
	fakeClient := fake.NewSimpleClientset()
	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, 0)
	leaseInformer := fakeInformerFactory.Coordination().V1().Leases()

	fakeReconciler := newFakeReconciler()
	negotiatedSerializer := serializer.NewCodecFactory(runtime.NewScheme())
	loopbackConfig := &rest.Config{}
	proxyConfig := &transport.Config{
		TLS: transport.TLSConfig{Insecure: true},
	}

	h, err := NewPeerProxyHandler(
		localServerID,
		labelSelector,
		leaseInformer,
		fakeReconciler,
		negotiatedSerializer,
		loopbackConfig,
		proxyConfig,
	)
	if err != nil {
		t.Fatalf("failed to create handler: %v", err)
	}

	return h, fakeReconciler, leaseInformer.Informer(), fakeClient, fakeInformerFactory
}

// newTestTLSServer creates a new httptest.NewTLSServer that serves discovery endpoints.
func newTestTLSServer(t *testing.T) *httptest.Server {
	return httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/apis" || r.URL.Path == "/api" {
			discoveryResponse := &apidiscoveryv2.APIGroupDiscoveryList{
				Items: []apidiscoveryv2.APIGroupDiscovery{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "testgroup",
						},
						Versions: []apidiscoveryv2.APIVersionDiscovery{
							{
								Version: "v1",
								Resources: []apidiscoveryv2.APIResourceDiscovery{
									{Resource: "testresources"},
								},
							},
						},
					},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(discoveryResponse); err != nil {
				t.Fatalf("error recording discovery response")
			}
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

type fakeReconciler struct {
	endpoints map[string]string
}

func newFakeReconciler() *fakeReconciler {
	return &fakeReconciler{
		endpoints: make(map[string]string),
	}
}

func (f *fakeReconciler) UpdateLease(serverID string, publicIP string, ports []corev1.EndpointPort) error {
	return nil
}

func (f *fakeReconciler) DeleteLease(serverID string) error {
	return nil
}

func (f *fakeReconciler) Destroy() {
}

func (f *fakeReconciler) GetEndpoint(serverID string) (string, error) {
	endpoint, ok := f.endpoints[serverID]
	if !ok {
		return "", fmt.Errorf("endpoint not found for serverID: %s", serverID)
	}
	return endpoint, nil
}

func (f *fakeReconciler) RemoveLease(serverID string) error {
	return nil
}

func (f *fakeReconciler) StopReconciling() {
}

func (f *fakeReconciler) setEndpoint(serverID, endpoint string) {
	f.endpoints[serverID] = endpoint
}

func makePeerDiscoveryCacheEntry(group, version, resource string) PeerDiscoveryCacheEntry {
	return PeerDiscoveryCacheEntry{
		GVRs: map[schema.GroupVersionResource]bool{
			{Group: group, Version: version, Resource: resource}: true,
		},
		GroupDiscovery: []apidiscoveryv2.APIGroupDiscovery{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: group,
				},
				Versions: []apidiscoveryv2.APIVersionDiscovery{
					{
						Version: version,
						Resources: []apidiscoveryv2.APIResourceDiscovery{
							{Resource: resource},
						},
					},
				},
			},
		},
	}
}
