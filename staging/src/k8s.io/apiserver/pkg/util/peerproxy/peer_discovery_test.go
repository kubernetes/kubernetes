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

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRunPeerDiscoveryCacheSync(t *testing.T) {
	localServerID := "local-server"

	testCases := []struct {
		desc                string
		leases              []*v1.Lease
		labelSelectorString string
		updatedLease        *v1.Lease
		deletedLeaseNames   []string
		wantCache           map[string]map[schema.GroupVersionResource]bool
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
			wantCache: map[string]map[schema.GroupVersionResource]bool{
				"remote-1": {
					{Group: "testgroup", Version: "v1", Resource: "testresources"}: true,
				},
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
			wantCache: map[string]map[schema.GroupVersionResource]bool{
				"remote-1": {
					{Group: "testgroup", Version: "v1", Resource: "testresources"}: true,
				},
				"remote-2": {
					{Group: "testgroup", Version: "v1", Resource: "testresources"}: true,
				},
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
			wantCache: map[string]map[schema.GroupVersionResource]bool{
				"remote-1": {
					{Group: "testgroup", Version: "v1", Resource: "testresources"}: true,
				},
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
			wantCache:         map[string]map[schema.GroupVersionResource]bool{},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
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
				tt.labelSelectorString,
				leaseInformer,
				fakeReconciler,
				negotiatedSerializer,
				loopbackConfig,
				proxyConfig,
			)
			if err != nil {
				t.Fatalf("failed to create handler: %v", err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Add leases to the fake client and informer.
			for _, lease := range tt.leases {
				_, err := fakeClient.CoordinationV1().Leases("default").Create(ctx, lease, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create lease: %v", err)
				}
				if err = leaseInformer.Informer().GetIndexer().Add(lease); err != nil {
					t.Fatalf("failed to create lease: %v", err)
				}
			}

			go fakeInformerFactory.Start(ctx.Done())
			cache.WaitForCacheSync(ctx.Done(), leaseInformer.Informer().HasSynced)

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

			// Wait for initial cache update.
			initialCache := map[string]map[schema.GroupVersionResource]bool{}
			for _, lease := range tt.leases {
				initialCache[lease.Name] = map[schema.GroupVersionResource]bool{
					{Group: "testgroup", Version: "v1", Resource: "testresources"}: true,
				}
			}
			err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
				select {
				case <-ctx.Done():
					return false, ctx.Err()
				default:
				}
				gotCache := h.peerDiscoveryInfoCache.Load()
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
				if err = leaseInformer.Informer().GetIndexer().Update(updatedLease); err != nil {
					t.Fatalf("failed to update lease: %v", err)
				}
			}

			// Delete leases if indicated.
			if len(tt.deletedLeaseNames) > 0 {
				for _, leaseName := range tt.deletedLeaseNames {
					lease, exists, err := leaseInformer.Informer().GetIndexer().GetByKey("default/" + leaseName)
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
					if err = leaseInformer.Informer().GetIndexer().Delete(deletedLease); err != nil {
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
				gotCache := h.peerDiscoveryInfoCache.Load()
				return assert.ObjectsAreEqual(tt.wantCache, gotCache), nil
			})
			if err != nil {
				t.Errorf("cache doesnt match expectation: %v", err)
			}

		})
	}
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
