/*
Copyright 2017 The Kubernetes Authors.

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

package reconcilers

import (
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/google/uuid"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func init() {
	var scheme = runtime.NewScheme()

	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(scheme.SetVersionPriority(corev1.SchemeGroupVersion))

	codecs = serializer.NewCodecFactory(scheme)
}

var codecs serializer.CodecFactory

type serverInfo struct {
	existingIP     string
	id             string
	ports          []corev1.EndpointPort
	newIP          string
	removeLease    bool
	expectEndpoint string
}

func NewFakePeerEndpointReconciler(t *testing.T, s storage.Interface) peerEndpointLeaseReconciler {
	// use the same base key used by the controlplane, but add a random
	// prefix so we can reuse the etcd instance for subtests independently.
	base := "/" + uuid.New().String() + "/peerserverleases/"
	return peerEndpointLeaseReconciler{serverLeases: &peerEndpointLeases{
		storage:   s,
		destroyFn: func() {},
		baseKey:   base,
		leaseTime: 1 * time.Minute, // avoid the lease to timeout on tests
	}}
}

func (f *peerEndpointLeaseReconciler) SetKeys(servers []serverInfo) error {
	for _, server := range servers {
		if err := f.UpdateLease(server.id, server.existingIP, server.ports); err != nil {
			return err
		}
	}
	return nil
}

func TestPeerEndpointLeaseReconciler(t *testing.T) {
	// enable feature flags
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)

	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	t.Cleanup(func() { server.Terminate(t) })

	newFunc := func() runtime.Object { return &corev1.Endpoints{} }
	newListFunc := func() runtime.Object { return &corev1.EndpointsList{} }
	sc.Codec = apitesting.TestStorageCodec(codecs, corev1.SchemeGroupVersion)

	s, dFunc, err := factory.Create(*sc.ForResource(schema.GroupResource{Resource: "endpoints"}), newFunc, newListFunc, "")
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	t.Cleanup(dFunc)

	tests := []struct {
		testName     string
		servers      []serverInfo
		expectLeases []string
	}{
		{
			testName: "existing IP satisfy",
			servers: []serverInfo{{
				existingIP:     "4.3.2.1",
				id:             "server-1",
				ports:          []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				expectEndpoint: "4.3.2.1:8080",
			}, {
				existingIP:     "1.2.3.4",
				id:             "server-2",
				ports:          []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				expectEndpoint: "1.2.3.4:8080",
			}},
			expectLeases: []string{"4.3.2.1", "1.2.3.4"},
		},
		{
			testName: "existing IP + new IP = should return the new IP",
			servers: []serverInfo{{
				existingIP:     "4.3.2.2",
				id:             "server-1",
				ports:          []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				newIP:          "4.3.2.1",
				expectEndpoint: "4.3.2.1:8080",
			}, {
				existingIP:     "1.2.3.4",
				id:             "server-2",
				ports:          []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				newIP:          "1.1.1.1",
				expectEndpoint: "1.1.1.1:8080",
			}},
			expectLeases: []string{"4.3.2.1", "1.1.1.1"},
		},
		{
			testName: "no existing IP, should return new IP",
			servers: []serverInfo{{
				id:             "server-1",
				ports:          []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				newIP:          "1.2.3.4",
				expectEndpoint: "1.2.3.4:8080",
			}},
			expectLeases: []string{"1.2.3.4"},
		},
	}
	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			fakeReconciler := NewFakePeerEndpointReconciler(t, s)
			err := fakeReconciler.SetKeys(test.servers)
			if err != nil {
				t.Errorf("unexpected error creating keys: %v", err)
			}

			for _, server := range test.servers {
				if server.newIP != "" {
					err = fakeReconciler.UpdateLease(server.id, server.newIP, server.ports)
					if err != nil {
						t.Errorf("unexpected error reconciling: %v", err)
					}
				}
			}

			leases, err := fakeReconciler.ListLeases()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			// sort for comparison
			sort.Strings(leases)
			sort.Strings(test.expectLeases)
			if !reflect.DeepEqual(leases, test.expectLeases) {
				t.Errorf("expected %v got: %v", test.expectLeases, leases)
			}

			for _, server := range test.servers {
				endpoint, err := fakeReconciler.GetLease(server.id)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if endpoint != server.expectEndpoint {
					t.Errorf("expected %v got: %v", server.expectEndpoint, endpoint)
				}
			}
		})
	}
}

func TestPeerLeaseRemoveEndpoints(t *testing.T) {
	// enable feature flags
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)

	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	t.Cleanup(func() { server.Terminate(t) })

	newFunc := func() runtime.Object { return &corev1.Endpoints{} }
	newListFunc := func() runtime.Object { return &corev1.EndpointsList{} }
	sc.Codec = apitesting.TestStorageCodec(codecs, corev1.SchemeGroupVersion)

	s, dFunc, err := factory.Create(*sc.ForResource(schema.GroupResource{Resource: "pods"}), newFunc, newListFunc, "")
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	t.Cleanup(dFunc)

	stopTests := []struct {
		testName         string
		servers          []serverInfo
		expectLeases     []string
		apiServerStartup bool
	}{
		{
			testName: "successful remove previous endpoints before apiserver starts",
			servers: []serverInfo{
				{
					existingIP:  "1.2.3.4",
					id:          "test-server-1",
					ports:       []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					removeLease: true,
				},
				{
					existingIP: "2.4.6.8",
					id:         "test-server-2",
					ports:      []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			expectLeases:     []string{"2.4.6.8"},
			apiServerStartup: true,
		},
		{
			testName: "stop reconciling with new IP not in existing ip list",
			servers: []serverInfo{{
				existingIP: "1.2.3.4",
				newIP:      "4.6.8.9",
				id:         "test-server-1",
				ports:      []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			},
				{
					existingIP:  "2.4.6.8",
					id:          "test-server-2",
					ports:       []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					removeLease: true,
				}},
			expectLeases: []string{"1.2.3.4"},
		},
	}
	for _, test := range stopTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeReconciler := NewFakePeerEndpointReconciler(t, s)
			err := fakeReconciler.SetKeys(test.servers)
			if err != nil {
				t.Errorf("unexpected error creating keys: %v", err)
			}
			if !test.apiServerStartup {
				fakeReconciler.StopReconciling()
			}
			for _, server := range test.servers {
				if server.removeLease {
					err = fakeReconciler.RemoveLease(server.id)
					// if the ip is not on the endpoints, it must return an storage error and stop reconciling
					if err != nil {
						t.Errorf("unexpected error reconciling: %v", err)
					}
				}
			}

			leases, err := fakeReconciler.ListLeases()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			// sort for comparison
			sort.Strings(leases)
			sort.Strings(test.expectLeases)
			if !reflect.DeepEqual(leases, test.expectLeases) {
				t.Errorf("expected %v got: %v", test.expectLeases, leases)
			}
		})
	}
}
