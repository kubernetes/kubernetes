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

package servicecidr

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/test/integration/framework"
)

// populateEtcdForRepairTest populates etcd with namespaces, services, and ServiceCIDR
// to simulate an existing v1.32 cluster that needs repair during upgrade to v1.33.
func populateEtcdForRepairTest(t *testing.T, etcdOptions *storagebackend.Config, apiServerOptions *kubeapiservertesting.TestServerInstanceOptions, numNamespaces int) string {
	t.Logf("Populating etcd with %d namespaces and services (simulating v1.32 cluster)", numNamespaces)

	// We need a temporary server just to get the etcd client
	tempServer := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--service-cluster-ip-range=10.0.0.0/24",
			"--advertise-address=10.1.1.1",
		},
		etcdOptions)

	// Create test namespace directly in etcd
	namespace := "test-repair-race"
	ns := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              namespace,
			CreationTimestamp: metav1.Now(),
			UID:               types.UID("test-namespace-uid"),
		},
	}
	nsJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), ns)
	if err != nil {
		t.Fatalf("Failed to encode namespace: %v", err)
	}
	nsKey := "/" + etcdOptions.Prefix + "/namespaces/" + namespace
	if _, err := tempServer.EtcdClient.Put(context.Background(), nsKey, string(nsJSON)); err != nil {
		t.Fatalf("Failed to store namespace in etcd: %v", err)
	}
	t.Logf("Created namespace %s in etcd", namespace)

	// Create many namespaces directly in etcd to simulate a large cluster
	// This causes the namespace informer to take significant time to sync on startup
	for i := range numNamespaces {
		ns := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:              fmt.Sprintf("bulk-ns-%d", i),
				CreationTimestamp: metav1.Now(),
				UID:               types.UID(fmt.Sprintf("bulk-ns-uid-%d", i)),
			},
		}
		nsJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), ns)
		if err != nil {
			t.Fatalf("Failed to encode namespace: %v", err)
		}
		nsKey := "/" + etcdOptions.Prefix + "/namespaces/" + ns.Name
		if _, err := tempServer.EtcdClient.Put(context.Background(), nsKey, string(nsJSON)); err != nil {
			t.Fatalf("Failed to store namespace in etcd: %v", err)
		}
		if numNamespaces > 1000 && i%1000 == 0 && i > 0 {
			t.Logf("Created %d/%d bulk namespaces in etcd", i, numNamespaces)
		}
	}
	t.Logf("Created %d additional namespaces in etcd to simulate large cluster", numNamespaces)

	// Create services directly in etcd (simulating existing services from v1.32)
	// These services will need IPAddress objects created by the repair controller
	numServices := 5
	for i := 1; i <= numServices; i++ {
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:              fmt.Sprintf("test-service-%d", i),
				Namespace:         namespace,
				CreationTimestamp: metav1.Now(),
				UID:               types.UID(fmt.Sprintf("test-svc-uid-%d", i)),
			},
			Spec: v1.ServiceSpec{
				Type:      v1.ServiceTypeClusterIP,
				ClusterIP: fmt.Sprintf("10.0.0.%d", i+10),
				Ports: []v1.ServicePort{
					{
						Name:     "http",
						Port:     80,
						Protocol: v1.ProtocolTCP,
					},
				},
			},
		}
		svcJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), svc)
		if err != nil {
			t.Fatalf("Failed to encode service: %v", err)
		}
		svcKey := "/" + etcdOptions.Prefix + "/services/specs/" + namespace + "/" + svc.Name
		if _, err := tempServer.EtcdClient.Put(context.Background(), svcKey, string(svcJSON)); err != nil {
			t.Fatalf("Failed to store service in etcd: %v", err)
		}
		t.Logf("Created service %s with ClusterIP %s in etcd", svc.Name, svc.Spec.ClusterIP)
	}

	// Create ServiceCIDR in etcd - this is required for the repair controller to run
	serviceCIDR := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "kubernetes",
		},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{"10.0.0.0/24"},
		},
	}
	serviceCIDRJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(networkingv1.SchemeGroupVersion), serviceCIDR)
	if err != nil {
		t.Fatalf("Failed to encode ServiceCIDR: %v", err)
	}
	serviceCIDRKey := "/" + etcdOptions.Prefix + "/servicecidrs/" + serviceCIDR.Name
	if _, err := tempServer.EtcdClient.Put(context.Background(), serviceCIDRKey, string(serviceCIDRJSON)); err != nil {
		t.Fatalf("Failed to store ServiceCIDR in etcd: %v", err)
	}
	t.Logf("Created ServiceCIDR in etcd")

	// Tear down the temporary server
	tempServer.TearDownFn()
	t.Logf("Etcd population complete: %d namespaces, %d services, and ServiceCIDR", numNamespaces+1, numServices)

	return namespace
}

// TestServiceIPRepairRaceCondition tests the race condition with real namespace load.
//
// This test reproduces the race condition described in https://github.com/kubernetes/kubernetes/issues/136288
// using a real large number of namespaces (330000) without artificial delays.
//
// This test simulates the exact scenario where:
//  1. A cluster is upgraded from v1.32 to v1.33 with MultiCIDRServiceAllocator enabled
//  2. Services exist in etcd that need IPAddress objects to be created (repair scenario)
//  3. Many namespaces (330000) exist in etcd, causing the namespace informer to take time to sync naturally
//  4. During apiserver startup, the repair controller (in PostStartHook) tries to create IPAddress objects
//  5. Built-in admission plugins (like NamespaceLifecycle) reject these requests with "not yet ready"
//     because the namespace informer hasn't synced yet
//  6. The PostStartHook fails, preventing the apiserver from becoming ready
//
// This test takes a long time to run (~30 minutes to populate etcd) but reproduces the issue with realistic load.
func TestServiceIPRepairRaceCondition(t *testing.T) {
	t.Skip("reproducer for https://issues.k8s.io/136288")

	// Setup etcd
	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()

	// Phase 1: Populate etcd directly to simulate an existing v1.32 cluster
	// We write directly to etcd to avoid starting an apiserver, making the test faster
	// Use 330000 namespaces to reproduce the issue with real load (no artificial delay)
	t.Logf("Phase 1: Populating etcd with namespaces and services (simulating v1.32 cluster)")
	populateEtcdForRepairTest(t, etcdOptions, apiServerOptions, 330000)

	// Phase 2: Restart apiserver with MultiCIDRServiceAllocator enabled
	// This simulates the v1.32 -> v1.33 upgrade scenario
	t.Logf("Phase 2: Starting apiserver with MultiCIDRServiceAllocator enabled (simulating v1.33 upgrade)")

	// Start the apiserver with MultiCIDRServiceAllocator enabled
	// The repair controller will run in the PostStartHook and attempt to create IPAddress objects
	// Due to the many namespaces, the namespace informer will take time to sync
	// Admission plugins that depend on the namespace informer will reject requests with "not yet ready"
	testServer := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1beta1=true",
			"--service-cluster-ip-range=10.0.0.0/24",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
			// Enable verbose logging to see repair controller logs
			"-v=5",
		},
		etcdOptions)
	defer testServer.TearDownFn()
}
