/*
Copyright 2023 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/servicecidrs"
	"k8s.io/kubernetes/pkg/controlplane/controller/defaultservicecidr"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestMigrateServiceCIDR validates the steps necessary to migrate a cluster default ServiceCIDR
// including the existing kubernetes.default service.
// 1. start apiserver with --service-cluster-ip-range 192.168.0.0/29"
// 2. create services to use some addresses on the cidr
// 3. create a temporary new ServiceCIDR 10.168.0.0/24 to migrate the cluster to it
// 4. delete the default service CIDR so the allocators ignore it (it will be pending because of the finalizer and having still IPs)
// 5. recreate the services, the allocator should pick the temporary ServiceCIDR
// 6. start the new apiserver with the new ServiceCIDRs on the flags and shutdown the old one
// 7. delete the kubernetes.default service, the new apiserver will recreate it within the new ServiceCIDR
func TestMigrateServiceCIDR(t *testing.T) {
	tCtx := ktesting.Init(t)

	cidr1 := "192.168.0.0/29"
	cidr2 := "10.168.0.0/24"

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s1 := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1beta1=true",
			"--service-cluster-ip-range=" + cidr1,
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
			fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
		},
		etcdOptions)

	client1, err := clientset.NewForConfig(s1.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client1, "test-migrate-service-cidr", t)

	resyncPeriod := 12 * time.Hour
	informers1 := informers.NewSharedInformerFactory(client1, resyncPeriod)
	// ServiceCIDR controller
	go servicecidrs.NewController(
		tCtx,
		informers1.Networking().V1().ServiceCIDRs(),
		informers1.Networking().V1().IPAddresses(),
		client1,
	).Run(tCtx, 5)
	informers1.Start(tCtx.Done())
	informers1.WaitForCacheSync(tCtx.Done())

	// the default serviceCIDR should have a finalizer and ready condition set to true
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		if len(cidr.Finalizers) == 0 {
			return false, nil
		}

		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to false: %v", err)
	}

	svc := func(i int) *v1.Service {
		return &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("svc-%v", i),
			},
			Spec: v1.ServiceSpec{
				Type: v1.ServiceTypeClusterIP,
				Ports: []v1.ServicePort{
					{Port: 80},
				},
			},
		}
	}

	// make 2 services , there will be still 3 free addresses
	for i := 0; i < 2; i++ {
		if _, err := client1.CoreV1().Services(ns.Name).Create(context.TODO(), svc(i), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	// Add a new service CIDR to be able to migrate the apiserver
	if _, err := client1.NetworkingV1().ServiceCIDRs().Create(context.Background(), makeServiceCIDR("migration-cidr", cidr2, ""), metav1.CreateOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// wait ServiceCIDR is ready
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1().ServiceCIDRs().Get(context.TODO(), "migration-cidr", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to false: %v", err)
	}

	// delete the default ServiceCIDR so is no longer used for allocating IPs
	if err := client1.NetworkingV1().ServiceCIDRs().Delete(context.Background(), defaultservicecidr.DefaultServiceCIDRName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// the default serviceCIDR should be pending deletion with Ready condition set to false
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		for _, condition := range cidr.Status.Conditions {
			if condition.Type == networkingv1.ServiceCIDRConditionReady {
				return condition.Status == metav1.ConditionFalse, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to false: %v", err)
	}

	// Migrate the services, delete the existing ones and recreate without specifying the ClusterIP
	services, err := client1.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, svc := range services.Items {
		// skip the default service since is managed by the apiserver
		// and we want the new apiserver with the new service cidr to take over
		if svc.Name == "kubernetes" {
			continue
		}

		if err := client1.CoreV1().Services(svc.Namespace).Delete(context.Background(), svc.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("got unexpected error: %v", err)
		}
		t.Logf("Deleted Service with IP %s", svc.Spec.ClusterIP)

		// wipe the necessary fields so we can recreate the Service
		svc.ResourceVersion = ""
		svc.Spec.ClusterIP = ""
		svc.Spec.ClusterIPs = nil
		svc.Status = v1.ServiceStatus{}
		svc, err := client1.CoreV1().Services(svc.Namespace).Create(context.Background(), &svc, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("got unexpected error: %v", err)
		}
		t.Logf("Created Service with IP %s", svc.Spec.ClusterIP)
		if !cidrContainsIP(cidr2, svc.Spec.ClusterIP) {
			t.Fatalf("Service expected to have an ip in range 10.168.0.0/24, got %s", svc.Spec.ClusterIP)
		}
	}

	// start second apiserver with the new range and new service cidr controller
	s2 := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1beta1=true",
			"--service-cluster-ip-range=" + cidr2,
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
		},
		etcdOptions)
	defer s2.TearDownFn()

	client2, err := clientset.NewForConfig(s2.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer framework.DeleteNamespaceOrDie(client2, ns, t)

	// switch the controller to the new apiserver
	tCtx.Cancel("tearing down ServiceCIDR controller 1")
	s1.TearDownFn()

	// ServiceCIDR controller
	tCtx2 := ktesting.Init(t)
	defer tCtx2.Cancel("tearing down ServiceCIDR controller 2")
	informers2 := informers.NewSharedInformerFactory(client2, resyncPeriod)
	go servicecidrs.NewController(
		tCtx2,
		informers2.Networking().V1().ServiceCIDRs(),
		informers2.Networking().V1().IPAddresses(),
		client2,
	).Run(tCtx2, 5)
	informers2.Start(tCtx2.Done())
	informers2.WaitForCacheSync(tCtx.Done())

	// delete the kubernetes.default service so the old DefaultServiceCIDR can be deleted
	// and the new apiserver can take over
	if err := client2.CoreV1().Services(metav1.NamespaceDefault).Delete(context.Background(), "kubernetes", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	// the default serviceCIDR should  be the new one
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client2.NetworkingV1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}

		if len(cidr.Spec.CIDRs) == 0 {
			t.Logf("No CIDR available")
			return false, nil
		}

		if cidr.Spec.CIDRs[0] != cidr2 {
			t.Logf("CIDR expected %s got %s", cidr2, cidr.Spec.CIDRs[0])
			return false, nil
		}

		if len(cidr.Finalizers) == 0 {
			t.Logf("Expected finalizer to be set")
			return false, nil
		}

		for _, condition := range cidr.Status.Conditions {
			if condition.Type == networkingv1.ServiceCIDRConditionReady {
				t.Logf("Expected Condition %s to be %s", condition.Status, metav1.ConditionTrue)
				return condition.Status == metav1.ConditionTrue, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to true: %v", err)
	}

	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		svc, err := client2.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}

		if svc.Spec.ClusterIP != "10.168.0.1" {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("waiting for default service kubernetes.default to be migrated: %v", err)
	}

	// The temporary ServiceCIDR can be deleted now since the Default ServiceCIDR will cover it
	if err := client2.NetworkingV1().ServiceCIDRs().Delete(context.Background(), "migration-cidr", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// wait ServiceCIDR no longer exist
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		_, err := client2.NetworkingV1().ServiceCIDRs().Get(context.TODO(), "migration-cidr", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("waiting for the migration service cidr to be deleted: %v", err)
	}

}

// TestServiceCIDRMigrationScenarios tests various migration paths for ServiceCIDRs.
func TestServiceCIDRMigrationScenarios(t *testing.T) {
	ipv4CIDRSmall := "10.0.0.0/29" // 6 IPs
	ipv4CIDRBig := "10.1.0.0/16"
	ipv6CIDRSmall := "2001:db8:1::/125" // 6 IPs
	ipv6CIDRBig := "2001:db8:2::/112"

	testCases := []struct {
		name                          string
		initialCIDRs                  []string
		migratedCIDRs                 []string
		preMigrationSvcName           string
		postMigrationSvcName          string
		expectedPostMigrationSvcError bool // Changing the primary IP family and retaining the old allocator
		expectInconsistentState       bool // New Service CIDR configured by flags are not applied
	}{
		// --- No Change ---
		{
			name:                 "IPv4 -> IPv4 (no change)",
			initialCIDRs:         []string{ipv4CIDRSmall},
			migratedCIDRs:        []string{ipv4CIDRSmall},
			preMigrationSvcName:  "svc-pre-v4-v4",
			postMigrationSvcName: "svc-post-v4-v4",
		},
		{
			name:                 "IPv6 -> IPv6 (no change)",
			initialCIDRs:         []string{ipv6CIDRSmall},
			migratedCIDRs:        []string{ipv6CIDRSmall},
			preMigrationSvcName:  "svc-pre-v6-v6",
			postMigrationSvcName: "svc-post-v6-v6",
		},
		{
			name:                 "IPv4,IPv6 -> IPv4,IPv6 (no change)",
			initialCIDRs:         []string{ipv4CIDRSmall, ipv6CIDRSmall},
			migratedCIDRs:        []string{ipv4CIDRSmall, ipv6CIDRSmall},
			preMigrationSvcName:  "svc-pre-v4v6-v4v6",
			postMigrationSvcName: "svc-post-v4v6-v4v6",
		},
		{
			name:                 "IPv6,IPv4 -> IPv6,IPv4 (no change)",
			initialCIDRs:         []string{ipv6CIDRSmall, ipv4CIDRSmall},
			migratedCIDRs:        []string{ipv6CIDRSmall, ipv4CIDRSmall},
			preMigrationSvcName:  "svc-pre-v6v4-v6v4",
			postMigrationSvcName: "svc-post-v6v4-v6v4",
		},
		// --- Valid Upgrades ---
		{
			name:                 "IPv4 -> IPv4,IPv6 (upgrade)",
			initialCIDRs:         []string{ipv4CIDRSmall},
			migratedCIDRs:        []string{ipv4CIDRSmall, ipv6CIDRBig},
			preMigrationSvcName:  "svc-pre-v4-v4v6",
			postMigrationSvcName: "svc-post-v4-v4v6",
		},
		{
			name:                 "IPv6 -> IPv6,IPv4 (upgrade)",
			initialCIDRs:         []string{ipv6CIDRSmall},
			migratedCIDRs:        []string{ipv6CIDRSmall, ipv4CIDRBig},
			preMigrationSvcName:  "svc-pre-v6-v6v4",
			postMigrationSvcName: "svc-post-v6-v6v4",
		},
		// --- Invalid Migrations (Require manual intervention) ---
		{
			name:                          "IPv4,IPv6 -> IPv6,IPv4 (change primary)",
			initialCIDRs:                  []string{ipv4CIDRSmall, ipv6CIDRSmall},
			migratedCIDRs:                 []string{ipv6CIDRSmall, ipv4CIDRSmall},
			preMigrationSvcName:           "svc-pre-v4v6-v6v4",
			postMigrationSvcName:          "svc-post-v4v6-v6v4",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                          "IPv6,IPv4 -> IPv4,IPv6 (change primary)",
			initialCIDRs:                  []string{ipv6CIDRSmall, ipv4CIDRSmall},
			migratedCIDRs:                 []string{ipv4CIDRSmall, ipv6CIDRSmall},
			preMigrationSvcName:           "svc-pre-v6v4-v4v6",
			postMigrationSvcName:          "svc-post-v6v4-v4v6",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                    "IPv4,IPv6 -> IPv4 (downgrade)",
			initialCIDRs:            []string{ipv4CIDRSmall, ipv6CIDRSmall},
			migratedCIDRs:           []string{ipv4CIDRSmall},
			preMigrationSvcName:     "svc-pre-v4v6-v4",
			postMigrationSvcName:    "svc-post-v4v6-v4",
			expectInconsistentState: true,
		},
		{
			name:                          "IPv4,IPv6 -> IPv6 (downgrade)",
			initialCIDRs:                  []string{ipv4CIDRSmall, ipv6CIDRSmall},
			migratedCIDRs:                 []string{ipv6CIDRSmall},
			preMigrationSvcName:           "svc-pre-v4v6-v6",
			postMigrationSvcName:          "svc-post-v4v6-v6",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                          "IPv4 -> IPv6 (change family)",
			initialCIDRs:                  []string{ipv4CIDRSmall},
			migratedCIDRs:                 []string{ipv6CIDRSmall},
			preMigrationSvcName:           "svc-pre-v4-v6",
			postMigrationSvcName:          "svc-post-v4-v6",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                          "IPv6 -> IPv4 (change family)",
			initialCIDRs:                  []string{ipv6CIDRSmall},
			migratedCIDRs:                 []string{ipv4CIDRSmall},
			preMigrationSvcName:           "svc-pre-v6-v4",
			postMigrationSvcName:          "svc-post-v6-v4",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                          "IPv4 -> IPv6,IPv4 (upgrade, change primary)",
			initialCIDRs:                  []string{ipv4CIDRSmall},
			migratedCIDRs:                 []string{ipv6CIDRBig, ipv4CIDRSmall}, // Change primary during upgrade
			preMigrationSvcName:           "svc-pre-v4-v6v4",
			postMigrationSvcName:          "svc-post-v4-v6v4",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
		{
			name:                          "IPv6 -> IPv4,IPv6 (upgrade, change primary)",
			initialCIDRs:                  []string{ipv6CIDRSmall},
			migratedCIDRs:                 []string{ipv4CIDRBig, ipv6CIDRSmall}, // Change primary during upgrade
			preMigrationSvcName:           "svc-pre-v6-v4v6",
			postMigrationSvcName:          "svc-post-v6-v4v6",
			expectedPostMigrationSvcError: true,
			expectInconsistentState:       true,
		},
	}

	for i, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			resyncPeriod := 12 * time.Hour

			// --- Initial Setup ---
			initialFlags := []string{
				"--service-cluster-ip-range=" + strings.Join(tc.initialCIDRs, ","),
				"--advertise-address=" + strings.Split(tc.initialCIDRs[0], "/")[0], // the advertise address MUST match the cluster primary ip family
				"--disable-admission-plugins=ServiceAccount",
				// fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
			}
			t.Logf("Starting API server with CIDRs: %v", tc.initialCIDRs)
			s1 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, initialFlags, etcdOptions)
			client1, err := clientset.NewForConfig(s1.ClientConfig)
			if err != nil {
				s1.TearDownFn()
				t.Fatalf("Failed to create client for initial server: %v", err)
			}

			ns := framework.CreateNamespaceOrDie(client1, fmt.Sprintf("migrate-%d", i), t)

			informers1 := informers.NewSharedInformerFactory(client1, resyncPeriod)
			controllerCtx1, cancelController1 := context.WithCancel(tCtx)
			go servicecidrs.NewController(
				controllerCtx1,
				informers1.Networking().V1().ServiceCIDRs(),
				informers1.Networking().V1().IPAddresses(),
				client1,
			).Run(controllerCtx1, 5)
			informers1.Start(controllerCtx1.Done())
			informers1.WaitForCacheSync(controllerCtx1.Done())

			// Wait for default ServiceCIDR to be ready
			if err := waitForServiceCIDRState(tCtx, client1, tc.initialCIDRs, true); err != nil {
				s1.TearDownFn()
				cancelController1()
				t.Fatalf("Initial default ServiceCIDR did not become ready: %v", err)
			}

			// Create pre-migration service
			preSvc, err := client1.CoreV1().Services(ns.Name).Create(tCtx, makeService(tc.preMigrationSvcName), metav1.CreateOptions{})
			if err != nil {
				s1.TearDownFn()
				cancelController1()
				t.Fatalf("Failed to create pre-migration service: %v", err)
			}
			t.Logf("Pre-migration service %s created with ClusterIPs: %v", preSvc.Name, preSvc.Spec.ClusterIPs)

			// Basic verification of pre-migration service IP
			if len(preSvc.Spec.ClusterIPs) == 0 {
				s1.TearDownFn()
				cancelController1()
				t.Fatalf("Pre-migration service %s has no ClusterIPs", preSvc.Name)
			}
			if !cidrContainsIP(tc.initialCIDRs[0], preSvc.Spec.ClusterIPs[0]) {
				s1.TearDownFn()
				cancelController1()
				t.Fatalf("Pre-migration service %s primary IP %s not in expected range %s", preSvc.Name, preSvc.Spec.ClusterIPs[0], tc.initialCIDRs[0])
			}

			// --- Migration ---
			t.Logf("Shutting down initial API server and controller")
			cancelController1()
			s1.TearDownFn()

			t.Logf("Starting migrated API server with CIDRs: %v", tc.migratedCIDRs)
			migratedFlags := []string{
				"--service-cluster-ip-range=" + strings.Join(tc.migratedCIDRs, ","),
				"--advertise-address=" + strings.Split(tc.migratedCIDRs[0], "/")[0], // the advertise address MUST match the cluster configured primary ip family
				"--disable-admission-plugins=ServiceAccount",
				// fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
			}
			s2 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, migratedFlags, etcdOptions)
			defer s2.TearDownFn() // Ensure cleanup even on test failure

			client2, err := clientset.NewForConfig(s2.ClientConfig)
			if err != nil {
				t.Fatalf("Failed to create client for migrated server: %v", err)
			}
			defer framework.DeleteNamespaceOrDie(client2, ns, t)

			informers2 := informers.NewSharedInformerFactory(client2, resyncPeriod)
			controllerCtx2, cancelController2 := context.WithCancel(tCtx)
			defer cancelController2() // Ensure controller context is cancelled
			go servicecidrs.NewController(
				controllerCtx2,
				informers2.Networking().V1().ServiceCIDRs(),
				informers2.Networking().V1().IPAddresses(),
				client2,
			).Run(controllerCtx2, 5)
			informers2.Start(controllerCtx2.Done())
			informers2.WaitForCacheSync(controllerCtx2.Done())

			// Wait for default ServiceCIDR to reflect migrated state
			// For inconsistent states, we expect to keep existing CIDRs.
			expectedCIDRs := tc.migratedCIDRs
			if tc.expectInconsistentState {
				expectedCIDRs = tc.initialCIDRs
			}
			if err := waitForServiceCIDRState(tCtx, client2, expectedCIDRs, true); err != nil {
				t.Fatalf("Migrated default ServiceCIDR did not reach expected state : %v", err)
			}

			// --- Post-Migration Verification ---

			// Verify pre-migration service still exists and retains its IP(s)
			preSvcMigrated, err := client2.CoreV1().Services(ns.Name).Get(tCtx, tc.preMigrationSvcName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get pre-migration service after migration: %v", err)
			}
			if !reflect.DeepEqual(preSvcMigrated.Spec.ClusterIPs, preSvc.Spec.ClusterIPs) {
				t.Errorf("Pre-migration service %s ClusterIPs changed after migration. Before: %v, After: %v",
					preSvcMigrated.Name, preSvc.Spec.ClusterIPs, preSvcMigrated.Spec.ClusterIPs)
			}
			// Create post-migration service
			postSvc, err := client2.CoreV1().Services(ns.Name).Create(tCtx, makeService(tc.postMigrationSvcName), metav1.CreateOptions{})
			if err != nil && !tc.expectedPostMigrationSvcError {
				t.Fatalf("Failed to create post-migration service: %v", err)
			} else if err == nil && tc.expectedPostMigrationSvcError {
				return
			}

			t.Logf("Post-migration service %s created with ClusterIPs: %v, Families: %v", postSvc.Name, postSvc.Spec.ClusterIPs, postSvc.Spec.IPFamilies)
			// Check if IPs are within the migrated CIDR ranges
			if len(postSvc.Spec.ClusterIPs) > 0 && !cidrContainsIP(expectedCIDRs[0], postSvc.Spec.ClusterIPs[0]) {
				t.Errorf("Post-migration service %s primary IP %s not in expected range %s", postSvc.Name, postSvc.Spec.ClusterIPs[0], expectedCIDRs[0])
			}
		})
	}
}

// waitForServiceCIDRState waits for the named ServiceCIDR to exist, match the expected CIDRs,
// and have the specified Ready condition status.
func waitForServiceCIDRState(ctx context.Context, client clientset.Interface, expectedCIDRs []string, expectReady bool) error {
	pollCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	return wait.PollUntilContextCancel(pollCtx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return true, fmt.Errorf("default ServiceCIDR must exist")
			}
			return false, nil
		}

		// Check CIDRs match
		if !reflect.DeepEqual(cidr.Spec.CIDRs, expectedCIDRs) {
			klog.Infof("Waiting for ServiceCIDR %s CIDRs to match %v, current: %v", defaultservicecidr.DefaultServiceCIDRName, expectedCIDRs, cidr.Spec.CIDRs)
			return false, nil
		}

		// Check Ready condition
		isReady := false
		foundReadyCondition := false
		for _, condition := range cidr.Status.Conditions {
			if condition.Type == networkingv1.ServiceCIDRConditionReady {
				foundReadyCondition = true
				isReady = condition.Status == metav1.ConditionTrue
				break
			}
		}

		if !foundReadyCondition && expectReady {
			klog.Infof("Waiting for ServiceCIDR %s Ready condition to be set...", defaultservicecidr.DefaultServiceCIDRName)
			return false, nil // Ready condition not found yet
		}

		if isReady != expectReady {
			klog.Infof("Waiting for ServiceCIDR %s Ready condition to be %v, current: %v", defaultservicecidr.DefaultServiceCIDRName, expectReady, isReady)
			return false, nil // Ready condition doesn't match expectation
		}

		klog.Infof("ServiceCIDR %s reached desired state (Ready: %v, CIDRs: %v)", defaultservicecidr.DefaultServiceCIDRName, expectReady, expectedCIDRs)
		return true, nil // All conditions met
	})
}
