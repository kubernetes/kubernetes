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
	"net/netip"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/servicecidrs"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestServiceAllocNewServiceCIDR(t *testing.T) {
	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1=true",
			"--service-cluster-ip-range=192.168.0.0/29",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
			fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
		},
		etcdOptions)
	defer s.TearDownFn()

	client, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	// ServiceCIDR controller
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resyncPeriod := 12 * time.Hour
	informerFactory := informers.NewSharedInformerFactory(client, resyncPeriod)
	go servicecidrs.NewController(
		ctx,
		informerFactory.Networking().V1().ServiceCIDRs(),
		informerFactory.Networking().V1().IPAddresses(),
		client,
	).Run(ctx, 5)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	// /29 = 6 services, kubernetes.default takes the first address
	// make 5 more services to take up all IPs
	for i := 0; i < 5; i++ {
		if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.Background(), makeService(fmt.Sprintf("service-%d", i)), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	// Make another service. It will fail because we're out of cluster IPs
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.Background(), makeService("fail"), metav1.CreateOptions{}); err != nil {
		if !strings.Contains(err.Error(), "range is full") {
			t.Fatalf("unexpected error text: %v", err)
		}
	} else {
		svcs, err := client.CoreV1().Services(metav1.NamespaceAll).List(context.Background(), metav1.ListOptions{})
		if err != nil {
			t.Fatalf("unexpected error getting the services: %v", err)
		}
		allIPs := []string{}
		for _, s := range svcs.Items {
			allIPs = append(allIPs, s.Spec.ClusterIP)
		}
		t.Fatalf("unexpected creation success. The following IPs exist: %#v. It should only be possible to allocate 6 IP addresses in this cluster.\n\n%#v", allIPs, svcs)
	}

	// Add a new service CIDR to be able to create new IPs.
	cidr := makeServiceCIDR("test2", "10.168.0.0/24", "")
	if _, err := client.NetworkingV1().ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}
	// wait ServiceCIDR is ready
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(context.TODO(), cidr.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to false: %v", err)
	}
	// This time creating more Services should work
	for i := 10; i < 150; i++ {
		if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.Background(), makeService(fmt.Sprintf("service-%d", i)), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
}

// A ServiceCIDR can be deleted if there are no orphan IPs or if the existing IPs are contained in other ServiceCIDR
// that is not being deleted.
// The test starts the apiserver with the range "192.168.0.0/29"
// Create Services to fill the range
// Creates a new ServiceCIDR cidr1 with the same range as the one defined in the apiserver
// Deletes cidr1 object will work since its range is covered by the default ServiceCIDR created by the apiserver flags
// Creates a new cidr2 with a different range than cidr1
// Creates a new service so it picks an IPAddress on this range because "192.168.0.0/29" is full at this point
// Creates a new cidr3 that contains cidr2
// Deletes cidr2 since it is covered by cidr3
// Tries to delete cidr3 but is blocked since there is an IPAddress
// Deletes the Service with the IPAddress blocking the deletion
// cidr3 must not exist at this point
func TestServiceCIDRDeletion(t *testing.T) {
	cidr1 := "192.168.0.0/29" // same as the default
	cidr2 := "10.0.0.0/24"    // new range
	cidr3 := "10.0.0.0/16"    // contains cidr2

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1=true",
			"--service-cluster-ip-range=" + cidr1,
			"--advertise-address=172.16.1.1",
			"--disable-admission-plugins=ServiceAccount",
			fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
		},
		etcdOptions)
	defer s.TearDownFn()

	client, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-cidr-deletion", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// ServiceCIDR controller
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resyncPeriod := 12 * time.Hour
	informerFactory := informers.NewSharedInformerFactory(client, resyncPeriod)
	go servicecidrs.NewController(
		ctx,
		informerFactory.Networking().V1().ServiceCIDRs(),
		informerFactory.Networking().V1().IPAddresses(),
		client,
	).Run(ctx, 5)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	// /29 = 6 services, kubernetes.default takes the first address
	// make 5 more services to take up all IPs
	for i := 0; i < 5; i++ {
		if _, err := client.CoreV1().Services(ns.Name).Create(context.Background(), makeService(fmt.Sprintf("service-%d", i)), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	// create a new ServiceCIDRs that overlaps the default one
	_, err = client.NetworkingV1().ServiceCIDRs().Create(ctx, makeServiceCIDR("cidr1", cidr1, ""), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Wait until is ready.
	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr1", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("cidr1 is not ready")
	}
	// we should be able to delete the ServiceCIDR despite it contains IP addresses as it overlaps with the default ServiceCIDR
	err = client.NetworkingV1().ServiceCIDRs().Delete(ctx, "cidr1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		_, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr1", metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("cidr1 has not been deleted")
	}

	// add a new ServiceCIDR with a new range
	_, err = client.NetworkingV1().ServiceCIDRs().Create(ctx, makeServiceCIDR("cidr2", cidr2, ""), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// wait the allocator process the new ServiceCIDR
	// Wait until is ready.
	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr2", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("cidr2 is not ready")
	}
	// create a new Service so it will take a new IP address from the new range
	svc, err := client.CoreV1().Services(ns.Name).Create(context.Background(), makeService("new-cidr-service"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if !cidrContainsIP(cidr2, svc.Spec.ClusterIP) {
		t.Fatalf("Service %s expected to have an IP on range %s, got %s", svc.Name, cidr2, svc.Spec.ClusterIP)
	}

	// add a new ServiceCIDR that overlaps the existing one
	_, err = client.NetworkingV1().ServiceCIDRs().Create(ctx, makeServiceCIDR("cidr3", cidr3, ""), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Wait until is ready.
	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr3", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("cidr3 is not ready")
	}
	// we should be able to delete the ServiceCIDR2 despite it contains IP addresses as it is contained on ServiceCIDR3
	err = client.NetworkingV1().ServiceCIDRs().Delete(ctx, "cidr2", metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		_, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr2", metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("cidr2 has not been deleted")
	}

	// serviceCIDR3 will not be able to be deleted until the IPAddress is removed
	err = client.NetworkingV1().ServiceCIDRs().Delete(ctx, "cidr3", metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Wait until is not ready.
	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		cidr, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr3", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		for _, condition := range cidr.Status.Conditions {
			if condition.Type == networkingv1.ServiceCIDRConditionReady {
				return condition.Status == metav1.ConditionStatus(metav1.ConditionFalse), nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("cidr3 is ready")
	}

	// delete the service blocking the deletion
	if err := client.CoreV1().Services(ns.Name).Delete(context.Background(), "new-cidr-service", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	// cidr3 must not exist
	if err := wait.PollUntilContextTimeout(context.Background(), 250*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		_, err := client.NetworkingV1().ServiceCIDRs().Get(ctx, "cidr3", metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("cidr3 has not been deleted")
	}
}

func makeServiceCIDR(name, primary, secondary string) *networkingv1.ServiceCIDR {
	serviceCIDR := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1.ServiceCIDRSpec{},
	}
	serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, primary)
	if secondary != "" {
		serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, secondary)
	}
	return serviceCIDR
}

func makeService(name string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{
				{Port: 80},
			},
		},
	}
}

// returns true of the ServiceCIDRConditionReady is true
func isServiceCIDRReady(serviceCIDR *networkingv1.ServiceCIDR) bool {
	if serviceCIDR == nil {
		return false
	}

	for _, condition := range serviceCIDR.Status.Conditions {
		if condition.Type == networkingv1.ServiceCIDRConditionReady {
			return condition.Status == metav1.ConditionStatus(metav1.ConditionTrue)
		}
	}

	return false
}

func cidrContainsIP(cidr, ip string) bool {
	prefix := netip.MustParsePrefix(cidr)
	address := netip.MustParseAddr(ip)
	return prefix.Contains(address)
}
