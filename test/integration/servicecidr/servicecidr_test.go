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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/servicecidrs"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestServiceAllocNewServiceCIDR(t *testing.T) {
	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--service-cluster-ip-range=192.168.0.0/29",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
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
	for i := range 5 {
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
			"--service-cluster-ip-range=" + cidr1,
			"--advertise-address=172.16.1.1",
			"--disable-admission-plugins=ServiceAccount",
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
	for i := range 5 {
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

// TestValidationAdmissionPolicyServiceCIDR tests that a ValidatingAdmissionPolicy
// can be used to prevent updates to the default ServiceCIDR from users other than the
// apiserver, and that the apiserver can update it on startup.
func TestValidationAdmissionPolicyServiceCIDR(t *testing.T) {
	// This test restarts the apiserver, so it cannot run in parallel.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	etcdOptions := framework.SharedEtcd()
	// 1. Start an apiserver as a single stack.
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1=true",
			"--service-cluster-ip-range=192.168.0.0/24",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
			"--enable-admission-plugins=ValidatingAdmissionPolicy",
		},
		etcdOptions)

	clientset, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		s.TearDownFn()
		t.Fatalf("Failed to create clientset: %v", err)
	}

	// fake user
	newCfg := *s.ClientConfig
	newCfg.Impersonate = rest.ImpersonationConfig{
		UserName: "fake-admin",
		Groups:   []string{"system:authenticated"},
	}
	kc, err := kubernetes.NewForConfig(&newCfg)
	if err != nil {
		t.Fatalf("Unexpected error creating kubernetes client impersonating %q", newCfg.Impersonate.UserName)
	}

	// 2. Install a validationadmissionpolicy that only allows the user apiserver to update the default servicecidr to dual stack
	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "deny-non-apiserver-servicecidr-updates"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistrationv1.RuleWithOperations{
							Operations: []admissionregistrationv1.OperationType{"CREATE", "UPDATE"},
							Rule:       admissionregistrationv1.Rule{APIGroups: []string{"networking.k8s.io"}, APIVersions: []string{"v1"}, Resources: []string{"servicecidrs"}},
						},
					},
				},
			},
			Validations: []admissionregistrationv1.Validation{{
				// only allow to CREATE or UPDATE the default kubernetes ServiceCIDR to the apiserver
				Expression: "request.userInfo.username == 'system:apiserver' && 'system:masters' in request.userInfo.groups",
				Message:    "only apiserver can update and create servicecidrs",
			}, {
				Expression: "object.metadata.name == 'kubernetes'",
				Message:    "only allow changes on the default servicecidr",
			}},
		},
	}
	policy, err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
	if err != nil {
		s.TearDownFn()
		t.Fatalf("Failed to create policy: %v", err)
	}
	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "deny-non-apiserver-servicecidr-updates-binding"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: policy.Name,
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
			MatchResources: &admissionregistrationv1.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{}, // Match cluster-scoped resources
			},
		},
	}
	_, err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
	if err != nil {
		s.TearDownFn()
		t.Fatalf("Failed to create binding: %v", err)
	}

	// Wait for policy and binding to become active.
	time.Sleep(2 * time.Second)

	// 3. Try to update the defaultservicecidr to dual stack from the test and check that fails
	defaultCIDR, err := clientset.NetworkingV1().ServiceCIDRs().Get(ctx, "kubernetes", metav1.GetOptions{})
	if err != nil {
		s.TearDownFn()
		t.Fatalf("Failed to get default ServiceCIDR: %v", err)
	}

	defaultCIDR.Spec.CIDRs = append(defaultCIDR.Spec.CIDRs, "2001:db8::/112")
	_, err = kc.NetworkingV1().ServiceCIDRs().Update(ctx, defaultCIDR, metav1.UpdateOptions{})
	if err == nil {
		s.TearDownFn()
		t.Fatal("Expected an error updating default ServiceCIDR but got none")
	}
	if !strings.Contains(err.Error(), "only apiserver can update and create servicecidrs") {
		s.TearDownFn()
		t.Fatalf("Expected error to contain 'only apiserver can update and create servicecidrs', but got: %v", err)
	}

	// 4. add a new ServiceCIDR
	_, err = clientset.NetworkingV1().ServiceCIDRs().Create(ctx, makeServiceCIDR("cidrnew", "192.168.0.0/28", ""), metav1.CreateOptions{})
	if err == nil {
		s.TearDownFn()
		t.Fatal("Expected an error creating a new ServiceCIDR but got none")
	}
	if !strings.Contains(err.Error(), "only allow changes on the default servicecidr") {
		s.TearDownFn()
		t.Fatalf("Expected error to contain 'only allow changes on the default servicecidr', but got: %v", err)
	}

	// 5. Stop current apiserver and start a new one with dual stack cidrs in the flags
	s.TearDownFn()
	apiServerOptionsDual := kubeapiservertesting.NewDefaultTestServerOptions()
	sDual := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptionsDual,
		[]string{
			"--runtime-config=networking.k8s.io/v1=true",
			"--service-cluster-ip-range=192.168.0.0/24,2001:db8::/112",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
			"--enable-admission-plugins=ValidatingAdmissionPolicy",
		},
		etcdOptions)
	defer sDual.TearDownFn()

	clientsetDual, err := kubernetes.NewForConfig(sDual.ClientConfig)
	if err != nil {
		t.Fatalf("Failed to create clientset for dual-stack server: %v", err)
	}

	// 6. Validate that the default service cidr has been updated
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		cidr, err := clientsetDual.NetworkingV1().ServiceCIDRs().Get(ctx, "kubernetes", metav1.GetOptions{})
		if err != nil {
			// The API server may not be fully ready, so retry on error.
			t.Logf("failed to get service cidr, retrying: %v", err)
			return false, nil
		}
		if len(cidr.Spec.CIDRs) == 2 {
			return true, nil
		}
		t.Logf("service cidr not updated yet, have: %v", cidr.Spec.CIDRs)
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed to wait for default ServiceCIDR to be updated to dual-stack: %v", err)
	}

	updatedDefaultCIDR, err := clientsetDual.NetworkingV1().ServiceCIDRs().Get(ctx, "kubernetes", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get updated default ServiceCIDR: %v", err)
	}

	expectedCIDRs := []string{"192.168.0.0/24", "2001:db8::/112"}
	sort.Strings(updatedDefaultCIDR.Spec.CIDRs)
	sort.Strings(expectedCIDRs)
	if !reflect.DeepEqual(updatedDefaultCIDR.Spec.CIDRs, expectedCIDRs) {
		t.Errorf("Expected ServiceCIDR to be %v, but got %v", expectedCIDRs, updatedDefaultCIDR.Spec.CIDRs)
	}
}
