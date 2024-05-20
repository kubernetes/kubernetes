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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRServiceAllocator, true)
	tCtx := ktesting.Init(t)

	cidr1 := "192.168.0.0/29"
	cidr2 := "10.168.0.0/24"

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s1 := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1alpha1=true",
			"--service-cluster-ip-range=" + cidr1,
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
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
		informers1.Networking().V1alpha1().ServiceCIDRs(),
		informers1.Networking().V1alpha1().IPAddresses(),
		client1,
	).Run(tCtx, 5)
	informers1.Start(tCtx.Done())

	// the default serviceCIDR should have a finalizer and ready condition set to true
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1alpha1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
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
	if _, err := client1.NetworkingV1alpha1().ServiceCIDRs().Create(context.Background(), makeServiceCIDR("migration-cidr", cidr2, ""), metav1.CreateOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// wait ServiceCIDR is ready
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1alpha1().ServiceCIDRs().Get(context.TODO(), "migration-cidr", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return isServiceCIDRReady(cidr), nil
	}); err != nil {
		t.Fatalf("waiting for default service cidr ready condition set to false: %v", err)
	}

	// delete the default ServiceCIDR so is no longer used for allocating IPs
	if err := client1.NetworkingV1alpha1().ServiceCIDRs().Delete(context.Background(), defaultservicecidr.DefaultServiceCIDRName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// the default serviceCIDR should be pending deletion with Ready condition set to false
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client1.NetworkingV1alpha1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		for _, condition := range cidr.Status.Conditions {
			if condition.Type == networkingv1alpha1.ServiceCIDRConditionReady {
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
			"--runtime-config=networking.k8s.io/v1alpha1=true",
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
		informers2.Networking().V1alpha1().ServiceCIDRs(),
		informers2.Networking().V1alpha1().IPAddresses(),
		client2,
	).Run(tCtx2, 5)
	informers2.Start(tCtx2.Done())

	// delete the kubernetes.default service so the old DefaultServiceCIDR can be deleted
	// and the new apiserver can take over
	if err := client2.CoreV1().Services(metav1.NamespaceDefault).Delete(context.Background(), "kubernetes", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	// the default serviceCIDR should  be the new one
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		cidr, err := client2.NetworkingV1alpha1().ServiceCIDRs().Get(context.TODO(), defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
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
			if condition.Type == networkingv1alpha1.ServiceCIDRConditionReady {
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
	if err := client2.NetworkingV1alpha1().ServiceCIDRs().Delete(context.Background(), "migration-cidr", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// wait ServiceCIDR no longer exist
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
		_, err := client2.NetworkingV1alpha1().ServiceCIDRs().Get(context.TODO(), "migration-cidr", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("waiting for the migration service cidr to be deleted: %v", err)
	}

}
