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
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilversion "k8s.io/apiserver/pkg/util/version"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
)

func TestServiceAlloc(t *testing.T) {
	// Create an IPv4 single stack control-plane
	serviceCIDR := "192.168.0.0/29"

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
		},
	})
	defer tearDownFn()

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

	// Wait until the default "kubernetes" service is created.
	if err := wait.Poll(250*time.Millisecond, time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return !apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("creating kubernetes service timed out")
	}

	// make 5 more services to take up all IPs
	for i := 0; i < 5; i++ {
		if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(i), metav1.CreateOptions{}); err != nil {
			t.Error(err)
		}
	}

	// Make another service. It will fail because we're out of cluster IPs
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(8), metav1.CreateOptions{}); err != nil {
		if !strings.Contains(err.Error(), "range is full") {
			t.Errorf("unexpected error text: %v", err)
		}
	} else {
		svcs, err := client.CoreV1().Services(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			t.Fatalf("unexpected success, and error getting the services: %v", err)
		}
		allIPs := []string{}
		for _, s := range svcs.Items {
			allIPs = append(allIPs, s.Spec.ClusterIP)
		}
		t.Fatalf("unexpected creation success. The following IPs exist: %#v. It should only be possible to allocate 2 IP addresses in this cluster.\n\n%#v", allIPs, svcs)
	}

	// Delete the first service.
	if err := client.CoreV1().Services(metav1.NamespaceDefault).Delete(context.TODO(), svc(1).ObjectMeta.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}

	// This time creating the second service should work.
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(8), metav1.CreateOptions{}); err != nil {
		t.Fatalf("got unexpected error: %v", err)
	}
}

func TestServiceAllocIPAddress(t *testing.T) {
	// Create an IPv6 single stack control-plane with a large range
	serviceCIDR := "2001:db8::/64"
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRServiceAllocator, true)

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
			opts.GenericServerRunOptions.AdvertiseAddress = netutils.ParseIPSloppy("2001:db8::10")
			opts.GenericServerRunOptions.EffectiveVersion = utilversion.NewEffectiveVersion("1.31")
			opts.APIEnablement.RuntimeConfig.Set("networking.k8s.io/v1alpha1=true")
		},
	})
	defer tearDownFn()

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

	// Wait until the default "kubernetes" service is created.
	if err := wait.Poll(250*time.Millisecond, time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(tCtx, "kubernetes", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return !apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("creating kubernetes service timed out")
	}

	// create 5 random services and check that the Services have an IP associated
	for i := 0; i < 5; i++ {
		svc, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(tCtx, svc(i), metav1.CreateOptions{})
		if err != nil {
			t.Error(err)
		}
		_, err = client.NetworkingV1alpha1().IPAddresses().Get(tCtx, svc.Spec.ClusterIP, metav1.GetOptions{})
		if err != nil {
			t.Error(err)
		}
	}

	// Make a service in the top of the range to verify we can allocate in the whole range
	// because it is not reasonable to create 2^64 services
	lastSvc := svc(8)
	lastSvc.Spec.ClusterIP = "2001:db8::ffff:ffff:ffff:ffff"
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), lastSvc, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error text: %v", err)
	}
	_, err := client.NetworkingV1alpha1().IPAddresses().Get(context.TODO(), lastSvc.Spec.ClusterIP, metav1.GetOptions{})
	if err != nil {
		t.Error(err)
	}

}

func TestMigrateService(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRServiceAllocator, true)
	//logs.GlogSetter("7")

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s := kubeapiservertesting.StartTestServerOrDie(t,
		apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1alpha1=true",
			"--service-cluster-ip-range=10.0.0.0/24",
			"--advertise-address=10.1.1.1",
			"--disable-admission-plugins=ServiceAccount",
		},
		etcdOptions)
	defer s.TearDownFn()
	serviceName := "test-old-service"
	namespace := "old-service-ns"
	// Create a service and store it in etcd
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         namespace,
			CreationTimestamp: metav1.Now(),
			UID:               "08675309-9376-9376-9376-086753099999",
		},
		Spec: v1.ServiceSpec{
			ClusterIP: "10.0.0.11",
			Ports: []v1.ServicePort{
				{
					Name: "test-port",
					Port: 81,
				},
			},
		},
	}
	svcJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), svc)
	if err != nil {
		t.Fatalf("Failed creating service JSON: %v", err)
	}
	key := "/" + etcdOptions.Prefix + "/services/specs/" + namespace + "/" + serviceName
	if _, err := s.EtcdClient.Put(context.Background(), key, string(svcJSON)); err != nil {
		t.Error(err)
	}
	t.Logf("Service stored in etcd %v", string(svcJSON))

	kubeclient, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	ns := framework.CreateNamespaceOrDie(kubeclient, namespace, t)
	defer framework.DeleteNamespaceOrDie(kubeclient, ns, t)

	// TODO: Understand why the Service can not be obtained with a List, it only works if we trigger an event
	// by updating the Service.
	_, err = kubeclient.CoreV1().Services(namespace).Update(context.Background(), svc, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}

	err = wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		// The repair loop must create the IP address associated
		_, err = kubeclient.NetworkingV1alpha1().IPAddresses().Get(context.TODO(), svc.Spec.ClusterIP, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Error(err)
	}

}

func TestSkewedAllocators(t *testing.T) {
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

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	// s1 uses IPAddress allocator
	s1 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1alpha1=true",
			"--service-cluster-ip-range=10.0.0.0/24",
			"--disable-admission-plugins=ServiceAccount",
			fmt.Sprintf("--feature-gates=%s=true", features.MultiCIDRServiceAllocator)},
		etcdOptions)
	defer s1.TearDownFn()

	kubeclient1, err := kubernetes.NewForConfig(s1.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// create 5 random services and check that the Services have an IP associated
	for i := 0; i < 5; i++ {
		service, err := kubeclient1.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(i), metav1.CreateOptions{})
		if err != nil {
			t.Error(err)
			continue
		}
		_, err = kubeclient1.NetworkingV1alpha1().IPAddresses().Get(context.TODO(), service.Spec.ClusterIP, metav1.GetOptions{})
		if err != nil {
			t.Error(err)
		}
	}

	// s2 uses bitmap allocator
	s2 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1alpha1=false",
			"--service-cluster-ip-range=10.0.0.0/24",
			"--disable-admission-plugins=ServiceAccount",
			fmt.Sprintf("--feature-gates=%s=false", features.MultiCIDRServiceAllocator)},
		etcdOptions)
	defer s2.TearDownFn()

	kubeclient2, err := kubernetes.NewForConfig(s2.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// create 5 random services and check that the Services have an IP associated
	for i := 5; i < 10; i++ {
		service, err := kubeclient2.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(i), metav1.CreateOptions{})
		if err != nil {
			t.Error(err)
		}

		err = wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
			// The repair loop must create the IP address associated
			_, err = kubeclient1.NetworkingV1alpha1().IPAddresses().Get(context.TODO(), service.Spec.ClusterIP, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			t.Error(err)
		}

	}

}

func TestFlagsIPAllocator(t *testing.T) {
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

	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	// s1 uses IPAddress allocator
	s1 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--runtime-config=networking.k8s.io/v1alpha1=true",
			"--service-cluster-ip-range=10.0.0.0/24",
			fmt.Sprintf("--feature-gates=%s=true", features.MultiCIDRServiceAllocator)},
		etcdOptions)
	defer s1.TearDownFn()

	kubeclient1, err := kubernetes.NewForConfig(s1.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// create 5 random services and check that the Services have an IP associated
	for i := 0; i < 5; i++ {
		service, err := kubeclient1.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc(i), metav1.CreateOptions{})
		if err != nil {
			t.Error(err)
			continue
		}
		_, err = kubeclient1.NetworkingV1alpha1().IPAddresses().Get(context.TODO(), service.Spec.ClusterIP, metav1.GetOptions{})
		if err != nil {
			t.Error(err)
		}
	}

}
