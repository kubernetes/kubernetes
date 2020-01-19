/*
Copyright 2020 The Kubernetes Authors.

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

package network

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
)

var serviceTypes = []*v1.Service{
	&v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeClusterIP, Ports: []v1.ServicePort{{Port: 1000}}}},
	&v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeClusterIP, Ports: []v1.ServicePort{{Port: 1000}}, ClusterIP: v1.ClusterIPNone}},
	&v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeNodePort, Ports: []v1.ServicePort{{Port: 1000}}}},
	&v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer, Ports: []v1.ServicePort{{Port: 1000}}}},
	&v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeExternalName, ExternalName: "www.google.com"}},
}

// TestServicesIPFamily tests the IP family field behavior on DualStack clusters
func TestServicesIPFamily(t *testing.T) {
	var testcases = []struct {
		name            string
		serviceCIDR     string
		defaultIPFamily v1.IPFamily
		dualStack       bool
		expectErr       string
	}{
		{
			name:            "Single Stack IPv4",
			serviceCIDR:     "10.0.0.0/16",
			defaultIPFamily: v1.IPv4Protocol,
			dualStack:       false,
		},
		{
			name:            "Single Stack IPv6",
			serviceCIDR:     "2001:db8:1::/48",
			defaultIPFamily: v1.IPv6Protocol,
			dualStack:       false,
		},
		{
			name:            "DualStack enabled and IPv4only",
			serviceCIDR:     "10.0.0.0/16",
			defaultIPFamily: v1.IPv4Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv6only",
			serviceCIDR:     "2001:db8:1::/48",
			defaultIPFamily: v1.IPv6Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv4 IPv6",
			serviceCIDR:     "10.0.0.0/16,2001:db8:1::/48",
			defaultIPFamily: v1.IPv4Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv6 IPv4",
			serviceCIDR:     "2001:db8:1::/48,10.0.0.0/16",
			defaultIPFamily: v1.IPv6Protocol,
			dualStack:       true,
		},
	}

	for _, tc := range testcases {
		tc := tc

		for _, svc := range serviceTypes {
			t.Run(fmt.Sprintf("%s/%v", tc.name, svc.Spec.Type), func(t *testing.T) {
				ipFamily := tc.defaultIPFamily
				otherIPFamily := v1.IPv6Protocol
				if ipFamily == otherIPFamily {
					otherIPFamily = v1.IPv4Protocol
				}
				etcd := framework.SharedEtcd()
				// cleanup the registry storage
				defer registry.CleanupStorage()
				// start a kube-apiserver
				server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
					"--service-cluster-ip-range", tc.serviceCIDR,
					"--advertise-address", "10.0.0.1",
					"--feature-gates", fmt.Sprintf("IPv6DualStack=%t", tc.dualStack),
				}, etcd)
				defer server.TearDownFn()
				// create a client
				client, err := kubernetes.NewForConfig(server.ClientConfig)
				if err != nil {
					t.Errorf("error creating client: %v", err)
				}

				// verify client is working
				if err := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
					_, err = client.CoreV1().Endpoints("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
					if err != nil {
						t.Logf("error fetching endpoints: %v", err)
						return false, nil
					}
					return true, nil
				}); err != nil {
					t.Errorf("server without enabled endpoints failed to register: %v", err)
				}

				// Create a service setting the IPFamily and check that it's created with the correct IP family
				svc.Name = "svc-ipfamily-default"
				svc.Spec.IPFamily = &ipFamily
				if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{}); err != nil {
					t.Errorf("unexpected error text: %v", err)
				}

				// Create a service without IPFamily and check that it's created with the default IP family
				svc.Name = "svc-ipfamily-nil"
				svc.Spec.IPFamily = nil
				if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{}); err != nil {
					t.Errorf("unexpected error text: %v", err)
				}

				// obtain the services and check the Spec.IPFamily matches the default IP family
				svcs, err := client.CoreV1().Services(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					t.Errorf("unexpected error getting the services: %v", err)
				}
				for _, s := range svcs.Items {
					// single family Spec.IPFamily must be nil
					if !tc.dualStack || s.Spec.Type == v1.ServiceTypeExternalName {
						if s.Spec.IPFamily != nil {
							t.Errorf("error service %s expected service IP family nil: %v ", s.Name, *s.Spec.IPFamily)
						}
					} else if err = validateServiceAndClusterIPFamily(&s, ipFamily); err != nil {
						t.Errorf("error service %s expected service IP family nil: %v, err: %v ", s.Name, *s.Spec.IPFamily, err)
					}
				}

				// Create a service with the non-default IPfamily, it must fail on non-dualstack clusters
				svc.Name = "svc-ipfamily-other"
				svc.Spec.IPFamily = &otherIPFamily
				_, err = client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{})
				if !tc.dualStack && err == nil {
					t.Errorf("unexpected success creating service: %v", svc)
				}
				if tc.dualStack && err != nil {
					t.Errorf("unexpected error text: %v", err)
				}
			})
		}
	}
}

// TestDualStackSkew spawns one API server with DualStack enable and other with DualStack disable
// and test te behavior of IPFamily with skewed clients
func TestDualStackSkew(t *testing.T) {

	var testcases = []struct {
		name            string
		serviceCIDR     string
		defaultIPFamily v1.IPFamily
		dualStack       bool
		expectErr       string
	}{
		{
			name:            "DualStack enabled and IPv4only",
			serviceCIDR:     "10.0.0.0/16",
			defaultIPFamily: v1.IPv4Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv6only",
			serviceCIDR:     "2001:db8:1::/48",
			defaultIPFamily: v1.IPv6Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv4 IPv6",
			serviceCIDR:     "10.0.0.0/16,2001:db8:1::/48",
			defaultIPFamily: v1.IPv4Protocol,
			dualStack:       true,
		},
		{
			name:            "DualStack enabled and IPv6 IPv4",
			serviceCIDR:     "2001:db8:1::/48,10.0.0.0/16",
			defaultIPFamily: v1.IPv6Protocol,
			dualStack:       true,
		},
	}

	for _, tc := range testcases {
		tc := tc
		for _, svc := range serviceTypes {
			t.Run(fmt.Sprintf("%s/%v", tc.name, svc.Spec.Type), func(t *testing.T) {
				ipFamily := tc.defaultIPFamily
				etcd := framework.SharedEtcd()

				instanceOptions := &kubeapiservertesting.TestServerInstanceOptions{
					DisableStorageCleanup: true,
				}

				// cleanup the registry storage
				defer registry.CleanupStorage()

				// start api server without dual stack enabled
				server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, []string{
					"--endpoint-reconciler-type", "master-count",
					"--advertise-address", "10.0.1.1",
					"--apiserver-count", "1",
					"--feature-gates", "IPv6DualStack=false",
				}, etcd)
				defer server.TearDownFn()

				client, err := kubernetes.NewForConfig(server.ClientConfig)
				if err != nil {
					t.Errorf("error creating client: %v", err)
				}

				// verify client
				if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
					_, err = client.CoreV1().Endpoints("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
					if err != nil {
						t.Logf("error fetching endpoints: %v", err)
						return false, nil
					}
					return true, nil
				}); err != nil {
					t.Errorf("server without DualStack enabled endpoints failed to register: %v", err)
				}

				// start api server with dual stack enabled
				serverDual := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, []string{
					"--endpoint-reconciler-type", "master-count",
					"--advertise-address", "10.0.1.1",
					"--apiserver-count", "2",
					"--feature-gates", "IPv6DualStack=true",
				}, etcd)
				defer serverDual.TearDownFn()

				clientDual, err := kubernetes.NewForConfig(serverDual.ClientConfig)
				if err != nil {
					t.Errorf("error creating client: %v", err)

				}
				// verify client
				if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
					_, err = clientDual.CoreV1().Endpoints("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
					if err != nil {
						t.Logf("error fetching endpoints: %v", err)
						return false, nil
					}
					return true, nil
				}); err != nil {
					t.Errorf("server with DualStack enabled endpoints failed to register: %v", err)
				}

				// Create a service from a dual-stack apiserver setting the IPFamily and check that it's created with the correct IP family
				svc.Name = "svc-dual-ipfamily"
				svc.Spec.IPFamily = &ipFamily

				if _, err := clientDual.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{}); err != nil {
					t.Errorf("unexpected error text: %v", err)
				}

				// Create a service without IPFamily from a dual-stack apiserver and check that it's created with the default IP family
				svc.Name = "svc-dual-ipfamily-nil"
				svc.Spec.IPFamily = nil
				if _, err := clientDual.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{}); err != nil {
					t.Errorf("unexpected error text: %v", err)
				}

				// Create a service from a non dual-stack apiserver and check that it doesn't have IP family
				svc.Name = "svc-single-stack"
				svc.Spec.IPFamily = nil
				if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{}); err != nil {
					t.Errorf("unexpected error text: %v", err)
				}

				// obtain the services and check the Spec.IPFamily is nil for non-dualstack clients
				svcs, err := client.CoreV1().Services(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					t.Errorf("unexpected success, and error getting the services: %v", err)
				}
				for _, s := range svcs.Items {
					if s.Spec.IPFamily != nil {
						t.Errorf("error service %s expected service IP family nil: %v ", s.Name, *s.Spec.IPFamily)
					}
				}
			})
		}
	}
}

// helper functions obtained from test/e2e/network/dual_stack.go

// validateServiceAndClusterIPFamily checks that the service
// belongs to the IPfamily passed as a parameter
func validateServiceAndClusterIPFamily(svc *v1.Service, expectedIPFamily v1.IPFamily) error {
	if svc.Spec.IPFamily == nil {
		return fmt.Errorf("service ip family nil for service %s/%s", svc.Namespace, svc.Name)
	}
	if *svc.Spec.IPFamily != expectedIPFamily {
		return fmt.Errorf("ip family mismatch for service: %s/%s, expected: %s, actual: %s", svc.Namespace, svc.Name, expectedIPFamily, *svc.Spec.IPFamily)
	}

	if svc.Spec.ClusterIP != v1.ClusterIPNone {
		isIPv6ClusterIP := netutils.IsIPv6String(svc.Spec.ClusterIP)
		if (expectedIPFamily == v1.IPv4Protocol && isIPv6ClusterIP) || (expectedIPFamily == v1.IPv6Protocol && !isIPv6ClusterIP) {
			return fmt.Errorf("got unexpected service ip %s, should belong to %s ip family", svc.Spec.ClusterIP, expectedIPFamily)
		}
	}
	return nil
}
