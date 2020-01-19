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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

var (
	IPv4Family = v1.IPv4Protocol
	IPv6Family = v1.IPv6Protocol
)

// TestNodePortSetIPFamily tests the NodePort allocation of the same port with
// different IP families
func TestNodePortSetIPFamily(t *testing.T) {
	etcd := framework.SharedEtcd()
	// cleanup the registry storage
	defer registry.CleanupStorage()
	// start a kube-apiserver with dual stack enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--service-cluster-ip-range", "10.0.0.0/24",
		"--advertise-address", "10.0.0.1",
		"--feature-gates", "IPv6DualStack=True",
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

	// Create a NodePort service listening on the IPv4 family
	svcNodePort := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "svc",
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       8443,
				NodePort:   30443,
				TargetPort: intstr.FromInt(8443),
				Protocol:   v1.ProtocolTCP,
			}},
			Type: v1.ServiceTypeNodePort,
		},
	}

	svcNodePort.Name = "svc-ipv4"
	svcNodePort.Spec.IPFamily = &IPv4Family
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error text: %v", err)
	}
	svcNodePort.Name = "svc-ipv6"
	svcNodePort.Spec.IPFamily = &IPv6Family
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error text: %v", err)
	}
}

// TestNodePortDefaultIPFamily tests the NodePort allocation if one service does not set the
// ipFamily and other with the ipFamily set tries to use the same port
func TestNodePortDefaultIPFamily(t *testing.T) {
	etcd := framework.SharedEtcd()
	// cleanup the registry storage
	defer registry.CleanupStorage()
	// start a kube-apiserver with dual stack enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--service-cluster-ip-range", "10.0.0.0/24",
		"--advertise-address", "10.0.0.1",
		"--feature-gates", "IPv6DualStack=True",
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

	// Create a NodePort service listening on the IPv4 family
	svcNodePort := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "svc",
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       8443,
				NodePort:   30443,
				TargetPort: intstr.FromInt(8443),
				Protocol:   v1.ProtocolTCP,
			}},
			Type: v1.ServiceTypeNodePort,
		},
	}

	svcNodePort.Name = "svc-no-ipfamily"
	svcNodePort.Spec.IPFamily = nil
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error text: %v", err)
	}
	svcNodePort.Name = "svc-ipv6"
	svcNodePort.Spec.IPFamily = &IPv6Family
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error text: %v", err)
	}
}
