/*
Copyright 2024 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// TODO: remove --emulated-version and --feature-gates in 1.37
func TestEnableDisableServiceCIDR(t *testing.T) {
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
	// start etcd instance
	etcdOptions := framework.SharedEtcd()
	// apiserver with the feature disabled
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	s1 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--service-cluster-ip-range=10.0.0.0/24",
			"--disable-admission-plugins=ServiceAccount",
			"--emulated-version=1.33",
			fmt.Sprintf("--feature-gates=%s=false", features.MultiCIDRServiceAllocator)},
		etcdOptions)

	client1, err := clientset.NewForConfig(s1.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client1, "test-enable-disable-service-cidr", t)
	// make 2 services , there will be 3 services counting the kubernetes.default
	for i := 0; i < 2; i++ {
		if _, err := client1.CoreV1().Services(ns.Name).Create(context.TODO(), svc(i), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	services, err := client1.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(services.Items) != 3 {
		t.Fatalf("expected 3 Services got %d", len(services.Items))
	}

	// shutdown s1
	s1.TearDownFn()

	// apiserver with the feature enabled
	s2 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--service-cluster-ip-range=10.0.0.0/24",
			"--disable-admission-plugins=ServiceAccount"},
		etcdOptions)

	client2, err := clientset.NewForConfig(s2.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// make 2 services , there will be 5 services now
	for i := 2; i < 4; i++ {
		if _, err := client2.CoreV1().Services(ns.Name).Create(context.TODO(), svc(i), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	services, err = client2.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(services.Items) != 5 {
		t.Fatalf("expected 5 Services got %d", len(services.Items))
	}

	// shutdown apiserver with the feature enabled
	s2.TearDownFn()
	// start an apiserver with the feature disabled
	s3 := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions,
		[]string{
			"--service-cluster-ip-range=10.0.0.0/24",
			"--disable-admission-plugins=ServiceAccount",
			"--emulated-version=1.33",
			fmt.Sprintf("--feature-gates=%s=false", features.MultiCIDRServiceAllocator)},
		etcdOptions)
	defer s3.TearDownFn()

	client3, err := clientset.NewForConfig(s3.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	// make 2 services , there will be 7 services now
	for i := 4; i < 6; i++ {
		if _, err := client3.CoreV1().Services(ns.Name).Create(context.TODO(), svc(i), metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	services, err = client3.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(services.Items) != 7 {
		t.Fatalf("expected 5 Services got %d", len(services.Items))
	}

}
