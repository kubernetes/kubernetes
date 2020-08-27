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

package endpoints

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEndpointUpdates(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, server, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	informers := informers.NewSharedInformerFactory(client, 0)

	epController := endpoint.NewEndpointController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go epController.Run(1, stopCh)

	// Create namespace
	ns := framework.CreateTestingNamespace("test-endpoints-updates", server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	// Create a pod with labels
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: ns.Name,
			Labels:    labelMap(),
		},
		Spec: v1.PodSpec{
			NodeName: "fakenode",
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
	}

	createdPod, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}

	// Set pod IPs
	createdPod.Status = v1.PodStatus{
		Phase:  v1.PodRunning,
		PodIPs: []v1.PodIP{{IP: "1.1.1.1"}, {IP: "2001:db8::"}},
	}
	_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}

	// Create a service associated to the pod
	svc := newService(ns.Name, "foo1")
	svc1, err := client.CoreV1().Services(ns.Name).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	// Obtain ResourceVersion of the new endpoint created
	var resVersion string
	if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(context.TODO(), svc.Name, metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		resVersion = endpoints.ObjectMeta.ResourceVersion
		return true, nil
	}); err != nil {
		t.Fatalf("endpoints not found: %v", err)
	}

	// Force recomputation on the endpoint controller
	svc1.SetAnnotations(map[string]string{"foo": "bar"})
	_, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), svc1, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update service %s: %v", svc1.Name, err)
	}

	// Create a new service and wait until it has been processed,
	// this way we can be sure that the endpoint for the original service
	// was recomputed before asserting, since we only have 1 worker
	// in the endpoint controller
	svc2 := newService(ns.Name, "foo2")
	_, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), svc2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := client.CoreV1().Endpoints(ns.Name).Get(context.TODO(), svc2.Name, metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("endpoints not found: %v", err)
	}

	// the endpoint controller should not update the endpoint created for the original
	// service since nothing has changed, the resource version has to be the same
	endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(context.TODO(), svc.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching endpoints: %v", err)
	}
	if resVersion != endpoints.ObjectMeta.ResourceVersion {
		t.Fatalf("endpoints resource version does not match, expected %s received %s", resVersion, endpoints.ObjectMeta.ResourceVersion)
	}

}

func labelMap() map[string]string {
	return map[string]string{"foo": "bar"}
}

// newService returns a service with selector and exposing ports
func newService(namespace, name string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labelMap(),
		},
		Spec: v1.ServiceSpec{
			Selector: labelMap(),
			Ports: []v1.ServicePort{
				{Name: "port-1338", Port: 1338, Protocol: "TCP", TargetPort: intstr.FromInt(1338)},
				{Name: "port-1337", Port: 1337, Protocol: "TCP", TargetPort: intstr.FromInt(1337)},
			},
		},
	}

}
