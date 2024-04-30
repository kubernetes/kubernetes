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
	"errors"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestEndpointUpdates(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	informers := informers.NewSharedInformerFactory(client, 0)

	tCtx := ktesting.Init(t)
	epController := endpoint.NewEndpointController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epController.Run(tCtx, 1)

	// Create namespace
	ns := framework.CreateNamespaceOrDie(client, "test-endpoints-updates", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

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

	createdPod, err := client.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}

	// Set pod IPs
	createdPod.Status = v1.PodStatus{
		Phase:  v1.PodRunning,
		PodIPs: []v1.PodIP{{IP: "1.1.1.1"}, {IP: "2001:db8::"}},
	}
	_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}

	// Create a service associated to the pod
	svc := newService(ns.Name, "foo1")
	svc1, err := client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	// Obtain ResourceVersion of the new endpoint created
	var resVersion string
	if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
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
	_, err = client.CoreV1().Services(ns.Name).Update(tCtx, svc1, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update service %s: %v", svc1.Name, err)
	}

	// Create a new service and wait until it has been processed,
	// this way we can be sure that the endpoint for the original service
	// was recomputed before asserting, since we only have 1 worker
	// in the endpoint controller
	svc2 := newService(ns.Name, "foo2")
	_, err = client.CoreV1().Services(ns.Name).Create(tCtx, svc2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc2.Name, metav1.GetOptions{})
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
	endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching endpoints: %v", err)
	}
	if resVersion != endpoints.ObjectMeta.ResourceVersion {
		t.Fatalf("endpoints resource version does not match, expected %s received %s", resVersion, endpoints.ObjectMeta.ResourceVersion)
	}

}

// TestExternalNameToClusterIPTransition tests that Service of type ExternalName
// does not get endpoints, and after transition to ClusterIP, service gets endpoint,
// without headless label
func TestExternalNameToClusterIPTransition(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	informers := informers.NewSharedInformerFactory(client, 0)

	tCtx := ktesting.Init(t)
	epController := endpoint.NewEndpointController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epController.Run(tCtx, 1)

	// Create namespace
	ns := framework.CreateNamespaceOrDie(client, "test-endpoints-updates", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

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

	createdPod, err := client.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}

	// Set pod IPs
	createdPod.Status = v1.PodStatus{
		Phase:  v1.PodRunning,
		PodIPs: []v1.PodIP{{IP: "1.1.1.1"}, {IP: "2001:db8::"}},
	}
	_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}

	// Create an ExternalName service associated to the pod
	svc := newExternalNameService(ns.Name, "foo1")
	svc1, err := client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	err = wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
		if err == nil {
			t.Errorf("expected no endpoints for externalName service, got: %v", endpoints)
			return true, nil
		}
		return false, nil
	})
	if err == nil {
		t.Errorf("expected error waiting for endpoints")
	}

	// update service to ClusterIP type and verify endpoint was created
	svc1.Spec.Type = v1.ServiceTypeClusterIP
	_, err = client.CoreV1().Services(ns.Name).Update(tCtx, svc1, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update service %s: %v", svc1.Name, err)
	}

	if err := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		ep, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc1.Name, metav1.GetOptions{})
		if err != nil {
			t.Logf("no endpoints found, error: %v", err)
			return false, nil
		}
		t.Logf("endpoint %s was successfully created", svc1.Name)
		if _, ok := ep.Labels[v1.IsHeadlessService]; ok {
			t.Errorf("ClusterIP endpoint should not have headless label, got: %v", ep)
		}
		return true, nil
	}); err != nil {
		t.Fatalf("endpoints not found: %v", err)
	}
}

// TestEndpointWithTerminatingPod tests that terminating pods are NOT included in Endpoints.
// This capability is only available in the newer EndpointSlice API and there are no plans to
// include it for Endpoints. This test can be removed in the future if we decide to include
// terminating endpoints in Endpoints, but in the mean time this test ensures we do not change
// this behavior accidentally.
func TestEndpointWithTerminatingPod(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	informers := informers.NewSharedInformerFactory(client, 0)

	tCtx := ktesting.Init(t)
	epController := endpoint.NewEndpointController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epController.Run(tCtx, 1)

	// Create namespace
	ns := framework.CreateNamespaceOrDie(client, "test-endpoints-terminating", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create a pod with labels
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-pod",
			Labels: labelMap(),
		},
		Spec: v1.PodSpec{
			NodeName: "fake-node",
			Containers: []v1.Container{
				{
					Name:  "fakename",
					Image: "fakeimage",
					Ports: []v1.ContainerPort{
						{
							Name:          "port-443",
							ContainerPort: 443,
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: v1.ConditionTrue,
				},
			},
			PodIP: "10.0.0.1",
			PodIPs: []v1.PodIP{
				{
					IP: "10.0.0.1",
				},
			},
		},
	}

	createdPod, err := client.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}

	createdPod.Status = pod.Status
	_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}

	// Create a service associated to the pod
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-service",
			Namespace: ns.Name,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Ports: []v1.ServicePort{
				{Name: "port-443", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt32(443)},
			},
		},
	}
	_, err = client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create service %s: %v", svc.Name, err)
	}

	// poll until associated Endpoints to the previously created Service exists
	if err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		numEndpoints := 0
		for _, subset := range endpoints.Subsets {
			numEndpoints += len(subset.Addresses)
		}

		if numEndpoints == 0 {
			return false, nil
		}

		return true, nil
	}); err != nil {
		t.Fatalf("endpoints not found: %v", err)
	}

	err = client.CoreV1().Pods(ns.Name).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("error deleting test pod: %v", err)
	}

	// poll until endpoint for deleted Pod is no longer in Endpoints.
	if err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		// Ensure that the recently deleted Pod exists but with a deletion timestamp. If the Pod does not exist,
		// we should fail the test since it is no longer validating against a terminating pod.
		pod, err := client.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, fmt.Errorf("expected Pod %q to exist with deletion timestamp but was not found: %v", pod.Name, err)
		}
		if err != nil {
			return false, nil
		}

		if pod.DeletionTimestamp == nil {
			return false, errors.New("pod did not have deletion timestamp set")
		}

		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		numEndpoints := 0
		for _, subset := range endpoints.Subsets {
			numEndpoints += len(subset.Addresses)
		}

		if numEndpoints > 0 {
			return false, nil
		}

		return true, nil
	}); err != nil {
		t.Fatalf("error checking for no endpoints with terminating pods: %v", err)
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
				{Name: "port-1338", Port: 1338, Protocol: "TCP", TargetPort: intstr.FromInt32(1338)},
				{Name: "port-1337", Port: 1337, Protocol: "TCP", TargetPort: intstr.FromInt32(1337)},
			},
		},
	}
}

// newExternalNameService returns an ExternalName service with selector and exposing ports
func newExternalNameService(namespace, name string) *v1.Service {
	svc := newService(namespace, name)
	svc.Spec.Type = v1.ServiceTypeExternalName
	svc.Spec.ExternalName = "google.com"
	return svc
}
