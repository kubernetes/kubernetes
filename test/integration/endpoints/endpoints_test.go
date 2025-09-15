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
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
)

func TestEndpointUpdates(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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

// Regression test for https://issues.k8s.io/125638
func TestEndpointWithMultiplePodUpdates(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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

	// Process 10 services in parallel to increase likelihood of outdated informer cache.
	concurrency := 10
	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epController.Run(tCtx, concurrency)

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

	pod, err = client.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
	}

	// Set pod status
	pod.Status = v1.PodStatus{
		Phase:      v1.PodRunning,
		Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}},
		PodIPs:     []v1.PodIP{{IP: "1.1.1.1"}},
	}
	pod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}

	var services []*v1.Service
	// Create services associated to the pod
	for i := 0; i < concurrency; i++ {
		svc := newService(ns.Name, fmt.Sprintf("foo%d", i))
		_, err = client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create service %s: %v", svc.Name, err)
		}
		services = append(services, svc)
	}

	for _, service := range services {
		// Ensure the new endpoints are created.
		if err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Second, true, func(context.Context) (bool, error) {
			_, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, service.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return true, nil
		}); err != nil {
			t.Fatalf("endpoints not found: %v", err)
		}
	}

	// Update pod's status and revert it immediately. The endpoints should be in-sync with the pod's status eventually.
	pod.Status.Conditions[0].Status = v1.ConditionFalse
	pod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod %s to not ready: %v", pod.Name, err)
	}

	pod.Status.Conditions[0].Status = v1.ConditionTrue
	pod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod %s to ready: %v", pod.Name, err)
	}

	// Some workers might update endpoints twice (Ready->NotReady->Ready), while others may not update endpoints at all
	// if they receive the 2nd pod update quickly. Consequently, we can't rely on endpoints resource version to
	// determine if the controller has processed the pod updates. Instead, we will wait for 1 second, assuming that this
	// provides enough time for the workers to process endpoints at least once.
	time.Sleep(1 * time.Second)
	expectedEndpointAddresses := []v1.EndpointAddress{
		{
			IP:       pod.Status.PodIP,
			NodeName: &pod.Spec.NodeName,
			TargetRef: &v1.ObjectReference{
				Kind:      "Pod",
				Namespace: pod.Namespace,
				Name:      pod.Name,
				UID:       pod.UID,
			},
		},
	}
	for _, service := range services {
		var endpoints *v1.Endpoints
		if err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Second, true, func(context.Context) (bool, error) {
			endpoints, err = client.CoreV1().Endpoints(ns.Name).Get(tCtx, service.Name, metav1.GetOptions{})
			if err != nil {
				t.Logf("Error fetching endpoints: %v", err)
				return false, nil
			}
			if len(endpoints.Subsets) == 0 {
				return false, nil
			}
			if !reflect.DeepEqual(expectedEndpointAddresses, endpoints.Subsets[0].Addresses) {
				return false, nil
			}
			return true, nil
		}); err != nil {
			t.Fatalf("Expected endpoints %v to contain ready endpoint addresses %v", endpoints, expectedEndpointAddresses)
		}
	}
}

// TestExternalNameToClusterIPTransition tests that Service of type ExternalName
// does not get endpoints, and after transition to ClusterIP, service gets endpoint,
// without headless label
func TestExternalNameToClusterIPTransition(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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

func TestEndpointTruncate(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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
	ns := framework.CreateNamespaceOrDie(client, "test-endpoints-truncate", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create a pod with labels
	basePod := &v1.Pod{
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

	// create 1001 Pods to reach endpoint max capacity that is set to 1000
	allPodNames := sets.New[string]()
	baseIP := netutils.BigForIP(netutils.ParseIPSloppy("10.0.0.1"))
	for i := 0; i < 1001; i++ {
		pod := basePod.DeepCopy()
		pod.Name = fmt.Sprintf("%s-%d", basePod.Name, i)
		allPodNames.Insert(pod.Name)
		podIP := netutils.AddIPOffset(baseIP, i).String()
		pod.Status.PodIP = podIP
		pod.Status.PodIPs[0] = v1.PodIP{IP: podIP}
		createdPod, err := client.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}

		createdPod.Status = pod.Status
		_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, createdPod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
		}
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

	var truncatedPodName string
	// poll until associated Endpoints to the previously created Service exists
	if err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Second, true, func(context.Context) (bool, error) {
		podNames := sets.New[string]()
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		for _, subset := range endpoints.Subsets {
			for _, address := range subset.Addresses {
				podNames.Insert(address.TargetRef.Name)
			}
		}

		if podNames.Len() != 1000 {
			return false, nil
		}

		truncated, ok := endpoints.Annotations[v1.EndpointsOverCapacity]
		if !ok || truncated != "truncated" {
			return false, nil
		}
		// There is only 1 truncated Pod.
		truncatedPodName, _ = allPodNames.Difference(podNames).PopAny()
		return true, nil
	}); err != nil {
		t.Fatalf("endpoints not found: %v", err)
	}

	// Update the truncated Pod several times to make endpoints controller resync the service.
	truncatedPod, err := client.CoreV1().Pods(ns.Name).Get(tCtx, truncatedPodName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod %s: %v", truncatedPodName, err)
	}
	for i := 0; i < 10; i++ {
		truncatedPod.Status.Conditions[0].Status = v1.ConditionFalse
		truncatedPod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, truncatedPod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status of pod %s: %v", truncatedPod.Name, err)
		}
		truncatedPod.Status.Conditions[0].Status = v1.ConditionTrue
		truncatedPod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, truncatedPod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status of pod %s: %v", truncatedPod.Name, err)
		}
	}

	// delete 501 Pods
	for i := 500; i < 1001; i++ {
		podName := fmt.Sprintf("%s-%d", basePod.Name, i)
		err = client.CoreV1().Pods(ns.Name).Delete(tCtx, podName, metav1.DeleteOptions{})
		if err != nil {
			t.Fatalf("error deleting test pod: %v", err)
		}
	}

	// poll until endpoints for deleted Pod are no longer in Endpoints.
	if err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Second, true, func(context.Context) (bool, error) {
		endpoints, err := client.CoreV1().Endpoints(ns.Name).Get(tCtx, svc.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		numEndpoints := 0
		for _, subset := range endpoints.Subsets {
			numEndpoints += len(subset.Addresses)
		}

		if numEndpoints != 500 {
			return false, nil
		}

		truncated, ok := endpoints.Annotations[v1.EndpointsOverCapacity]
		if ok || truncated == "truncated" {
			return false, nil
		}

		return true, nil
	}); err != nil {
		t.Fatalf("error checking for no endpoints with terminating pods: %v", err)
	}
}
