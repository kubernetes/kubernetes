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

package endpointslice

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/pkg/controller/endpointslicemirroring"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEndpointSliceMirroring(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	epController := endpoint.NewEndpointController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		1*time.Second)

	epsController := endpointslice.NewController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Nodes(),
		informers.Discovery().V1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second)

	epsmController := endpointslicemirroring.NewController(
		informers.Core().V1().Endpoints(),
		informers.Discovery().V1().EndpointSlices(),
		informers.Core().V1().Services(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informers.Start(ctx.Done())
	go epController.Run(ctx, 5)
	go epsController.Run(5, ctx.Done())
	go epsmController.Run(5, ctx.Done())

	testCases := []struct {
		testName                     string
		service                      *corev1.Service
		customEndpoints              *corev1.Endpoints
		expectEndpointSlice          bool
		expectEndpointSliceManagedBy string
	}{{
		testName: "Service with selector",
		service: &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-123",
			},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{
					Port: int32(80),
				}},
				Selector: map[string]string{
					"foo": "bar",
				},
			},
		},
		expectEndpointSlice:          true,
		expectEndpointSliceManagedBy: "endpointslice-controller.k8s.io",
	}, {
		testName: "Service without selector",
		service: &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-123",
			},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{
					Port: int32(80),
				}},
			},
		},
		customEndpoints: &corev1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-123",
			},
			Subsets: []corev1.EndpointSubset{{
				Ports: []corev1.EndpointPort{{
					Port: 80,
				}},
				Addresses: []corev1.EndpointAddress{{
					IP: "10.0.0.1",
				}},
			}},
		},
		expectEndpointSlice:          true,
		expectEndpointSliceManagedBy: "endpointslicemirroring-controller.k8s.io",
	}, {
		testName: "Service without Endpoints",
		service: &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-123",
			},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{
					Port: int32(80),
				}},
				Selector: map[string]string{
					"foo": "bar",
				},
			},
		},
		customEndpoints:              nil,
		expectEndpointSlice:          true,
		expectEndpointSliceManagedBy: "endpointslice-controller.k8s.io",
	}, {
		testName: "Endpoints without Service",
		service:  nil,
		customEndpoints: &corev1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-123",
			},
			Subsets: []corev1.EndpointSubset{{
				Ports: []corev1.EndpointPort{{
					Port: 80,
				}},
				Addresses: []corev1.EndpointAddress{{
					IP: "10.0.0.1",
				}},
			}},
		},
		expectEndpointSlice: false,
	}}

	for i, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			ns := framework.CreateTestingNamespace(fmt.Sprintf("test-endpointslice-mirroring-%d", i), server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)

			resourceName := ""
			if tc.service != nil {
				resourceName = tc.service.Name
				tc.service.Namespace = ns.Name
				_, err = client.CoreV1().Services(ns.Name).Create(ctx, tc.service, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating service: %v", err)
				}
			}

			if tc.customEndpoints != nil {
				resourceName = tc.customEndpoints.Name
				tc.customEndpoints.Namespace = ns.Name
				_, err = client.CoreV1().Endpoints(ns.Name).Create(ctx, tc.customEndpoints, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating endpoints: %v", err)
				}
			}

			err = wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				lSelector := discovery.LabelServiceName + "=" + resourceName
				esList, err := client.DiscoveryV1().EndpointSlices(ns.Name).List(ctx, metav1.ListOptions{LabelSelector: lSelector})
				if err != nil {
					t.Logf("Error listing EndpointSlices: %v", err)
					return false, err
				}

				if tc.expectEndpointSlice {
					if len(esList.Items) == 0 {
						t.Logf("Waiting for EndpointSlice to be created")
						return false, nil
					}
					if len(esList.Items) > 1 {
						return false, fmt.Errorf("Only expected 1 EndpointSlice, got %d", len(esList.Items))
					}
					endpointSlice := esList.Items[0]
					if tc.expectEndpointSliceManagedBy != "" {
						if endpointSlice.Labels[discovery.LabelManagedBy] != tc.expectEndpointSliceManagedBy {
							return false, fmt.Errorf("Expected EndpointSlice to be managed by %s, got %s", tc.expectEndpointSliceManagedBy, endpointSlice.Labels[discovery.LabelManagedBy])
						}
					}
				} else if len(esList.Items) > 0 {
					t.Logf("Waiting for EndpointSlices to be removed, still %d", len(esList.Items))
					return false, nil
				}

				return true, nil
			})
			if err != nil {
				t.Fatalf("Timed out waiting for conditions: %v", err)
			}
		})
	}

}

func TestEndpointSliceMirroringUpdates(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	epsmController := endpointslicemirroring.NewController(
		informers.Core().V1().Endpoints(),
		informers.Discovery().V1().EndpointSlices(),
		informers.Core().V1().Services(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informers.Start(ctx.Done())
	go epsmController.Run(1, ctx.Done())

	testCases := []struct {
		testName      string
		tweakEndpoint func(ep *corev1.Endpoints)
	}{
		{
			testName: "Update labels",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				ep.Labels["foo"] = "bar"
			},
		},
		{
			testName: "Update annotations",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				ep.Annotations["foo2"] = "bar2"
			},
		},
		{
			testName: "Update annotations but triggertime",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				ep.Annotations["foo2"] = "bar2"
				ep.Annotations[corev1.EndpointsLastChangeTriggerTime] = "date"
			},
		},
		{
			testName: "Update addresses",
			tweakEndpoint: func(ep *corev1.Endpoints) {
				ep.Subsets[0].Addresses = []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "1.2.3.6"}}
			},
		},
	}

	for i, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			ns := framework.CreateTestingNamespace(fmt.Sprintf("test-endpointslice-mirroring-%d", i), server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)

			service := &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-123",
					Namespace: ns.Name,
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{{
						Port: int32(80),
					}},
				},
			}

			customEndpoints := &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-123",
					Namespace:   ns.Name,
					Labels:      map[string]string{},
					Annotations: map[string]string{},
				},
				Subsets: []corev1.EndpointSubset{{
					Ports: []corev1.EndpointPort{{
						Port: 80,
					}},
					Addresses: []corev1.EndpointAddress{{
						IP: "10.0.0.1",
					}},
				}},
			}

			_, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating service: %v", err)
			}

			_, err = client.CoreV1().Endpoints(ns.Name).Create(ctx, customEndpoints, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating endpoints: %v", err)
			}

			// update endpoint
			tc.tweakEndpoint(customEndpoints)
			_, err = client.CoreV1().Endpoints(ns.Name).Update(ctx, customEndpoints, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Error updating endpoints: %v", err)
			}

			// verify the endpoint updates were mirrored
			err = wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				lSelector := discovery.LabelServiceName + "=" + service.Name
				esList, err := client.DiscoveryV1().EndpointSlices(ns.Name).List(ctx, metav1.ListOptions{LabelSelector: lSelector})
				if err != nil {
					t.Logf("Error listing EndpointSlices: %v", err)
					return false, err
				}

				if len(esList.Items) == 0 {
					t.Logf("Waiting for EndpointSlice to be created")
					return false, nil
				}

				for _, endpointSlice := range esList.Items {
					if endpointSlice.Labels[discovery.LabelManagedBy] != "endpointslicemirroring-controller.k8s.io" {
						return false, fmt.Errorf("Expected EndpointSlice to be managed by endpointslicemirroring-controller.k8s.io, got %s", endpointSlice.Labels[discovery.LabelManagedBy])
					}

					// compare addresses
					epAddresses := []string{}
					for _, address := range customEndpoints.Subsets[0].Addresses {
						epAddresses = append(epAddresses, address.IP)
					}

					sliceAddresses := []string{}
					for _, sliceEndpoint := range endpointSlice.Endpoints {
						sliceAddresses = append(sliceAddresses, sliceEndpoint.Addresses...)
					}

					sort.Strings(epAddresses)
					sort.Strings(sliceAddresses)

					if !apiequality.Semantic.DeepEqual(epAddresses, sliceAddresses) {
						t.Logf("Expected EndpointSlice to have the same IP addresses, expected %v got %v", epAddresses, sliceAddresses)
						return false, nil
					}

					// check labels were mirrored
					if !isSubset(customEndpoints.Labels, endpointSlice.Labels) {
						t.Logf("Expected EndpointSlice to mirror labels, expected %v to be in received %v", customEndpoints.Labels, endpointSlice.Labels)
						return false, nil
					}

					// check annotations but endpoints.kubernetes.io/last-change-trigger-time were mirrored
					annotations := map[string]string{}
					for k, v := range customEndpoints.Annotations {
						if k == corev1.EndpointsLastChangeTriggerTime {
							continue
						}
						annotations[k] = v
					}
					if !apiequality.Semantic.DeepEqual(annotations, endpointSlice.Annotations) {
						t.Logf("Expected EndpointSlice to mirror annotations, expected %v received %v", customEndpoints.Annotations, endpointSlice.Annotations)
						return false, nil
					}
				}
				return true, nil
			})
			if err != nil {
				t.Fatalf("Timed out waiting for conditions: %v", err)
			}
		})
	}
}

func TestEndpointSliceMirroringSelectorTransition(t *testing.T) {
	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, server, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	defer closeFn()

	config := restclient.Config{Host: server.URL}
	client, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	epsmController := endpointslicemirroring.NewController(
		informers.Core().V1().Endpoints(),
		informers.Discovery().V1().EndpointSlices(),
		informers.Core().V1().Services(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go epsmController.Run(1, stopCh)

	testCases := []struct {
		testName               string
		startingSelector       map[string]string
		startingMirroredSlices int
		endingSelector         map[string]string
		endingMirroredSlices   int
	}{
		{
			testName:               "nil -> {foo: bar} selector",
			startingSelector:       nil,
			startingMirroredSlices: 1,
			endingSelector:         map[string]string{"foo": "bar"},
			endingMirroredSlices:   0,
		},
		{
			testName:               "{foo: bar} -> nil selector",
			startingSelector:       map[string]string{"foo": "bar"},
			startingMirroredSlices: 0,
			endingSelector:         nil,
			endingMirroredSlices:   1,
		},
		{
			testName:               "{} -> {foo: bar} selector",
			startingSelector:       map[string]string{},
			startingMirroredSlices: 1,
			endingSelector:         map[string]string{"foo": "bar"},
			endingMirroredSlices:   0,
		},
		{
			testName:               "{foo: bar} -> {} selector",
			startingSelector:       map[string]string{"foo": "bar"},
			startingMirroredSlices: 0,
			endingSelector:         map[string]string{},
			endingMirroredSlices:   1,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			ns := framework.CreateTestingNamespace(fmt.Sprintf("test-endpointslice-mirroring-%d", i), server, t)
			defer framework.DeleteTestingNamespace(ns, server, t)
			meta := metav1.ObjectMeta{Name: "test-123", Namespace: ns.Name}

			service := &corev1.Service{
				ObjectMeta: meta,
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{{
						Port: int32(80),
					}},
					Selector: tc.startingSelector,
				},
			}

			customEndpoints := &corev1.Endpoints{
				ObjectMeta: meta,
				Subsets: []corev1.EndpointSubset{{
					Ports: []corev1.EndpointPort{{
						Port: 80,
					}},
					Addresses: []corev1.EndpointAddress{{
						IP: "10.0.0.1",
					}},
				}},
			}

			_, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating service: %v", err)
			}

			_, err = client.CoreV1().Endpoints(ns.Name).Create(context.TODO(), customEndpoints, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating endpoints: %v", err)
			}

			// verify the expected number of mirrored slices exist
			err = waitForMirroredSlices(t, client, ns.Name, service.Name, tc.startingMirroredSlices)
			if err != nil {
				t.Fatalf("Timed out waiting for initial mirrored slices to match expectations: %v", err)
			}

			service.Spec.Selector = tc.endingSelector
			_, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Error updating service: %v", err)
			}

			// verify the expected number of mirrored slices exist
			err = waitForMirroredSlices(t, client, ns.Name, service.Name, tc.endingMirroredSlices)
			if err != nil {
				t.Fatalf("Timed out waiting for final mirrored slices to match expectations: %v", err)
			}
		})
	}
}

func waitForMirroredSlices(t *testing.T, client *kubernetes.Clientset, nsName, svcName string, num int) error {
	t.Helper()
	return wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		lSelector := discovery.LabelServiceName + "=" + svcName
		lSelector += "," + discovery.LabelManagedBy + "=endpointslicemirroring-controller.k8s.io"
		esList, err := client.DiscoveryV1().EndpointSlices(nsName).List(context.TODO(), metav1.ListOptions{LabelSelector: lSelector})
		if err != nil {
			t.Logf("Error listing EndpointSlices: %v", err)
			return false, err
		}

		if len(esList.Items) != num {
			t.Logf("Expected %d slices to be mirrored, got %d", num, len(esList.Items))
			return false, nil
		}

		return true, nil
	})
}

// isSubset check if all the elements in a exist in b
func isSubset(a, b map[string]string) bool {
	if len(a) > len(b) {
		return false
	}
	for k, v1 := range a {
		if v2, ok := b[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}
