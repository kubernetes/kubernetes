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
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/pkg/controller/endpointslicemirroring"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEndpointSliceMirroring(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, server, closeFn := framework.RunAMaster(masterConfig)
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
		informers.Discovery().V1beta1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second)

	epsmController := endpointslicemirroring.NewController(
		informers.Core().V1().Endpoints(),
		informers.Discovery().V1beta1().EndpointSlices(),
		informers.Core().V1().Services(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go epController.Run(5, stopCh)
	go epsController.Run(5, stopCh)
	go epsmController.Run(5, stopCh)

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
				_, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), tc.service, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating service: %v", err)
				}
			}

			if tc.customEndpoints != nil {
				resourceName = tc.customEndpoints.Name
				tc.customEndpoints.Namespace = ns.Name
				_, err = client.CoreV1().Endpoints(ns.Name).Create(context.TODO(), tc.customEndpoints, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating endpoints: %v", err)
				}
			}

			err = wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				lSelector := discovery.LabelServiceName + "=" + resourceName
				esList, err := client.DiscoveryV1beta1().EndpointSlices(ns.Name).List(context.TODO(), metav1.ListOptions{LabelSelector: lSelector})
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
