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
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/testutils/ktesting"
)

// TestEndpointsControllersLabelSemantics tests that the corresponding endpoints controller
// creates an Endpoint or EndpointSlice with the service.kubernetes.io/headless label when
// a headless service is created.
// It also verifies that Service labels are propagated to the Endpoints and EndpointSlice.
func TestEndpointsControllersLabelSemantics(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	tCtx := ktesting.Init(t)
	epController := endpoint.NewEndpointController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	epsController := endpointslice.NewController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Nodes(),
		informers.Discovery().V1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epsController.Run(tCtx, 1)
	go epController.Run(tCtx, 1)

	ns := framework.CreateNamespaceOrDie(client, "test-endpoints-semantics", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
			Labels: map[string]string{
				"app": "test",
			},
		},
		Spec: corev1.ServiceSpec{
			Type:      corev1.ServiceTypeClusterIP,
			ClusterIP: corev1.ClusterIPNone, // Headless service
			Selector: map[string]string{
				"app": "test",
			},
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt(8080),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating service: %v", err)
	}

	validateLabelsOnEndpointAndEndpointSlice(t, tCtx, client, ns.Name, svc.Name,
		[]string{corev1.IsHeadlessService, "app"}, nil)

	svc.Labels["new-label"] = "new-label-value"
	delete(svc.Labels, "app")

	_, err = client.CoreV1().Services(ns.Name).Update(tCtx, svc, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating service: %v", err)
	}

	// Validate that the new labels are propagated.
	validateLabelsOnEndpointAndEndpointSlice(t, tCtx, client, ns.Name, svc.Name,
		[]string{corev1.IsHeadlessService, "new-label"}, []string{"app"})
}

// TestEndpointsHeadlessLabel tests how the service.kubernetes.io/headless
// label is applied to Endpoints and EndpointSlices based on Service properties.
func TestEndpointsHeadlessLabel(t *testing.T) {
	tCtx := ktesting.Init(t)

	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	epController := endpoint.NewEndpointController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Endpoints(),
		client,
		0)

	epsController := endpointslice.NewController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Nodes(),
		informers.Discovery().V1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second)

	// Start informer and controllers
	informers.Start(tCtx.Done())
	go epController.Run(tCtx, 1)
	go epsController.Run(tCtx, 1)

	testCases := []struct {
		name          string
		serviceLabels map[string]string
		isHeadless    bool
	}{
		{
			name:          "non-headless service with no labels",
			serviceLabels: nil,
			isHeadless:    false,
		},
		{
			name:          "headless service with no labels",
			serviceLabels: nil,
			isHeadless:    true,
		},
		{
			name:          "non-headless service with headless label",
			serviceLabels: map[string]string{corev1.IsHeadlessService: ""},
			isHeadless:    false,
		},
		{
			name:          "headless service with headless label",
			serviceLabels: map[string]string{corev1.IsHeadlessService: ""},
			isHeadless:    true,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nsName := fmt.Sprintf("test-headless-label-%d", i)
			ns := framework.CreateNamespaceOrDie(client, nsName, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			svcName := fmt.Sprintf("test-svc-%d", i)
			svc := &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   svcName,
					Labels: tc.serviceLabels,
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "test"},
					Ports: []corev1.ServicePort{{
						Port:       80,
						TargetPort: intstr.FromInt(8080),
					}},
				},
			}

			if tc.isHeadless {
				svc.Spec.ClusterIP = corev1.ClusterIPNone
			}

			_, err := client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating service: %v", err)
			}

			if tc.isHeadless {
				validateLabelsOnEndpointAndEndpointSlice(t, tCtx, client, ns.Name, svc.Name,
					[]string{corev1.IsHeadlessService}, nil)
			} else {
				validateLabelsOnEndpointAndEndpointSlice(t, tCtx, client, ns.Name, svc.Name,
					nil, []string{corev1.IsHeadlessService})
			}

		})
	}
}

// validateLabelsOnEndpointAndEndpointSlice is a helper function to check for the
// presence of expected labels and absence of unexpected labels on an Endpoint
// and its corresponding EndpointSlice.
func validateLabelsOnEndpointAndEndpointSlice(t *testing.T, tCtx context.Context, client clientset.Interface, nsName, svcName string, wantLabels, notWantLabels []string) {
	t.Helper()
	// Validate Endpoint
	err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 1*time.Minute, true, func(context.Context) (bool, error) {
		endpoint, err := client.CoreV1().Endpoints(nsName).Get(tCtx, svcName, metav1.GetOptions{})
		if err != nil {
			t.Logf("Error getting endpoints: %v", err)
			return false, nil
		}

		for _, label := range wantLabels {
			if _, ok := endpoint.Labels[label]; !ok {
				t.Logf("Expected %s label on Endpoints", label)
				return false, nil
			}
		}
		for _, label := range notWantLabels {
			if _, ok := endpoint.Labels[label]; ok {
				t.Logf("Unexpected %s label on Endpoints", label)
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for Endpoint labels: %v", err)
	}

	// Validate EndpointSlice
	err = wait.PollUntilContextTimeout(tCtx, 1*time.Second, 1*time.Minute, true, func(context.Context) (bool, error) {
		lSelector := discovery.LabelServiceName + "=" + svcName
		esList, err := client.DiscoveryV1().EndpointSlices(nsName).List(tCtx, metav1.ListOptions{LabelSelector: lSelector})
		if err != nil {
			t.Logf("Error listing EndpointSlices: %v", err)
			return false, err
		}

		if len(esList.Items) == 0 {
			t.Logf("Waiting for EndpointSlice to be created")
			return false, nil
		}
		if len(esList.Items) != 1 {
			t.Logf("Only expected 1 EndpointSlice, got %d", len(esList.Items))
			return false, nil
		}

		endpointSlice := esList.Items[0]
		for _, label := range wantLabels {
			if _, ok := endpointSlice.Labels[label]; !ok {
				t.Logf("Expected %s label on EndpointSlice", label)
				return false, nil
			}
		}
		for _, label := range notWantLabels {
			if _, ok := endpointSlice.Labels[label]; ok {
				t.Logf("Unexpected %s label on EndpointSlice", label)
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for EndpointSlice labels: %v", err)
	}
}
