/*
Copyright 2022 The Kubernetes Authors.

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

package service

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// Test_ExternalNameServiceStopsDefaultingInternalTrafficPolicy tests that Services no longer default
// the internalTrafficPolicy field when Type is ExternalName. This test exists due to historic reasons where
// the internalTrafficPolicy field was being defaulted in older versions. New versions stop defaulting the
// field and drop on read, but for compatibility reasons we still accept the field.
func Test_ExternalNameServiceStopsDefaultingInternalTrafficPolicy(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-external-name-drops-internal-traffic-policy", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: "foo.bar.com",
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}
}

// Test_ExternalNameServiceDropsInternalTrafficPolicy tests that Services accepts the internalTrafficPolicy field on Create,
// but drops the field on read. This test exists due to historic reasons where the internalTrafficPolicy field was being defaulted
// in older versions. New versions stop defaulting the field and drop on read, but for compatibility reasons we still accept the field.
func Test_ExternalNameServiceDropsInternalTrafficPolicy(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-external-name-drops-internal-traffic-policy", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	internalTrafficPolicy := corev1.ServiceInternalTrafficPolicyCluster
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                  corev1.ServiceTypeExternalName,
			ExternalName:          "foo.bar.com",
			InternalTrafficPolicy: &internalTrafficPolicy,
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}
}

// Test_ConvertingToExternalNameServiceDropsInternalTrafficPolicy tests that converting a Service to Type=ExternalName
// results in the internalTrafficPolicy field being dropped.This test exists due to historic reasons where the internalTrafficPolicy
// field was being defaulted in older versions. New versions stop defaulting the field and drop on read, but for compatibility reasons
// we still accept the field.
func Test_ConvertingToExternalNameServiceDropsInternalTrafficPolicy(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-external-name-drops-internal-traffic-policy", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if *service.Spec.InternalTrafficPolicy != corev1.ServiceInternalTrafficPolicyCluster {
		t.Error("service internalTrafficPolicy was not set for clusterIP Service")
	}

	newService := service.DeepCopy()
	newService.Spec.Type = corev1.ServiceTypeExternalName
	newService.Spec.ExternalName = "foo.bar.com"

	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), newService, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("error updating service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.InternalTrafficPolicy != nil {
		t.Errorf("service internalTrafficPolicy should be droppped but is set: %v", service.Spec.InternalTrafficPolicy)
	}
}

// Test_RemovingExternalIPsFromClusterIPServiceDropsExternalTrafficPolicy tests that removing externalIPs from a
// ClusterIP Service results in the externalTrafficPolicy field being dropped.
func Test_RemovingExternalIPsFromClusterIPServiceDropsExternalTrafficPolicy(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-removing-external-ips-drops-external-traffic-policy", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
			Selector: map[string]string{
				"foo": "bar",
			},
			ExternalIPs: []string{"1.1.1.1"},
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	if service.Spec.ExternalTrafficPolicy != corev1.ServiceExternalTrafficPolicyCluster {
		t.Error("service externalTrafficPolicy was not set for clusterIP Service with externalIPs")
	}

	// externalTrafficPolicy should be dropped after removing externalIPs.
	newService := service.DeepCopy()
	newService.Spec.ExternalIPs = []string{}

	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), newService, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("error updating service: %v", err)
	}

	if service.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("service externalTrafficPolicy should be droppped but is set: %v", service.Spec.ExternalTrafficPolicy)
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("service externalTrafficPolicy should be droppped but is set: %v", service.Spec.ExternalTrafficPolicy)
	}

	// externalTrafficPolicy should be set after adding externalIPs again.
	newService = service.DeepCopy()
	newService.Spec.ExternalIPs = []string{"1.1.1.1"}

	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), newService, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("error updating service: %v", err)
	}

	if service.Spec.ExternalTrafficPolicy != corev1.ServiceExternalTrafficPolicyCluster {
		t.Error("service externalTrafficPolicy was not set for clusterIP Service with externalIPs")
	}

	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}

	if service.Spec.ExternalTrafficPolicy != corev1.ServiceExternalTrafficPolicyCluster {
		t.Error("service externalTrafficPolicy was not set for clusterIP Service with externalIPs")
	}
}

// Test transitions involving the `trafficDistribution` field in Service spec.
func Test_TransitionsForTrafficDistribution(t *testing.T) {

	////////////////////////////////////////////////////////////////////////////
	// Setup components, like kube-apiserver and EndpointSlice controller.
	////////////////////////////////////////////////////////////////////////////

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceTrafficDistribution, true)()

	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(client, resyncPeriod)

	ctx := ktesting.Init(t)
	defer ctx.Cancel("test has completed")
	epsController := endpointslice.NewController(
		ctx,
		informers.Core().V1().Pods(),
		informers.Core().V1().Services(),
		informers.Core().V1().Nodes(),
		informers.Discovery().V1().EndpointSlices(),
		int32(100),
		client,
		1*time.Second,
	)

	informers.Start(ctx.Done())
	go epsController.Run(ctx, 1)

	////////////////////////////////////////////////////////////////////////////
	// Create a namespace, node, pod in the node, and a service exposing the pod.
	////////////////////////////////////////////////////////////////////////////

	ns := framework.CreateNamespaceOrDie(client, "test-service-traffic-distribution", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fake-node",
			Labels: map[string]string{
				corev1.LabelTopologyZone: "fake-zone-1",
			},
		},
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: ns.GetName(),
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: node.GetName(),
			Containers: []corev1.Container{
				{
					Name:  "fake-name",
					Image: "fake-image",
					Ports: []corev1.ContainerPort{
						{
							Name:          "port-443",
							ContainerPort: 443,
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{
					Type:   corev1.PodReady,
					Status: corev1.ConditionTrue,
				},
			},
			PodIP: "10.0.0.1",
			PodIPs: []corev1.PodIP{
				{
					IP: "10.0.0.1",
				},
			},
		},
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-service",
			Namespace: ns.GetName(),
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Ports: []corev1.ServicePort{
				{Name: "port-443", Port: 443, Protocol: "TCP", TargetPort: intstr.FromInt32(443)},
			},
		},
	}

	_, err = client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test node: %v", err)
	}
	_, err = client.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test ready pod: %v", err)
	}
	_, err = client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status for test pod to Ready: %v", err)
	}
	_, err = client.CoreV1().Services(ns.Name).Create(ctx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test service: %v", err)
	}

	////////////////////////////////////////////////////////////////////////////
	// Assert that without the presence of `trafficDistribution` field and the
	// service.kubernetes.io/topology-mode=Auto annotation, there are no zone
	// hints in EndpointSlice.
	////////////////////////////////////////////////////////////////////////////

	// logsBuffer captures logs during assertions which multiple retires. These
	// will only be printed if the assertion failed.
	logsBuffer := &bytes.Buffer{}

	endpointSlicesHaveNoHints := func(ctx context.Context) (bool, error) {
		slices, err := client.DiscoveryV1().EndpointSlices(ns.GetName()).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svc.GetName())})
		if err != nil {
			fmt.Fprintf(logsBuffer, "failed to list EndpointSlices for service %q: %v\n", svc.GetName(), err)
			return false, nil
		}
		if slices == nil || len(slices.Items) == 0 {
			fmt.Fprintf(logsBuffer, "no EndpointSlices returned for service %q\n", svc.GetName())
			return false, nil
		}
		fmt.Fprintf(logsBuffer, "EndpointSlices=\n%v\n", format.Object(slices, 1 /* indent one level */))

		for _, slice := range slices.Items {
			for _, endpoint := range slice.Endpoints {
				var ip string
				if len(endpoint.Addresses) > 0 {
					ip = endpoint.Addresses[0]
				}
				if endpoint.Hints != nil && len(endpoint.Hints.ForZones) != 0 {
					fmt.Fprintf(logsBuffer, "endpoint with ip %v has hint %+v, want no hint\n", ip, endpoint.Hints)
					return false, nil
				}
			}
		}
		return true, nil
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, endpointSlicesHaveNoHints)
	if err != nil {
		t.Logf("logsBuffer=\n%v", logsBuffer)
		t.Fatalf("Error waiting for EndpointSlices to have same zone hints: %v", err)
	}
	logsBuffer.Reset()

	////////////////////////////////////////////////////////////////////////////
	// Update the service by setting the `trafficDistribution: PreferLocal` field
	//
	// Assert that the respective EndpointSlices get the same-zone hints.
	////////////////////////////////////////////////////////////////////////////

	trafficDist := corev1.ServiceTrafficDistributionPreferClose
	svc.Spec.TrafficDistribution = &trafficDist
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, svc, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update test service with 'trafficDistribution: PreferLocal': %v", err)
	}

	endpointSlicesHaveSameZoneHints := func(ctx context.Context) (bool, error) {
		slices, err := client.DiscoveryV1().EndpointSlices(ns.GetName()).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svc.GetName())})
		if err != nil {
			fmt.Fprintf(logsBuffer, "failed to list EndpointSlices for service %q: %v\n", svc.GetName(), err)
			return false, nil
		}
		if slices == nil || len(slices.Items) == 0 {
			fmt.Fprintf(logsBuffer, "no EndpointSlices returned for service %q\n", svc.GetName())
			return false, nil
		}
		fmt.Fprintf(logsBuffer, "EndpointSlices=\n%v\n", format.Object(slices, 1 /* indent one level */))

		for _, slice := range slices.Items {
			for _, endpoint := range slice.Endpoints {
				var ip string
				if len(endpoint.Addresses) > 0 {
					ip = endpoint.Addresses[0]
				}
				var zone string
				if endpoint.Zone != nil {
					zone = *endpoint.Zone
				}
				if endpoint.Hints == nil || len(endpoint.Hints.ForZones) != 1 || endpoint.Hints.ForZones[0].Name != zone {
					fmt.Fprintf(logsBuffer, "endpoint with ip %v does not have the correct hint, want hint for zone %q\n", ip, zone)
					return false, nil
				}
			}
		}
		return true, nil
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, endpointSlicesHaveSameZoneHints)
	if err != nil {
		t.Logf("logsBuffer=\n%v", logsBuffer)
		t.Fatalf("Error waiting for EndpointSlices to have same zone hints: %v", err)
	}
	logsBuffer.Reset()

	////////////////////////////////////////////////////////////////////////////
	// Update the service with the service.kubernetes.io/topology-mode=Auto
	// annotation.
	//
	// Assert that the EndpointSlice for service have no hints once
	// service.kubernetes.io/topology-mode=Auto takes affect, since topology
	// annotation would not work with only one service pod.
	////////////////////////////////////////////////////////////////////////////
	svc.Annotations = map[string]string{corev1.AnnotationTopologyMode: "Auto"}
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, svc, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update test service with 'service.kubernetes.io/topology-mode=Auto' annotation: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, endpointSlicesHaveNoHints)
	if err != nil {
		t.Logf("logsBuffer=\n%v", logsBuffer)
		t.Fatalf("Error waiting for EndpointSlices to have no hints: %v", err)
	}
	logsBuffer.Reset()

	////////////////////////////////////////////////////////////////////////////
	// Remove the annotation service.kubernetes.io/topology-mode=Auto from the
	// service.
	//
	// Assert that EndpointSlice for service again has the correct same-zone
	// hints because of the `trafficDistribution: PreferLocal` field.
	////////////////////////////////////////////////////////////////////////////
	svc.Annotations = map[string]string{}
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, svc, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to remove annotation 'service.kubernetes.io/topology-mode=Auto' from service: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, endpointSlicesHaveSameZoneHints)
	if err != nil {
		t.Logf("logsBuffer=\n%v", logsBuffer)
		t.Fatalf("Error waiting for EndpointSlices to have same zone hints: %v", err)
	}
	logsBuffer.Reset()

	////////////////////////////////////////////////////////////////////////////
	// Remove the field `trafficDistribution: PreferLocal` from the service.
	//
	// Assert that EndpointSlice for service again has no zone hints.
	////////////////////////////////////////////////////////////////////////////
	svc.Spec.TrafficDistribution = nil
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, svc, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to remove annotation 'service.kubernetes.io/topology-mode=Auto' from service: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, endpointSlicesHaveNoHints)
	if err != nil {
		t.Logf("logsBuffer=\n%v", logsBuffer)
		t.Fatalf("Error waiting for EndpointSlices to have no hints: %v", err)
	}
	logsBuffer.Reset()
}
