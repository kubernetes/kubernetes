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

package service

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	servicecontroller "k8s.io/cloud-provider/controllers/service"
	fakecloud "k8s.io/cloud-provider/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	controllersmetrics "k8s.io/component-base/metrics/prometheus/controllers"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
)

// Test_ServiceLoadBalancerAllocateNodePorts tests that a Service with spec.allocateLoadBalancerNodePorts=false
// does not allocate node ports for the Service.
func Test_ServiceLoadBalancerDisableAllocateNodePorts(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-allocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(false),
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

	if serviceHasNodePorts(service) {
		t.Error("found node ports when none was expected")
	}
}

// Test_ServiceUpdateLoadBalancerAllocateNodePorts tests that a Service that is updated from ClusterIP to LoadBalancer
// with spec.allocateLoadBalancerNodePorts=false does not allocate node ports for the Service
func Test_ServiceUpdateLoadBalancerDisableAllocateNodePorts(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-allocate-node-ports", t)
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

	if serviceHasNodePorts(service) {
		t.Error("found node ports when none was expected")
	}

	service.Spec.Type = corev1.ServiceTypeLoadBalancer
	service.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(false)
	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if serviceHasNodePorts(service) {
		t.Error("found node ports when none was expected")
	}
}

// Test_ServiceLoadBalancerSwitchToDeallocatedNodePorts test that switching a Service
// to spec.allocateLoadBalancerNodePorts=false, does not de-allocate existing node ports.
func Test_ServiceLoadBalancerEnableThenDisableAllocatedNodePorts(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-deallocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(true),
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

	if !serviceHasNodePorts(service) {
		t.Error("expected node ports but found none")
	}

	service.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(false)
	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if !serviceHasNodePorts(service) {
		t.Error("node ports were unexpectedly deallocated")
	}
}

// Test_ServiceLoadBalancerDisableAllocatedNodePort test that switching a Service
// to spec.allocateLoadBalancerNodePorts=false can de-allocate existing node ports.
func Test_ServiceLoadBalancerDisableAllocatedNodePort(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-deallocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(true),
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

	if !serviceHasNodePorts(service) {
		t.Error("expected node ports but found none")
	}

	service.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(false)
	service.Spec.Ports[0].NodePort = 0
	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if serviceHasNodePorts(service) {
		t.Error("node ports were expected to be deallocated")
	}
}

// Test_ServiceLoadBalancerDisableAllocatedNodePorts test that switching a Service
// to spec.allocateLoadBalancerNodePorts=false can de-allocate one of existing node ports.
func Test_ServiceLoadBalancerDisableAllocatedNodePorts(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-deallocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(true),
			Ports: []corev1.ServicePort{{
				Name: "np-1",
				Port: int32(80),
			}, {
				Name: "np-2",
				Port: int32(81),
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

	if !serviceHasNodePorts(service) {
		t.Error("expected node ports but found none")
	}

	service.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(false)
	service.Spec.Ports[0].NodePort = 0
	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if service.Spec.Ports[0].NodePort != 0 {
		t.Error("node ports[0] was expected to be deallocated")
	}
	if service.Spec.Ports[1].NodePort == 0 {
		t.Error("node ports was not expected to be deallocated")
	}
}

// Test_ServiceLoadBalancerDisableAllocatedNodePortsByPatch test that switching a Service
// to spec.allocateLoadBalancerNodePorts=false with path can de-allocate one of existing node ports.
func Test_ServiceLoadBalancerDisableAllocatedNodePortsByPatch(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-deallocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(true),
			Ports: []corev1.ServicePort{{
				Name: "np-1",
				Port: int32(80),
			}, {
				Name: "np-2",
				Port: int32(81),
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

	if !serviceHasNodePorts(service) {
		t.Error("expected node ports but found none")
	}

	clone := service.DeepCopy()
	clone.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(false)
	clone.Spec.Ports[0].NodePort = 0

	oldData, err := json.Marshal(service)
	if err != nil {
		t.Fatalf("Error marshalling test service: %v", err)
	}
	newData, err := json.Marshal(clone)
	if err != nil {
		t.Fatalf("Error marshalling test service: %v", err)
	}
	patch, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, corev1.Service{})
	if err != nil {
		t.Fatalf("Error creating patch: %v", err)
	}

	service, err = client.CoreV1().Services(ns.Name).Patch(context.TODO(), service.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if service.Spec.Ports[0].NodePort != 0 {
		t.Error("node ports[0] was expected to be deallocated")
	}
	if service.Spec.Ports[1].NodePort == 0 {
		t.Error("node ports was not expected to be deallocated")
	}
}

// Test_ServiceLoadBalancerDisableThenEnableAllocatedNodePorts test that switching a Service
// to spec.allocateLoadBalancerNodePorts=true from false, allocate new node ports.
func Test_ServiceLoadBalancerDisableThenEnableAllocatedNodePorts(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-reallocate-node-ports", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type:                          corev1.ServiceTypeLoadBalancer,
			AllocateLoadBalancerNodePorts: utilpointer.Bool(false),
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

	if serviceHasNodePorts(service) {
		t.Error("not expected node ports but found one")
	}

	service.Spec.AllocateLoadBalancerNodePorts = utilpointer.Bool(true)
	service, err = client.CoreV1().Services(ns.Name).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating test service: %v", err)
	}

	if !serviceHasNodePorts(service) {
		t.Error("expected node ports but found none")
	}
}

func serviceHasNodePorts(svc *corev1.Service) bool {
	for _, port := range svc.Spec.Ports {
		if port.NodePort > 0 {
			return true
		}
	}

	return false
}

// Test_ServiceLoadBalancerEnableLoadBalancerClass tests that when a LoadBalancer
// type of service has spec.LoadBalancerClass set, cloud provider should not create default load balancer.
func Test_ServiceLoadBalancerEnableLoadBalancerClass(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-load-balancer-class", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	controller, cloud, informer := newServiceController(t, client)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informer.Start(ctx.Done())
	go controller.Run(ctx, 1, controllersmetrics.NewControllerManagerMetrics("loadbalancer-test"))

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-load-balancer-class",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeLoadBalancer,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
			LoadBalancerClass: utilpointer.String("test.com/test"),
		},
	}

	_, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	time.Sleep(5 * time.Second) // sleep 5 second to wait for the service controller reconcile
	if len(cloud.Calls) > 0 {
		t.Errorf("Unexpected cloud provider calls: %v", cloud.Calls)
	}
}

// Test_SetLoadBalancerClassThenUpdateLoadBalancerClass tests that when a LoadBalancer
// type of service has spec.LoadBalancerClass set, it should be immutable as long as the service type
// is still LoadBalancer.
func Test_SetLoadBalancerClassThenUpdateLoadBalancerClass(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-immutable-load-balancer-class", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	controller, cloud, informer := newServiceController(t, client)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informer.Start(ctx.Done())
	go controller.Run(ctx, 1, controllersmetrics.NewControllerManagerMetrics("loadbalancer-test"))

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-load-balancer-class",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeLoadBalancer,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
			LoadBalancerClass: utilpointer.String("test.com/test"),
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	service.Spec.LoadBalancerClass = utilpointer.String("test.com/update")
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, service, metav1.UpdateOptions{})
	if err == nil {
		t.Fatal("Error: updating test service load balancer class should throw error, field is immutable")
	}

	time.Sleep(5 * time.Second) // sleep 5 second to wait for the service controller reconcile
	if len(cloud.Calls) > 0 {
		t.Errorf("Unexpected cloud provider calls: %v", cloud.Calls)
	}
}

// Test_UpdateLoadBalancerWithLoadBalancerClass tests that when a Load Balancer type of Service that
// is updated from non loadBalancerClass set to loadBalancerClass set, it should be not allowed.
func Test_UpdateLoadBalancerWithLoadBalancerClass(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-update-load-balancer-class", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	controller, cloud, informer := newServiceController(t, client)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informer.Start(ctx.Done())
	go controller.Run(ctx, 1, controllersmetrics.NewControllerManagerMetrics("loadbalancer-test"))

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-update-load-balancer-class",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeLoadBalancer,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
		},
	}

	service, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	service.Spec.LoadBalancerClass = utilpointer.String("test.com/test")
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, service, metav1.UpdateOptions{})
	if err == nil {
		t.Fatal("Error: updating test service load balancer class should throw error, field is immutable")
	}

	time.Sleep(5 * time.Second) // sleep 5 second to wait for the service controller reconcile
	if len(cloud.Calls) == 0 {
		t.Errorf("expected cloud provider calls to create load balancer")
	}
}

// Test_ServiceLoadBalancerMixedProtocolSetup tests that a LoadBalancer Service with different protocol values
// can be created.
func Test_ServiceLoadBalancerMixedProtocolSetup(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-mixed-protocols", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	controller, cloud, informer := newServiceController(t, client)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informer.Start(ctx.Done())
	go controller.Run(ctx, 1, controllersmetrics.NewControllerManagerMetrics("loadbalancer-test"))

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeLoadBalancer,
			Ports: []corev1.ServicePort{
				{
					Name:     "tcpport",
					Port:     int32(53),
					Protocol: corev1.ProtocolTCP,
				},
				{
					Name:     "udpport",
					Port:     int32(53),
					Protocol: corev1.ProtocolUDP,
				},
			},
		},
	}

	_, err = client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	time.Sleep(5 * time.Second) // sleep 5 second to wait for the service controller reconcile
	if len(cloud.Calls) == 0 {
		t.Errorf("expected cloud provider calls to create load balancer")
	}
}

func newServiceController(t *testing.T, client *clientset.Clientset) (*servicecontroller.Controller, *fakecloud.Cloud, informers.SharedInformerFactory) {
	cloud := &fakecloud.Cloud{}
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	serviceInformer := informerFactory.Core().V1().Services()
	nodeInformer := informerFactory.Core().V1().Nodes()

	controller, err := servicecontroller.New(cloud,
		client,
		serviceInformer,
		nodeInformer,
		"test-cluster",
		utilfeature.DefaultFeatureGate)
	if err != nil {
		t.Fatalf("Error creating service controller: %v", err)
	}
	cloud.ClearCalls() // ignore any cloud calls made in init()
	return controller, cloud, informerFactory
}

// Test_ServiceLoadBalancerIPMode tests whether the cloud provider has correctly updated the ipMode field.
func Test_ServiceLoadBalancerIPMode(t *testing.T) {
	ipModeVIP := corev1.LoadBalancerIPModeVIP
	testCases := []struct {
		ipModeEnabled  bool
		externalIP     string
		expectedIPMode *corev1.LoadBalancerIPMode
	}{
		{
			ipModeEnabled:  false,
			externalIP:     "1.2.3.4",
			expectedIPMode: nil,
		},
		{
			ipModeEnabled:  true,
			externalIP:     "1.2.3.5",
			expectedIPMode: &ipModeVIP,
		},
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, tc.ipModeEnabled)
			server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
			defer server.TearDownFn()

			client, err := clientset.NewForConfig(server.ClientConfig)
			if err != nil {
				t.Fatalf("Error creating clientset: %v", err)
			}

			ns := framework.CreateNamespaceOrDie(client, "test-service-update-load-balancer-ip-mode", t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			controller, cloud, informer := newServiceController(t, client)
			cloud.ExternalIP = net.ParseIPSloppy(tc.externalIP)

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			informer.Start(ctx.Done())
			go controller.Run(ctx, 1, controllersmetrics.NewControllerManagerMetrics("loadbalancer-test"))

			service := &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-update-load-balancer-ip-mode",
				},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{{
						Port: int32(80),
					}},
				},
			}

			service, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error creating test service: %v", err)
			}

			time.Sleep(5 * time.Second) // sleep 5 second to wait for the service controller reconcile
			service, err = client.CoreV1().Services(ns.Name).Get(ctx, service.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Error getting test service: %v", err)
			}

			if len(service.Status.LoadBalancer.Ingress) == 0 {
				t.Fatalf("unexpected load balancer status")
			}

			gotIngress := service.Status.LoadBalancer.Ingress[0]
			if gotIngress.IP != tc.externalIP || !reflect.DeepEqual(gotIngress.IPMode, tc.expectedIPMode) {
				t.Errorf("unexpected load balancer ingress, got ingress %v, expected IP %v, expected ipMode %v",
					gotIngress, tc.externalIP, tc.expectedIPMode)
			}
		})
	}
}
