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
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
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

func Test_ServiceClusterIPSelector(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-external-name-drops-internal-traffic-policy", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// create headless service
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-headless",
			Namespace: ns.Name,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Type:      corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{{
				Port: int32(80),
			}},
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	_, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	// informer to watch only non-headless services
	kubeInformers := informers.NewSharedInformerFactoryWithOptions(client, 0, informers.WithTweakListOptions(func(options *metav1.ListOptions) {
		options.FieldSelector = fields.OneTermNotEqualSelector("spec.clusterIP", corev1.ClusterIPNone).String()
	}))

	serviceInformer := kubeInformers.Core().V1().Services().Informer()
	serviceLister := kubeInformers.Core().V1().Services().Lister()
	serviceHasSynced := serviceInformer.HasSynced
	if _, err = serviceInformer.AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				svc := obj.(*corev1.Service)
				t.Logf("Added Service %#v", svc)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldSvc := oldObj.(*corev1.Service)
				newSvc := newObj.(*corev1.Service)
				t.Logf("Updated Service %#v to %#v", oldSvc, newSvc)
			},
			DeleteFunc: func(obj interface{}) {
				svc := obj.(*corev1.Service)
				t.Logf("Deleted Service %#v", svc)
			},
		},
	); err != nil {
		t.Fatalf("Error adding service informer handler: %v", err)
	}
	kubeInformers.Start(ctx.Done())
	cache.WaitForCacheSync(ctx.Done(), serviceHasSynced)
	svcs, err := serviceLister.List(labels.Everything())
	if err != nil {
		t.Fatalf("Error listing services: %v", err)
	}
	// only the kubernetes.default service expected
	if len(svcs) != 1 || svcs[0].Name != "kubernetes" {
		t.Fatalf("expected 1 services, got %d", len(svcs))
	}

	// create a new service with ClusterIP
	service2 := service.DeepCopy()
	service2.Spec.ClusterIP = ""
	service2.Name = "test-clusterip"
	_, err = client.CoreV1().Services(ns.Name).Create(ctx, service2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		svc, err := serviceLister.Services(service2.Namespace).Get(service2.Name)
		if svc == nil || err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error waiting for test service test-clusterip: %v", err)
	}

	// mutate the Service to drop the ClusterIP, theoretically ClusterIP is inmutable but ...
	service.Spec.ExternalName = "test"
	service.Spec.Type = corev1.ServiceTypeExternalName
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		svc, err := serviceLister.Services(service.Namespace).Get(service.Name)
		if svc == nil || err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error waiting for test service without ClusterIP: %v", err)
	}

	// mutate the Service to get the ClusterIP again
	service.Spec.ExternalName = ""
	service.Spec.ClusterIP = ""
	service.Spec.Type = corev1.ServiceTypeClusterIP
	_, err = client.CoreV1().Services(ns.Name).Update(ctx, service, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		svc, err := serviceLister.Services(service.Namespace).Get(service.Name)
		if svc == nil || err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error waiting for test service with ClusterIP: %v", err)
	}
}
