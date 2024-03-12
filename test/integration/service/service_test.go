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
	"encoding/json"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/util/retry"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
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

// Repro https://github.com/kubernetes/kubernetes/issues/123853
func Test_ServiceWatchUntil(t *testing.T) {
	svcReadyTimeout := 1 * time.Minute

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(client, "test-service-watchuntil", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	testSvcName := "test-service-" + utilrand.String(5)
	testSvcLabels := map[string]string{"test-service-static": "true"}
	testSvcLabelsFlat := "test-service-static=true"

	w := &cache.ListWatch{
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.LabelSelector = testSvcLabelsFlat
			return client.CoreV1().Services(ns.Name).Watch(ctx, options)
		},
	}

	svcList, err := client.CoreV1().Services("").List(ctx, metav1.ListOptions{LabelSelector: testSvcLabelsFlat})
	if err != nil {
		t.Fatalf("failed to list Services: %v", err)
	}
	// create  service
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:   testSvcName,
			Labels: testSvcLabels,
		},
		Spec: corev1.ServiceSpec{
			Type: "LoadBalancer",
			Ports: []corev1.ServicePort{{
				Name:       "http",
				Protocol:   corev1.ProtocolTCP,
				Port:       int32(80),
				TargetPort: intstr.FromInt32(80),
			}},
			LoadBalancerClass: ptr.To[string]("example.com/internal-vip"),
		},
	}
	_, err = client.CoreV1().Services(ns.Name).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}

	ctxUntil, cancel := context.WithTimeout(ctx, svcReadyTimeout)
	defer cancel()
	_, err = watchtools.Until(ctxUntil, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		if svc, ok := event.Object.(*corev1.Service); ok {
			found := svc.ObjectMeta.Name == service.ObjectMeta.Name &&
				svc.ObjectMeta.Namespace == ns.Name &&
				svc.Labels["test-service-static"] == "true"
			if !found {
				t.Logf("observed Service %v in namespace %v with labels: %v & ports %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Spec.Ports)
				return false, nil
			}
			t.Logf("Found Service %v in namespace %v with labels: %v & ports %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Spec.Ports)
			return found, nil
		}
		t.Logf("Observed event: %+v", event.Object)
		return false, nil
	})
	if err != nil {
		t.Fatalf("Error service not found: %v", err)
	}

	t.Log("patching the ServiceStatus")
	lbStatus := corev1.LoadBalancerStatus{
		Ingress: []corev1.LoadBalancerIngress{{IP: "203.0.113.1"}},
	}
	lbStatusJSON, err := json.Marshal(lbStatus)
	if err != nil {
		t.Fatalf("Error marshalling status: %v", err)
	}
	_, err = client.CoreV1().Services(ns.Name).Patch(ctx, testSvcName, types.MergePatchType,
		[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"loadBalancer":`+string(lbStatusJSON)+`}}`),
		metav1.PatchOptions{}, "status")
	if err != nil {
		t.Fatalf("Could not patch service status: %v", err)
	}

	t.Log("watching for the Service to be patched")
	ctxUntil, cancel = context.WithTimeout(ctx, svcReadyTimeout)
	defer cancel()

	_, err = watchtools.Until(ctxUntil, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		if svc, ok := event.Object.(*corev1.Service); ok {
			found := svc.ObjectMeta.Name == service.ObjectMeta.Name &&
				svc.ObjectMeta.Namespace == ns.Name &&
				svc.Annotations["patchedstatus"] == "true"
			if !found {
				t.Logf("observed Service %v in namespace %v with annotations: %v & LoadBalancer: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
				return false, nil
			}
			t.Logf("Found Service %v in namespace %v with annotations: %v & LoadBalancer: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
			return found, nil
		}
		t.Logf("Observed event: %+v", event.Object)
		return false, nil
	})
	if err != nil {
		t.Fatalf("failed to locate Service %v in namespace %v", service.ObjectMeta.Name, ns)
	}
	t.Logf("Service %s has service status patched", testSvcName)

	t.Log("updating the ServiceStatus")

	var statusToUpdate, updatedStatus *corev1.Service
	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		statusToUpdate, err = client.CoreV1().Services(ns.Name).Get(ctx, testSvcName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, metav1.Condition{
			Type:    "StatusUpdate",
			Status:  metav1.ConditionTrue,
			Reason:  "E2E",
			Message: "Set from e2e test",
		})

		updatedStatus, err = client.CoreV1().Services(ns.Name).UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		t.Fatalf("\n\n Failed to UpdateStatus. %v\n\n", err)
	}
	t.Logf("updatedStatus.Conditions: %#v", updatedStatus.Status.Conditions)

	t.Log("watching for the Service to be updated")
	ctxUntil, cancel = context.WithTimeout(ctx, svcReadyTimeout)
	defer cancel()
	_, err = watchtools.Until(ctxUntil, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		if svc, ok := event.Object.(*corev1.Service); ok {
			found := svc.ObjectMeta.Name == service.ObjectMeta.Name &&
				svc.ObjectMeta.Namespace == ns.Name &&
				svc.Annotations["patchedstatus"] == "true"
			if !found {
				t.Logf("Observed Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
				return false, nil
			}
			for _, cond := range svc.Status.Conditions {
				if cond.Type == "StatusUpdate" &&
					cond.Reason == "E2E" &&
					cond.Message == "Set from e2e test" {
					t.Logf("Found Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.Conditions)
					return found, nil
				} else {
					t.Logf("Observed Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
					return false, nil
				}
			}
		}
		t.Logf("Observed event: %+v", event.Object)
		return false, nil
	})
	if err != nil {
		t.Fatalf("failed to locate Service %v in namespace %v", service.ObjectMeta.Name, ns)
	}
	t.Logf("Service %s has service status updated", testSvcName)

	t.Log("patching the service")
	servicePatchPayload, err := json.Marshal(corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"test-service": "patched",
			},
		},
	})

	_, err = client.CoreV1().Services(ns.Name).Patch(ctx, testSvcName, types.StrategicMergePatchType, []byte(servicePatchPayload), metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("failed to patch service. %v", err)
	}

	t.Log("watching for the Service to be patched")
	ctxUntil, cancel = context.WithTimeout(ctx, svcReadyTimeout)
	defer cancel()
	_, err = watchtools.Until(ctxUntil, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		if svc, ok := event.Object.(*corev1.Service); ok {
			found := svc.ObjectMeta.Name == service.ObjectMeta.Name &&
				svc.ObjectMeta.Namespace == ns.Name &&
				svc.Labels["test-service"] == "patched"
			if !found {
				t.Logf("observed Service %v in namespace %v with labels: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels)
				return false, nil
			}
			t.Logf("Found Service %v in namespace %v with labels: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels)
			return found, nil
		}
		t.Logf("Observed event: %+v", event.Object)
		return false, nil
	})
	if err != nil {
		t.Fatalf("failed to locate Service %v in namespace %v", service.ObjectMeta.Name, ns)
	}

	t.Logf("Service %s patched", testSvcName)

	t.Log("deleting the service")
	err = client.CoreV1().Services(ns.Name).Delete(ctx, testSvcName, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("failed to delete the Service. %v", err)
	}

	t.Log("watching for the Service to be deleted")
	ctxUntil, cancel = context.WithTimeout(ctx, svcReadyTimeout)
	defer cancel()
	_, err = watchtools.Until(ctxUntil, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			if svc, ok := event.Object.(*corev1.Service); ok {
				found := svc.ObjectMeta.Name == service.ObjectMeta.Name &&
					svc.ObjectMeta.Namespace == ns.Name &&
					svc.Labels["test-service-static"] == "true"
				if !found {
					t.Logf("observed Service %v in namespace %v with labels: %v & annotations: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Annotations)
					return false, nil
				}
				t.Logf("Found Service %v in namespace %v with labels: %v & annotations: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Annotations)
				return found, nil
			}
		default:
			t.Logf("Observed event: %+v", event.Type)
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("failed to delete Service %v in namespace %v", service.ObjectMeta.Name, ns)
	}
	t.Logf("Service %s deleted", testSvcName)
}
