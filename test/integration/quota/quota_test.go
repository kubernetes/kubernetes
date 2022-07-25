/*
Copyright 2015 The Kubernetes Authors.

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

package quota

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	resourceQuotaTimeout = 10 * time.Second
)

// 1.2 code gets:
// 	quota_test.go:95: Took 4.218619579s to scale up without quota
// 	quota_test.go:199: unexpected error: timed out waiting for the condition, ended with 342 pods (1 minute)
// 1.3+ code gets:
// 	quota_test.go:100: Took 4.196205966s to scale up without quota
// 	quota_test.go:115: Took 12.021640372s to scale up with quota
func TestQuota(t *testing.T) {
	// Set up a API server
	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
		},
	})
	defer tearDownFn()

	clientset := clientset.NewForConfigOrDie(kubeConfig)

	ns := framework.CreateNamespaceOrDie(clientset, "quotaed", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)
	ns2 := framework.CreateNamespaceOrDie(clientset, "non-quotaed", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns2, t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	rm := replicationcontroller.NewReplicationManager(
		informers.Core().V1().Pods(),
		informers.Core().V1().ReplicationControllers(),
		clientset,
		replicationcontroller.BurstReplicas,
	)
	go rm.Run(ctx, 3)

	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     informers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(ctx, 2)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, ctx.Done())

	informers.Start(ctx.Done())
	close(informersStarted)

	startTime := time.Now()
	scale(t, ns2.Name, clientset)
	endTime := time.Now()
	t.Logf("Took %v to scale up without quota", endTime.Sub(startTime))

	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods: resource.MustParse("1000"),
			},
		},
	}
	waitForQuota(t, quota, clientset)

	startTime = time.Now()
	scale(t, "quotaed", clientset)
	endTime = time.Now()
	t.Logf("Took %v to scale up with quota", endTime.Sub(startTime))
}

func waitForQuota(t *testing.T, quota *v1.ResourceQuota, clientset *clientset.Clientset) {
	w, err := clientset.CoreV1().ResourceQuotas(quota.Namespace).Watch(context.TODO(), metav1.SingleObject(metav1.ObjectMeta{Name: quota.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := clientset.CoreV1().ResourceQuotas(quota.Namespace).Create(context.TODO(), quota, metav1.CreateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}
		switch cast := event.Object.(type) {
		case *v1.ResourceQuota:
			if len(cast.Status.Hard) > 0 {
				return true, nil
			}
		}

		return false, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// waitForUsedResourceQuota polls a ResourceQuota status for an expected used value
func waitForUsedResourceQuota(t *testing.T, c clientset.Interface, ns, quotaName string, used v1.ResourceList) {
	err := wait.Poll(1*time.Second, resourceQuotaTimeout, func() (bool, error) {
		resourceQuota, err := c.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// used may not yet be calculated
		if resourceQuota.Status.Used == nil {
			return false, nil
		}

		// verify that the quota shows the expected used resource values
		for k, v := range used {
			actualValue, found := resourceQuota.Status.Used[k]
			if !found {
				t.Logf("resource %s was not found in ResourceQuota status", k)
				return false, nil
			}

			if !actualValue.Equal(v) {
				t.Logf("resource %s, expected %s, actual %s", k, v.String(), actualValue.String())
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Errorf("error waiting or ResourceQuota status: %v", err)
	}
}

func scale(t *testing.T, namespace string, clientset *clientset.Clientset) {
	target := int32(100)
	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &target,
			Selector: map[string]string{"foo": "bar"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container",
							Image: "busybox",
						},
					},
				},
			},
		},
	}

	w, err := clientset.CoreV1().ReplicationControllers(namespace).Watch(context.TODO(), metav1.SingleObject(metav1.ObjectMeta{Name: rc.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := clientset.CoreV1().ReplicationControllers(namespace).Create(context.TODO(), rc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}

		switch cast := event.Object.(type) {
		case *v1.ReplicationController:
			fmt.Printf("Found %v of %v replicas\n", int(cast.Status.Replicas), target)
			if cast.Status.Replicas == target {
				return true, nil
			}
		}

		return false, nil
	})
	if err != nil {
		pods, _ := clientset.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String(), FieldSelector: fields.Everything().String()})
		t.Fatalf("unexpected error: %v, ended with %v pods", err, len(pods.Items))
	}
}

func TestQuotaLimitedResourceDenial(t *testing.T) {
	// Create admission configuration with ResourceQuota configuration.
	admissionConfigFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())
	if err := os.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ResourceQuota
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: ResourceQuotaConfiguration
    limitedResources:
    - resource: pods
      matchContains:
      - pods
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// Set up an API server
	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
			opts.Admission.GenericAdmission.ConfigFile = admissionConfigFile.Name()

		},
	})
	defer tearDownFn()

	clientset := clientset.NewForConfigOrDie(kubeConfig)

	ns := framework.CreateNamespaceOrDie(clientset, "quota", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	rm := replicationcontroller.NewReplicationManager(
		informers.Core().V1().Pods(),
		informers.Core().V1().ReplicationControllers(),
		clientset,
		replicationcontroller.BurstReplicas,
	)
	go rm.Run(ctx, 3)

	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     informers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(ctx, 2)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, ctx.Done())

	informers.Start(ctx.Done())
	close(informersStarted)

	// try to create a pod
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container",
					Image: "busybox",
				},
			},
		},
	}
	if _, err := clientset.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{}); err == nil {
		t.Fatalf("expected error for insufficient quota")
	}

	// now create a covering quota
	// note: limited resource does a matchContains, so we now have "pods" matching "pods" and "count/pods"
	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods:               resource.MustParse("1000"),
				v1.ResourceName("count/pods"): resource.MustParse("1000"),
			},
		},
	}
	waitForQuota(t, quota, clientset)

	// attempt to create a new pod once the quota is propagated
	err = wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		// retry until we succeed (to allow time for all changes to propagate)
		if _, err := clientset.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{}); err == nil {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestQuotaLimitService(t *testing.T) {
	// Create admission configuration with ResourceQuota configuration.
	admissionConfigFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())
	if err := os.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ResourceQuota
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: ResourceQuotaConfiguration
    limitedResources:
    - resource: pods
      matchContains:
      - pods
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// Set up an API server
	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
			opts.Admission.GenericAdmission.ConfigFile = admissionConfigFile.Name()

		},
	})
	defer tearDownFn()

	clientset := clientset.NewForConfigOrDie(kubeConfig)

	ns := framework.CreateNamespaceOrDie(clientset, "quota", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	rm := replicationcontroller.NewReplicationManager(
		informers.Core().V1().Pods(),
		informers.Core().V1().ReplicationControllers(),
		clientset,
		replicationcontroller.BurstReplicas,
	)
	go rm.Run(ctx, 3)

	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     informers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(ctx, 2)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, ctx.Done())

	informers.Start(ctx.Done())
	close(informersStarted)

	// now create a covering quota
	// note: limited resource does a matchContains, so we now have "pods" matching "pods" and "count/pods"
	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceServices:              resource.MustParse("4"),
				v1.ResourceServicesNodePorts:     resource.MustParse("2"),
				v1.ResourceServicesLoadBalancers: resource.MustParse("2"),
			},
		},
	}

	waitForQuota(t, quota, clientset)

	// Creating the first node port service should succeed
	nodePortService := newService("np-svc", v1.ServiceTypeNodePort, true)
	_, err = clientset.CoreV1().Services(ns.Name).Create(ctx, nodePortService, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("creating first node port Service should not have returned error: %v", err)
	}

	// Creating the first loadbalancer service should succeed
	lbServiceWithNodePort1 := newService("lb-svc-withnp1", v1.ServiceTypeLoadBalancer, true)
	_, err = clientset.CoreV1().Services(ns.Name).Create(ctx, lbServiceWithNodePort1, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("creating first loadbalancer Service should not have returned error: %v", err)
	}

	// wait for ResourceQuota status to be updated before proceeding, otherwise the test will race with resource quota controller
	expectedQuotaUsed := v1.ResourceList{
		v1.ResourceServices:              resource.MustParse("2"),
		v1.ResourceServicesNodePorts:     resource.MustParse("2"),
		v1.ResourceServicesLoadBalancers: resource.MustParse("1"),
	}
	waitForUsedResourceQuota(t, clientset, quota.Namespace, quota.Name, expectedQuotaUsed)

	// Creating another loadbalancer Service using node ports should fail because node prot quota is exceeded
	lbServiceWithNodePort2 := newService("lb-svc-withnp2", v1.ServiceTypeLoadBalancer, true)
	testServiceForbidden(clientset, ns.Name, lbServiceWithNodePort2, t)

	// Creating a loadbalancer Service without node ports should succeed
	lbServiceWithoutNodePort1 := newService("lb-svc-wonp1", v1.ServiceTypeLoadBalancer, false)
	_, err = clientset.CoreV1().Services(ns.Name).Create(ctx, lbServiceWithoutNodePort1, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("creating another loadbalancer Service without node ports should not have returned error: %v", err)
	}

	// wait for ResourceQuota status to be updated before proceeding, otherwise the test will race with resource quota controller
	expectedQuotaUsed = v1.ResourceList{
		v1.ResourceServices:              resource.MustParse("3"),
		v1.ResourceServicesNodePorts:     resource.MustParse("2"),
		v1.ResourceServicesLoadBalancers: resource.MustParse("2"),
	}
	waitForUsedResourceQuota(t, clientset, quota.Namespace, quota.Name, expectedQuotaUsed)

	// Creating another loadbalancer Service without node ports should fail because loadbalancer quota is exceeded
	lbServiceWithoutNodePort2 := newService("lb-svc-wonp2", v1.ServiceTypeLoadBalancer, false)
	testServiceForbidden(clientset, ns.Name, lbServiceWithoutNodePort2, t)

	// Creating a ClusterIP Service should succeed
	clusterIPService1 := newService("clusterip-svc1", v1.ServiceTypeClusterIP, false)
	_, err = clientset.CoreV1().Services(ns.Name).Create(ctx, clusterIPService1, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("creating a cluster IP Service should not have returned error: %v", err)
	}

	// wait for ResourceQuota status to be updated before proceeding, otherwise the test will race with resource quota controller
	expectedQuotaUsed = v1.ResourceList{
		v1.ResourceServices:              resource.MustParse("4"),
		v1.ResourceServicesNodePorts:     resource.MustParse("2"),
		v1.ResourceServicesLoadBalancers: resource.MustParse("2"),
	}
	waitForUsedResourceQuota(t, clientset, quota.Namespace, quota.Name, expectedQuotaUsed)

	// Creating a ClusterIP Service should fail because Service quota has been exceeded.
	clusterIPService2 := newService("clusterip-svc2", v1.ServiceTypeClusterIP, false)
	testServiceForbidden(clientset, ns.Name, clusterIPService2, t)
}

// testServiceForbidden attempts to create a Service expecting 403 Forbidden due to resource quota limits being exceeded.
func testServiceForbidden(clientset clientset.Interface, namespace string, service *v1.Service, t *testing.T) {
	pollErr := wait.PollImmediate(2*time.Second, 30*time.Second, func() (bool, error) {
		_, err := clientset.CoreV1().Services(namespace).Create(context.TODO(), service, metav1.CreateOptions{})
		if apierrors.IsForbidden(err) {
			return true, nil
		}

		if err == nil {
			return false, errors.New("creating Service should have returned error but got nil")
		}

		return false, nil

	})
	if pollErr != nil {
		t.Errorf("creating Service should return Forbidden due to resource quota limits but got: %v", pollErr)
	}
}

func newService(name string, svcType v1.ServiceType, allocateNodePort bool) *v1.Service {
	var allocateNPs *bool
	// Only set allocateLoadBalancerNodePorts when service type is LB
	if svcType == v1.ServiceTypeLoadBalancer {
		allocateNPs = &allocateNodePort
	}
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Type:                          svcType,
			AllocateLoadBalancerNodePorts: allocateNPs,
			Ports: []v1.ServicePort{{
				Port:       int32(80),
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
}
