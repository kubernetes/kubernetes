// +build integration,!no-etcd

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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/plugin/pkg/admission/resourcequota"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func init() {
	integration.RequireEtcd()
}

// 1.2 code gets:
// 	quota_test.go:95: Took 4.218619579s to scale up without quota
// 	quota_test.go:199: unexpected error: timed out waiting for the condition, ended with 342 pods (1 minute)
// 1.3+ code gets:
// 	quota_test.go:100: Took 4.196205966s to scale up without quota
// 	quota_test.go:115: Took 12.021640372s to scale up with quota
func TestQuota(t *testing.T) {
	// Set up a master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	admissionCh := make(chan struct{})
	clientset := clientset.NewForConfigOrDie(&restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})
	admission, err := resourcequota.NewResourceQuota(clientset, quotainstall.NewRegistry(clientset, nil), 5, admissionCh)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer close(admissionCh)

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.AdmissionControl = admission
	framework.RunAMasterUsingServer(masterConfig, s, h)

	ns := framework.CreateTestingNamespace("quotaed", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	ns2 := framework.CreateTestingNamespace("non-quotaed", s, t)
	defer framework.DeleteTestingNamespace(ns2, s, t)

	controllerCh := make(chan struct{})
	defer close(controllerCh)

	go replicationcontroller.NewReplicationManagerFromClientForIntegration(clientset, controller.NoResyncPeriodFunc, replicationcontroller.BurstReplicas, 4096).
		Run(3, controllerCh)

	resourceQuotaRegistry := quotainstall.NewRegistry(clientset, nil)
	groupKindsToReplenish := []unversioned.GroupKind{
		api.Kind("Pod"),
	}
	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		KubeClient:                clientset,
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		Registry:                  resourceQuotaRegistry,
		GroupKindsToReplenish:     groupKindsToReplenish,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactoryFromClient(clientset),
	}
	go resourcequotacontroller.NewResourceQuotaController(resourceQuotaControllerOptions).Run(2, controllerCh)

	startTime := time.Now()
	scale(t, ns2.Name, clientset)
	endTime := time.Now()
	t.Logf("Took %v to scale up without quota", endTime.Sub(startTime))

	quota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourcePods: resource.MustParse("1000"),
			},
		},
	}
	waitForQuota(t, quota, clientset)

	startTime = time.Now()
	scale(t, "quotaed", clientset)
	endTime = time.Now()
	t.Logf("Took %v to scale up with quota", endTime.Sub(startTime))
}

func waitForQuota(t *testing.T, quota *api.ResourceQuota, clientset *clientset.Clientset) {
	w, err := clientset.Core().ResourceQuotas(quota.Namespace).Watch(api.SingleObject(api.ObjectMeta{Name: quota.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := clientset.Core().ResourceQuotas(quota.Namespace).Create(quota); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = watch.Until(1*time.Minute, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}

		switch cast := event.Object.(type) {
		case *api.ResourceQuota:
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

func scale(t *testing.T, namespace string, clientset *clientset.Clientset) {
	target := 100
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: int32(target),
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "busybox",
						},
					},
				},
			},
		},
	}

	w, err := clientset.Core().ReplicationControllers(namespace).Watch(api.SingleObject(api.ObjectMeta{Name: rc.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := clientset.Core().ReplicationControllers(namespace).Create(rc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = watch.Until(3*time.Minute, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}

		switch cast := event.Object.(type) {
		case *api.ReplicationController:
			fmt.Printf("Found %v of %v replicas\n", int(cast.Status.Replicas), target)
			if int(cast.Status.Replicas) == target {
				return true, nil
			}
		}

		return false, nil
	})
	if err != nil {
		pods, _ := clientset.Core().Pods(namespace).List(api.ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything()})
		t.Fatalf("unexpected error: %v, ended with %v pods", err, len(pods.Items))
	}
}
