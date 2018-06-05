/*
Copyright 2018 The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/restmapper"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	"k8s.io/kubernetes/pkg/quota/generic"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCustomResourceQuota(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	clientset, err := clientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	dynamicclient, err := dynamic.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf("Starting resourcequota controller")
	controllerCh := make(chan struct{})
	defer close(controllerCh)
	informers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     informers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		SharedInformerFactory:     informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
		RESTMapper:                restmapper.NewDeferredDiscoveryRESTMapper(cached.NewMemCacheClient(clientset.Discovery())),
		DynamicClient:             dynamicclient,
	}
	resourceQuotaController, err := resourcequotacontroller.NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	go resourceQuotaController.Run(2, controllerCh)
	go resourceQuotaController.Sync(clientset.Discovery(), 1*time.Second, controllerCh)

	go informers.Start(controllerCh)
	informers.WaitForCacheSync(controllerCh)
	close(informersStarted)

	t.Logf("Creating CRD")
	apiextensionsclient, err := apiextensionsclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	crd, err := apiextensionsclient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(newFooCRD())
	if err != nil {
		t.Fatalf("Failed to create crd: %v", err)
	}
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		_, err = apiextensionsclient.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + crd.Spec.Version)
		if err == nil {
			return true, nil
		}
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}); err != nil {
		t.Fatalf("Failed to see crd in discovery: %v", err)
	}

	t.Logf("Creating namespaces")
	if _, err := clientset.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "quotaed"}}); err != nil {
		t.Fatalf("Failed to create test namespace: %v", err)
	}
	if _, err := clientset.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "non-quotaed"}}); err != nil {
		t.Fatalf("Failed to create test namespace: %v", err)
	}

	t.Logf("Creating quota")
	quota, err := clientset.CoreV1().ResourceQuotas("quotaed").Create(newTestResourceQuota("crd-quota"))
	if err != nil {
		t.Fatalf("Failed to create crd-quota: %v", err)
	}

	t.Logf("Creating Foo object and check count")
	fooclient := dynamicclient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural})
	if _, err := fooclient.Namespace(quota.Namespace).Create(newFoo("one")); err != nil {
		t.Fatalf("Failed to create CR: %v", err)
	}

	if err := wait.Poll(300*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		quota, err = clientset.CoreV1().ResourceQuotas(quota.Namespace).Get(quota.Name, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("failed to get quota: %v", err)
		}

		if used, ok := quota.Status.Used["count/foos.quota.integration.test.k8s.io"]; ok && used.Value() == 1 {
			return true, nil
		}

		return false, nil
	}); err != nil {
		t.Fatalf("Failed to wait for quota count to increase: %v", err)
	}

	t.Logf("Creating Foo object until quota hits limit")
	i := 0
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		if _, err := fooclient.Namespace(quota.Namespace).Create(newFoo(fmt.Sprintf("instance-%d", i))); err != nil {
			if strings.Contains(err.Error(), "exceeded quota") {
				return true, nil
			}
			return false, fmt.Errorf("failed to create CR: %v", err)
		}
		i++
		return false, nil
	}); err != nil {
		t.Fatalf("Failed hit quota limit: %v", err)
	}
}

func newFoo(name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "quota.integration.test.k8s.io/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": name,
			},
		},
	}
}

func newFooCRD() *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "foos.quota.integration.test.k8s.io"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "quota.integration.test.k8s.io",
			Version: "v1",
			Scope:   apiextensionsv1beta1.NamespaceScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "foos",
				Singular: "foo",
				Kind:     "Foo",
				ListKind: "FooList",
			},
		},
	}
}

func newTestResourceQuota(name string) *v1.ResourceQuota {
	hard := v1.ResourceList{}
	hard[v1.ResourcePods] = resource.MustParse("5")
	hard[v1.ResourceName("count/foos.quota.integration.test.k8s.io")] = resource.MustParse("3")
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.ResourceQuotaSpec{Hard: hard},
	}
}
