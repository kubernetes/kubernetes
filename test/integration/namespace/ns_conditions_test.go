/*
Copyright 2019 The Kubernetes Authors.

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

package namespace

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestNamespaceCondition(t *testing.T) {
	closeFn, nsController, informers, kubeClient, dynamicClient := namespaceLifecycleSetup(t)
	defer closeFn()
	nsName := "test-namespace-conditions"
	_, err := kubeClient.CoreV1().Namespaces().Create(&corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: nsName,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Start informer and controllers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers.Start(stopCh)
	go nsController.Run(5, stopCh)

	data := etcd.GetEtcdStorageDataForNamespace(nsName)
	podJSON, err := jsonToUnstructured(data[corev1.SchemeGroupVersion.WithResource("pods")].Stub, "v1", "Pod")
	if err != nil {
		t.Fatal(err)
	}
	_, err = dynamicClient.Resource(corev1.SchemeGroupVersion.WithResource("pods")).Namespace(nsName).Create(podJSON, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	deploymentJSON, err := jsonToUnstructured(data[appsv1.SchemeGroupVersion.WithResource("deployments")].Stub, "apps/v1", "Deployment")
	if err != nil {
		t.Fatal(err)
	}
	deploymentJSON.SetFinalizers([]string{"custom.io/finalizer"})
	_, err = dynamicClient.Resource(appsv1.SchemeGroupVersion.WithResource("deployments")).Namespace(nsName).Create(deploymentJSON, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if err = kubeClient.CoreV1().Namespaces().Delete(nsName, nil); err != nil {
		t.Fatal(err)
	}

	err = wait.PollImmediate(1*time.Second, 60*time.Second, func() (bool, error) {
		curr, err := kubeClient.CoreV1().Namespaces().Get(nsName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		conditionsFound := 0
		for _, condition := range curr.Status.Conditions {
			if condition.Type == corev1.NamespaceDeletionGVParsingFailure && condition.Message == `All legacy kube types successfully parsed` {
				conditionsFound++
			}
			if condition.Type == corev1.NamespaceDeletionDiscoveryFailure && condition.Message == `All resources successfully discovered` {
				conditionsFound++
			}
			if condition.Type == corev1.NamespaceDeletionContentFailure && condition.Message == `All content successfully deleted, may be waiting on finalization` {
				conditionsFound++
			}
			if condition.Type == corev1.NamespaceContentRemaining && condition.Message == `Some resources are remaining: deployments.apps has 1 resource instances` {
				conditionsFound++
			}
			if condition.Type == corev1.NamespaceFinalizersRemaining && condition.Message == `Some content in the namespace has finalizers remaining: custom.io/finalizer in 1 resource instances` {
				conditionsFound++
			}
		}

		t.Log(spew.Sdump(curr))
		return conditionsFound == 5, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// JSONToUnstructured converts a JSON stub to unstructured.Unstructured and
// returns a dynamic resource client that can be used to interact with it
func jsonToUnstructured(stub, version, kind string) (*unstructured.Unstructured, error) {
	typeMetaAdder := map[string]interface{}{}
	if err := json.Unmarshal([]byte(stub), &typeMetaAdder); err != nil {
		return nil, err
	}

	// we don't require GVK on the data we provide, so we fill it in here.  We could, but that seems extraneous.
	typeMetaAdder["apiVersion"] = version
	typeMetaAdder["kind"] = kind

	return &unstructured.Unstructured{Object: typeMetaAdder}, nil
}

func namespaceLifecycleSetup(t *testing.T) (framework.CloseFunc, *namespace.NamespaceController, informers.SharedInformerFactory, clientset.Interface, dynamic.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	config.QPS = 10000
	config.Burst = 10000
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "deployment-informers")), resyncPeriod)

	metadataClient, err := metadata.NewForConfig(&config)
	if err != nil {
		panic(err)
	}

	discoverResourcesFn := clientSet.Discovery().ServerPreferredNamespacedResources

	controller := namespace.NewNamespaceController(
		clientSet,
		metadataClient,
		discoverResourcesFn,
		informers.Core().V1().Namespaces(),
		10*time.Hour,
		corev1.FinalizerKubernetes)
	if err != nil {
		t.Fatalf("error creating Deployment controller: %v", err)
	}
	return closeFn, controller, informers, clientSet, dynamic.NewForConfigOrDie(&config)
}
