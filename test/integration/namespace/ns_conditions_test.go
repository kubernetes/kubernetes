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
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestNamespaceCondition(t *testing.T) {
	closeFn, nsController, informers, kubeClient, dynamicClient := namespaceLifecycleSetup(t)
	defer closeFn()
	nsName := "test-namespace-conditions"
	_, err := kubeClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: nsName,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Start informer and controllers
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	informers.Start(ctx.Done())
	go nsController.Run(ctx, 5)

	data := etcd.GetEtcdStorageDataForNamespace(nsName)
	podJSON, err := jsonToUnstructured(data[corev1.SchemeGroupVersion.WithResource("pods")].Stub, "v1", "Pod")
	if err != nil {
		t.Fatal(err)
	}
	_, err = dynamicClient.Resource(corev1.SchemeGroupVersion.WithResource("pods")).Namespace(nsName).Create(context.TODO(), podJSON, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	deploymentJSON, err := jsonToUnstructured(data[appsv1.SchemeGroupVersion.WithResource("deployments")].Stub, "apps/v1", "Deployment")
	if err != nil {
		t.Fatal(err)
	}
	deploymentJSON.SetFinalizers([]string{"custom.io/finalizer"})
	_, err = dynamicClient.Resource(appsv1.SchemeGroupVersion.WithResource("deployments")).Namespace(nsName).Create(context.TODO(), deploymentJSON, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if err = kubeClient.CoreV1().Namespaces().Delete(context.TODO(), nsName, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	err = wait.PollImmediate(1*time.Second, 60*time.Second, func() (bool, error) {
		curr, err := kubeClient.CoreV1().Namespaces().Get(context.TODO(), nsName, metav1.GetOptions{})
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

		t.Log(dump.Pretty(curr))
		return conditionsFound == 5, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestNamespaceLabels tests for default labels added in https://github.com/kubernetes/kubernetes/pull/96968
func TestNamespaceLabels(t *testing.T) {
	closeFn, nsController, _, kubeClient, _ := namespaceLifecycleSetup(t)
	defer closeFn()

	// Even though nscontroller isn't used in this test, its creation is already
	// spawning some goroutines. So we need to run it to ensure they won't leak.
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go nsController.Run(ctx, 5)

	nsName := "test-namespace-labels-generated"
	// Create a new namespace w/ no name
	ns, err := kubeClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: nsName,
		},
	}, metav1.CreateOptions{})

	if err != nil {
		t.Fatal(err)
	}

	if ns.Name != ns.Labels[corev1.LabelMetadataName] {
		t.Fatal(fmt.Errorf("expected %q, got %q", ns.Name, ns.Labels[corev1.LabelMetadataName]))
	}

	nsList, err := kubeClient.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})

	if err != nil {
		t.Fatal(err)
	}

	for _, ns := range nsList.Items {
		if ns.Name != ns.Labels[corev1.LabelMetadataName] {
			t.Fatal(fmt.Errorf("expected %q, got %q", ns.Name, ns.Labels[corev1.LabelMetadataName]))
		}
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

func namespaceLifecycleSetup(t *testing.T) (kubeapiservertesting.TearDownFunc, *namespace.NamespaceController, informers.SharedInformerFactory, clientset.Interface, dynamic.Interface) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = 10000
	config.Burst = 10000
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "deployment-informers")), resyncPeriod)

	metadataClient, err := metadata.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	discoverResourcesFn := clientSet.Discovery().ServerPreferredNamespacedResources
	_, ctx := ktesting.NewTestContext(t)
	controller := namespace.NewNamespaceController(
		ctx,
		clientSet,
		metadataClient,
		discoverResourcesFn,
		informers.Core().V1().Namespaces(),
		10*time.Hour,
		corev1.FinalizerKubernetes)

	return server.TearDownFn, controller, informers, clientSet, dynamic.NewForConfigOrDie(config)
}
