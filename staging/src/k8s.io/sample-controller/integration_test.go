/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	extensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	exttesting "k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	apitesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	clientset "k8s.io/sample-controller/pkg/client/clientset/versioned"
	informers "k8s.io/sample-controller/pkg/client/informers/externalversions"
)

func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
}

func TestController(t *testing.T) {
	// Start an API server
	result := apitesting.StartTestServerOrDie(t, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	// Install the CRD
	// TODO: Load this from artifacts/examples/crd.yaml
	apiExtensionClient, err := extensionclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating extension clientset: %v", err)
	}
	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "foos.samplecontroller.k8s.io"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "samplecontroller.k8s.io",
			Version: "v1alpha1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "foos",
				Singular: "foo",
				Kind:     "Foo",
			},
			Scope: apiextensionsv1beta1.NamespaceScoped,
		},
	}

	pool := dynamic.NewDynamicClientPool(result.ClientConfig)
	dynamicClient, err := exttesting.CreateNewCustomResourceDefinition(crd, apiExtensionClient, pool)
	if err != nil {
		t.Fatalf("error while creating CRD: %v", err)
	}

	// Create clients
	kubeClient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating kubernetes clientset: %v", err.Error())
	}

	exampleClient, err := clientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	// Create a test namespace
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "namespace-for-test",
		},
	}
	_, err = kubeClient.Core().Namespaces().Create(ns)
	if err != nil {
		t.Fatalf("error creating namespace: %v", err)
	}

	// XXX: Can't use the generated client to create Foo resources.
	// It fails with:
	//	integration_test.go:92: error creating example foo: object *v1alpha1.Foo does not implement the protobuf marshalling interface and cannot be encoded to a protobuf message
	// replicas := int32(1)
	// exampleFoo := &sampleapi.Foo{
	//	ObjectMeta: metav1.ObjectMeta{
	//		Name: "example-foo",
	//	},
	//	Spec: sampleapi.FooSpec{
	//		DeploymentName: "example-foo",
	//		Replicas:       &replicas,
	//	},
	// }
	// _, err = exampleClient.SamplecontrollerV1alpha1().Foos(ns.Name).Create(exampleFoo)
	// if err != nil {
	//	t.Fatalf("error creating example foo: %v", err)
	// }

	// Create an instance of the CRD using dynamic client.
	// Following example from https://github.com/kubernetes/apiextensions-apiserver/blob/release-1.8/test/integration/registration_test.go#L42
	exampleFoo := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "samplecontroller.k8s.io/v1alpha1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"namespace": ns.Name,
				"name":      "example-foo",
			},
			"spec": map[string]interface{}{
				"deploymentName": "example-foo",
				"replicas":       123,
			},
		},
	}
	resourceClient := dynamicClient.Resource(
		&metav1.APIResource{
			Name:       crd.Spec.Names.Plural,
			Namespaced: crd.Spec.Scope != apiextensionsv1beta1.ClusterScoped,
		},
		ns.Name,
	)
	_, err = resourceClient.Create(exampleFoo)
	if err != nil {
		t.Fatalf("error creating example foo: %v", err)
	}

	// The example client can find the created Foo resource
	found, err := exampleClient.SamplecontrollerV1alpha1().Foos(ns.Name).Get(
		"example-foo",
		metav1.GetOptions{},
	)
	if err != nil {
		t.Fatalf("error getting example-foo: %v", err)
	}
	t.Logf("Found example-foo: %v", found)

	// Create and start the sample-controller and its informers
	kubeInformerFactory := kubeinformers.NewSharedInformerFactory(kubeClient, 0)
	exampleInformerFactory := informers.NewSharedInformerFactory(exampleClient, 0)

	controller := NewController(
		kubeClient,
		exampleClient,
		kubeInformerFactory,
		exampleInformerFactory,
	)

	stopCh := make(chan struct{})
	controllerFinished := make(chan struct{})
	defer func() {
		close(stopCh)
		<-controllerFinished
	}()

	kubeInformerFactory.Start(stopCh)
	exampleInformerFactory.Start(stopCh)

	go func() {
		defer close(controllerFinished)
		err := controller.Run(2, stopCh)
		if err != nil {
			t.Fatalf("error running controller: %v", err)
		}
	}()

	// Wait 60s for the controller to create a deployment
	err = wait.PollImmediate(
		5*time.Second,
		60*time.Second,
		func() (bool, error) {
			deployment, err := kubeClient.AppsV1beta2().Deployments(ns.Name).Get(
				"example-foo",
				metav1.GetOptions{},
			)
			if err != nil {
				if errors.IsNotFound(err) {
					return false, nil
				}
				return false, err
			}
			if *deployment.Spec.Replicas != 123 {
				return false, fmt.Errorf(
					"deployment had unexpected replicas value: %v",
					*deployment.Spec.Replicas,
				)
			}
			return true, nil
		},
	)
	if err != nil {
		t.Fatalf("controller failed to create expected deployment: %v", err)
	}
}
