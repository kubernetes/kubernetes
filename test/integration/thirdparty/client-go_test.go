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

package thirdparty

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/kubernetes"
	apiv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration/framework"

	exampletprv1 "k8s.io/client-go/examples/third-party-resources-deprecated/apis/tpr/v1"
	exampleclient "k8s.io/client-go/examples/third-party-resources-deprecated/client"
	examplecontroller "k8s.io/client-go/examples/third-party-resources-deprecated/controller"
)

func TestClientGoThirdPartyResourceExample(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()

	scheme := runtime.NewScheme()
	if err := exampletprv1.AddToScheme(scheme); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	config := &rest.Config{Host: s.URL, ContentConfig: rest.ContentConfig{
		NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: serializer.NewCodecFactory(scheme)},
	}}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Creating TPR %q", exampletprv1.ExampleResourcePlural)
	if err := exampleclient.CreateTPR(clientset); err != nil {
		t.Fatalf("unexpected error creating the ThirdPartyResource: %v", err)
	}

	exampleClient, exampleScheme, err := exampleclient.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Waiting for TPR %q to show up", exampletprv1.ExampleResourcePlural)
	if err := exampleclient.WaitForExampleResource(exampleClient); err != nil {
		t.Fatalf("TPR examples did not show up: %v", err)
	}
	t.Logf("TPR %q is active", exampletprv1.ExampleResourcePlural)

	t.Logf("Starting a controller on instances of TPR %q", exampletprv1.ExampleResourcePlural)
	controller := examplecontroller.ExampleController{
		ExampleClient: exampleClient,
		ExampleScheme: exampleScheme,
	}

	ctx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()
	go controller.Run(ctx)

	// Create an instance of our TPR
	t.Logf("Creating instance of TPR %q", exampletprv1.ExampleResourcePlural)
	example := &exampletprv1.Example{
		ObjectMeta: metav1.ObjectMeta{
			Name: "example1",
		},
		Spec: exampletprv1.ExampleSpec{
			Foo: "hello",
			Bar: true,
		},
		Status: exampletprv1.ExampleStatus{
			State:   exampletprv1.ExampleStateCreated,
			Message: "Created, not processed yet",
		},
	}
	var result exampletprv1.Example
	err = exampleClient.Post().
		Resource(exampletprv1.ExampleResourcePlural).
		Namespace(apiv1.NamespaceDefault).
		Body(example).
		Do().Into(&result)
	if err != nil {
		t.Fatalf("Failed to create an instance of TPR: %v", err)
	}

	t.Logf("Waiting for TPR %q instance to be processed", exampletprv1.ExampleResourcePlural)
	if err := exampleclient.WaitForExampleInstanceProcessed(exampleClient, "example1"); err != nil {
		t.Fatalf("TPR example was not processed correctly: %v", err)
	}
	t.Logf("TPR %q instance is processed", exampletprv1.ExampleResourcePlural)
}
