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

package integration

import (
	"context"
	"testing"

	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiv1 "k8s.io/client-go/pkg/api/v1"

	examplecrv1 "k8s.io/apiextensions-apiserver/examples/client-go/apis/cr/v1"
	exampleclient "k8s.io/apiextensions-apiserver/examples/client-go/client"
	examplecontroller "k8s.io/apiextensions-apiserver/examples/client-go/controller"
)

func TestClientGoCustomResourceExample(t *testing.T) {
	t.Logf("Creating apiextensions apiserver")
	config, err := testserver.DefaultServerConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	stopCh, apiExtensionClient, _, err := testserver.StartServer(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer close(stopCh)

	t.Logf("Creating CustomResourceDefinition")
	crd, err := exampleclient.CreateCustomResourceDefinition(apiExtensionClient)
	if err != nil {
		t.Fatalf("unexpected error creating the CustomResourceDefinition: %v", err)
	}
	defer apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Delete(crd.Name, nil)

	exampleClient, exampleScheme, err := exampleclient.NewClient(config.GenericConfig.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Starting a controller on instances of custom resource %q", examplecrv1.ExampleResourcePlural)
	controller := examplecontroller.ExampleController{
		ExampleClient: exampleClient,
		ExampleScheme: exampleScheme,
	}

	ctx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()
	go controller.Run(ctx)

	// Create an instance of our custom resource
	t.Logf("Creating custom resource instance")
	example := &examplecrv1.Example{
		ObjectMeta: metav1.ObjectMeta{
			Name: "example1",
		},
		Spec: examplecrv1.ExampleSpec{
			Foo: "hello",
			Bar: true,
		},
		Status: examplecrv1.ExampleStatus{
			State:   examplecrv1.ExampleStateCreated,
			Message: "Created, not processed yet",
		},
	}
	var result examplecrv1.Example
	err = exampleClient.Post().
		Resource(examplecrv1.ExampleResourcePlural).
		Namespace(apiv1.NamespaceDefault).
		Body(example).
		Do().Into(&result)
	if err != nil {
		t.Fatalf("Failed to create an instance of the custom resource: %v", err)
	}

	t.Logf("Waiting instance to be processed by the controller")
	if err := exampleclient.WaitForExampleInstanceProcessed(exampleClient, "example1"); err != nil {
		t.Fatalf("Instance was not processed correctly: %v", err)
	}
	t.Logf("Instance is processed")
}
