/*
Copyright 2023 The Kubernetes Authors.

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

package openapi

import (
	"context"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	kubernetes "k8s.io/client-go/kubernetes"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestOpenAPICRDGenerationNumber(t *testing.T) {
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// Create a new CRD with group mygroup.example.com
	crd := fixtures.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	_, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Delete(context.TODO(), crd.Name, metav1.DeleteOptions{})
	}()

	crd, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Update OpenAPI schema and ensure it's reflected in the publishing
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{
		OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
			Type:       "object",
			Properties: map[string]apiextensionsv1.JSONSchemaProps{"num": {Type: "integer", Description: "description"}},
		},
	}

	updatedCRD, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Update(context.TODO(), crd, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if updatedCRD.Generation <= crd.Generation {
		t.Fatalf("Expected updated CRD to increment Generation counter. got preupdate: %d, postupdate %d", crd.Generation, updatedCRD.Generation)
	}

	err = wait.Poll(time.Second*1, wait.ForeverTestTimeout, func() (bool, error) {
		body, err := clientset.RESTClient().Get().AbsPath("/openapi/v2").Do(context.TODO()).Raw()
		if err != nil {
			t.Fatal(err)
		}
		swagger := &spec.Swagger{}
		if err := swagger.UnmarshalJSON(body); err != nil {
			t.Error(err)
		}

		// Ensure that OpenAPI schema updated is reflected
		if description := swagger.Definitions["com.example.mygroup.v1beta1."+crd.Spec.Names.Kind].Properties["num"].Description; description == "description" {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		t.Errorf("Expected description to be updated, err: %s", err)
	}

}
