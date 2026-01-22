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

package openapi

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/kube-openapi/pkg/validation/spec"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestOpenAPIV2CRDMergeNoDuplicateTypes(t *testing.T) {
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
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	foo := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foosubs.cr.bar.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "cr.bar.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "foosubs",
				Kind:   "FooSub",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "integer",
										},
									},
								},
								"status": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "integer",
										},
									},
								},
							},
						},
					},
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
						},
					},
				},
			},
		},
	}

	baz := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bazsubs.cr.bar.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "cr.bar.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "bazsubs",
				Kind:   "BazSub",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "integer",
										},
									},
								},
								"status": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "integer",
										},
									},
								},
							},
						},
					},
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
						},
					},
				},
			},
		},
	}

	_, err = fixtures.CreateNewV1CustomResourceDefinition(foo, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	_, err = fixtures.CreateNewV1CustomResourceDefinition(baz, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	var openAPISpec spec.Swagger
	// Poll for the OpenAPI to be updated with the new CRD
	wait.Poll(time.Second*1, wait.ForeverTestTimeout, func() (bool, error) {
		jsonData, err := clientset.RESTClient().Get().AbsPath("/openapi/v2").Do(context.TODO()).Raw()
		if err != nil {
			t.Fatal(err)
		}
		openAPISpec = spec.Swagger{}
		err = json.Unmarshal(jsonData, &openAPISpec)
		if err != nil {
			t.Fatal(err)
		}
		for schemaName := range openAPISpec.Definitions {
			if strings.HasPrefix(schemaName, "com.bar.cr.v1.BazSub") {
				return true, nil
			}
		}
		return false, nil
	})

	for schemaName := range openAPISpec.Definitions {
		if strings.HasSuffix(schemaName, "_v2") {
			t.Errorf("Error: Expected no _v2 types, got %s", schemaName)
		}
	}
}
