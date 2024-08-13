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

package apiserver

import (
	"context"
	"fmt"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

// Regression test for https://issues.k8s.io/109099
func TestCRDExponentialRecursionBug(t *testing.T) {
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

	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "crontabs.stable.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "stable.example.com",
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "crontabs",
				Singular: "crontab",
				Kind:     "CronTab",
				ListKind: "CronTabList",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							XPreserveUnknownFields: pointer.BoolPtr(true),
							Type:                   "object",
							Properties:             map[string]apiextensionsv1.JSONSchemaProps{},
						},
					},
				},
			},
		},
	}

	crd, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), crd, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Wait until the CRD exist in discovery
	err = wait.PollImmediate(100*time.Millisecond, 15*time.Second, func() (bool, error) {
		groupResource, err := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + crd.Spec.Versions[0].Name)
		if err != nil {
			return false, nil
		}
		for _, g := range groupResource.APIResources {
			if g.Name == crd.Spec.Names.Plural {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	gvr := schema.GroupVersionResource{
		Group:    crd.Spec.Group,
		Version:  crd.Spec.Versions[0].Name,
		Resource: crd.Spec.Names.Plural,
	}

	crClient := dynamicClient.Resource(gvr)

	instance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name": "instanceame",
			},
			"spec": map[string]interface{}{},
		},
	}

	// create a object with nested fields to trigger the bug
	var m map[string]interface{}
	m = instance.Object["spec"].(map[string]interface{})
	for i := 0; i < 50; i++ {
		m[fmt.Sprintf("field%d", i)] = map[string]interface{}{}
		m = m[fmt.Sprintf("field%d", i)].(map[string]interface{})
	}

	_, err = crClient.Create(context.TODO(), instance, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create custom resource: %v", err)
	}

}
