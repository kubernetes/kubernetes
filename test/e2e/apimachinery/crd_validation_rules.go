/*
Copyright 2021 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"strings"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/runtime/schema"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("CustomResourceValidationRules [Privileged:ClusterAdmin][Alpha][Feature:CustomResourceValidationExpressions]", func() {
	f := framework.NewDefaultFramework("crd-validation-expressions")

	var apiExtensionClient *clientset.Clientset
	ginkgo.BeforeEach(func() {
		var err error
		apiExtensionClient, err = clientset.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err, "initializing apiExtensionClient")
	})

	customResourceClient := func(crd *v1.CustomResourceDefinition) (dynamic.NamespaceableResourceInterface, schema.GroupVersionResource) {
		gvrs := fixtures.GetGroupVersionResourcesOfCustomResource(crd)
		if len(gvrs) != 1 {
			ginkgo.Fail("Expected one version in custom resource definition")
		}
		gvr := gvrs[0]
		return f.DynamicClient.Resource(gvr), gvr
	}
	unmarshallSchema := func(schemaJson []byte) *v1.JSONSchemaProps {
		var c v1.JSONSchemaProps
		err := json.Unmarshal(schemaJson, &c)
		framework.ExpectNoError(err, "unmarshalling OpenAPIv3 schema")
		return &c
	}

	ginkgo.It("MUST NOT fail validation for create of a custom resource that satisfies the x-kubernetes-validator rules", func() {
		ginkgo.By("Creating a custom resource definition with validation rules")
		var schemaWithValidationExpression = unmarshallSchema([]byte(`{
			"type":"object",
			"properties":{
			   "spec":{
				  "type":"object",
				  "x-kubernetes-validations":[
					{ "rule":"self.x + self.y > 0" }
				  ],
				  "properties":{
					 "x":{ "type":"integer" },
					 "y":{ "type":"integer" }
				  }
			   },
			   "status":{
				  "type":"object",
				  "x-kubernetes-validations":[
					 { "rule":"self.health == 'ok' || self.health == 'unhealthy'" }
				  ],
				  "properties":{
					 "health":{ "type":"string" }
				  }
			   }
			}
		 }`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithValidationExpression, false)
		crd, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectNoError(err, "creating CustomResourceDefinition")
		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		ginkgo.By("Creating a custom resource with values that are allowed by the validation rules set on the custom resource definition")
		crClient, gvr := customResourceClient(crd)
		name1 := names.SimpleNameGenerator.GenerateName("cr-1")
		_, err = crClient.Namespace(f.Namespace.Name).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name":      name1,
				"namespace": f.Namespace.Name,
			},
			"spec": map[string]interface{}{
				"x": int64(1),
				"y": int64(0),
			},
		}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "validation rules satisfied")
	})
	ginkgo.It("MUST fail validation for create of a custom resource that does not satisfy the x-kubernetes-validator rules", func() {
		ginkgo.By("Creating a custom resource definition with validation rules")
		var schemaWithValidationExpression = unmarshallSchema([]byte(`{
			"type":"object",
			"properties":{
			   "spec":{
				  "type":"object",
				  "x-kubernetes-validations":[
					{ "rule":"self.x + self.y > 0" }
				  ],
				  "properties":{
					 "x":{ "type":"integer" },
					 "y":{ "type":"integer" }
				  }
			   },
			   "status":{
				  "type":"object",
				  "x-kubernetes-validations":[
					 { "rule":"self.health == 'ok' || self.health == 'unhealthy'" }
				  ],
				  "properties":{
					 "health":{ "type":"string" }
				  }
			   }
			}
		 }`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithValidationExpression, false)
		crd, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectNoError(err, "creating CustomResourceDefinition")
		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		ginkgo.By("Creating a custom resource with values that fail the validation rules set on the custom resource definition")
		crClient, gvr := customResourceClient(crd)
		name1 := names.SimpleNameGenerator.GenerateName("cr-1")
		_, err = crClient.Namespace(f.Namespace.Name).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name":      name1,
				"namespace": f.Namespace.Name,
			},
			"spec": map[string]interface{}{
				"x": int64(0),
				"y": int64(0),
			},
		}}, metav1.CreateOptions{})
		framework.ExpectError(err, "validation rules not satisfied")
		expectedErrMsg := "failed rule"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
		}
	})

	ginkgo.It("MUST fail create of a custom resource definition that contains a x-kubernetes-validator rule that refers to a property that do not exist", func() {
		ginkgo.By("Defining a custom resource definition with a validation rule that refers to a property that do not exist")
		var schemaWithInvalidValidationRule = unmarshallSchema([]byte(`{
		   "type":"object",
		   "properties":{
			  "spec":{
				 "type":"object",
				 "x-kubernetes-validations":[
				   { "rule":"self.z == 100" }
				 ],
				 "properties":{
					"x":{ "type":"integer" }
				 }
			  }
		   }
		}`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithInvalidValidationRule, false)
		_, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectError(err, "creating CustomResourceDefinition with a validation rule that refers to a property that do not exist")
		expectedErrMsg := "undefined field 'z'"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
		}
	})
})
