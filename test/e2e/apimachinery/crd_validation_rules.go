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

	"github.com/onsi/ginkgo/v2"
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
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("CustomResourceValidationRules [Privileged:ClusterAdmin][Alpha][Feature:CustomResourceValidationExpressions]", func() {
	f := framework.NewDefaultFramework("crd-validation-expressions")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

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

	// for all new CRD validation features that should be E2E-tested, add them
	// into this schema and then add CR requests to the end of the test
	// below ("MUST NOT fail validation...") instead of writing a new and
	// separate test
	var schemaWithValidationExpression = unmarshallSchema([]byte(`{
	   "type":"object",
	   "properties":{
		  "spec":{
			 "type":"object",
			 "x-kubernetes-validations":[
			   { "rule":"self.x + self.y > 0" },
			   { "rule":"self.firstArray.isSorted() && self.secondArray.isSorted() && ((self.firstArray.sum() + self.secondArray.sum()) % 2 == 0)" },
			   { "rule":"self.largeArray.all(x, self.largeArray.all(y, y == x))" }
	         ],
			 "properties":{
				"x":{ "type":"integer" },
				"y":{ "type":"integer" },
				"firstArray":{ "type":"array", "maxItems": 1000, "items":{ "type": "integer"} },
				"secondArray":{ "type":"array", "maxItems": 1000, "items":{ "type": "integer"} },
				"largeArray":{ "type":"array", "maxItems": 725, "items":{ "type": "integer"} }
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
	ginkgo.It("MUST NOT fail validation for create of a custom resource that satisfies the x-kubernetes-validations rules", func() {
		ginkgo.By("Creating a custom resource definition with validation rules")
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
				"x":           int64(1),
				"y":           int64(0),
				"firstArray":  []int64{3, 4},
				"secondArray": []int64{5, 10},
				"largeArray":  []int64{2, 2},
			},
		}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "validation rules satisfied")
	})
	ginkgo.It("MUST fail validation for create of a custom resource that does not satisfy the x-kubernetes-validations rules", func() {
		ginkgo.By("Creating a custom resource definition with validation rules")
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

	ginkgo.It("MUST fail create of a custom resource definition that contains a x-kubernetes-validations rule that refers to a property that do not exist", func() {
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

	ginkgo.It("MUST fail create of a custom resource definition that contains an x-kubernetes-validations rule that contains a syntax error", func() {
		ginkgo.By("Defining a custom resource definition that contains a validation rule with a syntax error")
		var schemaWithSyntaxErrorRule = unmarshallSchema([]byte(`{
		   "type":"object",
		   "properties":{
		      "spec":{
			    "type":"object",
				"x-kubernetes-validations":[
				  { "rule":"self = 42" }
				]
			  }
			}
		}`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithSyntaxErrorRule, false)
		_, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectError(err, "creating a CustomResourceDefinition with a validation rule that contains a syntax error")
		expectedErrMsg := "Syntax error"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expected error message to contain %q, got %q", expectedErrMsg, err.Error())
		}
	})

	ginkgo.It("MUST fail create of a custom resource definition that contains an x-kubernetes-validations rule that exceeds the estimated cost limit", func() {
		ginkgo.By("Defining a custom resource definition that contains a validation rule that exceeds the cost limit")
		var schemaWithExpensiveRule = unmarshallSchema([]byte(`{
		   "type":"object",
		   "properties":{
			  "spec":{
			    "type":"object",
			    "properties":{
				  "x":{
				    "type":"array",
				    "items":{
				      "type":"array",
					  "items":{
					    "type":"string"
					  },
					  "x-kubernetes-validations":[
					    { "rule":"self.all(s, s == 'string constant')" }
					  ]
				    }
				  }
			    }
			  }
		    }
		}`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithExpensiveRule, false)
		_, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectError(err, "creating a CustomResourceDefinition with a validation rule that exceeds the cost limit")
		expectedErrMsg := "exceeds budget"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expected error message to contain %q, got %q", expectedErrMsg, err.Error())
		}
	})

	ginkgo.It("MUST fail create of a custom resource that exceeds the runtime cost limit for x-kubernetes-validations rule execution", func() {
		ginkgo.By("Defining a custom resource definition including an expensive rule on a large amount of data")
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithValidationExpression, false)
		_, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectNoError(err, "creating CustomResourceDefinition including an expensive rule on a large amount of data")
		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()
		ginkgo.By("Attempting to create a custom resource that will exceed the runtime cost limit")
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
				"largeArray": genLargeArray(725, 20),
			},
		}}, metav1.CreateOptions{})
		framework.ExpectError(err, "custom resource creation should be prohibited by runtime cost limit")
		expectedErrMsg := "call cost exceeds limit"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
		}
	})

	ginkgo.It("MUST fail update of a custom resource that does not satisfy a x-kubernetes-validations transition rule", func() {
		ginkgo.By("Defining a custom resource definition with a x-kubernetes-validations transition rule")
		var schemaWithTransitionRule = unmarshallSchema([]byte(`{
		   "type":"object",
		   "properties":{
			  "spec":{
			    "type":"object",
			    "properties":{
				  "num":{
				    "type":"integer",
					  "x-kubernetes-validations":[
					    { "rule":"self > oldSelf" }
					  ]
				    }
				  }
			    }
			  }
		}`))
		crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(v1.NamespaceScoped, schemaWithTransitionRule, false)
		_, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectNoError(err, "creating CustomResourceDefinition including an x-kubernetes-validations transition rule")
		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()
		ginkgo.By("Attempting to create a custom resource")
		crClient, gvr := customResourceClient(crd)
		name1 := names.SimpleNameGenerator.GenerateName("cr-1")
		unstruct, err := crClient.Namespace(f.Namespace.Name).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name":      name1,
				"namespace": f.Namespace.Name,
			},
			"spec": map[string]interface{}{
				"num": int64(10),
			},
		}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "transition rules do not apply to create operations")
		ginkgo.By("Updating a custom resource with a value that does not satisfy an x-kubernetes-validations transition rule")
		_, err = crClient.Namespace(f.Namespace.Name).Update(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name":            name1,
				"namespace":       f.Namespace.Name,
				"resourceVersion": unstruct.GetResourceVersion(),
			},
			"spec": map[string]interface{}{
				"num": int64(9),
			},
		}}, metav1.UpdateOptions{})
		framework.ExpectError(err, "custom resource update should be prohibited by transition rule")
		expectedErrMsg := "failed rule"
		if !strings.Contains(err.Error(), expectedErrMsg) {
			framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
		}
	})
})

func genLargeArray(n, x int64) []int64 {
	arr := make([]int64, n)
	for i := int64(0); i < n; i++ {
		arr[i] = x
	}
	return arr
}
