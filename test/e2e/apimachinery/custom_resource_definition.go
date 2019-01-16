/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/types"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/go-openapi/spec"
	. "github.com/onsi/ginkgo"
)

var crdVersion = utilversion.MustParseSemantic("v1.7.0")
var crdPublishOpenAPIVersion = utilversion.MustParseSemantic("v1.13.0")

var _ = SIGDescribe("CustomResourceDefinition resources", func() {

	f := framework.NewDefaultFramework("custom-resource-definition")

	Context("Simple CustomResourceDefinition", func() {
		/*
			Release : v1.9
			Testname: Custom Resource Definition, create
			Description: Create a API extension client, define a random custom resource definition, create the custom resource. API server MUST be able to create the custom resource.
		*/
		framework.ConformanceIt("creating/deleting custom resource definition objects works ", func() {

			framework.SkipUnlessServerVersionGTE(crdVersion, f.ClientSet.Discovery())

			config, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("failed to load config: %v", err)
			}

			apiExtensionClient, err := clientset.NewForConfig(config)
			if err != nil {
				framework.Failf("failed to initialize apiExtensionClient: %v", err)
			}

			randomDefinition := fixtures.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)

			//create CRD and waits for the resource to be recognized and available.
			randomDefinition, err = fixtures.CreateNewCustomResourceDefinition(randomDefinition, apiExtensionClient, f.DynamicClient)
			if err != nil {
				framework.Failf("failed to create CustomResourceDefinition: %v", err)
			}

			defer func() {
				err = fixtures.DeleteCustomResourceDefinition(randomDefinition, apiExtensionClient)
				if err != nil {
					framework.Failf("failed to delete CustomResourceDefinition: %v", err)
				}
			}()
		})

		It("has OpenAPI spec served with CRD Validation chema", func() {
			framework.SkipUnlessServerVersionGTE(crdPublishOpenAPIVersion, f.ClientSet.Discovery())

			testcrd, err := framework.CreateTestCRD(f)
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}
			testcrd2, err := framework.CreateTestCRD(f)
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}

			// Make sure test CRDs are of same GV but different kinds
			if testcrd.ApiGroup != testcrd2.ApiGroup || testcrd.Versions[0].Name != testcrd2.Versions[0].Name {
				framework.Failf("unexpected: test CRDs should share the same GV %v, %v", testcrd, testcrd2)
			}
			if testcrd.Kind == testcrd2.Kind {
				framework.Failf("unexpected: test CRDs should have different kinds %v, %v", testcrd, testcrd2)
			}

			crdDefinitionKey := fmt.Sprintf("%s.%s.%s", testcrd.ApiGroup, testcrd.Versions[0].Name, testcrd.Kind)
			crd2DefinitionKey := fmt.Sprintf("%s.%s.%s", testcrd2.ApiGroup, testcrd2.Versions[0].Name, testcrd2.Kind)

			// Test CRD don't have ValidationSchema setup. Here we patch a
			// simple Schema for kubectl behavior testing.
			if err := patchCRDSchema(crdValidationSchema, testcrd); err != nil {
				framework.Failf("failed to patch CRD schema: %v", err)
			}
			if err := patchCRDSchema(crd2ValidationSchema, testcrd2); err != nil {
				framework.Failf("failed to patch CRD schema: %v", err)
			}

			// We use a wait.Poll block here because the kube-aggregator openapi
			// controller takes time to rotate the queue and resync apiextensions-apiserver's spec
			By("Waiting for test CRD's Schema to show up in OpenAPI spec")
			lastMsg := ""
			if err := wait.Poll(5*time.Second, 120*time.Second, func() (bool, error) {
				bs, err := f.ClientSet.CoreV1().RESTClient().Get().AbsPath("openapi", "v2").DoRaw()
				if err != nil {
					return false, err
				}
				spec := spec.Swagger{}
				if err := json.Unmarshal(bs, &spec); err != nil {
					return false, err
				}
				if spec.SwaggerProps.Paths == nil {
					lastMsg = "spec.SwaggerProps.Paths is nil"
					return false, nil
				}
				d, ok := spec.SwaggerProps.Definitions[crdDefinitionKey]
				if !ok {
					lastMsg = fmt.Sprintf("crd1 spec.SwaggerProps.Definitions[\"%s\"] not found", crdDefinitionKey)
					return false, nil
				}
				_, ok = d.Properties["status"]
				if !ok {
					lastMsg = fmt.Sprintf("crd1 spec.SwaggerProps.Definitions[\"%s\"].Properties[\"status\"] not found", crdDefinitionKey)
					return false, nil
				}
				d2, ok := spec.SwaggerProps.Definitions[crd2DefinitionKey]
				if !ok {
					lastMsg = fmt.Sprintf("crd2 spec.SwaggerProps.Definitions[\"%s\"] not found", crd2DefinitionKey)
					return false, nil
				}
				_, ok = d2.Properties["status"]
				if !ok {
					lastMsg = fmt.Sprintf("crd2 spec.SwaggerProps.Definitions[\"%s\"].Properties[\"status\"] not found", crd2DefinitionKey)
					return false, nil
				}
				return true, nil
			}); err != nil {
				framework.Failf("failed to wait for apiserver to serve openapi spec for registered CRD: %v; lastMsg: %s", err, lastMsg)
			}

			By("Having kubectl understand the schema")

			result, err := framework.RunKubectl("explain", testcrd.GetPluralName())
			if err != nil {
				framework.Failf("failed to explain CRD: %v", err)
			}
			// Example of expected result:
			//
			// KIND:     E2e-test-custom-resource-definition-1717-crd
			// VERSION:  custom-resource-definition-crd-test.k8s.io/v1
			//
			// DESCRIPTION: Foo CRD for Testing
			//
			// FIELDS:
			//    apiVersion	<string>
			//      APIVersion defines the versioned schema of this representation of an
			//      object. Servers should convert recognized schemas to the latest internal
			//      value, and may reject unrecognized values. More info:
			//      https://git.k8s.io/community/contributors/devel/api-conventions.md#resources
			//
			//    kind	<string>
			//      Kind is a string value representing the REST resource this object
			//      represents. Servers may infer this from the endpoint the client submits
			//      requests to. Cannot be updated. In CamelCase. More info:
			//      https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
			//
			//    metadata	<Object>
			//      Standard object's metadata. More info:
			//      https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
			//
			//    spec	<Object>
			//      Specification of Foo
			//
			//    status	<Object>
			//      Status of Foo
			pattern := regexp.MustCompile(`(?s)DESCRIPTION:.*Foo CRD for Testing.*FIELDS:.*apiVersion.*<string>.*APIVersion defines.*spec.*<Object>.*Specification of Foo`)
			if !pattern.Match([]byte(result)) {
				framework.Failf("CRD explain result doesn't contain proper object description and field description: %s", result)
			}

			result, err = framework.RunKubectl("explain", testcrd.GetPluralName()+".metadata")
			if err != nil {
				framework.Failf("failed to explain CRD.metadata: %v", err)
			}
			// Omitted the example because the result for explain CRD.metadata is quite long
			pattern = regexp.MustCompile(`(?s)DESCRIPTION:.*Standard object's metadata.*FIELDS:.*creationTimestamp.*<string>.*CreationTimestamp is a timestamp`)
			if !pattern.Match([]byte(result)) {
				framework.Failf("CRD.metadata explain result doesn't contain proper object description and field description: %s", result)
			}

			result, err = framework.RunKubectl("explain", testcrd.GetPluralName()+".spec")
			if err != nil {
				framework.Failf("failed to explain CRD.spec: %v", err)
			}
			// Example of expected result:
			//
			// KIND:     E2e-test-custom-resource-definition-1717-crd
			// VERSION:  custom-resource-definition-crd-test.k8s.io/v1
			//
			// RESOURCE: spec <Object>
			//
			// DESCRIPTION:
			//      Specification of Foo
			//
			// FIELDS:
			//    bars	<[]Object>
			//      List of Bars and their specs.
			pattern = regexp.MustCompile(`(?s)DESCRIPTION:.*Specification of Foo.*FIELDS:.*bars.*<\[\]Object>.*List of Bars and their specs`)
			if !pattern.Match([]byte(result)) {
				framework.Failf("CRD.spec explain result doesn't contain proper object description and field description: %s", result)
			}

			result, err = framework.RunKubectl("explain", testcrd.GetPluralName()+".spec.bars")
			if err != nil {
				framework.Failf("failed to explain CRD.spec.bars: %v", err)
			}
			// Example of expected result:
			//
			// KIND:     E2e-test-custom-resource-definition-1717-crd
			// VERSION:  custom-resource-definition-crd-test.k8s.io/v1
			//
			// RESOURCE: bars <[]Object>
			//
			// DESCRIPTION:
			//      List of Bars and their specs.
			//
			// FIELDS:
			//    bazs	<[]string>
			//      List of Bazs.
			//
			//    name	<string>
			//      Name of Bar.
			pattern = regexp.MustCompile(`(?s)RESOURCE:.*bars.*<\[\]Object>.*DESCRIPTION:.*List of Bars and their specs.*FIELDS:.*bazs.*<\[\]string>.*List of Bazs.*name.*<string>.*Name of Bar`)
			if !pattern.Match([]byte(result)) {
				framework.Failf("CRD.spec.bars explain result doesn't contain proper object description and field description: %s", result)
			}

			// kubectl should return error when explaining property that doesn't exist
			if _, err := framework.RunKubectl("explain", testcrd.GetPluralName()+".spec.bars2"); err == nil || !strings.Contains(err.Error(), `field "bars2" does not exist`) {
				framework.Failf("unexpected no error when explaining property that doesn't exist")
			}

			// kubectl should be able to explain testcrd2
			result, err = framework.RunKubectl("explain", testcrd2.GetPluralName())
			if err != nil {
				framework.Failf("failed to explain CRD2: %v", err)
			}
			pattern = regexp.MustCompile(`(?s)DESCRIPTION:.*Waldo CRD for Testing.*FIELDS:.*apiVersion.*<string>.*APIVersion defines.*spec.*<Object>.*Specification of Waldo`)
			if !pattern.Match([]byte(result)) {
				framework.Failf("CRD2 explain result doesn't contain proper object description and field description: %s", result)
			}

			By("Having kubectl perform client-side validation")
			input := fmt.Sprintf("{\"kind\":\"%s\",\"apiVersion\":\"%s/%s\",\"metadata\":{\"name\":\"foo\"},\"spec\":{\"foo\":true}}", testcrd.Kind, testcrd.ApiGroup, testcrd.Versions[0].Name)
			if _, err := framework.RunKubectlInput(input, "create", "-f", "-"); err == nil || !strings.Contains(err.Error(), `unknown field "foo"`) {
				framework.Failf("unexpected no error when creating CR with unknown field")
			}
			input = fmt.Sprintf("{\"kind\":\"%s\",\"apiVersion\":\"%s/%s\",\"metadata\":{\"name\":\"foo\"},\"spec\":{\"bars\":[{\"name\":\"test\"}]}}", testcrd.Kind, testcrd.ApiGroup, testcrd.Versions[0].Name)
			if _, err := framework.RunKubectlInput(input, "create", "-f", "-"); err != nil {
				framework.Failf("failed to create valid CR: %v", err)
			}
			if _, err := framework.RunKubectl("delete", testcrd.GetPluralName(), "foo"); err != nil {
				framework.Failf("failed to delete valid CR: %v", err)
			}

			// Delete test CRD
			testcrd.CleanUp()

			// We use a wait.Poll block here because the kube-aggregator openapi
			// controller takes time to rotate the queue and resync apiextensions-apiserver's spec
			By("Waiting for test CRD's Schema to be removed from OpenAPI spec")
			if err := wait.Poll(5*time.Second, 120*time.Second, func() (bool, error) {
				bs, err := f.ClientSet.CoreV1().RESTClient().Get().AbsPath("openapi", "v2").DoRaw()
				if err != nil {
					return false, err
				}
				spec := spec.Swagger{}
				if err := json.Unmarshal(bs, &spec); err != nil {
					return false, err
				}
				if spec.SwaggerProps.Paths == nil {
					lastMsg = "spec.SwaggerProps.Paths is nil"
					return false, nil
				}
				if _, ok := spec.SwaggerProps.Definitions[crdDefinitionKey]; ok {
					return false, nil
				}
				return true, nil
			}); err != nil {
				framework.Failf("failed to wait for apiserver to remove openapi spec for deleted CRD: %v", err)
			}
		})
	})
})

// patchCRDSchema takes CRD validation schema in yaml format, and patches it to given TestCrd object
func patchCRDSchema(schema []byte, crd *framework.TestCrd) error {
	patch, err := utilyaml.ToJSON(schema)
	if err != nil {
		return fmt.Errorf("failed to create json patch: %v", err)
	}
	crd.Crd, err = crd.ApiExtensionClient.ApiextensionsV1beta1().
		CustomResourceDefinitions().Patch(crd.GetMetaName(), types.MergePatchType, patch)
	if err != nil {
		return fmt.Errorf("failed to patch CustomResourceDefinition: %v", err)
	}
	return nil
}

var crdValidationSchema = []byte(`spec:
  validation:
    openAPIV3Schema:
      description: Foo CRD for Testing
      properties:
        spec:
          description: Specification of Foo
          properties:
            bars:
              description: List of Bars and their specs.
              type: array
              items:
                properties:
                  name:
                    description: Name of Bar.
                    type: string
                  bazs:
                    description: List of Bazs.
                    items:
                      type: string
                    type: array
        status:
          description: Status of Foo
          properties:
            bars:
              description: List of Bars and their statuses.
              type: array
              items:
                properties:
                  name:
                    description: Name of Bar.
                    type: string
                  available:
                    description: Whether the Bar is installed.
                    type: boolean
                  quxType:
                    description: Indicates to external qux type.
                    pattern: in-tree|out-of-tree
                    type: string`)

var crd2ValidationSchema = []byte(`spec:
  validation:
    openAPIV3Schema:
      description: Waldo CRD for Testing
      properties:
        spec:
          description: Specification of Waldo
          type: object
          properties:
            dummy:
              description: Dummy property.
        status:
          description: Status of Waldo
          type: object
          properties:
            bars:
              description: List of Bars and their statuses.`)
