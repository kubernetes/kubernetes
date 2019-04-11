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

package apimachinery

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/go-openapi/spec"
	. "github.com/onsi/ginkgo"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	k8sclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	openapiutil "k8s.io/kube-openapi/pkg/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/crd"
	"sigs.k8s.io/yaml"
)

var (
	crdPublishOpenAPIVersion = utilversion.MustParseSemantic("v1.14.0")
	metaPattern              = `"kind":"%s","apiVersion":"%s/%s","metadata":{"name":"%s"}`
)

var _ = SIGDescribe("CustomResourcePublishOpenAPI [Feature:CustomResourcePublishOpenAPI]", func() {
	f := framework.NewDefaultFramework("crd-publish-openapi")

	BeforeEach(func() {
		framework.SkipUnlessServerVersionGTE(crdPublishOpenAPIVersion, f.ClientSet.Discovery())
	})

	It("works for CRD with validation schema", func() {
		crd, err := setupCRD(f, schemaFoo, "foo", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Kind, crd.APIGroup, crd.Versions[0].Name, "test-foo")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		By("client-side validation (kubectl create and apply) allows request with known and required properties")
		validCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}]}}`, meta)
		if _, err := framework.RunKubectlInput(validCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create valid CR %s: %v", validCR, err)
		}
		if _, err := framework.RunKubectl(ns, "delete", crd.GetPluralName(), "test-foo"); err != nil {
			framework.Failf("failed to delete valid CR: %v", err)
		}
		if _, err := framework.RunKubectlInput(validCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply valid CR %s: %v", validCR, err)
		}
		if _, err := framework.RunKubectl(ns, "delete", crd.GetPluralName(), "test-foo"); err != nil {
			framework.Failf("failed to delete valid CR: %v", err)
		}

		By("client-side validation (kubectl create and apply) rejects request with unknown properties when disallowed by the schema")
		unknownCR := fmt.Sprintf(`{%s,"spec":{"foo":true}}`, meta)
		if _, err := framework.RunKubectlInput(unknownCR, ns, "create", "-f", "-"); err == nil || !strings.Contains(err.Error(), `unknown field "foo"`) {
			framework.Failf("unexpected no error when creating CR with unknown field: %v", err)
		}
		if _, err := framework.RunKubectlInput(unknownCR, ns, "apply", "-f", "-"); err == nil || !strings.Contains(err.Error(), `unknown field "foo"`) {
			framework.Failf("unexpected no error when applying CR with unknown field: %v", err)
		}

		By("client-side validation (kubectl create and apply) rejects request without required properties")
		noRequireCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"age":"10"}]}}`, meta)
		if _, err := framework.RunKubectlInput(noRequireCR, ns, "create", "-f", "-"); err == nil || !strings.Contains(err.Error(), `missing required field "name"`) {
			framework.Failf("unexpected no error when creating CR without required field: %v", err)
		}
		if _, err := framework.RunKubectlInput(noRequireCR, ns, "apply", "-f", "-"); err == nil || !strings.Contains(err.Error(), `missing required field "name"`) {
			framework.Failf("unexpected no error when applying CR without required field: %v", err)
		}

		By("kubectl explain works to explain CR properties")
		if err := verifyKubectlExplain(crd.GetPluralName(), `(?s)DESCRIPTION:.*Foo CRD for Testing.*FIELDS:.*apiVersion.*<string>.*APIVersion defines.*spec.*<Object>.*Specification of Foo`); err != nil {
			framework.Failf("%v", err)
		}

		By("kubectl explain works to explain CR properties recursively")
		if err := verifyKubectlExplain(crd.GetPluralName()+".metadata", `(?s)DESCRIPTION:.*Standard object's metadata.*FIELDS:.*creationTimestamp.*<string>.*CreationTimestamp is a timestamp`); err != nil {
			framework.Failf("%v", err)
		}
		if err := verifyKubectlExplain(crd.GetPluralName()+".spec", `(?s)DESCRIPTION:.*Specification of Foo.*FIELDS:.*bars.*<\[\]Object>.*List of Bars and their specs`); err != nil {
			framework.Failf("%v", err)
		}
		if err := verifyKubectlExplain(crd.GetPluralName()+".spec.bars", `(?s)RESOURCE:.*bars.*<\[\]Object>.*DESCRIPTION:.*List of Bars and their specs.*FIELDS:.*bazs.*<\[\]string>.*List of Bazs.*name.*<string>.*Name of Bar`); err != nil {
			framework.Failf("%v", err)
		}

		By("kubectl explain works to return error when explain is called on property that doesn't exist")
		if _, err := framework.RunKubectl("explain", crd.GetPluralName()+".spec.bars2"); err == nil || !strings.Contains(err.Error(), `field "bars2" does not exist`) {
			framework.Failf("unexpected no error when explaining property that doesn't exist: %v", err)
		}

		if err := cleanupCRD(f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("works for CRD without validation schema", func() {
		crd, err := setupCRD(f, nil, "empty", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Kind, crd.APIGroup, crd.Versions[0].Name, "test-cr")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		By("client-side validation (kubectl create and apply) allows request with any unknown properties")
		randomCR := fmt.Sprintf(`{%s,"a":{"b":[{"c":"d"}]}}`, meta)
		if _, err := framework.RunKubectlInput(randomCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := framework.RunKubectl(ns, "delete", crd.GetPluralName(), "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}
		if _, err := framework.RunKubectlInput(randomCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := framework.RunKubectl(ns, "delete", crd.GetPluralName(), "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}

		By("kubectl explain works to explain CR without validation schema")
		if err := verifyKubectlExplain(crd.GetPluralName(), `(?s)DESCRIPTION:.*<empty>`); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("works for multiple CRDs of different groups", func() {
		By("CRs in different groups (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "foo", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "waldo", "v1beta1")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.APIGroup == crdWaldo.APIGroup {
			framework.Failf("unexpected: CRDs should be of different group %v, %v", crdFoo.APIGroup, crdWaldo.APIGroup)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v1beta1"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v1"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("works for multiple CRDs of same group but different versions", func() {
		By("CRs in the same group but different versions (one multiversion CRD) show up in OpenAPI documentation")
		crdMultiVer, err := setupCRD(f, schemaFoo, "multi-ver", "v2", "v3")
		if err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v3"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdMultiVer); err != nil {
			framework.Failf("%v", err)
		}

		By("CRs in the same group but different versions (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "common-group", "v4")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "common-group", "v5")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.APIGroup != crdWaldo.APIGroup {
			framework.Failf("unexpected: CRDs should be of the same group %v, %v", crdFoo.APIGroup, crdWaldo.APIGroup)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v5"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v4"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("works for multiple CRDs of same group and version but different kinds", func() {
		By("CRs in the same group and version but different kinds (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "common-group", "v6")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "common-group", "v6")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.APIGroup != crdWaldo.APIGroup {
			framework.Failf("unexpected: CRDs should be of the same group %v, %v", crdFoo.APIGroup, crdWaldo.APIGroup)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v6"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v6"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("updates the published spec when one versin gets renamed", func() {
		By("set up a multi version CRD")
		crdMultiVer, err := setupCRD(f, schemaFoo, "multi-ver", "v2", "v3")
		if err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v3"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		By("rename a version")
		patch := []byte(`{"spec":{"versions":[{"name":"v2","served":true,"storage":true},{"name":"v4","served":true,"storage":false}]}}`)
		crdMultiVer.Crd, err = crdMultiVer.APIExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Patch(crdMultiVer.GetMetaName(), types.MergePatchType, patch)
		if err != nil {
			framework.Failf("%v", err)
		}

		By("check the new version name is served")
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v4"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		By("check the old version name is removed")
		if err := waitForDefinitionCleanup(f.ClientSet, definitionName(crdMultiVer, "v3")); err != nil {
			framework.Failf("%v", err)
		}
		By("check the other version is not changed")
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		// TestCrd.Versions is different from TestCrd.Crd.Versions, we have to manually
		// update the name there. Used by cleanupCRD
		crdMultiVer.Versions[1].Name = "v4"
		if err := cleanupCRD(f, crdMultiVer); err != nil {
			framework.Failf("%v", err)
		}
	})

	It("removes definition from spec when one versin gets changed to not be served", func() {
		By("set up a multi version CRD")
		crd, err := setupCRD(f, schemaFoo, "multi-to-single-ver", "v5", "v6alpha1")
		if err != nil {
			framework.Failf("%v", err)
		}
		// just double check. setupCRD() checked this for us already
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v6alpha1"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v5"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		By("mark a version not serverd")
		crd.Crd.Spec.Versions[1].Served = false
		crd.Crd, err = crd.APIExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd.Crd)
		if err != nil {
			framework.Failf("%v", err)
		}

		By("check the unserved version gets removed")
		if err := waitForDefinitionCleanup(f.ClientSet, definitionName(crd, "v6alpha1")); err != nil {
			framework.Failf("%v", err)
		}
		By("check the other version is not changed")
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v5"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})
})

func setupCRD(f *framework.Framework, schema []byte, groupSuffix string, versions ...string) (*crd.TestCrd, error) {
	group := fmt.Sprintf("%s-test-%s.k8s.io", f.BaseName, groupSuffix)
	if len(versions) == 0 {
		return nil, fmt.Errorf("require at least one version for CRD")
	}
	apiVersions := []v1beta1.CustomResourceDefinitionVersion{}
	for _, version := range versions {
		v := v1beta1.CustomResourceDefinitionVersion{
			Name:    version,
			Served:  true,
			Storage: false,
		}
		apiVersions = append(apiVersions, v)
	}
	apiVersions[0].Storage = true

	crd, err := crd.CreateMultiVersionTestCRD(f, group, apiVersions, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create CRD: %v", err)
	}

	if schema != nil {
		// patch validation schema for all versions
		if err := patchSchema(schema, crd); err != nil {
			return nil, fmt.Errorf("failed to patch schema: %v", err)
		}
	} else {
		// change expectation if CRD doesn't have schema
		schema = []byte(`type: object`)
	}

	for _, v := range crd.Versions {
		if err := waitForDefinition(f.ClientSet, definitionName(crd, v.Name), schema); err != nil {
			return nil, fmt.Errorf("%v", err)
		}
	}
	return crd, nil
}

func cleanupCRD(f *framework.Framework, crd *crd.TestCrd) error {
	crd.CleanUp()
	for _, v := range crd.Versions {
		name := definitionName(crd, v.Name)
		if err := waitForDefinitionCleanup(f.ClientSet, name); err != nil {
			return fmt.Errorf("%v", err)
		}
	}
	return nil
}

// patchSchema takes schema in YAML and patches it to given CRD in given version
func patchSchema(schema []byte, crd *crd.TestCrd) error {
	s, err := utilyaml.ToJSON(schema)
	if err != nil {
		return fmt.Errorf("failed to create json patch: %v", err)
	}
	patch := []byte(fmt.Sprintf(`{"spec":{"validation":{"openAPIV3Schema":%s}}}`, string(s)))
	crd.Crd, err = crd.APIExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Patch(crd.GetMetaName(), types.MergePatchType, patch)
	return err
}

const waitSuccessThreshold = 10

// mustSucceedMultipleTimes calls f multiple times on success and only returns true if all calls are successful.
// This is necessary to avoid flaking tests where one call might hit a good apiserver while in HA other apiservers
// might be lagging behind. Calling f multiple times reduces the chance exponentially.
func mustSucceedMultipleTimes(n int, f func() (bool, error)) func() (bool, error) {
	return func() (bool, error) {
		for i := 0; i < n; i++ {
			ok, err := f()
			if err != nil || !ok {
				return ok, err
			}
		}
		return true, nil
	}
}

// waitForDefinition waits for given definition showing up in swagger with given schema
func waitForDefinition(c k8sclientset.Interface, name string, schema []byte) error {
	expect := spec.Schema{}
	if err := convertJSONSchemaProps(schema, &expect); err != nil {
		return err
	}

	err := waitForOpenAPISchema(c, func(spec *spec.Swagger) (bool, string) {
		d, ok := spec.SwaggerProps.Definitions[name]
		if !ok {
			return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] not found", name)
		}
		// drop properties and extension that we added
		dropDefaults(&d)
		if !apiequality.Semantic.DeepEqual(expect, d) {
			return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] not match; expect: %v, actual: %v", name, expect, d)
		}
		return true, ""
	})
	if err != nil {
		return fmt.Errorf("failed to wait for definition %q to be served with the right OpenAPI schema: %v", name, err)
	}
	return nil
}

// waitForDefinitionCleanup waits for given definition to be removed from swagger
func waitForDefinitionCleanup(c k8sclientset.Interface, name string) error {
	err := waitForOpenAPISchema(c, func(spec *spec.Swagger) (bool, string) {
		if _, ok := spec.SwaggerProps.Definitions[name]; ok {
			return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] still exists", name)
		}
		return true, ""
	})
	if err != nil {
		return fmt.Errorf("failed to wait for definition %q not to be served anymore: %v", name, err)
	}
	return nil
}

func waitForOpenAPISchema(c k8sclientset.Interface, pred func(*spec.Swagger) (bool, string)) error {
	client := c.CoreV1().RESTClient().(*rest.RESTClient).Client
	url := c.CoreV1().RESTClient().Get().AbsPath("openapi", "v2").URL()
	lastMsg := ""
	etag := ""
	var etagSpec *spec.Swagger
	if err := wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, mustSucceedMultipleTimes(waitSuccessThreshold, func() (bool, error) {
		// download spec with etag support
		spec := &spec.Swagger{}
		req, err := http.NewRequest("GET", url.String(), nil)
		if err != nil {
			return false, err
		}
		req.Close = true // enforce a new connection to hit different HA API servers
		if len(etag) > 0 {
			req.Header.Set("If-None-Match", fmt.Sprintf(`"%s"`, etag))
		}
		resp, err := client.Do(req)
		if err != nil {
			return false, err
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusNotModified {
			spec = etagSpec
		} else if resp.StatusCode != http.StatusOK {
			return false, fmt.Errorf("unexpected response: %d", resp.StatusCode)
		} else if bs, err := ioutil.ReadAll(resp.Body); err != nil {
			return false, err
		} else if err := json.Unmarshal(bs, spec); err != nil {
			return false, err
		} else {
			etag = strings.Trim(resp.Header.Get("ETag"), `"`)
			etagSpec = spec
		}

		var ok bool
		ok, lastMsg = pred(spec)
		return ok, nil
	})); err != nil {
		return fmt.Errorf("failed to wait for OpenAPI spec validating condition: %v; lastMsg: %s", err, lastMsg)
	}
	return nil
}

// convertJSONSchemaProps converts JSONSchemaProps in YAML to spec.Schema
func convertJSONSchemaProps(in []byte, out *spec.Schema) error {
	external := v1beta1.JSONSchemaProps{}
	if err := yaml.UnmarshalStrict(in, &external); err != nil {
		return err
	}
	internal := apiextensions.JSONSchemaProps{}
	if err := v1beta1.Convert_v1beta1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(&external, &internal, nil); err != nil {
		return err
	}
	if err := validation.ConvertJSONSchemaProps(&internal, out); err != nil {
		return err
	}
	return nil
}

// dropDefaults drops properties and extension that we added to a schema
func dropDefaults(s *spec.Schema) {
	delete(s.Properties, "metadata")
	delete(s.Properties, "apiVersion")
	delete(s.Properties, "kind")
	delete(s.Extensions, "x-kubernetes-group-version-kind")
}

func verifyKubectlExplain(name, pattern string) error {
	result, err := framework.RunKubectl("explain", name)
	if err != nil {
		return fmt.Errorf("failed to explain %s: %v", name, err)
	}
	r := regexp.MustCompile(pattern)
	if !r.Match([]byte(result)) {
		return fmt.Errorf("kubectl explain %s result {%s} doesn't match pattern {%s}", name, result, pattern)
	}
	return nil
}

// definitionName returns the openapi definition name for given CRD in given version
func definitionName(crd *crd.TestCrd, version string) string {
	return openapiutil.ToRESTFriendlyName(fmt.Sprintf("%s/%s/%s", crd.APIGroup, version, crd.Kind))
}

var schemaFoo = []byte(`description: Foo CRD for Testing
type: object
properties:
  spec:
    type: object
    description: Specification of Foo
    properties:
      bars:
        description: List of Bars and their specs.
        type: array
        items:
          type: object
          required:
          - name
          properties:
            name:
              description: Name of Bar.
              type: string
            age:
              description: Age of Bar.
              type: string
            bazs:
              description: List of Bazs.
              items:
                type: string
              type: array
  status:
    description: Status of Foo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object
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

var schemaWaldo = []byte(`description: Waldo CRD for Testing
type: object
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
