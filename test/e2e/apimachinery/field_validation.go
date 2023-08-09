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

package apimachinery

import (
	// ensure libs have a chance to initialize
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2"
	_ "github.com/stretchr/testify/assert"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("FieldValidation", func() {
	f := framework.NewDefaultFramework("field-validation")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var client clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment", metav1.DeleteOptions{})
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment-shared-unset", metav1.DeleteOptions{})
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment-shared-map-item-removal", metav1.DeleteOptions{})
		_ = client.CoreV1().Pods(ns).Delete(ctx, "test-pod", metav1.DeleteOptions{})
	})

	/*
		Release: v1.27
		Testname: Server side field validation, typed object
		Description: It should reject the request if a typed object has unknown or duplicate fields.
	*/
	framework.ConformanceIt("should detect unknown and duplicate fields of a typed object", func(ctx context.Context) {
		ginkgo.By("apply creating a deployment")
		invalidMetaDeployment := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "my-dep",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"unknownField": "foo",
			"replicas": 2,
			"replicas": 3,
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "nginx:latest"
					}]
				}
			}
		}
	}`
		_, err := client.CoreV1().RESTClient().Post().
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body([]byte(invalidMetaDeployment)).
			Do(ctx).
			Get()
		if !(strings.Contains(err.Error(), `strict decoding error: unknown field "spec.unknownField", duplicate field "spec.replicas"`)) {
			framework.Failf("error missing unknown/duplicate field field, got: %v", err)
		}

	})

	/*
		Release: v1.27
		Testname: Server side field validation, typed unknown metadata
		Description: It should reject the request if a typed object has unknown fields in the metadata.
	*/
	framework.ConformanceIt("should detect unknown metadata fields of a typed object", func(ctx context.Context) {
		ginkgo.By("apply creating a deployment")
		invalidMetaDeployment := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "my-dep",
			"unknownMeta": "foo",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "nginx:latest"
					}]
				}
			}
		}
	}`
		_, err := client.CoreV1().RESTClient().Post().
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body([]byte(invalidMetaDeployment)).
			Do(ctx).
			Get()
		if !(strings.Contains(err.Error(), `strict decoding error: unknown field "metadata.unknownMeta"`)) {
			framework.Failf("error missing unknown metadata field, got: %v", err)
		}

	})

	/*
		Release: v1.27
		Testname: Server side field validation, valid CR with validation schema
		Description: When a CRD has a validation schema, it should succeed when a valid CR is applied.
	*/
	framework.ConformanceIt("should create/apply a valid CR for CRD with validation schema", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		var c apiextensionsv1.CustomResourceValidation
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"properties": {
						"foo": {
							"type": "string"
						},
						"cronSpec": {
							"type": "string",
							"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
						},
						"ports": {
							"type": "array",
							"x-kubernetes-list-map-keys": [
								"containerPort",
								"protocol"
							],
							"x-kubernetes-list-type": "map",
							"items": {
								"properties": {
									"containerPort": {
										"format": "int32",
										"type": "integer"
									},
									"hostIP": {
										"type": "string"
									},
									"hostPort": {
										"format": "int32",
										"type": "integer"
									},
									"name": {
										"type": "string"
									},
									"protocol": {
										"type": "string"
									}
								},
								"required": [
									"containerPort",
									"protocol"
								],
								"type": "object"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%v", err)
		}
		for i := range noxuDefinition.Spec.Versions {
			noxuDefinition.Spec.Versions[i].Schema = &c
		}

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  foo: foo1
  cronSpec: "* * * * */5"
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
		_, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body(yamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.27
		Testname: Server side field validation, unknown fields CR no validation schema
		Description: When a CRD does not have a validation schema, it should succeed when a CR with unknown fields is applied.
	*/
	framework.ConformanceIt("should create/apply a CR with unknown fields for CRD with no validation schema", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  unknown: uk1
  cronSpec: "* * * * */5"
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
		_, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body(yamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("%v", err)
		}

	})

	/*
		Release: v1.27
		Testname: Server side field validation, unknown fields CR fails validation
		Description: When a CRD does have a validation schema, it should reject CRs with unknown fields.
	*/
	framework.ConformanceIt("should create/apply an invalid CR with extra properties for CRD with validation schema", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		var c apiextensionsv1.CustomResourceValidation
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"properties": {
						"foo": {
							"type": "string"
						},
						"cronSpec": {
							"type": "string",
							"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
						},
						"ports": {
							"type": "array",
							"x-kubernetes-list-map-keys": [
								"containerPort",
								"protocol"
							],
							"x-kubernetes-list-type": "map",
							"items": {
								"properties": {
									"containerPort": {
										"format": "int32",
										"type": "integer"
									},
									"hostIP": {
										"type": "string"
									},
									"hostPort": {
										"format": "int32",
										"type": "integer"
									},
									"name": {
										"type": "string"
									},
									"protocol": {
										"type": "string"
									}
								},
								"required": [
									"containerPort",
									"protocol"
								],
								"type": "object"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%v", err)
		}
		klog.Warningf("props: %v\n", c.OpenAPIV3Schema)
		for i := range noxuDefinition.Spec.Versions {
			noxuDefinition.Spec.Versions[i].Schema = &c
		}

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
unknownField: unknown
spec:
  foo: foo1
  cronSpec: "* * * * */5"
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
		result, err := rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body(yamlBody).
			DoRaw(ctx)
		if !(strings.Contains(string(result), `.unknownField: field not declared in schema`)) {
			framework.Failf("error missing unknown field: %v:\n%v", err, string(result))
		}
	})

	/*
		Release: v1.27
		Testname: Server side field validation, unknown metadata
		Description: The server should reject CRs with unknown metadata fields in both the root and embedded objects
		of a CR.
	*/
	framework.ConformanceIt("should detect unknown metadata fields in both the root and embedded object of a CR", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		var c apiextensionsv1.CustomResourceValidation
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"x-kubernetes-preserve-unknown-fields": true,
					"properties": {
						"template": {
							"type": "object",
							"x-kubernetes-embedded-resource": true,
							"properties": {
								"metadata": {
									"type": "object",
									"properties": {
										"name": {
											"type": "string"
										}
									}
								},
								"spec": {
									"type": "object"
								}
							}

						},
						"foo": {
							"type": "string"
						},
						"cronSpec": {
							"type": "string",
							"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
						},
						"ports": {
							"type": "array",
							"x-kubernetes-list-map-keys": [
								"containerPort",
								"protocol"
							],
							"x-kubernetes-list-type": "map",
							"items": {
								"properties": {
									"containerPort": {
										"format": "int32",
										"type": "integer"
									},
									"hostIP": {
										"type": "string"
									},
									"hostPort": {
										"format": "int32",
										"type": "integer"
									},
									"name": {
										"type": "string"
									},
									"protocol": {
										"type": "string"
									}
								},
								"required": [
									"containerPort",
									"protocol"
								],
								"type": "object"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%v", err)
		}
		for i := range noxuDefinition.Spec.Versions {
			noxuDefinition.Spec.Versions[i].Schema = &c
		}

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
  unknownMeta: unknown
spec:
  template:
    apiversion: foo/v1
    kind: Sub
    metadata:
        unknownSubMeta: unknown
        name: subobject
        namespace: %s
  foo: foo1
  cronSpec: "* * * * */5"
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name, ns))
		result, err := rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body(yamlBody).
			DoRaw(ctx)
		if !(strings.Contains(string(result), `.spec.template.metadata.unknownSubMeta: field not declared in schema`) || strings.Contains(string(result), `.metadata.unknownMeta: field not declared in schema`)) {
			framework.Failf("error missing duplicate field: %v:\n%v", err, string(result))
		}
	})

	/*
		Release: v1.27
		Testname: Server side field validation, CR duplicates
		Description: The server should reject CRs with duplicate fields even when preserving unknown fields.
	*/
	framework.ConformanceIt("should detect duplicates in a CR when preserving unknown fields", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		var c apiextensionsv1.CustomResourceValidation
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"x-kubernetes-preserve-unknown-fields": true,
					"properties": {
						"foo": {
							"type": "string"
						},
						"cronSpec": {
							"type": "string",
							"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
						},
						"ports": {
							"type": "array",
							"x-kubernetes-list-map-keys": [
								"containerPort",
								"protocol"
							],
							"x-kubernetes-list-type": "map",
							"items": {
								"properties": {
									"containerPort": {
										"format": "int32",
										"type": "integer"
									},
									"hostIP": {
										"type": "string"
									},
									"hostPort": {
										"format": "int32",
										"type": "integer"
									},
									"name": {
										"type": "string"
									},
									"protocol": {
										"type": "string"
									}
								},
								"required": [
									"containerPort",
									"protocol"
								],
								"type": "object"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%s", err)
		}
		for i := range noxuDefinition.Spec.Versions {
			noxuDefinition.Spec.Versions[i].Schema = &c
		}

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  unknown: uk1
  foo: foo1
  foo: foo2
  cronSpec: "* * * * */5"
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
		result, err := rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "field_validation_mgr").
			Param("fieldValidation", "Strict").
			Body(yamlBody).
			DoRaw(ctx)
		if !(strings.Contains(string(result), `line 9: key \"foo\" already set in map`)) {
			framework.Failf("error missing duplicate field: %v:\n%v", err, string(result))
		}
	})
})
