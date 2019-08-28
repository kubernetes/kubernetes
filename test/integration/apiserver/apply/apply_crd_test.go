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

package apiserver

import (
	"encoding/json"
	"fmt"
	"testing"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestApplyCRDNoSchema tests that CRDs and CRs can both be applied to with a PATCH request with the apply content type
// when there is no validation field provided.
func TestApplyCRDNoSchema(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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

	noxuDefinition := fixtures.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)

	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Version
	name := "mytest"

	rest := apiExtensionClient.Discovery().RESTClient()
	yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  replicas: 1`, apiVersion, kind, name))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}

	// Patch object to change the number of replicas
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"spec":{"replicas": 5}}`)).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
	}

	// Re-apply, we should get conflicts now, since the number of replicas was changed.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
	}
	status, ok := err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
	}
}

// TestApplyCRDStructuralSchema tests that when a CRD has a structural schema in its validation field,
// it will be used to construct the CR schema used by apply.
func TestApplyCRDStructuralSchema(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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

	noxuDefinition := fixtures.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)

	var c apiextensionsv1beta1.CustomResourceValidation
	err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"x-kubernetes-preserve-unknown-fields": true,
					"properties": {
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
									"containerPort"
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
		t.Fatal(err)
	}
	noxuDefinition.Spec.Validation = &c
	falseBool := false
	noxuDefinition.Spec.PreserveUnknownFields = &falseBool

	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Version
	name := "mytest"

	rest := apiExtensionClient.Discovery().RESTClient()
	yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
  finalizers:
  - test-finalizer
spec:
  cronSpec: "* * * * */5"
  replicas: 1
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}

	// Patch object to add another finalizer to the finalizers list
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"metadata":{"finalizers":["another-one"]}}`)).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to add finalizer with merge patch: %v:\n%v", err, string(result))
	}

	// Re-apply the same config, should work fine, since finalizers should have the list-type extension 'set'.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to apply same config after adding a finalizer: %v:\n%v", err, string(result))
	}

	// Patch object to change the number of replicas
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"spec":{"replicas": 5}}`)).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
	}

	// Re-apply, we should get conflicts now, since the number of replicas was changed.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
	}
	status, ok := err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
	}

	// New applier tries to edit an existing list item, we should get conflicts.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test_2").
		Body([]byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  ports:
  - name: "y"
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))).
		DoRaw()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when a different applier updates existing list item, got no error: %s", result)
	}
	status, ok = err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when a different applier updates existing list item, got: %v", status.Status().Details.Causes)
	}

	// New applier tries to add a new list item, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test_2").
		Body([]byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  ports:
  - name: "y"
    containerPort: 8080
    protocol: TCP`, apiVersion, kind, name))).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to add a new list item to the object as a different applier: %v:\n%v", err, string(result))
	}
}

// TestApplyCRDNonStructuralSchema tests that when a CRD has a non-structural schema in its validation field,
// it will be used to construct the CR schema used by apply, but any non-structural parts of the schema will be treated as
// nested maps (same as a CRD without a schema)
func TestApplyCRDNonStructuralSchema(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)

	var c apiextensionsv1beta1.CustomResourceValidation
	err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"anyOf": [
						{
							"type": "object",
							"properties": {
								"cronSpec": {
									"type": "string",
									"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
								}
							}
						}, {
							"type": "string"
						}
					]
				}
			}
		}
	}`), &c)
	if err != nil {
		t.Fatal(err)
	}
	noxuDefinition.Spec.Validation = &c

	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Version
	name := "mytest"

	rest := apiExtensionClient.Discovery().RESTClient()
	yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
  finalizers:
  - test-finalizer
spec:
  cronSpec: "* * * * */5"
  replicas: 1`, apiVersion, kind, name))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}

	// Patch object to add another finalizer to the finalizers list
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"metadata":{"finalizers":["another-one"]}}`)).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to add finalizer with merge patch: %v:\n%v", err, string(result))
	}

	// Re-apply the same config, should work fine, since finalizers should have the list-type extension 'set'.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to apply same config after adding a finalizer: %v:\n%v", err, string(result))
	}

	// Patch object to change the number of replicas
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"spec":{"replicas": 5}}`)).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
	}

	// Re-apply, we should get conflicts now, since the number of replicas was changed.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
	}
	status, ok := err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Version, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw()
	if err != nil {
		t.Fatalf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
	}
}
