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
	"context"
	"encoding/json"
	"fmt"
	"path"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"

	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"

	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestApplyCRDStructuralSchema tests that when a CRD has a structural schema in its validation field,
// it will be used to construct the CR schema used by apply.
func TestApplyCRDStructuralSchema(t *testing.T) {
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

	noxuDefinition := fixtures.NewMultipleVersionNoxuCRD(apiextensionsv1.ClusterScoped)

	var c apiextensionsv1.CustomResourceValidation
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
		t.Fatal(err)
	}
	for i := range noxuDefinition.Spec.Versions {
		noxuDefinition.Spec.Versions[i].Schema = &c
	}

	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
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
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}
	verifyNumFinalizers(t, result, 1)
	verifyFinalizersIncludes(t, result, "test-finalizer")
	verifyReplicas(t, result, 1)
	verifyNumPorts(t, result, 1)

	// Patch object to add another finalizer to the finalizers list
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"metadata":{"finalizers":["test-finalizer","another-one"]}}`)).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to add finalizer with merge patch: %v:\n%v", err, string(result))
	}
	verifyNumFinalizers(t, result, 2)
	verifyFinalizersIncludes(t, result, "test-finalizer")
	verifyFinalizersIncludes(t, result, "another-one")

	// Re-apply the same config, should work fine, since finalizers should have the list-type extension 'set'.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		SetHeader("Accept", "application/json").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to apply same config after adding a finalizer: %v:\n%v", err, string(result))
	}
	verifyNumFinalizers(t, result, 2)
	verifyFinalizersIncludes(t, result, "test-finalizer")
	verifyFinalizersIncludes(t, result, "another-one")

	// Patch object to change the number of replicas
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"spec":{"replicas": 5}}`)).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
	}
	verifyReplicas(t, result, 5)

	// Re-apply, we should get conflicts now, since the number of replicas was changed.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
	}
	status, ok := err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) != 1 {
		t.Fatalf("Expecting to get one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
	}
	verifyReplicas(t, result, 1)

	// New applier tries to edit an existing list item, we should get conflicts.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
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
		DoRaw(context.TODO())
	if err == nil {
		t.Fatalf("Expecting to get conflicts when a different applier updates existing list item, got no error: %s", result)
	}
	status, ok = err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) != 1 {
		t.Fatalf("Expecting to get one conflict when a different applier updates existing list item, got: %v", status.Status().Details.Causes)
	}

	// New applier tries to add a new list item, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
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
		SetHeader("Accept", "application/json").
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to add a new list item to the object as a different applier: %v:\n%v", err, string(result))
	}
	verifyNumPorts(t, result, 2)

	// UpdateOnCreate
	notExistingYAMLBody := []byte(fmt.Sprintf(`
	{
		"apiVersion": "%s",
		"kind": "%s",
		"metadata": {
		  "name": "%s",
		  "finalizers": [
			"test-finalizer"
		  ]
		},
		"spec": {
		  "cronSpec": "* * * * */5",
		  "replicas": 1,
		  "ports": [
			{
			  "name": "x",
			  "containerPort": 80
			}
		  ]
		},
		"protocol": "TCP"
	}`, apiVersion, kind, "should-not-exist"))
	_, err = rest.Put().
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name("should-not-exist").
		Param("fieldManager", "apply_test").
		Body(notExistingYAMLBody).
		DoRaw(context.TODO())
	if !apierrors.IsNotFound(err) {
		t.Fatalf("create on update should fail with notFound, got %v", err)
	}
}

// verifyNumFinalizers checks that len(.metadata.finalizers) == n
func verifyNumFinalizers(t *testing.T, b []byte, n int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if actual, expected := len(obj.GetFinalizers()), n; actual != expected {
		t.Fatalf("expected %v finalizers but got %v:\n%v", expected, actual, string(b))
	}
}

// verifyFinalizersIncludes checks that .metadata.finalizers includes e
func verifyFinalizersIncludes(t *testing.T, b []byte, e string) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	for _, a := range obj.GetFinalizers() {
		if a == e {
			return
		}
	}
	t.Fatalf("expected finalizers to include %q but got: %v", e, obj.GetFinalizers())
}

// verifyReplicas checks that .spec.replicas == r
func verifyReplicas(t *testing.T, b []byte, r int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		t.Fatalf("failed to find replicas number in response: %v:\n%v", err, string(b))
	}
	spec, ok := obj.Object["spec"]
	if !ok {
		t.Fatalf("failed to find replicas number in response:\n%v", string(b))
	}
	specMap, ok := spec.(map[string]interface{})
	if !ok {
		t.Fatalf("failed to find replicas number in response:\n%v", string(b))
	}
	replicas, ok := specMap["replicas"]
	if !ok {
		t.Fatalf("failed to find replicas number in response:\n%v", string(b))
	}
	replicasNumber, ok := replicas.(int64)
	if !ok {
		t.Fatalf("failed to find replicas number in response: expected int64 but got: %v", reflect.TypeOf(replicas))
	}
	if actual, expected := replicasNumber, int64(r); actual != expected {
		t.Fatalf("expected %v ports but got %v:\n%v", expected, actual, string(b))
	}
}

// verifyNumPorts checks that len(.spec.ports) == n
func verifyNumPorts(t *testing.T, b []byte, n int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		t.Fatalf("failed to find ports list in response: %v:\n%v", err, string(b))
	}
	spec, ok := obj.Object["spec"]
	if !ok {
		t.Fatalf("failed to find ports list in response:\n%v", string(b))
	}
	specMap, ok := spec.(map[string]interface{})
	if !ok {
		t.Fatalf("failed to find ports list in response:\n%v", string(b))
	}
	ports, ok := specMap["ports"]
	if !ok {
		t.Fatalf("failed to find ports list in response:\n%v", string(b))
	}
	portsList, ok := ports.([]interface{})
	if !ok {
		t.Fatalf("failed to find ports list in response: expected array but got: %v", reflect.TypeOf(ports))
	}
	if actual, expected := len(portsList), n; actual != expected {
		t.Fatalf("expected %v ports but got %v:\n%v", expected, actual, string(b))
	}
}

func findCRDCondition(crd *apiextensionsv1.CustomResourceDefinition, conditionType apiextensionsv1.CustomResourceDefinitionConditionType) *apiextensionsv1.CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

// TestApplyCRDUnhandledSchema tests that when a CRD has a schema that kube-openapi ToProtoModels cannot handle correctly,
// apply falls back to non-schema behavior
func TestApplyCRDUnhandledSchema(t *testing.T) {
	storageConfig := framework.SharedEtcd()
	tlsInfo := transport.TLSInfo{
		CertFile:      storageConfig.Transport.CertFile,
		KeyFile:       storageConfig.Transport.KeyFile,
		TrustedCAFile: storageConfig.Transport.TrustedCAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}
	etcdConfig := clientv3.Config{
		Endpoints:   storageConfig.Transport.ServerList,
		DialTimeout: 20 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: tlsConfig,
	}
	etcdclient, err := clientv3.New(etcdConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer etcdclient.Close()

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, storageConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// this has to be v1beta1, so we can have an item with validation that does not match.  v1 validation prevents this.

	noxuBetaDefinition := &apiextensionsv1beta1.CustomResourceDefinition{
		TypeMeta: metav1.TypeMeta{
			Kind:       "CustomResourceDefinition",
			APIVersion: "apiextensions.k8s.io/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{{
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
			}},
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
				Categories: []string{"all"},
			},
			Scope: apiextensionsv1beta1.ClusterScoped,
		},
	}

	// This is a schema that kube-openapi ToProtoModels does not handle correctly.
	// https://github.com/kubernetes/kubernetes/blob/38752f7f99869ed65fb44378360a517649dc2f83/vendor/k8s.io/kube-openapi/pkg/util/proto/document.go#L184
	var c apiextensionsv1beta1.CustomResourceValidation
	err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"properties": {
				"TypeFooBar": {
					"type": "array"
				}
			}
		}
	}`), &c)
	if err != nil {
		t.Fatal(err)
	}
	noxuBetaDefinition.Spec.Validation = &c

	betaBytes, err := json.Marshal(noxuBetaDefinition)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(betaBytes))
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	key := path.Join("/", storageConfig.Prefix, "apiextensions.k8s.io", "customresourcedefinitions", noxuBetaDefinition.Name)
	if _, err := etcdclient.Put(ctx, key, string(betaBytes)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	noxuDefinition, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), noxuBetaDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// wait until the CRD is established
	err = wait.Poll(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		localCrd, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), noxuBetaDefinition.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		condition := findCRDCondition(localCrd, apiextensionsv1.Established)
		if condition == nil {
			return false, nil
		}
		if condition.Status == apiextensionsv1.ConditionTrue {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}

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
  replicas: 1`, apiVersion, kind, name))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}
	verifyReplicas(t, result, 1)

	// Patch object to change the number of replicas
	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Body([]byte(`{"spec":{"replicas": 5}}`)).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
	}
	verifyReplicas(t, result, 5)

	// Re-apply, we should get conflicts now, since the number of replicas was changed.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
	}
	status, ok := err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) != 1 {
		t.Fatalf("Expecting to get one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
	}
	verifyReplicas(t, result, 1)
}

func getManagedFields(rawResponse []byte) ([]metav1.ManagedFieldsEntry, error) {
	obj := unstructured.Unstructured{}
	if err := obj.UnmarshalJSON(rawResponse); err != nil {
		return nil, err
	}
	return obj.GetManagedFields(), nil
}

func TestDefaultMissingKeyCRD(t *testing.T) {
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

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
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
										"default": "TCP",
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
	}`), &noxuDefinition.Spec.Versions[0].Schema)
	if err != nil {
		t.Fatal(err)
	}
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
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
    containerPort: 80`, apiVersion, kind, name))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name(name).
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, string(result))
	}

	// New applier tries to edit an existing list item, we should get conflicts.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
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
		DoRaw(context.TODO())
	if err == nil {
		t.Fatalf("Expecting to get conflicts when a different applier updates existing list item, got no error: %s", result)
	}
	status, ok := err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) != 1 {
		t.Fatalf("Expecting to get one conflict when a different applier updates existing list item, got: %v", status.Status().Details.Causes)
	}
}

func TestNoOpApplyWithDefaultsSameResourceVersionCRD(t *testing.T) {
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

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"properties": {
						"infrastructureRef": {
							"type": "object",
							"properties": {
								"name": {
									"type": "string"
								},
								"namespace": {
									"type": "string",
									"default": "default-namespace"
								}
							},
							"x-kubernetes-map-type": "atomic"
						}
					}
				}
			}
		}
	}`), &noxuDefinition.Spec.Versions[0].Schema)
	if err != nil {
		t.Fatal(err)
	}
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	group := noxuDefinition.Spec.Group
	kind := noxuDefinition.Spec.Names.Kind
	resource := noxuDefinition.Spec.Names.Plural
	version := noxuDefinition.Spec.Versions[0].Name
	apiVersion := noxuDefinition.Spec.Group + "/" + version
	gvr := schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
	name := "mytest"

	// fieldWithDefault will be defaulted
	applyConfiguration := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"name": name,
			},
			"spec": map[string]interface{}{"infrastructureRef": map[string]interface{}{"name": "infrastructure-machine-1\n"}},
		},
	}

	created, err := dynamicClient.Resource(gvr).Apply(context.TODO(), name, applyConfiguration, metav1.ApplyOptions{FieldManager: "apply_test"})

	if err != nil {
		t.Fatalf("failed to create custom resource with apply: %v:\n%v", err, created)
	}

	createdAccessor, err := meta.Accessor(created)
	if err != nil {
		t.Fatalf("Failed to get meta accessor for updated object: %v", err)
	}

	// Sleep for one second to make sure that the times of each update operation is different.
	time.Sleep(1 * time.Second)

	updated, err := dynamicClient.Resource(gvr).Apply(context.TODO(), name, applyConfiguration, metav1.ApplyOptions{FieldManager: "apply_test"})
	if err != nil {
		t.Fatalf("failed to update custom resource with apply: %v:\n%v", err, updated)
	}

	updatedAccessor, err := meta.Accessor(updated)
	if err != nil {
		t.Fatalf("Failed to get meta accessor for updated object: %v", err)
	}

	if createdAccessor.GetResourceVersion() != updatedAccessor.GetResourceVersion() {
		t.Fatalf("Expected same resource version to be %v but got: %v\nold object:\n%v\nnew object:\n%v",
			createdAccessor.GetResourceVersion(),
			updatedAccessor.GetResourceVersion(),
			created,
			updated,
		)
	}
}
