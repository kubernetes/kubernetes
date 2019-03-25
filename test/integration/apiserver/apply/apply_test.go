/*
Copyright 2018 The Kubernetes Authors.

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
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

func setup(t *testing.T, groupVersions ...schema.GroupVersion) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	if len(groupVersions) > 0 {
		resourceConfig := master.DefaultAPIResourceConfigSource()
		resourceConfig.EnableVersions(groupVersions...)
		masterConfig.ExtraConfig.APIResourceConfigSource = resourceConfig
	}
	masterConfig.GenericConfig.OpenAPIConfig = framework.DefaultOpenAPIConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	return s, clientSet, closeFn
}

// TestApplyAlsoCreates makes sure that PATCH requests with the apply content type
// will create the object if it doesn't already exist
// TODO: make a set of test cases in an easy-to-consume place (separate package?) so it's easy to test in both integration and e2e.
func TestApplyAlsoCreates(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	testCases := []struct {
		resource string
		name     string
		body     string
	}{
		{
			resource: "pods",
			name:     "test-pod",
			body: `{
				"apiVersion": "v1",
				"kind": "Pod",
				"metadata": {
					"name": "test-pod"
				},
				"spec": {
					"containers": [{
						"name":  "test-container",
						"image": "test-image"
					}]
				}
			}`,
		}, {
			resource: "services",
			name:     "test-svc",
			body: `{
				"apiVersion": "v1",
				"kind": "Service",
				"metadata": {
					"name": "test-svc"
				},
				"spec": {
					"ports": [{
						"port": 8080,
						"protocol": "UDP"
					}]
				}
			}`,
		},
	}

	for _, tc := range testCases {
		_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			Namespace("default").
			Resource(tc.resource).
			Name(tc.name).
			Param("fieldManager", "apply_test").
			Body([]byte(tc.body)).
			Do().
			Get()
		if err != nil {
			t.Fatalf("Failed to create object using Apply patch: %v", err)
		}

		_, err = client.CoreV1().RESTClient().Get().Namespace("default").Resource(tc.resource).Name(tc.name).Do().Get()
		if err != nil {
			t.Fatalf("Failed to retrieve object: %v", err)
		}

		// Test that we can re apply with a different field manager and don't get conflicts
		_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			Namespace("default").
			Resource(tc.resource).
			Name(tc.name).
			Param("fieldManager", "apply_test_2").
			Body([]byte(tc.body)).
			Do().
			Get()
		if err != nil {
			t.Fatalf("Failed to re-apply object using Apply patch: %v", err)
		}
	}
}

// TestCreateOnApplyFailsWithUID makes sure that PATCH requests with the apply content type
// will not create the object if it doesn't already exist and it specifies a UID
func TestCreateOnApplyFailsWithUID(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("pods").
		Name("test-pod-uid").
		Param("fieldManager", "apply_test").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "Pod",
			"metadata": {
				"name": "test-pod-uid",
				"uid":  "88e00824-7f0e-11e8-94a1-c8d3ffb15800"
			},
			"spec": {
				"containers": [{
					"name":  "test-container",
					"image": "test-image"
				}]
			}
		}`)).
		Do().
		Get()
	if !errors.IsConflict(err) {
		t.Fatalf("Expected conflict error but got: %v", err)
	}
}

func TestApplyUpdateApplyConflictForced(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
                        "labels": {"app": "nginx"}
		},
		"spec": {
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
	}`)

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("fieldManager", "apply_test").
		Body(obj).Do().Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		AbsPath("/apis/extensions/v1beta1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Body([]byte(`{"spec":{"replicas": 5}}`)).Do().Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("fieldManager", "apply_test").
		Body([]byte(obj)).Do().Get()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object")
	}
	status, ok := err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when applying object, got: %v", status.Status().Details.Causes)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body([]byte(obj)).Do().Get()
	if err != nil {
		t.Fatalf("Failed to apply object with force: %v", err)
	}
}

// TestApplyManagedFields makes sure that managedFields api does not change
func TestApplyManagedFields(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "apply_test").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "test-cm",
				"namespace": "default",
				"labels": {
					"test-label": "test"
				}
			},
			"data": {
				"key": "value"
			}
		}`)).
		Do().
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "updater").
		Body([]byte(`{"data":{"key": "new value"}}`)).Do().Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do().Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	actual, err := json.MarshalIndent(object, "\t", "\t")
	if err != nil {
		t.Fatalf("Failed to marshal object: %v", err)
	}

	expected := []byte(`{
		"metadata": {
			"name": "test-cm",
			"namespace": "default",
			"selfLink": "` + accessor.GetSelfLink() + `",
			"uid": "` + string(accessor.GetUID()) + `",
			"resourceVersion": "` + accessor.GetResourceVersion() + `",
			"creationTimestamp": "` + accessor.GetCreationTimestamp().UTC().Format(time.RFC3339) + `",
			"labels": {
				"test-label": "test"
			},
			"managedFields": [
				{
					"manager": "apply_test",
					"operation": "Apply",
					"apiVersion": "v1",
					"fields": {
						"f:metadata": {
							"f:labels": {
								"f:test-label": {}
							}
						}
					}
				},
				{
					"manager": "updater",
					"operation": "Update",
					"apiVersion": "v1",
					"time": "` + accessor.GetManagedFields()[1].Time.UTC().Format(time.RFC3339) + `",
					"fields": {
						"f:data": {
							"f:key": {}
						}
					}
				}
			]
		},
		"data": {
			"key": "new value"
		}
	}`)

	if string(expected) != string(actual) {
		t.Fatalf("Expected:\n%v\nGot:\n%v", string(expected), string(actual))
	}
}

// TestApplyRemovesEmptyManagedFields there are no empty managers in managedFields
func TestApplyRemovesEmptyManagedFields(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "v1",
		"kind": "ConfigMap",
		"metadata": {
			"name": "test-cm",
			"namespace": "default"
		}
	}`)

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "apply_test").
		Body(obj).
		Do().
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "apply_test").
		Body(obj).Do().Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do().Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	if managed := accessor.GetManagedFields(); managed != nil {
		t.Fatalf("Object contains unexpected managedFields: %v", managed)
	}
}

func TestApplyRequiresFieldManager(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "v1",
		"kind": "ConfigMap",
		"metadata": {
			"name": "test-cm",
			"namespace": "default"
		}
	}`)

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Body(obj).
		Do().
		Get()
	if err == nil {
		t.Fatalf("Apply should fail to create without fieldManager")
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "apply_test").
		Body(obj).
		Do().
		Get()
	if err != nil {
		t.Fatalf("Apply failed to create with fieldManager: %v", err)
	}
}

// TestApplyRemoveContainerPort removes a container port from a deployment
func TestApplyRemoveContainerPort(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
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
						"image": "nginx:latest",
						"ports": [{
							"containerPort": 80,
							"protocol": "TCP"
						}]
					}]
				}
			}
		}
	}`)

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("fieldManager", "apply_test").
		Body(obj).Do().Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	obj = []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
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
	}`)

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("fieldManager", "apply_test").
		Body(obj).Do().Get()
	if err != nil {
		t.Fatalf("Failed to remove container port using Apply patch: %v", err)
	}

	deployment, err := client.AppsV1().Deployments("default").Get("deployment", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	if len(deployment.Spec.Template.Spec.Containers[0].Ports) > 0 {
		t.Fatalf("Expected no container ports but got: %v", deployment.Spec.Template.Spec.Containers[0].Ports)
	}
}
