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
	"net/http/httptest"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
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

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("pods").
		Name("test-pod").
		Body([]byte(`{
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
		}`)).
		Do().
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Get().Namespace("default").Resource("pods").Name("test-pod").Do().Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
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
		Body([]byte(obj)).Do().Get()
	if err != nil {
		t.Fatalf("Failed to apply object with force: %v", err)
	}
}
