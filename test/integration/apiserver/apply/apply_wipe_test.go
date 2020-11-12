/*
Copyright 2020 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
	"sigs.k8s.io/yaml"
)

// TestWipeManagedFieldsForPod is the very basic and specific test used while development
func TestWipeManagedFieldsForPod(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	testCases := []struct {
		resource string
		name     string
		body     string
	}{
		{
			resource: "pods",
			name:     "test-pod-with-status",
			body: `{
				"apiVersion": "v1",
				"kind": "Pod",
				"metadata": {
					"name": "test-pod-with-status"
				},
				"spec": {
					"containers": [{
						"name":  "test-container",
						"image": "test-image"
					}]
				},
				"status": {
					"phase": "testing"
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
			Do(context.TODO()).
			Get()
		if err != nil {
			t.Fatalf("Failed to create object using Apply patch: %v", err)
		}

		pod, err := client.CoreV1().Pods("default").Get(context.TODO(), tc.name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to retrieve object: %v", err)
		}

		if string(pod.Status.Phase) == "testing" {
			t.Fatalf("Pod should not have .status.phase 'testing'")
		}

		if pod.Status.Phase == "testing" {
			t.Fatalf("Pod status should have been reset")
		}

		actualFieldsV1 := pod.ObjectMeta.ManagedFields[0].FieldsV1.Raw
		expectedFieldsV1 := []byte(`{"f:spec":{"f:containers":{"k:{\"name\":\"test-container\"}":{".":{},"f:image":{},"f:name":{}}}}}`)

		if !reflect.DeepEqual(actualFieldsV1, expectedFieldsV1) {
			t.Fatalf("expected managedFields to be:\n%s\ngot:\n%s", string(expectedFieldsV1), string(actualFieldsV1))
		}
	}
}

// namespace used for all tests, do not change this
const resetFieldsNamespace = "reset-fields-namespace"

// TestApplyWipeManagedFields makes sure that fieldManager does not own fields reset by the storage strategy.
func TestApplyWipeManagedFields(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), []string{"--disable-admission-plugins", "ServiceAccount,TaintNodesByCondition"}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// create CRDs so we can make sure that custom resources do not get lost
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: resetFieldsNamespace}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	createData := etcd.GetEtcdStorageDataForNamespace(resetFieldsNamespace)

	// gather resources to test
	_, resourceLists, err := client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources with error: %+v", err)
	}

	for _, resourceList := range resourceLists {
		for _, resource := range resourceList.APIResources {
			if !strings.HasSuffix(resource.Name, "/status") {
				continue
			}
			mapping, err := createMapping(resourceList.GroupVersion, resource)
			if err != nil {
				t.Fatal(err)
			}
			t.Run(mapping.Resource.String(), func(t *testing.T) {
				if _, ok := ignoreList[mapping.Resource]; ok {
					t.Skip()
				}

				// TODO: fix this test for customresourcedefinitions
				// CRD status gets updated by other actors which makes those own the conditions field,
				// which causes this test to get a conflict.
				if mapping.Resource.Resource == "customresourcedefinitions" {
					t.Skip()
				}

				status, ok := statusData[mapping.Resource]
				if !ok {
					status = statusDefault
				}
				newResource, ok := createData[mapping.Resource]
				if !ok {
					t.Fatalf("no test data for %s.  Please add a test for your new type to etcd.GetEtcdStorageData().", mapping.Resource)
				}

				newObj := unstructured.Unstructured{}
				if err := json.Unmarshal([]byte(newResource.Stub), &newObj.Object); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal([]byte(status), &newObj.Object); err != nil {
					t.Fatal(err)
				}

				namespace := resetFieldsNamespace
				if mapping.Scope == meta.RESTScopeRoot {
					namespace = ""
				}
				name := newObj.GetName()
				rsc := dynamicClient.Resource(mapping.Resource).Namespace(namespace)
				_, err := rsc.Create(context.TODO(), &newObj, metav1.CreateOptions{FieldManager: "create_test"})
				if err != nil {
					t.Fatal(err)
				}

				statusObj := unstructured.Unstructured{}
				if err := json.Unmarshal([]byte(status), &statusObj.Object); err != nil {
					t.Fatal(err)
				}
				statusObj.SetAPIVersion(mapping.GroupVersionKind.GroupVersion().String())
				statusObj.SetKind(mapping.GroupVersionKind.Kind)
				statusObj.SetName(name)
				statusYAML, err := yaml.Marshal(statusObj.Object)
				if err != nil {
					t.Fatal(err)
				}

				// an unforced apply should not conflict if the resetfields are set correctly
				_, err = dynamicClient.
					Resource(mapping.Resource).
					Namespace(namespace).
					Patch(context.TODO(), name, types.ApplyPatchType, statusYAML, metav1.PatchOptions{FieldManager: "apply_status_test"}, "status")
				if err != nil {
					t.Fatalf("Failed to apply: %v", err)
				}

				if err := rsc.Delete(context.TODO(), name, *metav1.NewDeleteOptions(0)); err != nil {
					t.Fatalf("deleting final object failed: %v", err)
				}
			})
		}
	}
}
