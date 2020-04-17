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
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

func setup(t testing.TB, groupVersions ...schema.GroupVersion) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	opts := framework.MasterConfigOptions{EtcdOptions: framework.DefaultEtcdOptions()}
	opts.EtcdOptions.DefaultStorageMediaType = "application/vnd.kubernetes.protobuf"
	masterConfig := framework.NewIntegrationTestMasterConfigWithOptions(&opts)
	if len(groupVersions) > 0 {
		resourceConfig := master.DefaultAPIResourceConfigSource()
		resourceConfig.EnableVersions(groupVersions...)
		masterConfig.ExtraConfig.APIResourceConfigSource = resourceConfig
	}
	masterConfig.GenericConfig.OpenAPIConfig = framework.DefaultOpenAPIConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL, QPS: -1})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	return s, clientSet, closeFn
}

// TestApplyAlsoCreates makes sure that PATCH requests with the apply content type
// will create the object if it doesn't already exist
// TODO: make a set of test cases in an easy-to-consume place (separate package?) so it's easy to test in both integration and e2e.
func TestApplyAlsoCreates(t *testing.T) {
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
			Do(context.TODO()).
			Get()
		if err != nil {
			t.Fatalf("Failed to create object using Apply patch: %v", err)
		}

		_, err = client.CoreV1().RESTClient().Get().Namespace("default").Resource(tc.resource).Name(tc.name).Do(context.TODO()).Get()
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
			Do(context.TODO()).
			Get()
		if err != nil {
			t.Fatalf("Failed to re-apply object using Apply patch: %v", err)
		}
	}
}

// TestNoOpUpdateSameResourceVersion makes sure that PUT requests which change nothing
// will not change the resource version (no write to etcd is done)
func TestNoOpUpdateSameResourceVersion(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	podName := "no-op"
	podResource := "pods"
	podBytes := []byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "` + podName + `",
			"labels": {
				"a": "one",
				"c": "two",
				"b": "three"
			}
		},
		"spec": {
			"containers": [{
				"name":  "test-container-a",
				"image": "test-image-one"
			},{
				"name":  "test-container-c",
				"image": "test-image-two"
			},{
				"name":  "test-container-b",
				"image": "test-image-three"
			}]
		}
	}`)

	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Param("fieldManager", "apply_test").
		Resource(podResource).
		Name(podName).
		Body(podBytes).
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object: %v", err)
	}

	// Sleep for one second to make sure that the times of each update operation is different.
	time.Sleep(1 * time.Second)

	createdObject, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource(podResource).Name(podName).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve created object: %v", err)
	}

	createdAccessor, err := meta.Accessor(createdObject)
	if err != nil {
		t.Fatalf("Failed to get meta accessor for created object: %v", err)
	}

	createdBytes, err := json.MarshalIndent(createdObject, "\t", "\t")
	if err != nil {
		t.Fatalf("Failed to marshal created object: %v", err)
	}

	// Test that we can put the same object and don't change the RV
	_, err = client.CoreV1().RESTClient().Put().
		Namespace("default").
		Resource(podResource).
		Name(podName).
		Body(createdBytes).
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to apply no-op update: %v", err)
	}

	updatedObject, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource(podResource).Name(podName).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve updated object: %v", err)
	}

	updatedAccessor, err := meta.Accessor(updatedObject)
	if err != nil {
		t.Fatalf("Failed to get meta accessor for updated object: %v", err)
	}

	updatedBytes, err := json.MarshalIndent(updatedObject, "\t", "\t")
	if err != nil {
		t.Fatalf("Failed to marshal updated object: %v", err)
	}

	if createdAccessor.GetResourceVersion() != updatedAccessor.GetResourceVersion() {
		t.Fatalf("Expected same resource version to be %v but got: %v\nold object:\n%v\nnew object:\n%v",
			createdAccessor.GetResourceVersion(),
			updatedAccessor.GetResourceVersion(),
			string(createdBytes),
			string(updatedBytes),
		)
	}
}

// TestCreateOnApplyFailsWithUID makes sure that PATCH requests with the apply content type
// will not create the object if it doesn't already exist and it specifies a UID
func TestCreateOnApplyFailsWithUID(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if !apierrors.IsConflict(err) {
		t.Fatalf("Expected conflict error but got: %v", err)
	}
}

func TestApplyUpdateApplyConflictForced(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Body([]byte(`{"spec":{"replicas": 5}}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Param("fieldManager", "apply_test").
		Body([]byte(obj)).Do(context.TODO()).Get()
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object")
	}
	status, ok := err.(*apierrors.StatusError)
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
		Body([]byte(obj)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to apply object with force: %v", err)
	}
}

// TestApplyGroupsManySeparateUpdates tests that when many different managers update the same object,
// the number of managedFields entries will only grow to a certain size.
func TestApplyGroupsManySeparateUpdates(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "admissionregistration.k8s.io/v1",
		"kind": "ValidatingWebhookConfiguration",
		"metadata": {
			"name": "webhook",
			"labels": {"applier":"true"},
		},
	}`)

	object, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		AbsPath("/apis/admissionregistration.k8s.io/v1").
		Resource("validatingwebhookconfigurations").
		Name("webhook").
		Param("fieldManager", "apply_test").
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	for i := 0; i < 20; i++ {
		unique := fmt.Sprintf("updater%v", i)
		version := "v1"
		if i%2 == 0 {
			version = "v1beta1"
		}
		object, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
			AbsPath("/apis/admissionregistration.k8s.io/"+version).
			Resource("validatingwebhookconfigurations").
			Name("webhook").
			Param("fieldManager", unique).
			Body([]byte(`{"metadata":{"labels":{"` + unique + `":"new"}}}`)).Do(context.TODO()).Get()
		if err != nil {
			t.Fatalf("Failed to patch object: %v", err)
		}
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	// Expect 11 entries, because the cap for update entries is 10, and 1 apply entry
	if actual, expected := len(accessor.GetManagedFields()), 11; actual != expected {
		if b, err := json.MarshalIndent(object, "\t", "\t"); err == nil {
			t.Fatalf("Object expected to contain %v entries in managedFields, but got %v:\n%v", expected, actual, string(b))
		} else {
			t.Fatalf("Object expected to contain %v entries in managedFields, but got %v: error marshalling object: %v", expected, actual, err)
		}
	}

	// Expect the first entry to have the manager name "apply_test"
	if actual, expected := accessor.GetManagedFields()[0].Manager, "apply_test"; actual != expected {
		t.Fatalf("Expected first manager to be named %v but got %v", expected, actual)
	}

	// Expect the second entry to have the manager name "ancient-changes"
	if actual, expected := accessor.GetManagedFields()[1].Manager, "ancient-changes"; actual != expected {
		t.Fatalf("Expected first manager to be named %v but got %v", expected, actual)
	}
}

// TestCreateVeryLargeObject tests that a very large object can be created without exceeding the size limit due to managedFields
func TestCreateVeryLargeObject(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	cfg := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "large-create-test-cm",
			Namespace: "default",
		},
		Data: map[string]string{},
	}

	for i := 0; i < 9999; i++ {
		unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
		cfg.Data[unique] = "A"
	}

	// Should be able to create an object near the object size limit.
	if _, err := client.CoreV1().ConfigMaps(cfg.Namespace).Create(context.TODO(), cfg, metav1.CreateOptions{}); err != nil {
		t.Errorf("unable to create large test configMap: %v", err)
	}

	// Applying to the same object should cause managedFields to go over the object size limit, and fail.
	_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace(cfg.Namespace).
		Resource("configmaps").
		Name(cfg.Name).
		Param("fieldManager", "apply_test").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "large-create-test-cm",
				"namespace": "default",
			}
		}`)).
		Do(context.TODO()).
		Get()
	if err == nil {
		t.Fatalf("expected to fail to update object using Apply patch, but succeeded")
	}
}

// TestUpdateVeryLargeObject tests that a small object can be updated to be very large without exceeding the size limit due to managedFields
func TestUpdateVeryLargeObject(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	cfg := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "large-update-test-cm",
			Namespace: "default",
		},
		Data: map[string]string{"k": "v"},
	}

	// Create a small config map.
	cfg, err := client.CoreV1().ConfigMaps(cfg.Namespace).Create(context.TODO(), cfg, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("unable to create configMap: %v", err)
	}

	// Should be able to update a small object to be near the object size limit.
	var updateErr error
	pollErr := wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		updateCfg, err := client.CoreV1().ConfigMaps(cfg.Namespace).Get(context.TODO(), cfg.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// Apply the large update, then attempt to push it to the apiserver.
		for i := 0; i < 9999; i++ {
			unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
			updateCfg.Data[unique] = "A"
		}

		if _, err = client.CoreV1().ConfigMaps(cfg.Namespace).Update(context.TODO(), updateCfg, metav1.UpdateOptions{}); err == nil {
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		t.Errorf("unable to update configMap: %v", updateErr)
	}

	// Applying to the same object should cause managedFields to go over the object size limit, and fail.
	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace(cfg.Namespace).
		Resource("configmaps").
		Name(cfg.Name).
		Param("fieldManager", "apply_test").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "large-update-test-cm",
				"namespace": "default",
			}
		}`)).
		Do(context.TODO()).
		Get()
	if err == nil {
		t.Fatalf("expected to fail to update object using Apply patch, but succeeded")
	}
}

// TestPatchVeryLargeObject tests that a small object can be patched to be very large without exceeding the size limit due to managedFields
func TestPatchVeryLargeObject(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	cfg := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "large-patch-test-cm",
			Namespace: "default",
		},
		Data: map[string]string{"k": "v"},
	}

	// Create a small config map.
	if _, err := client.CoreV1().ConfigMaps(cfg.Namespace).Create(context.TODO(), cfg, metav1.CreateOptions{}); err != nil {
		t.Errorf("unable to create configMap: %v", err)
	}

	patchString := `{"data":{"k":"v"`
	for i := 0; i < 9999; i++ {
		unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
		patchString = fmt.Sprintf("%s,%q:%q", patchString, unique, "A")
	}
	patchString = fmt.Sprintf("%s}}", patchString)

	// Should be able to update a small object to be near the object size limit.
	_, err := client.CoreV1().RESTClient().Patch(types.MergePatchType).
		AbsPath("/api/v1").
		Namespace(cfg.Namespace).
		Resource("configmaps").
		Name(cfg.Name).
		Body([]byte(patchString)).Do(context.TODO()).Get()
	if err != nil {
		t.Errorf("unable to patch configMap: %v", err)
	}

	// Applying to the same object should cause managedFields to go over the object size limit, and fail.
	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("large-patch-test-cm").
		Param("fieldManager", "apply_test").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "large-patch-test-cm",
				"namespace": "default",
			}
		}`)).
		Do(context.TODO()).
		Get()
	if err == nil {
		t.Fatalf("expected to fail to update object using Apply patch, but succeeded")
	}
}

// TestApplyManagedFields makes sure that managedFields api does not change
func TestApplyManagedFields(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "updater").
		Body([]byte(`{"data":{"new-key": "value"}}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	// Sleep for one second to make sure that the times of each update operation is different.
	// This will let us check that update entries with the same manager name are grouped together,
	// and that the most recent update time is recorded in the grouped entry.
	time.Sleep(1 * time.Second)

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "updater").
		Body([]byte(`{"data":{"key": "new value"}}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
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
					"time": "` + accessor.GetManagedFields()[0].Time.UTC().Format(time.RFC3339) + `",
					"fieldsType": "FieldsV1",
					"fieldsV1": {
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
					"fieldsType": "FieldsV1",
					"fieldsV1": {
						"f:data": {
							"f:key": {},
							"f:new-key": {}
						}
					}
				}
			]
		},
		"data": {
			"key": "new value",
			"new-key": "value"
		}
	}`)

	if string(expected) != string(actual) {
		t.Fatalf("Expected:\n%v\nGot:\n%v", string(expected), string(actual))
	}

	if accessor.GetManagedFields()[0].Time.UTC().Format(time.RFC3339) == accessor.GetManagedFields()[1].Time.UTC().Format(time.RFC3339) {
		t.Fatalf("Expected times to be different but got:\n%v", string(actual))
	}
}

// TestApplyRemovesEmptyManagedFields there are no empty managers in managedFields
func TestApplyRemovesEmptyManagedFields(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Param("fieldManager", "apply_test").
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Apply failed to create with fieldManager: %v", err)
	}
}

// TestApplyRemoveContainerPort removes a container port from a deployment
func TestApplyRemoveContainerPort(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Body(obj).Do(context.TODO()).Get()
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
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to remove container port using Apply patch: %v", err)
	}

	deployment, err := client.AppsV1().Deployments("default").Get(context.TODO(), "deployment", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	if len(deployment.Spec.Template.Spec.Containers[0].Ports) > 0 {
		t.Fatalf("Expected no container ports but got: %v, object: \n%#v", deployment.Spec.Template.Spec.Containers[0].Ports, deployment)
	}
}

// TestApplyFailsWithVersionMismatch ensures that a version mismatch between the
// patch object and the live object will error
func TestApplyFailsWithVersionMismatch(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	obj = []byte(`{
		"apiVersion": "extensions/v1beta",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 100,
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
		Body([]byte(obj)).Do(context.TODO()).Get()
	if err == nil {
		t.Fatalf("Expecting to get version mismatch when applying object")
	}
	status, ok := err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get version mismatch as API error")
	}
	if status.Status().Code != http.StatusBadRequest {
		t.Fatalf("expected status code to be %d but was %d", http.StatusBadRequest, status.Status().Code)
	}
}

// TestApplyConvertsManagedFieldsVersion checks that the apply
// converts the API group-version in the field manager
func TestApplyConvertsManagedFieldsVersion(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"},
			"managedFields": [
				{
					"manager": "sidecar_controller",
					"operation": "Apply",
					"apiVersion": "extensions/v1beta1",
					"fieldsV1": {
						"f:metadata": {
							"f:labels": {
								"f:sidecar_version": {}
							}
						},
						"f:spec": {
							"f:template": {
								"f: spec": {
									"f:containers": {
										"k:{\"name\":\"sidecar\"}": {
											".": {},
											"f:image": {}
										}
									}
								}
							}
						}
					}
				}
			]
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
	}`)

	_, err := client.CoreV1().RESTClient().Post().
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to create object: %v", err)
	}

	obj = []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"sidecar_version": "release"}
		},
		"spec": {
			"template": {
				"spec": {
					"containers": [{
						"name":  "sidecar",
						"image": "sidecar:latest"
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
		Param("fieldManager", "sidecar_controller").
		Body([]byte(obj)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to apply object: %v", err)
	}

	object, err := client.AppsV1().Deployments("default").Get(context.TODO(), "deployment", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	managed := accessor.GetManagedFields()
	if len(managed) != 2 {
		t.Fatalf("Expected 2 field managers, but got managed fields: %v", managed)
	}

	var actual *metav1.ManagedFieldsEntry
	for i := range managed {
		entry := &managed[i]
		if entry.Manager == "sidecar_controller" && entry.APIVersion == "apps/v1" {
			actual = entry
		}
	}

	if actual == nil {
		t.Fatalf("Expected managed fields to contain entry with manager '%v' with converted api version '%v', but got managed fields:\n%v", "sidecar_controller", "apps/v1", managed)
	}

	expected := &metav1.ManagedFieldsEntry{
		Manager:    "sidecar_controller",
		Operation:  metav1.ManagedFieldsOperationApply,
		APIVersion: "apps/v1",
		Time:       actual.Time,
		FieldsType: "FieldsV1",
		FieldsV1: &metav1.FieldsV1{
			Raw: []byte(`{"f:metadata":{"f:labels":{"f:sidecar_version":{}}},"f:spec":{"f:template":{"f:spec":{"f:containers":{"k:{\"name\":\"sidecar\"}":{".":{},"f:image":{},"f:name":{}}}}}}}`),
		},
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("expected:\n%v\nbut got:\n%v", expected, actual)
	}
}

// TestClearManagedFieldsWithMergePatch verifies it's possible to clear the managedFields
func TestClearManagedFieldsWithMergePatch(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Body([]byte(`{"metadata":{"managedFields": [{}]}}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	if managedFields := accessor.GetManagedFields(); len(managedFields) != 0 {
		t.Fatalf("Failed to clear managedFields, got: %v", managedFields)
	}
}

// TestClearManagedFieldsWithStrategicMergePatch verifies it's possible to clear the managedFields
func TestClearManagedFieldsWithStrategicMergePatch(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.StrategicMergePatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Body([]byte(`{"metadata":{"managedFields": [{}]}}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	if managedFields := accessor.GetManagedFields(); len(managedFields) != 0 {
		t.Fatalf("Failed to clear managedFields, got: %v", managedFields)
	}

	if labels := accessor.GetLabels(); len(labels) < 1 {
		t.Fatalf("Expected other fields to stay untouched, got: %v", object)
	}
}

// TestClearManagedFieldsWithJSONPatch verifies it's possible to clear the managedFields
func TestClearManagedFieldsWithJSONPatch(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Patch(types.JSONPatchType).
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Body([]byte(`[{"op": "replace", "path": "/metadata/managedFields", "value": [{}]}]`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	if managedFields := accessor.GetManagedFields(); len(managedFields) != 0 {
		t.Fatalf("Failed to clear managedFields, got: %v", managedFields)
	}
}

// TestClearManagedFieldsWithUpdate verifies it's possible to clear the managedFields
func TestClearManagedFieldsWithUpdate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

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
		Do(context.TODO()).
		Get()
	if err != nil {
		t.Fatalf("Failed to create object using Apply patch: %v", err)
	}

	_, err = client.CoreV1().RESTClient().Put().
		Namespace("default").
		Resource("configmaps").
		Name("test-cm").
		Body([]byte(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "test-cm",
				"namespace": "default",
				"managedFields": [{}],
				"labels": {
					"test-label": "test"
				}
			},
			"data": {
				"key": "value"
			}
		}`)).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to patch object: %v", err)
	}

	object, err := client.CoreV1().RESTClient().Get().Namespace("default").Resource("configmaps").Name("test-cm").Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to retrieve object: %v", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		t.Fatalf("Failed to get meta accessor: %v", err)
	}

	if managedFields := accessor.GetManagedFields(); len(managedFields) != 0 {
		t.Fatalf("Failed to clear managedFields, got: %v", managedFields)
	}

	if labels := accessor.GetLabels(); len(labels) < 1 {
		t.Fatalf("Expected other fields to stay untouched, got: %v", object)
	}
}

var podBytes = []byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: some-app
    plugin1: some-value
    plugin2: some-value
    plugin3: some-value
    plugin4: some-value
  name: some-name
  namespace: default
  ownerReferences:
  - apiVersion: apps/v1
    blockOwnerDeletion: true
    controller: true
    kind: ReplicaSet
    name: some-name
    uid: 0a9d2b9e-779e-11e7-b422-42010a8001be
spec:
  containers:
  - args:
    - one
    - two
    - three
    - four
    - five
    - six
    - seven
    - eight
    - nine
    env:
    - name: VAR_3
      valueFrom:
        secretKeyRef:
          key: some-other-key
          name: some-oher-name
    - name: VAR_2
      valueFrom:
        secretKeyRef:
          key: other-key
          name: other-name
    - name: VAR_1
      valueFrom:
        secretKeyRef:
          key: some-key
          name: some-name
    image: some-image-name
    imagePullPolicy: IfNotPresent
    name: some-name
    resources:
      requests:
        cpu: "0"
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
    - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      name: default-token-hu5jz
      readOnly: true
  dnsPolicy: ClusterFirst
  nodeName: node-name
  priority: 0
  restartPolicy: Always
  schedulerName: default-scheduler
  securityContext: {}
  serviceAccount: default
  serviceAccountName: default
  terminationGracePeriodSeconds: 30
  tolerations:
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
  volumes:
  - name: default-token-hu5jz
    secret:
      defaultMode: 420
      secretName: default-token-hu5jz
status:
  conditions:
  - lastProbeTime: null
    lastTransitionTime: "2019-07-08T09:31:18Z"
    status: "True"
    type: Initialized
  - lastProbeTime: null
    lastTransitionTime: "2019-07-08T09:41:59Z"
    status: "True"
    type: Ready
  - lastProbeTime: null
    lastTransitionTime: null
    status: "True"
    type: ContainersReady
  - lastProbeTime: null
    lastTransitionTime: "2019-07-08T09:31:18Z"
    status: "True"
    type: PodScheduled
  containerStatuses:
  - containerID: docker://885e82a1ed0b7356541bb410a0126921ac42439607c09875cd8097dd5d7b5376
    image: some-image-name
    imageID: docker-pullable://some-image-id
    lastState:
      terminated:
        containerID: docker://d57290f9e00fad626b20d2dd87a3cf69bbc22edae07985374f86a8b2b4e39565
        exitCode: 255
        finishedAt: "2019-07-08T09:39:09Z"
        reason: Error
        startedAt: "2019-07-08T09:38:54Z"
    name: name
    ready: true
    restartCount: 6
    state:
      running:
        startedAt: "2019-07-08T09:41:59Z"
  hostIP: 10.0.0.1
  phase: Running
  podIP: 10.0.0.1
  qosClass: BestEffort
  startTime: "2019-07-08T09:31:18Z"
`)

func decodePod(podBytes []byte) v1.Pod {
	pod := v1.Pod{}
	err := yaml.Unmarshal(podBytes, &pod)
	if err != nil {
		panic(err)
	}
	return pod
}

func encodePod(pod v1.Pod) []byte {
	podBytes, err := yaml.Marshal(pod)
	if err != nil {
		panic(err)
	}
	return podBytes
}

func BenchmarkNoServerSideApply(b *testing.B) {
	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, false)()

	_, client, closeFn := setup(b)
	defer closeFn()
	flag.Lookup("v").Value.Set("0")

	benchAll(b, client, decodePod(podBytes))
}

func getPodSizeWhenEnabled(b *testing.B, pod v1.Pod) int {
	return len(getPodBytesWhenEnabled(b, pod, "application/vnd.kubernetes.protobuf"))
}

func getPodBytesWhenEnabled(b *testing.B, pod v1.Pod, format string) []byte {
	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()
	_, client, closeFn := setup(b)
	defer closeFn()
	flag.Lookup("v").Value.Set("0")

	pod.Name = "size-pod"
	podB, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
		Name(pod.Name).
		Namespace("default").
		Param("fieldManager", "apply_test").
		Resource("pods").
		SetHeader("Accept", format).
		Body(encodePod(pod)).DoRaw(context.TODO())
	if err != nil {
		b.Fatalf("Failed to create object: %#v", err)
	}
	return podB
}

func BenchmarkNoServerSideApplyButSameSize(b *testing.B) {
	pod := decodePod(podBytes)

	ssaPodSize := getPodSizeWhenEnabled(b, pod)

	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, false)()
	_, client, closeFn := setup(b)
	defer closeFn()
	flag.Lookup("v").Value.Set("0")

	pod.Name = "size-pod"
	noSSAPod, err := client.CoreV1().RESTClient().Post().
		Namespace("default").
		Resource("pods").
		SetHeader("Content-Type", "application/yaml").
		SetHeader("Accept", "application/vnd.kubernetes.protobuf").
		Body(encodePod(pod)).DoRaw(context.TODO())
	if err != nil {
		b.Fatalf("Failed to create object: %v", err)
	}

	ssaDiff := ssaPodSize - len(noSSAPod)
	fmt.Printf("Without SSA: %v bytes, With SSA: %v bytes, Difference: %v bytes\n", len(noSSAPod), ssaPodSize, ssaDiff)
	annotations := pod.GetAnnotations()
	builder := strings.Builder{}
	for i := 0; i < ssaDiff; i++ {
		builder.WriteByte('0')
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations["x-ssa-difference"] = builder.String()
	pod.SetAnnotations(annotations)

	benchAll(b, client, pod)
}

func BenchmarkServerSideApply(b *testing.B) {
	podBytesWhenEnabled := getPodBytesWhenEnabled(b, decodePod(podBytes), "application/yaml")

	defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()

	_, client, closeFn := setup(b)
	defer closeFn()
	flag.Lookup("v").Value.Set("0")

	benchAll(b, client, decodePod(podBytesWhenEnabled))
}

func benchAll(b *testing.B, client kubernetes.Interface, pod v1.Pod) {
	// Make sure pod is ready to post
	pod.ObjectMeta.CreationTimestamp = metav1.Time{}
	pod.ObjectMeta.ResourceVersion = ""
	pod.ObjectMeta.UID = ""
	pod.ObjectMeta.SelfLink = ""

	// Create pod for repeated-updates
	pod.Name = "repeated-pod"
	_, err := client.CoreV1().RESTClient().Post().
		Namespace("default").
		Resource("pods").
		SetHeader("Content-Type", "application/yaml").
		Body(encodePod(pod)).Do(context.TODO()).Get()
	if err != nil {
		b.Fatalf("Failed to create object: %v", err)
	}

	b.Run("List1", benchListPod(client, pod, 1))
	b.Run("List20", benchListPod(client, pod, 20))
	b.Run("List200", benchListPod(client, pod, 200))
	b.Run("List2000", benchListPod(client, pod, 2000))

	b.Run("RepeatedUpdates", benchRepeatedUpdate(client, "repeated-pod"))
	b.Run("Post1", benchPostPod(client, pod, 1))
	b.Run("Post10", benchPostPod(client, pod, 10))
	b.Run("Post50", benchPostPod(client, pod, 50))
}

func benchPostPod(client kubernetes.Interface, pod v1.Pod, parallel int) func(*testing.B) {
	return func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			c := make(chan error)
			for j := 0; j < parallel; j++ {
				j := j
				i := i
				go func(pod v1.Pod) {
					pod.Name = fmt.Sprintf("post%d-%d-%d-%d", parallel, b.N, j, i)
					_, err := client.CoreV1().RESTClient().Post().
						Namespace("default").
						Resource("pods").
						SetHeader("Content-Type", "application/yaml").
						Body(encodePod(pod)).Do(context.TODO()).Get()
					c <- err
				}(pod)
			}
			for j := 0; j < parallel; j++ {
				err := <-c
				if err != nil {
					b.Fatal(err)
				}
			}
			close(c)
		}
	}
}

func createNamespace(client kubernetes.Interface, name string) error {
	namespace := v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	namespaceBytes, err := yaml.Marshal(namespace)
	if err != nil {
		return fmt.Errorf("Failed to marshal namespace: %v", err)
	}
	_, err = client.CoreV1().RESTClient().Get().
		Resource("namespaces").
		SetHeader("Content-Type", "application/yaml").
		Body(namespaceBytes).Do(context.TODO()).Get()
	if err != nil {
		return fmt.Errorf("Failed to create namespace: %v", err)
	}
	return nil
}

func benchListPod(client kubernetes.Interface, pod v1.Pod, num int) func(*testing.B) {
	return func(b *testing.B) {
		namespace := fmt.Sprintf("get-%d-%d", num, b.N)
		if err := createNamespace(client, namespace); err != nil {
			b.Fatal(err)
		}
		// Create pods
		for i := 0; i < num; i++ {
			pod.Name = fmt.Sprintf("get-%d-%d", b.N, i)
			pod.Namespace = namespace
			_, err := client.CoreV1().RESTClient().Post().
				Namespace(namespace).
				Resource("pods").
				SetHeader("Content-Type", "application/yaml").
				Body(encodePod(pod)).Do(context.TODO()).Get()
			if err != nil {
				b.Fatalf("Failed to create object: %v", err)
			}
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := client.CoreV1().RESTClient().Get().
				Namespace(namespace).
				Resource("pods").
				SetHeader("Accept", "application/vnd.kubernetes.protobuf").
				Do(context.TODO()).Get()
			if err != nil {
				b.Fatalf("Failed to patch object: %v", err)
			}
		}
	}
}

func benchRepeatedUpdate(client kubernetes.Interface, podName string) func(*testing.B) {
	return func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := client.CoreV1().RESTClient().Patch(types.JSONPatchType).
				Namespace("default").
				Resource("pods").
				Name(podName).
				Body([]byte(fmt.Sprintf(`[{"op": "replace", "path": "/spec/containers/0/image", "value": "image%d"}]`, i))).Do(context.TODO()).Get()
			if err != nil {
				b.Fatalf("Failed to patch object: %v", err)
			}
		}
	}
}
