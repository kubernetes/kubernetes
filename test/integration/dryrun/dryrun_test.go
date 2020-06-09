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

package dryrun

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

// Only add kinds to this list when this a virtual resource with get and create verbs that doesn't actually
// store into it's kind.  We've used this downstream for mappings before.
var kindWhiteList = sets.NewString()

// namespace used for all tests, do not change this
const testNamespace = "dryrunnamespace"

func DryRunCreateTest(t *testing.T, rsc dynamic.ResourceInterface, obj *unstructured.Unstructured, gvResource schema.GroupVersionResource) {
	createdObj, err := rsc.Create(context.TODO(), obj, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("failed to dry-run create stub for %s: %#v", gvResource, err)
	}
	if obj.GroupVersionKind() != createdObj.GroupVersionKind() {
		t.Fatalf("created object doesn't have the same gvk as original object: got %v, expected %v",
			createdObj.GroupVersionKind(),
			obj.GroupVersionKind())
	}

	if _, err := rsc.Get(context.TODO(), obj.GetName(), metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("object shouldn't exist: %v", err)
	}
}

func DryRunPatchTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	patch := []byte(`{"metadata":{"annotations":{"patch": "true"}}}`)
	obj, err := rsc.Patch(context.TODO(), name, types.MergePatchType, patch, metav1.PatchOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("failed to dry-run patch object: %v", err)
	}
	if v := obj.GetAnnotations()["patch"]; v != "true" {
		t.Fatalf("dry-run patched annotations should be returned, got: %v", obj.GetAnnotations())
	}
	obj, err = rsc.Get(context.TODO(), obj.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	if v := obj.GetAnnotations()["patch"]; v == "true" {
		t.Fatalf("dry-run patched annotations should not be persisted, got: %v", obj.GetAnnotations())
	}
}

func getReplicasOrFail(t *testing.T, obj *unstructured.Unstructured) int64 {
	t.Helper()
	replicas, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "spec", "replicas")
	if err != nil {
		t.Fatalf("failed to get int64 for replicas: %v", err)
	}
	if !found {
		t.Fatal("object doesn't have spec.replicas")
	}
	return replicas
}

func DryRunScalePatchTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	obj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{}, "scale")
	if apierrors.IsNotFound(err) {
		return
	}
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}

	replicas := getReplicasOrFail(t, obj)
	patch := []byte(`{"spec":{"replicas":10}}`)
	patchedObj, err := rsc.Patch(context.TODO(), name, types.MergePatchType, patch, metav1.PatchOptions{DryRun: []string{metav1.DryRunAll}}, "scale")
	if err != nil {
		t.Fatalf("failed to dry-run patch object: %v", err)
	}
	if newReplicas := getReplicasOrFail(t, patchedObj); newReplicas != 10 {
		t.Fatalf("dry-run patch to replicas didn't return new value: %v", newReplicas)
	}
	persistedObj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{}, "scale")
	if err != nil {
		t.Fatalf("failed to get scale sub-resource")
	}
	if newReplicas := getReplicasOrFail(t, persistedObj); newReplicas != replicas {
		t.Fatalf("number of replicas changed, expected %v, got %v", replicas, newReplicas)
	}
}

func DryRunScaleUpdateTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	obj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{}, "scale")
	if apierrors.IsNotFound(err) {
		return
	}
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}

	replicas := getReplicasOrFail(t, obj)
	if err := unstructured.SetNestedField(obj.Object, int64(10), "spec", "replicas"); err != nil {
		t.Fatalf("failed to set spec.replicas: %v", err)
	}
	updatedObj, err := rsc.Update(context.TODO(), obj, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}}, "scale")
	if err != nil {
		t.Fatalf("failed to dry-run update scale sub-resource: %v", err)
	}
	if newReplicas := getReplicasOrFail(t, updatedObj); newReplicas != 10 {
		t.Fatalf("dry-run update to replicas didn't return new value: %v", newReplicas)
	}
	persistedObj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{}, "scale")
	if err != nil {
		t.Fatalf("failed to get scale sub-resource")
	}
	if newReplicas := getReplicasOrFail(t, persistedObj); newReplicas != replicas {
		t.Fatalf("number of replicas changed, expected %v, got %v", replicas, newReplicas)
	}
}

func DryRunUpdateTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	var err error
	var obj *unstructured.Unstructured
	for i := 0; i < 3; i++ {
		obj, err = rsc.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to retrieve object: %v", err)
		}
		obj.SetAnnotations(map[string]string{"update": "true"})
		obj, err = rsc.Update(context.TODO(), obj, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
		if err == nil || !apierrors.IsConflict(err) {
			break
		}
	}
	if err != nil {
		t.Fatalf("failed to dry-run update resource: %v", err)
	}
	if v := obj.GetAnnotations()["update"]; v != "true" {
		t.Fatalf("dry-run updated annotations should be returned, got: %v", obj.GetAnnotations())
	}

	obj, err = rsc.Get(context.TODO(), obj.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	if v := obj.GetAnnotations()["update"]; v == "true" {
		t.Fatalf("dry-run updated annotations should not be persisted, got: %v", obj.GetAnnotations())
	}
}

func DryRunDeleteCollectionTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	err := rsc.DeleteCollection(context.TODO(), metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}}, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("dry-run delete collection failed: %v", err)
	}
	obj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	ts := obj.GetDeletionTimestamp()
	if ts != nil {
		t.Fatalf("object has a deletion timestamp after dry-run delete collection")
	}
}

func DryRunDeleteTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	err := rsc.Delete(context.TODO(), name, metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("dry-run delete failed: %v", err)
	}
	obj, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	ts := obj.GetDeletionTimestamp()
	if ts != nil {
		t.Fatalf("object has a deletion timestamp after dry-run delete")
	}
}

// TestDryRun tests dry-run on all types.
func TestDryRun(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DryRun, true)()

	// start API server
	s, err := kubeapiservertesting.StartTestServer(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount,StorageObjectInUseProtection",
		"--runtime-config=api/all=true",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer s.TearDownFn()

	client, err := kubernetes.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(s.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// create CRDs so we can make sure that custom resources do not get lost
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(s.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	dryrunData := etcd.GetEtcdStorageData()

	// dry run specific stub overrides
	for resource, stub := range map[schema.GroupVersionResource]string{
		// need to change event's namespace field to match dry run test
		gvr("", "v1", "events"): `{"involvedObject": {"namespace": "dryrunnamespace"}, "message": "some data here", "metadata": {"name": "event1"}}`,
	} {
		data := dryrunData[resource]
		data.Stub = stub
		dryrunData[resource] = data
	}

	// gather resources to test
	_, resources, err := client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources with error: %+v", err)
	}

	for _, resourceToTest := range etcd.GetResources(t, resources) {
		t.Run(resourceToTest.Mapping.Resource.String(), func(t *testing.T) {
			mapping := resourceToTest.Mapping
			gvk := resourceToTest.Mapping.GroupVersionKind
			gvResource := resourceToTest.Mapping.Resource
			kind := gvk.Kind

			if kindWhiteList.Has(kind) {
				t.Skip("whitelisted")
			}

			testData, hasTest := dryrunData[gvResource]

			if !hasTest {
				t.Fatalf("no test data for %s.  Please add a test for your new type to etcd.GetEtcdStorageData().", gvResource)
			}

			rsc, obj, err := etcd.JSONToUnstructured(testData.Stub, testNamespace, mapping, dynamicClient)
			if err != nil {
				t.Fatalf("failed to unmarshal stub (%v): %v", testData.Stub, err)
			}

			name := obj.GetName()

			DryRunCreateTest(t, rsc, obj, gvResource)

			if _, err := rsc.Create(context.TODO(), obj, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create stub for %s: %#v", gvResource, err)
			}

			DryRunUpdateTest(t, rsc, name)
			DryRunPatchTest(t, rsc, name)
			DryRunScalePatchTest(t, rsc, name)
			DryRunScaleUpdateTest(t, rsc, name)
			if resourceToTest.HasDeleteCollection {
				DryRunDeleteCollectionTest(t, rsc, name)
			}
			DryRunDeleteTest(t, rsc, name)

			if err = rsc.Delete(context.TODO(), obj.GetName(), *metav1.NewDeleteOptions(0)); err != nil {
				t.Fatalf("deleting final object failed: %v", err)
			}
		})
	}
}

func gvr(g, v, r string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
}
