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
	"testing"

	"k8s.io/kubernetes/test/integration/apiserver/apidefinitions"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// namespace used for all tests, do not change this
const testNamespace = "statusnamespace"

// TestApplyStatus makes sure that applying the status works for all known types.
func TestApplyStatus(t *testing.T) {
	testApplyStatus(t, func(testing.TB, *rest.Config) {})
}

// TestApplyStatus makes sure that applying the status works for all known types.
func TestApplyStatusWithCBOR(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsPreferCBOR, true)
	testApplyStatus(t, func(t testing.TB, config *rest.Config) {
		config.Wrap(framework.AssertRequestResponseAsCBOR(t))
	})
}

func testApplyStatus(t *testing.T, reconfigureClient func(testing.TB, *rest.Config)) {
	apidefinitions.TestAllDefinitions(t, testNamespace, func(t *testing.T, api apidefinitions.Definition) {
		if !api.HasStatus() {
			t.Skip()
		}
		if !api.HasVerb("patch") || !api.HasVerb("get") || !api.HasVerb("update") {
			t.Skip("Resource does not support patch, get, and update")
		}
		// both spec and status get wiped for CSRs,
		// nothing is expected to be managed for it, skip it
		if api.Mapping.Resource.Resource == "certificatesigningrequests" {
			t.Skip()
		}

		status := api.StorageData.GetStatusStub()
		newObj := unstructured.Unstructured{}
		if err := json.Unmarshal([]byte(api.StorageData.Stub), &newObj.Object); err != nil {
			t.Fatal(err)
		}

		namespace := api.Namespace
		if api.Mapping.Scope == meta.RESTScopeRoot {
			namespace = ""
		}
		name := newObj.GetName()

		// etcd test stub data doesn't contain apiVersion/kind (!), but apply requires it
		newObj.SetGroupVersionKind(api.Mapping.GroupVersionKind)

		dynamicClientConfig := rest.CopyConfig(api.Config)
		reconfigureClient(t, dynamicClientConfig)
		dynamicClient, err := dynamic.NewForConfig(dynamicClientConfig)
		if err != nil {
			t.Fatal(err)
		}

		rsc := dynamicClient.Resource(api.Mapping.Resource).Namespace(namespace)
		// apply to create
		_, err = rsc.Apply(context.TODO(), name, &newObj, metav1.ApplyOptions{FieldManager: "create_test"})
		if err != nil {
			t.Fatal(err)
		}

		statusObj := unstructured.Unstructured{}
		if err := json.Unmarshal([]byte(status), &statusObj.Object); err != nil {
			t.Fatal(err)
		}
		statusObj.SetAPIVersion(api.Mapping.GroupVersionKind.GroupVersion().String())
		statusObj.SetKind(api.Mapping.GroupVersionKind.Kind)
		statusObj.SetName(name)

		obj, err := dynamicClient.
			Resource(api.Mapping.Resource).
			Namespace(namespace).
			ApplyStatus(context.TODO(), name, &statusObj, metav1.ApplyOptions{FieldManager: "apply_status_test", Force: true})
		if err != nil {
			t.Fatalf("Failed to apply: %v", err)
		}

		accessor, err := meta.Accessor(obj)
		if err != nil {
			t.Fatalf("Failed to get meta accessor: %v:\n%v", err, obj)
		}

		managedFields := accessor.GetManagedFields()
		if managedFields == nil {
			t.Fatal("Empty managed fields")
		}
		if !findManager(managedFields, "apply_status_test") {
			t.Fatalf("Couldn't find apply_status_test: %v", managedFields)
		}
		if !findManager(managedFields, "create_test") {
			t.Fatalf("Couldn't find create_test: %v", managedFields)
		}

		if err := rsc.Delete(context.TODO(), name, *metav1.NewDeleteOptions(0)); err != nil {
			t.Fatalf("deleting final object failed: %v", err)
		}
	})
}

func findManager(managedFields []metav1.ManagedFieldsEntry, manager string) bool {
	for _, entry := range managedFields {
		if entry.Manager == manager {
			return true
		}
	}
	return false
}
