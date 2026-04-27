/*
Copyright The Kubernetes Authors.

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

package apidefinitions

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestGenerationManagement tests that metadata.generation is managed when a resource is updated.
//
// The test ensures:
// 1. Generation initializes to 1.
// 2. Generation monotonically increases for each spec update.
// 3. Generation does not increase for status updates.
func TestGenerationManagement(t *testing.T) {

	// DO NOT ADD NEW ENTRIES HERE.
	// This tracks resources that have status but do not manage generation.
	generationExempt := sets.New(
		"apiservices.apiregistration.k8s.io",
		"certificatesigningrequests.certificates.k8s.io",
		"namespaces",
		"nodes",
		"persistentvolumeclaims",
		"persistentvolumes",
		"resourceclaims.resource.k8s.io",
		"resourcequotas",
		"servicecidrs.networking.k8s.io",
		"services",
		"volumeattachments.storage.k8s.io",
	)

	TestAllDefinitions(t, "generation-namespace", func(t *testing.T, api Definition) {
		if !api.HasStatus() {
			t.Skip()
		}
		if !api.HasVerb("patch") || !api.HasVerb("get") || !api.HasVerb("update") {
			t.Skip("Resource does not support patch, get, and update")
		}

		differentSpec := api.StorageData.MutatedStub
		if differentSpec == "" {
			t.Skipf("No conflicting spec data defined for %v to test generation bump", api.Mapping.Resource)
		}

		status := api.StorageData.GetStatusStub()
		obj := TestObj(t, api.StorageData.Stub, status, api.Mapping.GroupVersionKind)
		name := obj.GetName()
		rsc := api.ResourceClient()

		_, err := rsc.Create(context.TODO(), obj, metav1.CreateOptions{FieldManager: "spec-manager"})
		if err != nil {
			t.Fatalf("Failed to create via SSA: %v", err)
		}

		baseline, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get baseline: %v", err)
		}

		// Verify that generation initializes to 1.
		if matchesException(api.Mapping.Resource, generationExempt) {
			if baseline.GetGeneration() != 0 {
				t.Errorf("Expected generation exempt resource always have generation 0, but got %v", baseline.GetGeneration())
			}
		} else if baseline.GetGeneration() != int64(1) {
			t.Errorf("Expected generation to be %v on create, got %v", 1, baseline.GetGeneration())
		}

		// Verify that updating status does NOT bump generation.
		update := api.StorageData.GetMutatedStatusStub()
		statusObj := TestObj(t, api.StorageData.Stub, update, api.Mapping.GroupVersionKind)
		statusObj.SetName(name)

		_, err = rsc.ApplyStatus(context.TODO(), name, statusObj, metav1.ApplyOptions{FieldManager: "status-manager", Force: true})
		if err != nil {
			t.Fatalf("Failed to apply status via SSA: %v", err)
		}

		afterStatus, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get after status update: %v", err)
		}

		if afterStatus.GetGeneration() != baseline.GetGeneration() {
			t.Errorf("Expected generation to remain %v after status update, but got %v", baseline.GetGeneration(), afterStatus.GetGeneration())
		}

		// Verify that updating spec does bump generation.
		result, err := rsc.Patch(context.TODO(), name, types.MergePatchType, []byte(differentSpec), metav1.PatchOptions{})
		if err != nil {
			t.Logf("Patch to main endpoint failed: %v", err)
		} else if result.GetGeneration() <= afterStatus.GetGeneration() {
			if matchesException(api.Mapping.Resource, generationExempt) {
				if result.GetGeneration() != 0 {
					t.Errorf("Expected generation exempt resource always have generation 0, but got %v", result.GetGeneration())
				}
			} else {
				t.Errorf("Expected generation to monotonically increase after spec update (was %v, got %v)", afterStatus.GetGeneration(), result.GetGeneration())
			}
		}
	})
}
