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
	"encoding/json"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestFieldsWipingConsistency(t *testing.T) {
	// Each allowlist tracks pre-existing APIs that do not implement the expected
	// field wiping behavior. DO NOT ADD NEW ENTRIES to any of these lists.

	// APIs where creating via the main endpoint does not wipe status.
	createDoesNotWipeStatus := sets.New(
		// Nodes allow all fields, including status, to be set on create.
		"nodes",
	)

	// APIs where updating via the /status endpoint does not wipe metadata.
	// All new APIs should use ResetObjectMetaForStatus.
	statusDoesNotWipeMetadata := sets.New(
		// https://github.com/kubernetes/kubernetes/issues/137681
		"customresourcedefinitions.apiextensions.k8s.io",

		// APIs that do not use ResetObjectMetaForStatus:
		"certificatesigningrequests.certificates.k8s.io",
		"cronjobs.batch",
		"daemonsets.apps",
		"horizontalpodautoscalers.autoscaling",
		"ingresses.networking.k8s.io",
		"jobs.batch",
		"namespaces",
		"nodes",
		"persistentvolumeclaims",
		"persistentvolumes",
		"poddisruptionbudgets.policy",
		"pods",
		"replicasets.apps",
		"replicationcontrollers",
		"resourcequotas",
		"services",
		"statefulsets.apps",
	)

	TestAllDefinitions(t, "reset-fields-test", func(t *testing.T, api Definition) {
		if !api.HasStatus() {
			t.Skip()
		}
		if !api.HasVerb("patch") || !api.HasVerb("get") || !api.HasVerb("update") {
			t.Skip("Resource does not support patch, get, and update")
		}

		status := api.StorageData.GetStatusStub()
		obj := TestObj(t, api.StorageData.Stub, status, api.Mapping.GroupVersionKind)
		name := obj.GetName()

		rsc := api.ResourceClient()

		// Create the resource and check if status is wiped on create.
		created, err := rsc.Apply(context.TODO(), name, obj, metav1.ApplyOptions{FieldManager: "spec-manager"})
		if err != nil {
			t.Fatalf("Failed to create via SSA: %v", err)
		}
		createWipesStatus := !checkPatch(t, status, "status", created.Object)

		// Apply to /status endpoint with mutated spec, status and metadata field changes.
		statusObj := TestObj(t, api.StorageData.Stub, status, api.Mapping.GroupVersionKind)
		if api.StorageData.MutatedStub != "" {
			if err := json.Unmarshal([]byte(api.StorageData.MutatedStub), &statusObj.Object); err != nil {
				t.Fatal(err)
			}
		}
		statusObj.SetName(name)
		statusLabels := statusObj.GetLabels()
		if statusLabels == nil {
			statusLabels = map[string]string{}
		}
		statusLabels["test-status-ssa"] = "true"
		statusObj.SetLabels(statusLabels)
		_, err = rsc.ApplyStatus(context.TODO(), name, statusObj, metav1.ApplyOptions{FieldManager: "status-manager", Force: true})
		if err != nil {
			t.Fatalf("Failed to apply status via SSA: %v", err)
		}

		// Read after writing to observe field wiping behavior and managedField state.
		baseline, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get baseline: %v", err)
		}
		baselineStatus := baseline.Object["status"]
		baselineSpec := baseline.Object["spec"]

		// Infer GetResetFields behavior from managedFields.
		ssaMainResetsStatus := true
		ssaStatusResetsSpec := true
		ssaStatusResetsMetadata := true
		for _, mf := range baseline.GetManagedFields() {
			if mf.Manager == "spec-manager" && mf.Subresource == "" {
				ssaMainResetsStatus = !managedFieldsOwnTopLevelField(t, mf.FieldsV1, "status")
			}
			if mf.Manager == "status-manager" && mf.Subresource == "status" {
				ssaStatusResetsSpec = !managedFieldsOwnTopLevelField(t, mf.FieldsV1, "spec")
				ssaStatusResetsMetadata = !managedFieldsOwnLabel(t, mf.FieldsV1, "test-status-ssa")
			}
		}

		// Check / PrepareForUpdate status wiping
		var mainWipesStatus bool
		if baselineStatus != nil {
			differentStatus := api.StorageData.GetMutatedStatusStub()
			result, err := rsc.Patch(context.TODO(), name, types.MergePatchType, []byte(differentStatus), metav1.PatchOptions{})
			if err != nil {
				t.Fatalf("Failed to patch main endpoint with different status: %v", err)
			}
			mainWipesStatus = !checkPatch(t, differentStatus, "status", result.Object)
		} else {
			mainWipesStatus = true
		}

		// Check /status PrepareForUpdate spec wiping
		var statusWipesSpec bool
		differentSpec := api.StorageData.MutatedStub
		if baselineSpec != nil && differentSpec != "" {
			result, err := rsc.Patch(context.TODO(), name, types.MergePatchType, []byte(differentSpec), metav1.PatchOptions{}, "status")
			if err != nil {
				statusWipesSpec = true
				t.Logf("Patch to status endpoint with different spec returned an error (OK if validation rejects it): %v", err)
			} else {
				statusWipesSpec = !checkPatch(t, differentSpec, "spec", result.Object)
			}
		} else {
			statusWipesSpec = true
		}

		// Check /status PrepareForUpdate metadata wiping
		var statusWipesMetadata bool
		labelPatch := []byte(`{"metadata": {"labels": {"test-wipe-label": "test-value"}}}`)
		result, err := rsc.Patch(context.TODO(), name, types.MergePatchType, labelPatch, metav1.PatchOptions{}, "status")
		if err != nil {
			t.Fatalf("Failed to patch status endpoint with different metadata labels: %v", err)
		}
		statusWipesMetadata = !checkPatch(t, string(labelPatch), "metadata", result.Object)

		if ssaMainResetsStatus != mainWipesStatus {
			t.Errorf("Main endpoint: SSA ResetStatus (%v) and PrepareForUpdate wipeStatus (%v) behaviors do not match.", ssaMainResetsStatus, mainWipesStatus)
		}
		if ssaStatusResetsSpec != statusWipesSpec {
			t.Errorf("Status endpoint: SSA ResetSpec (%v) and PrepareForUpdate wipeSpec (%v) behaviors do not match.", ssaStatusResetsSpec, statusWipesSpec)
		}
		if ssaStatusResetsMetadata != statusWipesMetadata {
			t.Errorf("Status endpoint: SSA ResetMetadata (%v) and PrepareForUpdate wipeMetadata (%v) behaviors do not match.", ssaStatusResetsMetadata, statusWipesMetadata)
		}

		// Enforce field wiping behaviors, with allowlists for pre-existing exceptions.
		assertWiped(t, api.Mapping.Resource, "create must wipe status", createWipesStatus, createDoesNotWipeStatus)
		assertWiped(t, api.Mapping.Resource, "update must wipe status", mainWipesStatus, sets.New[string]())      // No exemptions exist for this, please do not add any in the future.
		assertWiped(t, api.Mapping.Resource, "status update must wipe spec", statusWipesSpec, sets.New[string]()) // No exemptions exist for this, please do not add any in the future.
		assertWiped(t, api.Mapping.Resource, "status update must wipe metadata", statusWipesMetadata, statusDoesNotWipeMetadata)

		if err := rsc.Delete(context.TODO(), name, *metav1.NewDeleteOptions(0)); err != nil {
			t.Logf("Failed to delete %v: %v", name, err)
		}
	})
}

// checkPatch checks if field values under fieldScope (e.g. spec, status, metdata) in objData match the values
// in the applyManifest.
func checkPatch(t *testing.T, applyManifest string, fieldScope string, objData map[string]interface{}) bool {
	t.Helper()
	var applyObj map[string]interface{}
	if err := json.Unmarshal([]byte(applyManifest), &applyObj); err != nil {
		t.Fatalf("Failed to parse apply JSON: %v", err)
	}
	applyValue, ok := applyObj[fieldScope]
	if !ok {
		return false
	}
	objValue, ok := objData[fieldScope]
	if !ok {
		return false
	}
	return containsAll(applyValue, objValue)
}

// managedFieldsOwnTopLevelField checks whether a FieldsV1 set contains a given top-level field.
func managedFieldsOwnTopLevelField(t *testing.T, fieldsV1 *metav1.FieldsV1, field string) bool {
	t.Helper()
	if fieldsV1 == nil {
		return false
	}
	var fields map[string]interface{}
	if err := json.Unmarshal(fieldsV1.GetRawBytes(), &fields); err != nil {
		t.Logf("Failed to unmarshal FieldsV1: %v", err)
		return false
	}
	_, ok := fields["f:"+field]
	return ok
}

// assertWiped checks that a field wiping behavior holds, with an allowlist for known exceptions.
func assertWiped(t *testing.T, gvr schema.GroupVersionResource, msg string, wiped bool, allowed sets.Set[string]) {
	t.Helper()
	name := ResourceString(gvr)
	if matchesException(gvr, allowed) {
		if wiped {
			t.Errorf("%s: %s unexpectedly wiped. Remove it from the exception list.", name, msg)
		}
	} else {
		if !wiped {
			t.Errorf("%s: %s", name, msg)
		}
	}
}

// managedFieldsOwnLabel checks whether a FieldsV1 set contains a metadata label.
func managedFieldsOwnLabel(t *testing.T, fieldsV1 *metav1.FieldsV1, labelKey string) bool {
	t.Helper()
	if fieldsV1 == nil {
		return false
	}
	var fields map[string]interface{}
	if err := json.Unmarshal(fieldsV1.GetRawBytes(), &fields); err != nil {
		t.Logf("Failed to unmarshal FieldsV1: %v", err)
		return false
	}
	metadata, ok := fields["f:metadata"].(map[string]interface{})
	if !ok {
		return false
	}
	labels, ok := metadata["f:labels"].(map[string]interface{})
	if !ok {
		return false
	}
	_, ok = labels["f:"+labelKey]
	return ok
}
