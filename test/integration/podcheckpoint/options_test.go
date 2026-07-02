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

package podcheckpoint

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	nodev1 "k8s.io/api/node/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestRemovedRuntimeClassOptions(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("gate=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, enabled)
			ctx := ktesting.Init(t)
			client, _, closeFn := startAPIServer(ctx, t, enabled)
			defer closeFn()
			class := &nodev1.RuntimeClass{
				TypeMeta:   metav1.TypeMeta{APIVersion: nodev1.SchemeGroupVersion.String(), Kind: "RuntimeClass"},
				ObjectMeta: metav1.ObjectMeta{Name: "runtime-defaults"}, Handler: "runc",
			}
			testRemovedOptionField(t, client.CoreV1().RESTClient(), "/apis/node.k8s.io/v1/runtimeclasses", class,
				[]string{"podCheckpointOptions"}, map[string]interface{}{"allowedCheckpointOptions": []interface{}{"tcp"}})
		})
	}
}

func TestRemovedCheckpointOptions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "removed-checkpoint-options", t)
	source := createCheckpointSource(ctx, t, client, ns.Name, "source")
	checkpoint := newPodCheckpoint(ns.Name, "checkpoint-defaults", source.Name)
	checkpoint.APIVersion, checkpoint.Kind = nodev1alpha1.SchemeGroupVersion.String(), "PodCheckpoint"
	testRemovedOptionField(t, client.CoreV1().RESTClient(), "/apis/node.k8s.io/v1alpha1/namespaces/"+ns.Name+"/podcheckpoints", checkpoint,
		[]string{"spec", "checkpointOptions"}, map[string]interface{}{"tcp": "close"})
}

// Raw JSON is needed because typed clients cannot express the removed fields.
// Apply also rejects unknown fields during schema conversion, regardless of the
// field-validation mode; POST, PUT and merge patch can discard them.
func testRemovedOptionField(t *testing.T, client rest.Interface, collection string, prototype runtime.Object, field []string, value interface{}) {
	t.Helper()
	ctx := ktesting.Init(t)
	for _, operation := range []string{"post", "apply-create", "put", "merge-patch", "apply-update"} {
		for _, validation := range []string{metav1.FieldValidationStrict, metav1.FieldValidationWarn, ""} {
			for _, dryRun := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/validation=%s/dryrun=%t", operation, validation, dryRun), func(t *testing.T) {
					data, err := json.Marshal(prototype)
					require.NoError(t, err)
					var object map[string]interface{}
					require.NoError(t, json.Unmarshal(data, &object))
					name := fmt.Sprintf("defaults-%s-%s-%t", operation, strings.ToLower(validation), dryRun)
					require.NoError(t, unstructured.SetNestedField(object, name, "metadata", "name"))
					updating := operation == "put" || operation == "merge-patch" || operation == "apply-update"
					if updating {
						baseline, err := json.Marshal(object)
						require.NoError(t, err)
						created, err := client.Post().AbsPath(collection).SetHeader("Content-Type", "application/json").SetHeader("Accept", "application/json").Body(baseline).Do(ctx).Raw()
						require.NoError(t, err)
						require.NoError(t, json.Unmarshal(created, &object))
						unstructured.RemoveNestedField(object, "metadata", "managedFields")
					}
					require.NoError(t, unstructured.SetNestedField(object, value, field...))
					data, err = json.Marshal(object)
					require.NoError(t, err)
					request := client.Post().AbsPath(collection).SetHeader("Content-Type", "application/json")
					switch operation {
					case "apply-create", "apply-update":
						request = client.Patch(types.ApplyPatchType).AbsPath(collection, name).Param("fieldManager", "removed-options")
					case "put":
						request = client.Put().AbsPath(collection, name).SetHeader("Content-Type", "application/json")
					case "merge-patch":
						request = client.Patch(types.MergePatchType).AbsPath(collection, name)
					}
					if validation != "" {
						request.Param("fieldValidation", validation)
					}
					if dryRun {
						request.Param("dryRun", metav1.DryRunAll)
					}
					result := request.SetHeader("Accept", "application/json").Body(data).Do(ctx)
					response, _ := result.Raw()
					err = result.Error()
					rejected := validation == metav1.FieldValidationStrict || strings.HasPrefix(operation, "apply")
					if rejected {
						require.Error(t, err)
						require.Contains(t, err.Error(), strings.Join(field, "."))
					} else {
						require.NoError(t, err)
						var accepted map[string]interface{}
						require.NoError(t, json.Unmarshal(response, &accepted))
						_, present, err := unstructured.NestedFieldNoCopy(accepted, field...)
						require.NoError(t, err)
						require.False(t, present, "removed options must not survive decoding")
					}
					stored, getErr := client.Get().AbsPath(collection, name).SetHeader("Accept", "application/json").Do(ctx).Raw()
					if !updating && (dryRun || rejected) {
						require.True(t, apierrors.IsNotFound(getErr), "rejected or dry-run creates must not persist: %v", getErr)
					} else {
						require.NoError(t, getErr)
						var persisted map[string]interface{}
						require.NoError(t, json.Unmarshal(stored, &persisted))
						_, present, err := unstructured.NestedFieldNoCopy(persisted, field...)
						require.NoError(t, err)
						require.False(t, present, "removed options must not be persisted")
					}
				})
			}
		}
	}
}

func TestRemovedRestoreOptions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "removed-restore-options", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	pod := preparedRestorePod(checkpoint, "restore-defaults")
	pod.APIVersion, pod.Kind = "v1", "Pod"
	testRemovedOptionField(t, client.CoreV1().RESTClient(), "/api/v1/namespaces/"+ns.Name+"/pods", pod,
		[]string{"spec", "restoreFrom", "options"}, map[string]interface{}{"tcp": "close"})
}
