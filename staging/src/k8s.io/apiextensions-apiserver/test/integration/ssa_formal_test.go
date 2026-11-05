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

package integration

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

// This characterizes a counterexample to immediate Apply idempotence. The Lean
// reference model and the pinned SMD head/base corpus check the same ownership transitions.
func TestSSAEmptyAssociativeListPresence(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	for _, policy := range []string{"optional", "required", "cel"} {
		t.Run(policy, func(t *testing.T) {
			crSchema := &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"endpoints": {
						Type: "array", XListType: ptr.To("map"), XListMapKeys: []string{"name"},
						Items: &apiextensionsv1.JSONSchemaPropsOrArray{Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object", Required: []string{"name"},
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"name": {Type: "string"}, "host": {Type: "string"},
							},
						}},
					},
				},
			}
			if policy == "required" {
				crSchema.Required = []string{"endpoints"}
			}
			if policy == "cel" {
				crSchema.XValidations = apiextensionsv1.ValidationRules{{Rule: "has(self.endpoints)"}}
			}
			crd := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
			crd.Spec.Group = policy + ".formal.ssa"
			crd.Name = crd.Spec.Names.Plural + "." + crd.Spec.Group
			crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: crSchema}

			// This test uses GET rather than watches; the watch-primer creates a schema-incompatible object.
			crd, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiClient)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := apiClient.ApiextensionsV1().CustomResourceDefinitions().Delete(context.Background(), crd.Name, metav1.DeleteOptions{}); err != nil {
					t.Error(err)
				}
			})
			version := crd.Spec.Versions[0].Name
			resource := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: version, Resource: crd.Spec.Names.Plural})
			apply := func(name string, items []any) (*unstructured.Unstructured, error) {
				body, err := json.Marshal(map[string]any{
					"apiVersion": crd.Spec.Group + "/" + version, "kind": crd.Spec.Names.Kind,
					"metadata": map[string]any{"name": name}, "endpoints": items,
				})
				if err != nil {
					t.Fatal(err)
				}
				return resource.Patch(context.Background(), name, types.ApplyPatchType, body, metav1.PatchOptions{FieldManager: "A"})
			}

			// Presence policies permit an empty list when there are no old item claims.
			control, err := apply("created-empty", []any{})
			if err != nil {
				t.Fatalf("empty-list creation must succeed: %v", err)
			}
			if items, exists := control.Object["endpoints"]; !exists || !reflect.DeepEqual(items, []any{}) {
				t.Fatalf("empty-list creation changed presence/value: %#v", control.Object)
			}
			original, err := apply("example", []any{map[string]any{"name": "db", "host": "old"}})
			if err != nil {
				t.Fatal(err)
			}
			previous := original
			for attempt := 1; attempt <= 4; attempt++ {
				proposed, err := apply("example", []any{})
				if policy == "optional" {
					if err != nil {
						t.Fatal(err)
					}
					items, present := proposed.Object["endpoints"]
					if present != (attempt > 1) || (present && !reflect.DeepEqual(items, []any{})) {
						t.Fatalf("attempt %d: unexpected endpoints value/presence: %#v", attempt, proposed.Object)
					}

					// Once empty ownership vanishes, the default wrapper bootstraps an Update manager.
					owners := proposed.GetManagedFields()
					if attempt <= 2 && len(owners) != 0 {
						t.Fatalf("attempt %d unexpectedly retained ownership: %#v", attempt, owners)
					}
					if attempt >= 3 && (len(owners) != 1 || owners[0].Manager != "before-first-apply" ||
						owners[0].Operation != metav1.ManagedFieldsOperationUpdate || owners[0].FieldsV1 == nil ||
						string(owners[0].FieldsV1.Raw) != `{"f:endpoints":{}}`) {
						t.Fatalf("attempt %d did not preserve the bootstrapped empty-list claim: %#v", attempt, owners)
					}
					if (proposed.GetResourceVersion() == previous.GetResourceVersion()) != (attempt == 4) {
						t.Fatalf("attempt %d: unexpected resourceVersion transition %s -> %s", attempt, previous.GetResourceVersion(), proposed.GetResourceVersion())
					}
					previous = proposed
					continue
				}
				if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "endpoints") {
					t.Fatalf("attempt %d must fail endpoints validation: %v", attempt, err)
				}
				if policy == "cel" && !strings.Contains(err.Error(), "has(self.endpoints)") {
					t.Fatalf("unexpected CEL failure: %v", err)
				}

				// GET observes the committed state, including ownership and storage version.
				stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(stored.Object, original.Object) {
					t.Fatalf("rejected attempt %d changed the stored object or ownership", attempt)
				}
			}
		})
	}
}
