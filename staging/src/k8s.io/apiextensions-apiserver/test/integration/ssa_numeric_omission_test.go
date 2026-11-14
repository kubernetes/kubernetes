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
	"fmt"
	"reflect"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/utils/ptr"
)

// Raw exclusive ownership can turn into phantom sharing while decoding list keys.
func TestSSANumericListKeyOmission(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()
	for _, low := range []int64{41, 9007199254740992} {
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%d-%t", low, shared), func(t *testing.T) {
				resource := newSSAFormalResource(t, apiClient, dynamicClient, fmt.Sprintf("omit-%d-%t.formal.ssa", low, shared), ssaFormalNumericKeySchema())
				highItem := map[string]any{"id": low + 1, "payload": "high"}
				before, err := resource.apply(t, "example", "A", map[string]any{"endpoints": []any{map[string]any{"id": low, "payload": "low"}, highItem}})
				if err != nil {
					t.Fatal(err)
				}
				if shared {
					before, err = resource.apply(t, "example", "B", map[string]any{"endpoints": []any{highItem}})
					if err != nil {
						t.Fatal(err)
					}
				}
				initialItems := ssaNumericOmissionItems(t, before, low)
				if initialItems[low]["payload"] != "low" || initialItems[low+1]["payload"] != "high" {
					t.Fatalf("initial payloads changed: %#v", initialItems)
				}
				if owners := ssaNumericPayloadOwners(t, before, low); !reflect.DeepEqual(owners, map[string]bool{"A": true}) {
					t.Fatalf("lower payload must be exclusively A-owned on the wire: %v", owners)
				}
				if shared && !ssaNumericPayloadOwners(t, before, low+1)["B"] {
					t.Fatal("B did not acquire the exact upper-item payload claim")
				}
				after, err := resource.apply(t, "example", "A", map[string]any{"endpoints": []any{map[string]any{"id": low}, highItem}})
				if err != nil {
					t.Fatal(err)
				}
				items := ssaNumericOmissionItems(t, after, low)
				retained := shared && low == 9007199254740992
				payload, present := items[low]["payload"]
				if present != retained || (present && payload != "low") || items[low+1]["payload"] != "high" {
					t.Fatalf("unexpected exclusive omission result (retained=%t): %#v", retained, items)
				}
				owners := ssaNumericPayloadOwners(t, after, low)
				expectedOwners := map[string]bool{}
				if retained {
					expectedOwners["B"] = true
				}
				if !reflect.DeepEqual(owners, expectedOwners) {
					t.Fatalf("unexpected post-omission lower payload claims: %v", owners)
				}
				stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(stored.Object, after.Object) {
					t.Fatal("omission response differs from persisted data or claims")
				}
			})
		}
	}
}

func ssaFormalNumericKeySchema() *apiextensionsv1.JSONSchemaProps {
	return &apiextensionsv1.JSONSchemaProps{
		Type: "object",
		Properties: map[string]apiextensionsv1.JSONSchemaProps{
			"endpoints": {
				Type: "array", XListType: ptr.To("map"), XListMapKeys: []string{"id"},
				Items: &apiextensionsv1.JSONSchemaPropsOrArray{Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object", Required: []string{"id"},
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"id": {Type: "integer", Format: "int64"}, "payload": {Type: "string"},
					},
				}},
			},
		},
	}
}

func ssaNumericOmissionItems(t *testing.T, object *unstructured.Unstructured, low int64) map[int64]map[string]any {
	t.Helper()
	items, found, err := unstructured.NestedSlice(object.Object, "endpoints")
	if err != nil || !found || len(items) != 2 {
		t.Fatalf("omission lost an item: found=%t err=%v object=%#v", found, err, object.Object)
	}
	result := map[int64]map[string]any{}
	for _, value := range items {
		item, ok := value.(map[string]any)
		if !ok {
			t.Fatalf("non-object item: %T", value)
		}
		id, ok := item["id"].(int64)
		if !ok || (id != low && id != low+1) || result[id] != nil {
			t.Fatalf("unexpected, rounded or duplicate item key: %#v", item)
		}
		result[id] = item
	}
	return result
}

// Inspect the exact serialized selector rather than the mixed numeric comparator.
func ssaNumericPayloadOwners(t *testing.T, object *unstructured.Unstructured, key int64) map[string]bool {
	t.Helper()
	owners := map[string]bool{}
	for _, owner := range object.GetManagedFields() {
		if owner.FieldsV1 == nil || owner.Operation != metav1.ManagedFieldsOperationApply || owner.Subresource != "" {
			t.Fatalf("unexpected numeric-key owner: %+v", owner)
		}
		var fields map[string]map[string]map[string]json.RawMessage
		if err := json.Unmarshal(owner.FieldsV1.Raw, &fields); err != nil {
			t.Fatal(err)
		}
		if _, present := fields["f:endpoints"][fmt.Sprintf(`k:{"id":%d}`, key)]["f:payload"]; present {
			owners[owner.Manager] = true
		}
	}
	return owners
}
