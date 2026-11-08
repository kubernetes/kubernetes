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
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
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
			crSchema := ssaFormalListSchema()
			if policy == "required" {
				crSchema.Required = []string{"endpoints"}
			}
			if policy == "cel" {
				crSchema.XValidations = apiextensionsv1.ValidationRules{{Rule: "has(self.endpoints)"}}
			}
			resource := newSSAFormalResource(t, apiClient, dynamicClient, policy+".formal.ssa", crSchema)
			apply := func(name string, items []any) (*unstructured.Unstructured, error) {
				return resource.apply(t, name, "A", map[string]any{"endpoints": items})
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

func ssaFormalListSchema() *apiextensionsv1.JSONSchemaProps {
	return &apiextensionsv1.JSONSchemaProps{
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
}

// Discovery readiness suffices because these tests observe state through GET.
func newSSAFormalResource(t *testing.T, apiClient clientset.Interface, dynamicClient dynamic.Interface, group string, crSchema *apiextensionsv1.JSONSchemaProps) ssaFormalResource {
	t.Helper()
	return newSSAFormalResourceWithSubresources(t, apiClient, dynamicClient, group, crSchema, nil)
}

func newSSAFormalResourceWithSubresources(t *testing.T, apiClient clientset.Interface, dynamicClient dynamic.Interface, group string, crSchema *apiextensionsv1.JSONSchemaProps, subresources *apiextensionsv1.CustomResourceSubresources) ssaFormalResource {
	t.Helper()
	crd := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	crd.Spec.Group = group
	crd.Name = crd.Spec.Names.Plural + "." + group
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: crSchema}
	crd.Spec.Versions[0].Subresources = subresources
	crd, err := fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiClient)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := apiClient.ApiextensionsV1().CustomResourceDefinitions().Delete(context.Background(), crd.Name, metav1.DeleteOptions{}); err != nil {
			t.Error(err)
		}
	})
	version := crd.Spec.Versions[0].Name
	resource := dynamicClient.Resource(schema.GroupVersionResource{Group: group, Version: version, Resource: crd.Spec.Names.Plural})
	return ssaFormalResource{resource, group + "/" + version, crd.Spec.Names.Kind}
}

type ssaFormalResource struct {
	dynamic.ResourceInterface
	apiVersion, kind string
}

func (r ssaFormalResource) apply(t *testing.T, name, manager string, fields map[string]any, subresources ...string) (*unstructured.Unstructured, error) {
	t.Helper()
	object := map[string]any{"apiVersion": r.apiVersion, "kind": r.kind, "metadata": map[string]any{"name": name}}
	for key, value := range fields {
		object[key] = value
	}
	body, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	return r.Patch(context.Background(), name, types.ApplyPatchType, body, metav1.PatchOptions{FieldManager: manager}, subresources...)
}

// Empty-map intent has a presence claim, but granular pruning can still create null.
func TestSSAEmptyGranularMapPresence(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()
	for _, policy := range []string{"nonnullable", "nullable", "cel", "defaulted", "atomic"} {
		t.Run(policy, func(t *testing.T) {
			mapSchema := apiextensionsv1.JSONSchemaProps{
				Type: "object", Nullable: policy == "nullable" || policy == "cel",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{"left": {Type: "string"}},
			}
			if policy == "defaulted" {
				mapSchema.Default = &apiextensionsv1.JSON{Raw: []byte(`{}`)}
			}
			if policy == "atomic" {
				mapSchema.XMapType = ptr.To("atomic")
			}
			crSchema := &apiextensionsv1.JSONSchemaProps{Type: "object", Properties: map[string]apiextensionsv1.JSONSchemaProps{"settings": mapSchema}}
			if policy == "cel" {
				crSchema.XValidations = apiextensionsv1.ValidationRules{{Rule: "has(self.settings)"}}
			}
			resource := newSSAFormalResource(t, apiClient, dynamicClient, "map-"+policy+".formal.ssa", crSchema)
			empty := map[string]any{"settings": map[string]any{}}
			control, err := resource.apply(t, "created-empty", "A", empty)
			if err != nil {
				t.Fatalf("empty-map creation must succeed: %v", err)
			}
			if value, exists := control.Object["settings"]; !exists || !reflect.DeepEqual(value, map[string]any{}) {
				t.Fatalf("empty-map creation changed intent: %#v", control.Object)
			}
			original, err := resource.apply(t, "example", "A", map[string]any{"settings": map[string]any{"left": "old"}})
			if err != nil {
				t.Fatal(err)
			}
			previous := original
			for attempt := 1; attempt <= 3; attempt++ {
				proposed, err := resource.apply(t, "example", "A", empty)
				if policy == "nonnullable" || policy == "cel" {
					if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "settings") {
						t.Fatalf("attempt %d must fail map validation: %v", attempt, err)
					}
					if policy == "cel" && !strings.Contains(err.Error(), "has(self.settings)") {
						t.Fatalf("unexpected CEL failure: %v", err)
					}
					stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(stored.Object, original.Object) {
						t.Fatalf("rejected attempt %d changed data or ownership", attempt)
					}
					continue
				}
				if err != nil {
					t.Fatal(err)
				}
				var want any = map[string]any{}
				changes := 1
				if policy == "nullable" {
					changes = 2
					if attempt == 1 {
						want = nil
					}
				}
				if value, exists := proposed.Object["settings"]; !exists || !reflect.DeepEqual(value, want) {
					t.Fatalf("attempt %d: settings=%#v, want %#v", attempt, proposed.Object, want)
				}
				owners := proposed.GetManagedFields()
				if len(owners) != 1 || owners[0].Manager != "A" || owners[0].Operation != metav1.ManagedFieldsOperationApply ||
					owners[0].FieldsV1 == nil || string(owners[0].FieldsV1.Raw) != `{"f:settings":{}}` {
					t.Fatalf("attempt %d lost the explicit map claim: %#v", attempt, owners)
				}
				if (proposed.GetResourceVersion() != previous.GetResourceVersion()) != (attempt <= changes) {
					t.Fatalf("attempt %d: unexpected resourceVersion transition", attempt)
				}
				previous = proposed
				if policy == "nullable" && attempt == 1 {
					// B's identical intent conflicts with the null A stored; rejection is atomic.
					_, err := resource.apply(t, "example", "B", empty)
					if !apierrors.IsConflict(err) {
						t.Fatalf("B must conflict against the unexpected null: %v", err)
					}
					stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(stored.Object, proposed.Object) {
						t.Fatal("B's rejected Apply changed data or ownership")
					}
				}
			}
			if policy == "nullable" {
				// After A repairs its result, B can acquire the same empty-map value.
				shared, err := resource.apply(t, "example", "B", empty)
				if err != nil {
					t.Fatalf("B must acquire the repaired empty map without force: %v", err)
				}
				owners := shared.GetManagedFields()
				seen := map[string]bool{}
				for _, owner := range owners {
					seen[owner.Manager] = true
					if owner.Operation != metav1.ManagedFieldsOperationApply || owner.APIVersion != resource.apiVersion ||
						owner.FieldsV1 == nil || string(owner.FieldsV1.Raw) != `{"f:settings":{}}` {
						t.Fatalf("unexpected co-ownership after repair: %#v", owner)
					}
				}
				if len(owners) != 2 || !seen["A"] || !seen["B"] {
					t.Fatalf("expected co-ownership after repair: %#v", owners)
				}
			}
		})
	}
}

// A child Update claim does not keep the released item; nullability decides whether storage accepts the result.
func TestSSAUpdateOwnedListItemPruning(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	for _, policy := range []string{"nonnullable", "nullable"} {
		t.Run(policy, func(t *testing.T) {
			crSchema := ssaFormalListSchema()
			listSchema := crSchema.Properties["endpoints"]
			listSchema.Nullable = policy == "nullable"
			crSchema.Properties["endpoints"] = listSchema
			resource := newSSAFormalResource(t, apiClient, dynamicClient, "updater-"+policy+".formal.ssa", crSchema)

			// Omitting the optional list is a valid creation intent under both policies.
			if _, err := resource.apply(t, "created-without-list", "A", map[string]any{}); err != nil {
				t.Fatalf("creation without the list must succeed: %v", err)
			}
			created, err := resource.apply(t, "example", "A", map[string]any{
				"endpoints": []any{map[string]any{"name": "db", "host": "old"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			replacement := created.DeepCopy()
			if err := unstructured.SetNestedSlice(replacement.Object,
				[]any{map[string]any{"name": "db", "host": "new"}}, "endpoints"); err != nil {
				t.Fatal(err)
			}
			updated, err := resource.Update(context.Background(), replacement, metav1.UpdateOptions{FieldManager: "B"})
			if err != nil {
				t.Fatal(err)
			}

			// B acquires host by Update while A retains the item's membership.
			seen := map[string]bool{}
			for _, owner := range updated.GetManagedFields() {
				seen[owner.Manager] = true
				if owner.FieldsV1 == nil {
					t.Fatalf("missing fields for owner %s", owner.Manager)
				}
				var fields map[string]any
				if err := json.Unmarshal(owner.FieldsV1.Raw, &fields); err != nil {
					t.Fatal(err)
				}
				itemFields, exists, err := unstructured.NestedMap(fields, "f:endpoints", `k:{"name":"db"}`)
				if err != nil || !exists {
					t.Fatalf("missing item fields for owner %s: %s", owner.Manager, owner.FieldsV1.Raw)
				}
				_, membership := itemFields["."]
				_, host := itemFields["f:host"]
				switch owner.Manager {
				case "A":
					if owner.Operation != metav1.ManagedFieldsOperationApply || !membership || host {
						t.Fatalf("unexpected A ownership: %#v", owner)
					}
				case "B":
					if owner.Operation != metav1.ManagedFieldsOperationUpdate || membership || !host {
						t.Fatalf("unexpected B ownership: %#v", owner)
					}
				default:
					t.Fatalf("unexpected owner: %#v", owner)
				}
			}
			if len(updated.GetManagedFields()) != 2 || !seen["A"] || !seen["B"] {
				t.Fatalf("expected A item membership and B host ownership: %#v", updated.GetManagedFields())
			}

			for attempt := 1; attempt <= 3; attempt++ {
				proposed, err := resource.apply(t, "example", "A", map[string]any{})
				if policy == "nullable" {
					if err != nil {
						t.Fatal(err)
					}
					if value, exists := proposed.Object["endpoints"]; !exists || value != nil {
						t.Fatalf("expected the list to collapse to null: %#v", proposed.Object)
					}
					if len(proposed.GetManagedFields()) != 0 {
						t.Fatalf("removed item must lose its old claims: %#v", proposed.GetManagedFields())
					}
					if proposed.GetResourceVersion() == updated.GetResourceVersion() {
						t.Fatal("accepted deletion must change resourceVersion")
					}
					break
				}
				if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "endpoints") {
					t.Fatalf("attempt %d must reject the generated null list: %v", attempt, err)
				}
				stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(stored.Object, updated.Object) {
					t.Fatalf("rejected attempt %d changed data, ownership or resourceVersion", attempt)
				}
			}
		})
	}
}

// Status strategy restores every non-status root field, but its ownership filter names only spec and metadata.
func TestSSAStatusResetOwnership(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	for _, root := range []string{"spec", "settings"} {
		t.Run(root, func(t *testing.T) {
			crSchema := &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					root: {
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"desired": {Type: "string"},
						},
					},
					"status": {
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"observed": {Type: "string"},
						},
					},
				},
			}
			resource := newSSAFormalResourceWithSubresources(t, apiClient, dynamicClient,
				"status-"+root+".formal.ssa", crSchema,
				&apiextensionsv1.CustomResourceSubresources{Status: &apiextensionsv1.CustomResourceSubresourceStatus{}})
			created, err := resource.apply(t, "example", "A", map[string]any{
				root: map[string]any{"desired": "one"}, "status": map[string]any{"observed": "forged"},
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, exists := created.Object["status"]; exists {
				t.Fatalf("main-endpoint creation must reset status: %#v", created.Object)
			}

			// The same literal manager name has separate main/status identities.
			statusRoot := "must-not-change"
			if root == "settings" {
				statusRoot = "one"
			}
			withStatus, err := resource.apply(t, "example", "A", map[string]any{
				root: map[string]any{"desired": statusRoot}, "status": map[string]any{"observed": "ready"},
			}, "status")
			if err != nil {
				t.Fatal(err)
			}
			if desired, _, _ := unstructured.NestedString(withStatus.Object, root, "desired"); desired != "one" {
				t.Fatalf("status request changed %s: %#v", root, withStatus.Object)
			}
			if observed, _, _ := unstructured.NestedString(withStatus.Object, "status", "observed"); observed != "ready" {
				t.Fatalf("status request did not set status: %#v", withStatus.Object)
			}

			main, status := false, false
			for _, owner := range withStatus.GetManagedFields() {
				if owner.Manager != "A" || owner.Operation != metav1.ManagedFieldsOperationApply || owner.FieldsV1 == nil {
					t.Fatalf("unexpected ownership identity: %#v", owner)
				}
				var fields map[string]any
				if err := json.Unmarshal(owner.FieldsV1.Raw, &fields); err != nil {
					t.Fatal(err)
				}
				_, hasRoot := fields["f:"+root]
				_, hasStatus := fields["f:status"]
				if hasRoot && !reflect.DeepEqual(fields["f:"+root], map[string]any{"f:desired": map[string]any{}}) {
					t.Fatalf("unexpected desired-field ownership: %s", owner.FieldsV1.Raw)
				}
				if hasStatus && !reflect.DeepEqual(fields["f:status"], map[string]any{"f:observed": map[string]any{}}) {
					t.Fatalf("unexpected observed-field ownership: %s", owner.FieldsV1.Raw)
				}
				switch owner.Subresource {
				case "":
					main = true
					if !hasRoot || hasStatus {
						t.Fatalf("main manager acquired status ownership: %s", owner.FieldsV1.Raw)
					}
				case "status":
					status = true
					if !hasStatus || hasRoot != (root == "settings") {
						t.Fatalf("unexpected status root ownership: %s", owner.FieldsV1.Raw)
					}
				default:
					t.Fatalf("unexpected subresource: %#v", owner)
				}
			}
			if len(withStatus.GetManagedFields()) != 2 || !main || !status {
				t.Fatalf("same-name managers must remain distinct across subresources: %#v", withStatus.GetManagedFields())
			}

			// A status claim on settings blocks a later main-endpoint change that status cannot perform.
			mainIntent := map[string]any{root: map[string]any{"desired": "two"}}
			if root == "spec" {
				mainIntent["status"] = map[string]any{"observed": "forged"}
			}
			changed, err := resource.apply(t, "example", "A", mainIntent)
			if root == "settings" {
				if !apierrors.IsConflict(err) {
					t.Fatalf("expected a conflict against the inappropriate status claim: %v", err)
				}
				stored, err := resource.Get(context.Background(), "example", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(stored.Object, withStatus.Object) {
					t.Fatal("rejected root change modified data, ownership or resourceVersion")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if desired, _, _ := unstructured.NestedString(changed.Object, root, "desired"); desired != "two" {
				t.Fatalf("main request failed to change spec: %#v", changed.Object)
			}
			if observed, _, _ := unstructured.NestedString(changed.Object, "status", "observed"); observed != "ready" {
				t.Fatalf("main request changed status: %#v", changed.Object)
			}
		})
	}
}

// A rounded mixed-number equality can hide a field change from the conflict detector.
func TestSSANumericComparison(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	const initial = int64(9223372036854775807)
	for _, variant := range []string{"scalar-only", "with-note", "integer-control"} {
		t.Run(variant, func(t *testing.T) {
			resource := newSSAFormalResource(t, apiClient, dynamicClient, "numeric-"+variant+".formal.ssa", &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"amount": {Type: "number"},
					"note":   {Type: "string"},
				},
			})
			before, err := resource.apply(t, variant, "A", map[string]any{"amount": initial})
			if err != nil {
				t.Fatal(err)
			}
			if got, ok := before.Object["amount"].(int64); !ok || got != initial {
				t.Fatalf("initial Apply did not preserve the exact int64: %v (%T)", before.Object["amount"], before.Object["amount"])
			}
			intent := map[string]any{"amount": json.Number("9223372036854775808")}
			if variant != "scalar-only" {
				intent["note"] = "changed"
			}
			if variant == "integer-control" {
				intent["amount"] = initial - 1
			}
			after, err := resource.apply(t, variant, "B", intent)
			if variant == "integer-control" {
				if !apierrors.IsConflict(err) {
					t.Fatalf("exact integer changes must conflict: %v", err)
				}
				stored, err := resource.Get(context.Background(), variant, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(stored.Object, before.Object) {
					t.Fatal("rejected integer change modified storage")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			stored, err := resource.Get(context.Background(), variant, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(stored.Object, after.Object) {
				t.Fatal("numeric Apply response differs from persisted data, claims or resourceVersion")
			}

			if variant == "scalar-only" {
				if got, ok := after.Object["amount"].(int64); !ok || got != initial {
					t.Fatalf("expected lost numeric intent after false no-op: %v (%T)", after.Object["amount"], after.Object["amount"])
				}
			} else if got, ok := after.Object["amount"].(float64); !ok || got != float64(9223372036854775808) {
				t.Fatalf("expected rounded foreign-owned value to change without conflict: %v (%T)", after.Object["amount"], after.Object["amount"])
			}
			if variant == "with-note" && after.Object["note"] != "changed" {
				t.Fatalf("the independent note was not applied: %#v", after.Object)
			}
			owners := map[string]bool{}
			for _, owner := range after.GetManagedFields() {
				var fields map[string]any
				if owner.FieldsV1 == nil {
					t.Fatal("numeric owner has no FieldsV1")
				}
				if err := json.Unmarshal(owner.FieldsV1.Raw, &fields); err != nil {
					t.Fatal(err)
				}
				if _, found := fields["f:amount"]; found {
					if owner.Operation != metav1.ManagedFieldsOperationApply || owner.Subresource != "" {
						t.Fatalf("numeric claim has an unexpected manager identity: %+v", owner)
					}
					owners[owner.Manager] = true
				}
			}
			if !reflect.DeepEqual(owners, map[string]bool{"A": true, "B": true}) {
				t.Fatalf("both managers should retain the numeric claim: %#v", after.GetManagedFields())
			}
		})
	}
}

// Ownership decoding must keep distinct integer list keys distinct.
func TestSSANumericListKeyOwnership(t *testing.T) {
	tearDown, apiClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	for _, first := range []int64{42, 9007199254740993} {
		t.Run(fmt.Sprint(first), func(t *testing.T) {
			resource := newSSAFormalResource(t, apiClient, dynamicClient, "numeric-key-"+fmt.Sprint(first)+".formal.ssa", &apiextensionsv1.JSONSchemaProps{
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
			})
			before, err := resource.apply(t, "example", "A", map[string]any{
				"endpoints": []any{map[string]any{"id": first, "payload": "old"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			items, _, _ := unstructured.NestedSlice(before.Object, "endpoints")
			if len(items) != 1 || items[0].(map[string]any)["id"] != first {
				t.Fatalf("initial list key was rounded before storage: %#v", before.Object)
			}
			after, err := resource.apply(t, "example", "B", map[string]any{
				"endpoints": []any{map[string]any{"id": first - 1, "payload": "independent"}},
			})
			stored, getErr := resource.Get(context.Background(), "example", metav1.GetOptions{})
			if getErr != nil {
				t.Fatal(getErr)
			}
			if first == 9007199254740993 {
				if !apierrors.IsConflict(err) || !strings.Contains(err.Error(), "conflicts with \"A\"") {
					t.Fatalf("expected a false conflict against A's rounded key: %v", err)
				}
				if !reflect.DeepEqual(stored.Object, before.Object) {
					t.Fatal("false key conflict changed data, claims or resourceVersion")
				}
				foundA := false
				for _, owner := range before.GetManagedFields() {
					if owner.Manager == "A" {
						foundA = true
						if owner.FieldsV1 == nil || !strings.Contains(string(owner.FieldsV1.Raw), "9007199254740993") {
							t.Fatalf("initial wire ownership already lost the exact key: %+v", owner)
						}
					}
				}
				if !foundA {
					t.Fatal("initial wire ownership lost manager A")
				}
				return
			}
			if err != nil {
				t.Fatalf("small independent integer keys must succeed: %v", err)
			}
			if !reflect.DeepEqual(stored.Object, after.Object) {
				t.Fatal("independent-key response differs from persisted state")
			}
			items, _, _ = unstructured.NestedSlice(after.Object, "endpoints")
			if len(items) != 2 {
				t.Fatalf("independent-key creation lost a list item: %#v", after.Object)
			}
			actual := map[int64]string{}
			for _, item := range items {
				object := item.(map[string]any)
				key, ok := object["id"].(int64)
				if !ok {
					t.Fatalf("list key lost its int64 type: %#v", object)
				}
				actual[key] = object["payload"].(string)
			}
			if !reflect.DeepEqual(actual, map[int64]string{first: "old", first - 1: "independent"}) {
				t.Fatalf("independent-key contents changed: %#v", actual)
			}
		})
	}
}
