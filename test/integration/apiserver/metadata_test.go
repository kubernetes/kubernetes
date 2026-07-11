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

package apiserver

import (
	"fmt"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// TestCreateMetadataWiping tests that metadata wiping works as expected on all operations
// that can create a resource. This includes creates-via-update requests allowed for strategies
// that support AllowCreateOnUpdate.
func TestCreateMetadataWiping(t *testing.T) {
	ctx, client, _, tearDown := setup(t)
	defer tearDown()

	ns := framework.CreateNamespaceOrDie(client, "create-metadata-wiping", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	uidValue := types.UID("00000000-0000-0000-0000-000000000000")
	creationValue := metav1.NewTime(time.Date(1999, 1, 1, 0, 0, 0, 0, time.UTC))
	selfLinkValue := "/this/should/be/wiped"
	deletionValue := metav1.NewTime(time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC))
	gracePeriodValue := int64(30)

	assertCleared := func(t *testing.T, m metav1.Object) {
		t.Helper()
		if got := m.GetUID(); got == "" || got == uidValue {
			t.Errorf("UID not regenerated on create: got %q, input was %q", got, uidValue)
		}
		if got := m.GetCreationTimestamp(); got.IsZero() || got.Equal(&creationValue) {
			t.Errorf("CreationTimestamp not regenerated on create: got %v, input was %v", got.UTC(), creationValue.UTC())
		}
		if got := m.GetSelfLink(); got != "" {
			t.Errorf("SelfLink not cleared on create: got %q, want empty", got)
		}
		if got := m.GetDeletionTimestamp(); got != nil {
			t.Errorf("DeletionTimestamp not cleared on create: got %v, want nil", got.UTC())
		}
		if got := m.GetDeletionGracePeriodSeconds(); got != nil {
			t.Errorf("DeletionGracePeriodSeconds not cleared on create: got %d, want nil", *got)
		}
	}

	t.Run("POST create", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:                       "post-create",
				UID:                        uidValue,
				CreationTimestamp:          creationValue,
				SelfLink:                   selfLinkValue,
				DeletionTimestamp:          &deletionValue,
				DeletionGracePeriodSeconds: &gracePeriodValue,
			},
		}
		got, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("POST create failed: %v", err)
		}
		assertCleared(t, got)
	})

	t.Run("PUT create-via-update", func(t *testing.T) {
		// UID and ResourceVersion are omitted here as they are not unconditionally wiped. They
		// are used precondition checks, which are tested separtely.
		lease := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:                       "put-create",
				Namespace:                  ns.Name,
				CreationTimestamp:          creationValue,
				SelfLink:                   selfLinkValue,
				DeletionTimestamp:          &deletionValue,
				DeletionGracePeriodSeconds: &gracePeriodValue,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity: ptr.To("metadata-test"),
			},
		}
		got, err := client.CoordinationV1().Leases(ns.Name).Update(ctx, lease, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("PUT create-via-update failed: %v", err)
		}
		assertCleared(t, got)
	})

	t.Run("PATCH Apply create)", func(t *testing.T) {
		body := fmt.Sprintf(`{
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": "ssa-create",
				"namespace": %q,
				"creationTimestamp": %q,
				"selfLink": %q,
				"deletionTimestamp": %q,
				"deletionGracePeriodSeconds": 30
			}
		}`, ns.Name, creationValue.UTC().Format(time.RFC3339), selfLinkValue, deletionValue.UTC().Format(time.RFC3339))

		result, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			Namespace(ns.Name).
			Resource("configmaps").
			Name("ssa-create").
			Param("fieldManager", "metadata-test").
			Body([]byte(body)).
			Do(ctx).Get()
		if err != nil {
			t.Fatalf("SSA apply create failed: %v", err)
		}
		got, ok := result.(*corev1.ConfigMap)
		if !ok {
			t.Fatalf("expected *ConfigMap, got %T", result)
		}
		assertCleared(t, got)
	})

	// All other patch operations are NOT allowed to create via patch.

	patchCases := []struct {
		name      string
		patchType types.PatchType
		body      []byte
	}{
		{"json", types.JSONPatchType, []byte(`[{"op":"add","path":"/data","value":{"k":"v"}}]`)},
		{"merge", types.MergePatchType, []byte(`{"data":{"k":"v"}}`)},
		{"strategic-merge", types.StrategicMergePatchType, []byte(`{"data":{"k":"v"}}`)},
	}
	for _, tc := range patchCases {
		t.Run("PATCH "+tc.name+" create", func(t *testing.T) {
			err := client.CoreV1().RESTClient().Patch(tc.patchType).
				Namespace(ns.Name).
				Resource("configmaps").
				Name("missing-" + tc.name).
				Body(tc.body).
				Do(ctx).Error()
			if !apierrors.IsNotFound(err) {
				t.Errorf("expected NotFound from %s patch on missing object, got %v", tc.name, err)
			}
		})
	}
}
