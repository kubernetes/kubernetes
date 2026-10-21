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
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestManagedFieldsTrailingData ensures that any trailing data present in a
// metadata.managedFields.[*].fieldsV1 JSON payload causes decoding to fail,
// falling back to the live object's managed fields. On create, invalid
// managed fields are ignored and the system populates managed fields for the
// created data. On update, invalid managed fields are ignored and the existing
// managed fields from the live object are preserved and updated.
func TestManagedFieldsTrailingData(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	// Only possible to have trailing data when the request is protobuf.
	config := rest.CopyConfig(server.ClientConfig)
	config.ContentType = runtime.ContentTypeProtobuf
	config.AcceptContentTypes = runtime.ContentTypeProtobuf

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	ns := "default"
	ctx := context.Background()

	testCases := []struct {
		name     string
		fieldsV1 string
	}{
		{
			name:     "trailing data after FieldsV1 must cause decoding failure and be dropped",
			fieldsV1: `{"f:metadata":{"f:annotations":{"f:test.example.com/myannotation":{}}}}{"trailing":"data"}`,
		},
		{
			name:     "trailing data after set value must cause decoding failure and be dropped",
			fieldsV1: `{"f:metadata":{"f:finalizers":{"v:\"example.com/foo\" {\"trailing\":\"data\"}":{}}}}`,
		},
		{
			name:     "trailing data after map key must cause decoding failure and be dropped",
			fieldsV1: `{"f:metadata":{"f:ownerReferences":{"k:{\"uid\":\"abc\"} {\"trailing\":\"data\"}":{}}}}`,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fields := metav1.FieldsV1{}
			fields.SetRawBytes([]byte(tc.fieldsV1))

			invalidManagedFields := []metav1.ManagedFieldsEntry{
				{
					Manager:    "trailing-data-test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &fields,
					Time:       &metav1.Time{Time: time.Now()},
				},
			}

			t.Run("create with empty data", func(t *testing.T) {
				name := fmt.Sprintf("mf-create-empty-%d", i)
				cm := &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:          name,
						Namespace:     ns,
						ManagedFields: invalidManagedFields,
					},
					Data: map[string]string{},
				}

				created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Create failed: %v", err)
				}
				t.Cleanup(func() {
					_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, created.Name, metav1.DeleteOptions{})
				})

				got, err := client.CoreV1().ConfigMaps(ns).Get(ctx, created.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}

				if len(got.ManagedFields) != 0 {
					t.Fatalf("Expected exactly 0 managed fields entries, got %d: %#v", len(got.ManagedFields), got.ManagedFields)
				}
			})

			t.Run("create with data", func(t *testing.T) {
				name := fmt.Sprintf("mf-create-data-%d", i)
				cm := &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:          name,
						Namespace:     ns,
						ManagedFields: invalidManagedFields,
					},
					Data: map[string]string{"foo": "bar"},
				}

				created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Create failed: %v", err)
				}
				t.Cleanup(func() {
					_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, created.Name, metav1.DeleteOptions{})
				})

				got, err := client.CoreV1().ConfigMaps(ns).Get(ctx, created.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}

				if len(got.ManagedFields) != 1 {
					t.Fatalf("Expected exactly 1 managed fields entry, got %d: %#v", len(got.ManagedFields), got.ManagedFields)
				}
				expectedFields := `{"f:data":{".":{},"f:foo":{}}}`
				if string(got.ManagedFields[0].FieldsV1.GetRawBytes()) != expectedFields {
					t.Fatalf("Expected fields %s, got %s", expectedFields, string(got.ManagedFields[0].FieldsV1.GetRawBytes()))
				}
			})

			t.Run("update with data", func(t *testing.T) {
				name := fmt.Sprintf("mf-update-%d", i)
				cm := &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      name,
						Namespace: ns,
					},
					Data: map[string]string{"foo": "bar"},
				}

				created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Create failed: %v", err)
				}
				t.Cleanup(func() {
					_ = client.CoreV1().ConfigMaps(ns).Delete(ctx, created.Name, metav1.DeleteOptions{})
				})

				toUpdate := created.DeepCopy()
				toUpdate.Data["foo"] = "baz"
				toUpdate.Data["new-key"] = "new-value"
				toUpdate.ManagedFields = invalidManagedFields

				updated, err := client.CoreV1().ConfigMaps(ns).Update(ctx, toUpdate, metav1.UpdateOptions{})
				if err != nil {
					t.Fatalf("Update failed: %v", err)
				}

				got, err := client.CoreV1().ConfigMaps(ns).Get(ctx, updated.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}

				if len(got.ManagedFields) != 1 {
					t.Fatalf("Expected exactly 1 managed fields entry, got %d: %#v", len(got.ManagedFields), got.ManagedFields)
				}
				expectedFields := `{"f:data":{".":{},"f:foo":{},"f:new-key":{}}}`
				if string(got.ManagedFields[0].FieldsV1.GetRawBytes()) != expectedFields {
					t.Fatalf("Expected fields %s, got %s", expectedFields, string(got.ManagedFields[0].FieldsV1.GetRawBytes()))
				}
			})
		})
	}
}
