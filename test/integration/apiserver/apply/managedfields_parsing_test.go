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
	"bytes"
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
// metadata.managedFields.[*].fieldsV1 JSON payload is ignored and dropped
// for writes and is not persisted.
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

	testCases := []struct {
		name     string
		fieldsV1 string
		// expected is fieldsV1 with all trailing data is dropped from the data
		expected string
	}{
		{
			name:     "trailing data after FieldsV1 must be dropped",
			fieldsV1: `{"f:metadata":{"f:annotations":{"f:test.example.com/myannotation":{}}}}{"trailing":"data"}`,
			expected: `{"f:metadata":{"f:annotations":{"f:test.example.com/myannotation":{}}}}`,
		},
		{
			name:     "trailing data after set value must be dropped",
			fieldsV1: `{"f:metadata":{"f:finalizers":{"v:\"example.com/foo\" {\"trailing\":\"data\"}":{}}}}`,
			expected: `{"f:metadata":{"f:finalizers":{"v:\"example.com/foo\"":{}}}}`,
		},
		{
			name:     "trailing data after map key must be dropped",
			fieldsV1: `{"f:metadata":{"f:ownerReferences":{"k:{\"uid\":\"abc\"} {\"trailing\":\"data\"}":{}}}}`,
			expected: `{"f:metadata":{"f:ownerReferences":{"k:{\"uid\":\"abc\"}":{}}}}`,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fields := metav1.FieldsV1{}
			fields.SetRawBytes([]byte(tc.fieldsV1))

			name := fmt.Sprintf("managedfields-trailing-data-%d", i)
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: ns,
					ManagedFields: []metav1.ManagedFieldsEntry{
						{
							Manager:    "trailing-data-test",
							Operation:  metav1.ManagedFieldsOperationUpdate,
							APIVersion: "v1",
							FieldsType: "FieldsV1",
							FieldsV1:   &fields,
							Time:       &metav1.Time{Time: time.Now()},
						},
					},
				},
				Data: map[string]string{},
			}

			ctx := context.Background()
			created, err := client.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
			if err != nil {
				// We expect the trailing data to be accepted.
				t.Fatal(err)
			}
			t.Cleanup(func() {
				_ = client.CoreV1().ConfigMaps(ns).Delete(context.Background(), created.Name, metav1.DeleteOptions{})
			})

			got, err := client.CoreV1().ConfigMaps(ns).Get(ctx, created.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}

			if len(got.ManagedFields) != 1 {
				t.Fatal("Expected exactly 1 managed fields entry")
			}
			raw := got.ManagedFields[0].FieldsV1.GetRawBytes()
			if !bytes.Equal(raw, []byte(tc.expected)) {
				t.Errorf("expected %s but got %s", tc.expected, string(raw))
			}
		})
	}
}
