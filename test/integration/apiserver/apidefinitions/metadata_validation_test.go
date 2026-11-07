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
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
)

// TestObjectMetaValidation verifies that every resource rejects an update with invalid labels.
//
// ObjectMeta is validated by the strategy and, when the strategy reports no errors, by the
// generic REST layer. This guards against an update that neither of them validates.
// See https://github.com/kubernetes/kubernetes/issues/138654.
func TestObjectMetaValidation(t *testing.T) {
	TestAllDefinitions(t, "metadata-validation", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") {
			t.Skip("Resource does not support create, get, and update")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		created, err := client.Create(context.TODO(), obj, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create object: %v", err)
		}

		// Controllers may write the object concurrently, e.g. for CRDs.
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			latest, err := client.Get(context.TODO(), created.GetName(), metav1.GetOptions{})
			if err != nil {
				return err
			}
			latest.SetLabels(map[string]string{"invalid@key": "value"})
			_, err = client.Update(context.TODO(), latest, metav1.UpdateOptions{})
			return err
		})
		if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "metadata.labels") {
			t.Errorf("Expected an Invalid error for metadata.labels, got: %v", err)
		}
	})
}
