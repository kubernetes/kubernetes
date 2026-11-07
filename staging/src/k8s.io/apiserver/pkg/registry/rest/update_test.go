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

package rest

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// fakeUpdateStrategy validates ObjectMeta itself, as in-tree strategies do.
type fakeUpdateStrategy struct {
	RESTUpdateStrategy
}

func (fakeUpdateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return genericvalidation.ValidateObjectMetaAccessorUpdate(obj.(metav1.Object), old.(metav1.Object), field.NewPath("metadata"))
}

func (fakeUpdateStrategy) NamespaceScoped() bool { return true }

func TestValidateUpdateReportsMetadataErrorsOnce(t *testing.T) {
	oldObj := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "cm", Namespace: "default", ResourceVersion: "1"}}
	newObj := oldObj.DeepCopy()
	newObj.Annotations = map[string]string{"invalid@key": "value"}

	errs := ValidateUpdate(context.Background(), newObj, oldObj, fakeUpdateStrategy{})
	if len(errs) != 1 {
		t.Errorf("expected the invalid annotation to be reported once, got %d errors: %v", len(errs), errs)
	}
}
