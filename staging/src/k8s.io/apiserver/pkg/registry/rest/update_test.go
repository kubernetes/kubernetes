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

type fakeUpdateStrategy struct {
	RESTUpdateStrategy
	// validateMeta mimics in-tree strategies, which validate ObjectMeta themselves.
	validateMeta bool
}

func (s fakeUpdateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var errs field.ErrorList
	if s.validateMeta {
		errs = genericvalidation.ValidateObjectMetaAccessorUpdate(obj.(metav1.Object), old.(metav1.Object), field.NewPath("metadata"))
	}
	if _, ok := obj.(*v1.ConfigMap).Data["invalid"]; ok {
		errs = append(errs, field.Invalid(field.NewPath("data"), "invalid", "spec-level error"))
	}
	return errs
}

func (fakeUpdateStrategy) NamespaceScoped() bool { return true }

func TestValidateUpdate(t *testing.T) {
	invalidAnnotation := func(cm *v1.ConfigMap) { cm.Annotations = map[string]string{"invalid@key": "value"} }
	invalidData := func(cm *v1.ConfigMap) { cm.Data = map[string]string{"invalid": ""} }

	tests := []struct {
		name         string
		validateMeta bool
		mutate       []func(*v1.ConfigMap)
		wantFields   []string
	}{
		{
			name:         "valid update",
			validateMeta: true,
		},
		{
			name:         "invalid annotation reported once",
			validateMeta: true,
			mutate:       []func(*v1.ConfigMap){invalidAnnotation},
			wantFields:   []string{"metadata.annotations"},
		},
		{
			name:         "invalid data only",
			validateMeta: true,
			mutate:       []func(*v1.ConfigMap){invalidData},
			wantFields:   []string{"data"},
		},
		{
			name:         "invalid annotation and data both reported once",
			validateMeta: true,
			mutate:       []func(*v1.ConfigMap){invalidAnnotation, invalidData},
			wantFields:   []string{"metadata.annotations", "data"},
		},
		{
			// Matches ValidateCreate: common checks are skipped once the strategy has rejected the object.
			name:       "strategy errors short-circuit common checks",
			mutate:     []func(*v1.ConfigMap){invalidAnnotation, invalidData},
			wantFields: []string{"data"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			oldObj := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "cm", Namespace: "default", ResourceVersion: "1"}}
			newObj := oldObj.DeepCopy()
			for _, m := range tc.mutate {
				m(newObj)
			}

			errs := ValidateUpdate(context.Background(), newObj, oldObj, fakeUpdateStrategy{validateMeta: tc.validateMeta})
			if len(errs) != len(tc.wantFields) {
				t.Fatalf("expected %d errors for %v, got %d: %v", len(tc.wantFields), tc.wantFields, len(errs), errs)
			}
			for i, want := range tc.wantFields {
				if errs[i].Field != want {
					t.Errorf("error %d: expected field %q, got %q (%v)", i, want, errs[i].Field, errs[i])
				}
			}
		})
	}
}
