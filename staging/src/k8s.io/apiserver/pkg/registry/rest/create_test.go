/*
Copyright 2019 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/names"
)

type mockStrategy struct {
	names.NameGenerator
}

func (m mockStrategy) NamespaceScoped() bool {
	return false
}

func (m mockStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {

}
func (m mockStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return nil
}

func (m mockStrategy) Canonicalize(obj runtime.Object) {

}

func (m mockStrategy) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	return []schema.GroupVersionKind{{Group: "mygroup.example.com", Version: "v1beta1", Kind: "Secret"}}, true, nil
}
func (m mockStrategy) Recognizes(gvk schema.GroupVersionKind) bool {
	return false
}

func TestBeforeCreate(t *testing.T) {
	ctx := genericapirequest.NewContext()
	tests := []struct {
		resource *unstructured.Unstructured
		expected string
	}{
		{
			resource: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"name":         "foo",
						"generatename": "bar",
					},
				},
			},
		}, {
			resource: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generatename": "bar",
					},
				},
			},
		},
	}
	mock := mockStrategy{names.SimpleNameGenerator}
	for _, test := range tests {
		BeforeCreate(mock, ctx, test.resource)
		objectMeta, _, kerr := objectMetaAndKind(mock, test.resource)
		if kerr != nil {
			t.Fatal(kerr)
		}

		if len(objectMeta.GetName()) > 0 {
			if objectMeta.GetGenerateName() != "" {
				t.Fatal("GenerateName has to be empty, if the resource has name")
			}
		}

		if test.resource.GetGenerateName() != objectMeta.GetGenerateName() {
			t.Fatalf("error in GenerateName; expected %s, got %s", test.resource.GetGenerateName(), objectMeta.GetGenerateName())
		}
	}
}