/*
Copyright 2018 The Kubernetes Authors.

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

package fieldmanager

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

var testSchema = prototesting.Fake{
	Path: filepath.Join("testdata", "swagger.json"),
}

// TestVersionConverter tests the version converter
func TestVersionConverter(t *testing.T) {
	d, err := testSchema.OpenAPISchema()
	if err != nil {
		t.Fatalf("Failed to parse OpenAPI schema: %v", err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		t.Fatalf("Failed to build OpenAPI models: %v", err)
	}
	tc, err := NewTypeConverter(m, false)
	if err != nil {
		t.Fatalf("Failed to build TypeConverter: %v", err)
	}
	oc := fakeObjectConvertorForTestSchema{
		gvkForVersion("v1beta1"): objForGroupVersion("apps/v1beta1"),
		gvkForVersion("v1"):      objForGroupVersion("apps/v1"),
	}
	vc := newVersionConverter(tc, oc, schema.GroupVersion{Group: "apps", Version: runtime.APIVersionInternal})

	input, err := tc.ObjectToTyped(objForGroupVersion("apps/v1beta1"))
	if err != nil {
		t.Fatalf("error creating converting input object to a typed value: %v", err)
	}
	expected := objForGroupVersion("apps/v1")
	output, err := vc.Convert(input, fieldpath.APIVersion("apps/v1"))
	if err != nil {
		t.Fatalf("expected err to be nil but got %v", err)
	}
	actual, err := tc.TypedToObject(output)
	if err != nil {
		t.Fatalf("error converting output typed value to an object %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("expected to get %v but got %v", expected, actual)
	}
}

func gvkForVersion(v string) schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   "apps",
		Version: v,
		Kind:    "Deployment",
	}
}

func objForGroupVersion(gv string) runtime.Object {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": gv,
			"kind":       "Deployment",
		},
	}
}

type fakeObjectConvertorForTestSchema map[schema.GroupVersionKind]runtime.Object

var _ runtime.ObjectConvertor = fakeObjectConvertorForTestSchema{}

func (c fakeObjectConvertorForTestSchema) ConvertToVersion(_ runtime.Object, gv runtime.GroupVersioner) (runtime.Object, error) {
	allKinds := make([]schema.GroupVersionKind, 0)
	for kind := range c {
		allKinds = append(allKinds, kind)
	}
	gvk, _ := gv.KindForGroupVersionKinds(allKinds)
	return c[gvk], nil
}

func (fakeObjectConvertorForTestSchema) Convert(_, _, _ interface{}) error {
	return fmt.Errorf("function not implemented")
}

func (fakeObjectConvertorForTestSchema) ConvertFieldLabel(_ schema.GroupVersionKind, _, _ string) (string, string, error) {
	return "", "", fmt.Errorf("function not implemented")
}
