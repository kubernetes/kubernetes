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

package integration

import (
	"context"
	"fmt"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
)

func TestApplyBasic(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name

	rest := apiExtensionClient.Discovery().RESTClient()
	yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: mytest
values:
  numVal: 1
  boolVal: true
  stringVal: "1"`, apiVersion, kind))
	result, err := rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name("mytest").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatal(err, string(result))
	}

	result, err = rest.Patch(types.MergePatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name("mytest").
		Body([]byte(`{"values":{"numVal": 5}}`)).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatal(err, string(result))
	}

	// Re-apply the same object, we should get conflicts now.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name("mytest").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err == nil {
		t.Fatalf("Expecting to get conflicts when applying object, got no error: %s", result)
	}
	status, ok := err.(*errors.StatusError)
	if !ok {
		t.Fatalf("Expecting to get conflicts as API error")
	}
	if len(status.Status().Details.Causes) < 1 {
		t.Fatalf("Expecting to get at least one conflict when applying object, got: %v", status.Status().Details.Causes)
	}

	// Re-apply with force, should work fine.
	result, err = rest.Patch(types.ApplyPatchType).
		AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
		Name("mytest").
		Param("force", "true").
		Param("fieldManager", "apply_test").
		Body(yamlBody).
		DoRaw(context.TODO())
	if err != nil {
		t.Fatal(err, string(result))
	}

}

// TestApplyNullToObject ensures that when maps and slices are set to null using
// SSA, that the resulting field state is correct.
func TestApplyNullToObject(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	testCases := []struct {
		name             string
		nullable         bool
		required         bool
		subfieldRequired bool

		wantNull bool
		wantErr  string
	}{
		// This behavior is fussy, so we test a wide range of combinations.
		{name: "nullable and optional field with optional subfield", nullable: true, wantNull: true},
		{name: "nullable and required field with optional subfield", nullable: true, required: true, wantNull: true},
		{name: "nullable and optional field with required subfield", nullable: true, subfieldRequired: true, wantNull: true},
		{name: "nullable and required field with required subfield", nullable: true, subfieldRequired: true, required: true, wantNull: true},

		// Applying null to a non-nullable field is rejected by validation. To clear
		// a non-nullable field, it must be omitted from the apply request instead.
		{name: "non-nullable and optional field with optional subfield", wantErr: "must be of type object"},
		{name: "non-nullable and required field with optional subfield", required: true, wantErr: "must be of type object"},
		{name: "non-nullable and optional field with required subfield", subfieldRequired: true, wantErr: "must be of type object"},
		{name: "non-nullable and required field with required subfield", subfieldRequired: true, required: true, wantErr: "must be of type object"},
	}

	group, version, kind, plural := "stable.example.com", "v1", "Widget", "widgets"
	apiVersion := group + "/" + version
	gvr := schema.GroupVersionResource{Group: group, Version: version, Resource: plural}

	// Way more efficient to test if we build a single CRD to handle all the test cases.
	fieldName := func(i int) string { return fmt.Sprintf("field%d", i) }
	specProps := map[string]apiextensionsv1.JSONSchemaProps{}
	for i, tc := range testCases {
		wrapper := apiextensionsv1.JSONSchemaProps{
			Type:       "object",
			Properties: map[string]apiextensionsv1.JSONSchemaProps{"inner": mkObjectSchema(tc.nullable, tc.subfieldRequired)},
		}
		if tc.required {
			wrapper.Required = []string{"inner"}
		}
		specProps[fieldName(i)] = wrapper
	}

	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: plural + "." + group},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: group,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    version,
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{"spec": {Type: "object", Properties: specProps}},
					},
				},
			}},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   plural,
				Kind:     kind,
				ListKind: kind + "List",
			},
			Scope: apiextensionsv1.ClusterScoped,
		},
	}
	if _, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient); err != nil {
		t.Fatal(err)
	}

	apply := func(object, field string, inner interface{}) (*unstructured.Unstructured, error) {
		obj := &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata":   map[string]interface{}{"name": object},
			"spec":       map[string]interface{}{field: map[string]interface{}{"inner": inner}},
		}}
		return dynamicClient.Resource(gvr).Apply(context.TODO(), object, obj, metav1.ApplyOptions{FieldManager: "apply_test"})
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			object, field := fieldName(i), fieldName(i)
			if _, err := apply(object, field, map[string]interface{}{"a": "1", "b": "2"}); err != nil {
				t.Fatalf("populating apply failed: %v", err)
			}
			got, err := apply(object, field, nil)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("want apply to be rejected with %q, but it succeeded", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("want apply error to contain %q, got: %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("clearing apply was rejected: %v", err)
			}

			inner, _, err := unstructured.NestedFieldNoCopy(got.Object, "spec", field, "inner")
			if err != nil {
				t.Fatalf("reading spec.%s.inner: %v", field, err)
			}
			if tc.wantNull && inner != nil {
				t.Errorf("want inner to be null, got %#v", inner)
			}
		})
	}
}

func mkObjectSchema(nullable, subfieldRequired bool) apiextensionsv1.JSONSchemaProps {
	s := apiextensionsv1.JSONSchemaProps{
		Type:     "object",
		Nullable: nullable,
		Properties: map[string]apiextensionsv1.JSONSchemaProps{
			"a": {Type: "string"},
			"b": {Type: "string"},
		},
	}
	if subfieldRequired {
		s.Required = []string{"a", "b"}
	}
	return s
}
