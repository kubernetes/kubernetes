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

package integration

import (
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

var defaultingFixture = &apiextensionsv1beta1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "foos.tests.apiextensions.k8s.io"},
	Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
		Group:   "tests.apiextensions.k8s.io",
		Version: "v1beta1",
		Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
			Plural:   "foos",
			Singular: "foo",
			Kind:     "Foo",
			ListKind: "FooList",
		},
		Scope:                 apiextensionsv1beta1.ClusterScoped,
		PreserveUnknownFields: pointer.BoolPtr(false),
		Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
			Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
		},
	},
}

const defaultingFooSchema = `
type: object
properties:
  spec:
    type: object
    properties:
      a:
        type: string
        default: "A"
      b:
        type: string
        default: "B"
  status:
    type: object
    properties:
      a:
        type: string
        default: "A"
      b:
        type: string
        default: "B"
`

func TestCustomResourceDefaulting(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceDefaulting, true)()

	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := defaultingFixture.DeepCopy()
	crd.Spec.Validation = &apiextensionsv1beta1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(defaultingFooSchema), &crd.Spec.Validation.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	mustExist := func(obj map[string]interface{}, pths [][]string) {
		for _, pth := range pths {
			if _, found, _ := unstructured.NestedFieldNoCopy(obj, pth...); !found {
				t.Errorf("Expected '%s' field exist", strings.Join(pth, "."))
			}
		}
	}
	mustNotExist := func(obj map[string]interface{}, pths [][]string) {
		for _, pth := range pths {
			if fld, found, _ := unstructured.NestedFieldNoCopy(obj, pth...); found {
				t.Errorf("Expected '%s' field to not exist, but it does: %v", strings.Join(pth, "."), fld)
			}
		}
	}
	updateCRD := func(update func(*apiextensionsv1beta1.CustomResourceDefinition)) {
		var err error
		for retry := 0; retry < 10; retry++ {
			var obj *apiextensionsv1beta1.CustomResourceDefinition
			obj, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(crd.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			update(obj)
			obj, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(obj)
			if err != nil && apierrors.IsConflict(err) {
				continue
			} else if err != nil {
				t.Fatal(err)
			}
			crd = obj
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	addDefault := func(key string, value interface{}) {
		updateCRD(func(obj *apiextensionsv1beta1.CustomResourceDefinition) {
			for _, root := range []string{"spec", "status"} {
				obj.Spec.Validation.OpenAPIV3Schema.Properties[root].Properties[key] = apiextensionsv1beta1.JSONSchemaProps{
					Type:    "string",
					Default: jsonPtr(value),
				}
			}
		})
	}
	removeDefault := func(key string) {
		updateCRD(func(obj *apiextensionsv1beta1.CustomResourceDefinition) {
			for _, root := range []string{"spec", "status"} {
				props := obj.Spec.Validation.OpenAPIV3Schema.Properties[root].Properties[key]
				props.Default = nil
				obj.Spec.Validation.OpenAPIV3Schema.Properties[root].Properties[key] = props
			}
		})
	}

	t.Logf("Creating CR and expecting defaulted fields in spec, but status does not exist at all")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{crd.Spec.Group, crd.Spec.Version, crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(fooInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(foo.Object, "a", "spec", "a")
	unstructured.SetNestedField(foo.Object, "b", "status", "b")
	foo, err = fooClient.Create(foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}})
	mustNotExist(foo.Object, [][]string{{"status"}})

	t.Logf("Updating status and expecting 'a' and 'b' to show up.")
	unstructured.SetNestedField(foo.Object, map[string]interface{}{}, "status")
	if foo, err = fooClient.UpdateStatus(foo, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"status", "a"}, {"status", "b"}})

	t.Logf("Add 'c' default and wait until GET sees it in both status and spec")
	addDefault("c", "C")
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err := fooClient.Get(foo.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		_, found, _ := unstructured.NestedString(obj.Object, "spec", "c")
		foo = obj
		return found, nil
	}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})

	t.Logf("Updating status, expecting 'c' to be set in spec and status")
	if foo, err = fooClient.UpdateStatus(foo, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})

	t.Logf("Removing 'a', 'b' and `c` properties. Expecting that 'c' goes away in spec, but not in status. 'a' and 'b' were peristed.")
	removeDefault("a")
	removeDefault("b")
	removeDefault("c")
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err := fooClient.Get(foo.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		_, found, _ := unstructured.NestedString(obj.Object, "spec", "c")
		foo = obj
		return !found, nil
	}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})
	mustNotExist(foo.Object, [][]string{{"spec", "c"}})
}

func jsonPtr(x interface{}) *apiextensionsv1beta1.JSON {
	bs, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	ret := apiextensionsv1beta1.JSON{Raw: bs}
	return &ret
}
