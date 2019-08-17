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
	"fmt"
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
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/component-base/featuregate/testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

var defaultingFixture = &apiextensionsv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "foos.tests.example.com"},
	Spec: apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "tests.example.com",
		Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1beta1",
				Storage: false,
				Served:  true,
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
				},
			},
			{
				Name:    "v1beta2",
				Storage: true,
				Served:  false,
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
				},
			},
		},
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural:   "foos",
			Singular: "foo",
			Kind:     "Foo",
			ListKind: "FooList",
		},
		Scope:                 apiextensionsv1.ClusterScoped,
		PreserveUnknownFields: false,
	},
}

const defaultingFooV1beta1Schema = `
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
      c:
        type: string
      v1beta1:
        type: string
        default: "v1beta1"
      v1beta2:
        type: string
  status:
    type: object
    properties:
      a:
        type: string
        default: "A"
      b:
        type: string
        default: "B"
      c:
        type: string
      v1beta1:
        type: string
        default: "v1beta1"
      v1beta2:
        type: string
`

const defaultingFooV1beta2Schema = `
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
      c:
        type: string
      v1beta1:
        type: string
      v1beta2:
        type: string
        default: "v1beta2"
  status:
    type: object
    properties:
      a:
        type: string
        default: "A"
      b:
        type: string
        default: "B"
      c:
        type: string
      v1beta1:
        type: string
      v1beta2:
        type: string
        default: "v1beta2"
`

func TestCustomResourceDefaultingWithWatchCache(t *testing.T) {
	testDefaulting(t, true)
}

func TestCustomResourceDefaultingWithoutWatchCache(t *testing.T) {
	testDefaulting(t, false)
}

func testDefaulting(t *testing.T, watchCache bool) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceDefaulting, true)()

	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t, fmt.Sprintf("--watch-cache=%v", watchCache))
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := defaultingFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(defaultingFooV1beta1Schema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}
	crd.Spec.Versions[1].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(defaultingFooV1beta2Schema), &crd.Spec.Versions[1].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	mustExist := func(obj map[string]interface{}, pths [][]string) {
		t.Helper()
		for _, pth := range pths {
			if _, found, _ := unstructured.NestedFieldNoCopy(obj, pth...); !found {
				t.Errorf("Expected '%s' field was missing", strings.Join(pth, "."))
			}
		}
	}
	mustNotExist := func(obj map[string]interface{}, pths [][]string) {
		t.Helper()
		for _, pth := range pths {
			if fld, found, _ := unstructured.NestedFieldNoCopy(obj, pth...); found {
				t.Errorf("Expected '%s' field to not exist, but it does: %v", strings.Join(pth, "."), fld)
			}
		}
	}
	updateCRD := func(update func(*apiextensionsv1.CustomResourceDefinition)) {
		t.Helper()
		var err error
		for retry := 0; retry < 10; retry++ {
			var obj *apiextensionsv1.CustomResourceDefinition
			obj, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(crd.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			update(obj)
			obj, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Update(obj)
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
	addDefault := func(version string, key string, value interface{}) {
		t.Helper()
		updateCRD(func(obj *apiextensionsv1.CustomResourceDefinition) {
			for _, root := range []string{"spec", "status"} {
				for i := range obj.Spec.Versions {
					if obj.Spec.Versions[i].Name != version {
						continue
					}
					obj.Spec.Versions[i].Schema.OpenAPIV3Schema.Properties[root].Properties[key] = apiextensionsv1.JSONSchemaProps{
						Type:    "string",
						Default: jsonPtr(value),
					}
				}
			}
		})
	}
	removeDefault := func(version string, key string) {
		t.Helper()
		updateCRD(func(obj *apiextensionsv1.CustomResourceDefinition) {
			for _, root := range []string{"spec", "status"} {
				for i := range obj.Spec.Versions {
					if obj.Spec.Versions[i].Name != version {
						continue
					}
					props := obj.Spec.Versions[i].Schema.OpenAPIV3Schema.Properties[root].Properties[key]
					props.Default = nil
					obj.Spec.Versions[i].Schema.OpenAPIV3Schema.Properties[root].Properties[key] = props
				}
			}
		})
	}

	t.Logf("Creating CR and expecting defaulted fields in spec, but status does not exist at all")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{crd.Spec.Group, crd.Spec.Versions[0].Name, crd.Spec.Names.Plural})
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
	initialResourceVersion := foo.GetResourceVersion()
	t.Logf("CR created: %#v", foo.UnstructuredContent())
	// spec.a and spec.b are defaulted in both versions
	// spec.v1beta1 is defaulted when reading the incoming request
	// spec.v1beta2 is defaulted when reading the storage response
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "v1beta1"}, {"spec", "v1beta2"}})
	mustNotExist(foo.Object, [][]string{{"status"}})

	t.Logf("Updating status and expecting 'a' and 'b' to show up.")
	unstructured.SetNestedField(foo.Object, map[string]interface{}{}, "status")
	if foo, err = fooClient.UpdateStatus(foo, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"status", "a"}, {"status", "b"}})

	t.Logf("Add 'c' default to the storage version and wait until GET sees it in both status and spec")
	addDefault("v1beta2", "c", "C")

	t.Logf("wait until GET sees 'c' in both status and spec")
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err := fooClient.Get(foo.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if _, found, _ := unstructured.NestedString(obj.Object, "spec", "c"); !found {
			t.Log("will retry, did not find spec.c in the object")
			return false, nil
		}
		foo = obj
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})

	t.Logf("wait until GET sees 'c' in both status and spec of cached get")
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err := fooClient.Get(foo.GetName(), metav1.GetOptions{ResourceVersion: "0"})
		if err != nil {
			return false, err
		}
		if _, found, _ := unstructured.NestedString(obj.Object, "spec", "c"); !found {
			t.Logf("will retry, did not find spec.c in the cached object")
			return false, nil
		}
		foo = obj
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})

	t.Logf("verify LIST sees 'c' in both status and spec")
	foos, err := fooClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, foo := range foos.Items {
		mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})
	}

	t.Logf("verify LIST from cache sees 'c' in both status and spec")
	foos, err = fooClient.List(metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatal(err)
	}
	for _, foo := range foos.Items {
		mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})
	}

	// Omit this test when using the watch cache because changing the CRD resets the watch cache's minimum available resource version.
	// The watch cache is populated by list and watch, which are both tested by this test.
	// The contents of the watch cache are seen by list with rv=0, which is tested by this test.
	if !watchCache {
		t.Logf("verify WATCH sees 'c' in both status and spec")
		w, err := fooClient.Watch(metav1.ListOptions{ResourceVersion: initialResourceVersion})
		if err != nil {
			t.Fatal(err)
		}
		select {
		case event := <-w.ResultChan():
			if event.Type != watch.Modified {
				t.Fatalf("unexpected watch event: %v, %#v", event.Type, event.Object)
			}
			if e, a := "v1beta1", event.Object.GetObjectKind().GroupVersionKind().Version; e != a {
				t.Errorf("watch event for v1beta1 API returned %v", a)
			}
			mustExist(
				event.Object.(*unstructured.Unstructured).Object,
				[][]string{{"spec", "a"}, {"spec", "b"}, {"spec", "c"}, {"status", "a"}, {"status", "b"}, {"status", "c"}},
			)
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatal("timed out without getting watch event")
		}
	}

	t.Logf("Add 'c' default to the REST version, remove it from the storage version, and wait until GET no longer sees it in both status and spec")
	addDefault("v1beta1", "c", "C")
	removeDefault("v1beta2", "c")
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
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"status", "a"}, {"status", "b"}})
	mustNotExist(foo.Object, [][]string{{"spec", "c"}, {"status", "c"}})

	t.Logf("Updating status, expecting 'c' to be set in status only")
	if foo, err = fooClient.UpdateStatus(foo, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	mustExist(foo.Object, [][]string{{"spec", "a"}, {"spec", "b"}, {"status", "a"}, {"status", "b"}, {"status", "c"}})
	mustNotExist(foo.Object, [][]string{{"spec", "c"}})

	t.Logf("Removing 'a', 'b' and `c` properties from the REST version. Expecting that 'c' goes away in spec, but not in status. 'a' and 'b' were presisted.")
	removeDefault("v1beta1", "a")
	removeDefault("v1beta1", "b")
	removeDefault("v1beta1", "c")
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

func jsonPtr(x interface{}) *apiextensionsv1.JSON {
	bs, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	ret := apiextensionsv1.JSON{Raw: bs}
	return &ret
}
