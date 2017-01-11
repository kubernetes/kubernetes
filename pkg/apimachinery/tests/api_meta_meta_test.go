/*
Copyright 2014 The Kubernetes Authors.

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

package tests

import (
	"reflect"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestAPIObjectMeta(t *testing.T) {
	j := &api.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "/a", Kind: "b"},
		ObjectMeta: api.ObjectMeta{
			Namespace:       "bar",
			Name:            "foo",
			GenerateName:    "prefix",
			UID:             "uid",
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
			Labels:          map[string]string{"foo": "bar"},
			Annotations:     map[string]string{"x": "y"},
			Finalizers: []string{
				"finalizer.1",
				"finalizer.2",
			},
		},
	}
	var _ meta.Object = &j.ObjectMeta
	var _ meta.ObjectMetaAccessor = j
	accessor, err := meta.Accessor(j)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if accessor != meta.Object(j) {
		t.Fatalf("should have returned the same pointer: %#v\n\n%#v", accessor, j)
	}
	if e, a := "bar", accessor.GetNamespace(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "foo", accessor.GetName(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "prefix", accessor.GetGenerateName(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "uid", string(accessor.GetUID()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "1", accessor.GetResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "some/place/only/we/know", accessor.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := []string{"finalizer.1", "finalizer.2"}, accessor.GetFinalizers(); !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	typeAccessor, err := meta.TypeAccessor(j)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := "a", typeAccessor.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", typeAccessor.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	accessor.SetNamespace("baz")
	accessor.SetName("bar")
	accessor.SetGenerateName("generate")
	accessor.SetUID("other")
	typeAccessor.SetAPIVersion("c")
	typeAccessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")
	accessor.SetFinalizers([]string{"finalizer.3"})

	// Prove that accessor changes the original object.
	if e, a := "baz", j.Namespace; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "bar", j.Name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "generate", j.GenerateName; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := types.UID("other"), j.UID; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "c", j.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "d", j.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "2", j.ResourceVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "google.com", j.SelfLink; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := []string{"finalizer.3"}, j.Finalizers; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	typeAccessor.SetAPIVersion("d")
	typeAccessor.SetKind("e")
	if e, a := "d", j.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "e", j.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGenericTypeMeta(t *testing.T) {
	type TypeMeta struct {
		Kind              string                  `json:"kind,omitempty"`
		Namespace         string                  `json:"namespace,omitempty"`
		Name              string                  `json:"name,omitempty"`
		GenerateName      string                  `json:"generateName,omitempty"`
		UID               string                  `json:"uid,omitempty"`
		CreationTimestamp metav1.Time             `json:"creationTimestamp,omitempty"`
		SelfLink          string                  `json:"selfLink,omitempty"`
		ResourceVersion   string                  `json:"resourceVersion,omitempty"`
		APIVersion        string                  `json:"apiVersion,omitempty"`
		Labels            map[string]string       `json:"labels,omitempty"`
		Annotations       map[string]string       `json:"annotations,omitempty"`
		OwnerReferences   []metav1.OwnerReference `json:"ownerReferences,omitempty"`
		Finalizers        []string                `json:"finalizers,omitempty"`
	}

	j := struct{ TypeMeta }{TypeMeta{APIVersion: "a", Kind: "b"}}

	typeAccessor, err := meta.TypeAccessor(&j)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := "a", typeAccessor.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", typeAccessor.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	typeAccessor.SetAPIVersion("c")
	typeAccessor.SetKind("d")

	if e, a := "c", j.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "d", j.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	typeAccessor.SetAPIVersion("d")
	typeAccessor.SetKind("e")

	if e, a := "d", j.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "e", j.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

type InternalTypeMeta struct {
	Kind              string                  `json:"kind,omitempty"`
	Namespace         string                  `json:"namespace,omitempty"`
	Name              string                  `json:"name,omitempty"`
	GenerateName      string                  `json:"generateName,omitempty"`
	UID               string                  `json:"uid,omitempty"`
	CreationTimestamp metav1.Time             `json:"creationTimestamp,omitempty"`
	SelfLink          string                  `json:"selfLink,omitempty"`
	ResourceVersion   string                  `json:"resourceVersion,omitempty"`
	APIVersion        string                  `json:"apiVersion,omitempty"`
	Labels            map[string]string       `json:"labels,omitempty"`
	Annotations       map[string]string       `json:"annotations,omitempty"`
	Finalizers        []string                `json:"finalizers,omitempty"`
	OwnerReferences   []metav1.OwnerReference `json:"ownerReferences,omitempty"`
}

func (m *InternalTypeMeta) GetResourceVersion() string   { return m.ResourceVersion }
func (m *InternalTypeMeta) SetResourceVersion(rv string) { m.ResourceVersion = rv }
func (m *InternalTypeMeta) GetSelfLink() string          { return m.SelfLink }
func (m *InternalTypeMeta) SetSelfLink(link string)      { m.SelfLink = link }

type MyAPIObject struct {
	TypeMeta InternalTypeMeta `json:",inline"`
}

func (obj *MyAPIObject) GetListMeta() metav1.List { return &obj.TypeMeta }

func (obj *MyAPIObject) GetObjectKind() schema.ObjectKind { return obj }
func (obj *MyAPIObject) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.TypeMeta.APIVersion, obj.TypeMeta.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *MyAPIObject) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.TypeMeta.APIVersion, obj.TypeMeta.Kind)
}

type MyIncorrectlyMarkedAsAPIObject struct{}

func (obj *MyIncorrectlyMarkedAsAPIObject) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}

func TestResourceVersionerOfAPI(t *testing.T) {
	type T struct {
		runtime.Object
		Expected string
	}
	testCases := map[string]T{
		"empty api object":                   {&MyAPIObject{}, ""},
		"api object with version":            {&MyAPIObject{TypeMeta: InternalTypeMeta{ResourceVersion: "1"}}, "1"},
		"pointer to api object with version": {&MyAPIObject{TypeMeta: InternalTypeMeta{ResourceVersion: "1"}}, "1"},
	}
	versioning := meta.NewAccessor()
	for key, testCase := range testCases {
		actual, err := versioning.ResourceVersion(testCase.Object)
		if err != nil {
			t.Errorf("%s: unexpected error %#v", key, err)
		}
		if actual != testCase.Expected {
			t.Errorf("%s: expected %v, got %v", key, testCase.Expected, actual)
		}
	}

	failingCases := map[string]struct {
		runtime.Object
		Expected string
	}{
		"not a valid object to try": {&MyIncorrectlyMarkedAsAPIObject{}, "1"},
	}
	for key, testCase := range failingCases {
		_, err := versioning.ResourceVersion(testCase.Object)
		if err == nil {
			t.Errorf("%s: expected error, got nil", key)
		}
	}

	setCases := map[string]struct {
		runtime.Object
		Expected string
	}{
		"pointer to api object with version": {&MyAPIObject{TypeMeta: InternalTypeMeta{ResourceVersion: "1"}}, "1"},
	}
	for key, testCase := range setCases {
		if err := versioning.SetResourceVersion(testCase.Object, "5"); err != nil {
			t.Errorf("%s: unexpected error %#v", key, err)
		}
		actual, err := versioning.ResourceVersion(testCase.Object)
		if err != nil {
			t.Errorf("%s: unexpected error %#v", key, err)
		}
		if actual != "5" {
			t.Errorf("%s: expected %v, got %v", key, "5", actual)
		}
	}
}

func TestTypeMetaSelfLinker(t *testing.T) {
	table := map[string]struct {
		obj     runtime.Object
		expect  string
		try     string
		succeed bool
	}{
		"normal": {
			obj:     &MyAPIObject{TypeMeta: InternalTypeMeta{SelfLink: "foobar"}},
			expect:  "foobar",
			try:     "newbar",
			succeed: true,
		},
		"fail": {
			obj:     &MyIncorrectlyMarkedAsAPIObject{},
			succeed: false,
		},
	}

	linker := runtime.SelfLinker(meta.NewAccessor())
	for name, item := range table {
		got, err := linker.SelfLink(item.obj)
		if e, a := item.succeed, err == nil; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}
		if e, a := item.expect, got; item.succeed && e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}

		err = linker.SetSelfLink(item.obj, item.try)
		if e, a := item.succeed, err == nil; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}
		if item.succeed {
			got, err := linker.SelfLink(item.obj)
			if err != nil {
				t.Errorf("%v: expected no err, got %v", name, err)
			}
			if e, a := item.try, got; e != a {
				t.Errorf("%v: expected %v, got %v", name, e, a)
			}
		}
	}
}

type MyAPIObject2 struct {
	metav1.TypeMeta
	v1.ObjectMeta
}

func getObjectMetaAndOwnerRefereneces() (myAPIObject2 MyAPIObject2, metaOwnerReferences []metav1.OwnerReference) {
	fuzz.New().NilChance(.5).NumElements(1, 5).Fuzz(&myAPIObject2)
	references := myAPIObject2.ObjectMeta.OwnerReferences
	// This is necessary for the test to pass because the getter will return a
	// non-nil slice.
	metaOwnerReferences = make([]metav1.OwnerReference, 0)
	for i := 0; i < len(references); i++ {
		metaOwnerReferences = append(metaOwnerReferences, metav1.OwnerReference{
			Kind:       references[i].Kind,
			Name:       references[i].Name,
			UID:        references[i].UID,
			APIVersion: references[i].APIVersion,
			Controller: references[i].Controller,
		})
	}
	if len(references) == 0 {
		// This is necessary for the test to pass because the setter will make a
		// non-nil slice.
		myAPIObject2.ObjectMeta.OwnerReferences = make([]metav1.OwnerReference, 0)
	}
	return myAPIObject2, metaOwnerReferences
}

func testGetOwnerReferences(t *testing.T) {
	obj, expected := getObjectMetaAndOwnerRefereneces()
	accessor, err := meta.Accessor(&obj)
	if err != nil {
		t.Error(err)
	}
	references := accessor.GetOwnerReferences()
	if !reflect.DeepEqual(references, expected) {
		t.Errorf("expect %#v\n got %#v", expected, references)
	}
}

func testSetOwnerReferences(t *testing.T) {
	expected, references := getObjectMetaAndOwnerRefereneces()
	obj := MyAPIObject2{}
	accessor, err := meta.Accessor(&obj)
	if err != nil {
		t.Error(err)
	}
	accessor.SetOwnerReferences(references)
	if e, a := expected.ObjectMeta.OwnerReferences, obj.ObjectMeta.OwnerReferences; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %#v\n got %#v", e, a)
	}
}

func TestAccessOwnerReferences(t *testing.T) {
	fuzzIter := 5
	for i := 0; i < fuzzIter; i++ {
		testGetOwnerReferences(t)
		testSetOwnerReferences(t)
	}
}

// BenchmarkAccessorSetFastPath shows the interface fast path
func BenchmarkAccessorSetFastPath(b *testing.B) {
	obj := &api.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "/a", Kind: "b"},
		ObjectMeta: api.ObjectMeta{
			Namespace:       "bar",
			Name:            "foo",
			GenerateName:    "prefix",
			UID:             "uid",
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
			Labels:          map[string]string{"foo": "bar"},
			Annotations:     map[string]string{"x": "y"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc, err := meta.Accessor(obj)
		if err != nil {
			b.Fatal(err)
		}
		acc.SetNamespace("something")
	}
	b.StopTimer()
}
