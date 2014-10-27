/*
Copyright 2014 Google Inc. All rights reserved.

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

package meta

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestGenericTypeMeta(t *testing.T) {
	type TypeMeta struct {
		Kind              string    `json:"kind,omitempty" yaml:"kind,omitempty"`
		Namespace         string    `json:"namespace,omitempty" yaml:"namespace,omitempty"`
		Name              string    `json:"name,omitempty" yaml:"name,omitempty"`
		UID               string    `json:"uid,omitempty" yaml:"uid,omitempty"`
		CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
		SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
		ResourceVersion   string    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
		APIVersion        string    `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}
	type Object struct {
		TypeMeta `json:",inline" yaml:",inline"`
	}
	j := Object{
		TypeMeta{
			Namespace:       "bar",
			Name:            "foo",
			UID:             "uid",
			APIVersion:      "a",
			Kind:            "b",
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
		},
	}
	accessor, err := Accessor(&j)
	if err != nil {
		t.Fatalf("new err: %v", err)
	}
	if e, a := "bar", accessor.Namespace(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "foo", accessor.Name(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "uid", accessor.UID(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "a", accessor.APIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", accessor.Kind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "1", accessor.ResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "some/place/only/we/know", accessor.SelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	accessor.SetNamespace("baz")
	accessor.SetName("bar")
	accessor.SetUID("other")
	accessor.SetAPIVersion("c")
	accessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")

	// Prove that accessor changes the original object.
	if e, a := "baz", j.Namespace; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "bar", j.Name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "other", j.UID; e != a {
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
}

type InternalTypeMeta struct {
	Kind              string    `json:"kind,omitempty" yaml:"kind,omitempty"`
	Namespace         string    `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Name              string    `json:"name,omitempty" yaml:"name,omitempty"`
	UID               string    `json:"uid,omitempty" yaml:"uid,omitempty"`
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
	SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
	ResourceVersion   string    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
	APIVersion        string    `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
}
type InternalObject struct {
	TypeMeta InternalTypeMeta `json:",inline" yaml:",inline"`
}

func (*InternalObject) IsAnAPIObject() {}

func TestGenericTypeMetaAccessor(t *testing.T) {
	j := &InternalObject{
		InternalTypeMeta{
			Namespace:       "bar",
			Name:            "foo",
			UID:             "uid",
			APIVersion:      "a",
			Kind:            "b",
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
		},
	}
	accessor := NewAccessor()
	namespace, err := accessor.Namespace(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "bar", namespace; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	name, err := accessor.Name(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "foo", name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	uid, err := accessor.UID(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "uid", uid; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	apiVersion, err := accessor.APIVersion(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "a", apiVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	kind, err := accessor.Kind(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "b", kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	rv, err := accessor.ResourceVersion(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "1", rv; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	selfLink, err := accessor.SelfLink(j)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if e, a := "some/place/only/we/know", selfLink; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := accessor.SetNamespace(j, "baz"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetName(j, "bar"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetUID(j, "other"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetAPIVersion(j, "c"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetKind(j, "d"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetResourceVersion(j, "2"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := accessor.SetSelfLink(j, "google.com"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Prove that accessor changes the original object.
	if e, a := "baz", j.TypeMeta.Namespace; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "bar", j.TypeMeta.Name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "other", j.TypeMeta.UID; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "c", j.TypeMeta.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "d", j.TypeMeta.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "2", j.TypeMeta.ResourceVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "google.com", j.TypeMeta.SelfLink; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGenericObjectMeta(t *testing.T) {
	type TypeMeta struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}
	type ObjectMeta struct {
		Namespace         string    `json:"namespace,omitempty" yaml:"namespace,omitempty"`
		Name              string    `json:"name,omitempty" yaml:"name,omitempty"`
		UID               string    `json:"uid,omitempty" yaml:"uid,omitempty"`
		CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
		SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
		ResourceVersion   string    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
	}
	type Object struct {
		TypeMeta   `json:",inline" yaml:",inline"`
		ObjectMeta `json:"metadata" yaml:"metadata"`
	}
	j := Object{
		TypeMeta{
			APIVersion: "a",
			Kind:       "b",
		},
		ObjectMeta{
			Namespace:       "bar",
			Name:            "foo",
			UID:             "uid",
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
		},
	}
	accessor, err := Accessor(&j)
	if err != nil {
		t.Fatalf("new err: %v", err)
	}
	if e, a := "bar", accessor.Namespace(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "foo", accessor.Name(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "uid", accessor.UID(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "a", accessor.APIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", accessor.Kind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "1", accessor.ResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "some/place/only/we/know", accessor.SelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	accessor.SetNamespace("baz")
	accessor.SetName("bar")
	accessor.SetUID("other")
	accessor.SetAPIVersion("c")
	accessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")

	// Prove that accessor changes the original object.
	if e, a := "baz", j.Namespace; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "bar", j.Name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "other", j.UID; e != a {
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
}

func TestGenericListMeta(t *testing.T) {
	type TypeMeta struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}
	type ListMeta struct {
		SelfLink        string `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
		ResourceVersion string `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
	}
	type Object struct {
		TypeMeta `json:",inline" yaml:",inline"`
		ListMeta `json:"metadata" yaml:"metadata"`
	}
	j := Object{
		TypeMeta{
			APIVersion: "a",
			Kind:       "b",
		},
		ListMeta{
			ResourceVersion: "1",
			SelfLink:        "some/place/only/we/know",
		},
	}
	accessor, err := Accessor(&j)
	if err != nil {
		t.Fatalf("new err: %v", err)
	}
	if e, a := "", accessor.Name(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "", accessor.UID(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "a", accessor.APIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", accessor.Kind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "1", accessor.ResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "some/place/only/we/know", accessor.SelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	accessor.SetName("bar")
	accessor.SetUID("other")
	accessor.SetAPIVersion("c")
	accessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")

	// Prove that accessor changes the original object.
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
}

type MyAPIObject struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`
}

func (*MyAPIObject) IsAnAPIObject() {}

type MyIncorrectlyMarkedAsAPIObject struct {
}

func (*MyIncorrectlyMarkedAsAPIObject) IsAnAPIObject() {}

func TestResourceVersionerOfAPI(t *testing.T) {
	type T struct {
		runtime.Object
		Expected string
	}
	testCases := map[string]T{
		"empty api object":                   {&MyAPIObject{}, ""},
		"api object with version":            {&MyAPIObject{TypeMeta: runtime.TypeMeta{ResourceVersion: "1"}}, "1"},
		"pointer to api object with version": {&MyAPIObject{TypeMeta: runtime.TypeMeta{ResourceVersion: "1"}}, "1"},
	}
	versioning := NewAccessor()
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
		"pointer to api object with version": {&MyAPIObject{TypeMeta: runtime.TypeMeta{ResourceVersion: "1"}}, "1"},
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
			obj:     &MyAPIObject{TypeMeta: runtime.TypeMeta{SelfLink: "foobar"}},
			expect:  "foobar",
			try:     "newbar",
			succeed: true,
		},
		"fail": {
			obj:     &MyIncorrectlyMarkedAsAPIObject{},
			succeed: false,
		},
	}

	var linker runtime.SelfLinker = NewAccessor()
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
