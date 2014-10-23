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

	accessor.SetName("bar")
	accessor.SetUID("other")
	accessor.SetAPIVersion("c")
	accessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")

	// Prove that accessor changes the original object.
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

func TestGenericObjectMeta(t *testing.T) {
	type TypeMeta struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}
	type ObjectMeta struct {
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

	accessor.SetName("bar")
	accessor.SetUID("other")
	accessor.SetAPIVersion("c")
	accessor.SetKind("d")
	accessor.SetResourceVersion("2")
	accessor.SetSelfLink("google.com")

	// Prove that accessor changes the original object.
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
	versioning := NewResourceVersioner()
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

	linker := NewSelfLinker()
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
