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

package api

import (
	"reflect"
	"testing"
)

func TestGenericJSONBase(t *testing.T) {
	j := JSONBase{
		ID:              "foo",
		APIVersion:      "a",
		Kind:            "b",
		ResourceVersion: 1,
	}
	g, err := newGenericJSONBase(reflect.ValueOf(&j).Elem())
	if err != nil {
		t.Fatalf("new err: %v", err)
	}
	// Prove g supports JSONBaseInterface.
	jbi := JSONBaseInterface(g)
	if e, a := "foo", jbi.ID(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "a", jbi.APIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", jbi.Kind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := uint64(1), jbi.ResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	jbi.SetID("bar")
	jbi.SetAPIVersion("c")
	jbi.SetKind("d")
	jbi.SetResourceVersion(2)

	// Prove that jbi changes the original object.
	if e, a := "bar", j.ID; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "c", j.APIVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "d", j.Kind; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := uint64(2), j.ResourceVersion; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestResourceVersionerOfAPI(t *testing.T) {
	type T struct {
		Object   interface{}
		Expected uint64
	}
	testCases := map[string]T{
		"empty api object":                   {Service{}, 0},
		"api object with version":            {Service{JSONBase: JSONBase{ResourceVersion: 1}}, 1},
		"pointer to api object with version": {&Service{JSONBase: JSONBase{ResourceVersion: 1}}, 1},
	}
	versioning := NewJSONBaseResourceVersioner()
	for key, testCase := range testCases {
		actual, err := versioning.ResourceVersion(testCase.Object)
		if err != nil {
			t.Errorf("%s: unexpected error %#v", key, err)
		}
		if actual != testCase.Expected {
			t.Errorf("%s: expected %d, got %d", key, testCase.Expected, actual)
		}
	}

	failingCases := map[string]T{
		"not a valid object to try": {JSONBase{ResourceVersion: 1}, 1},
	}
	for key, testCase := range failingCases {
		_, err := versioning.ResourceVersion(testCase.Object)
		if err == nil {
			t.Errorf("%s: expected error, got nil", key)
		}
	}
}
