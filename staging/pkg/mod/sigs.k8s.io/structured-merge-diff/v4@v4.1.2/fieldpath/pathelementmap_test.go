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

package fieldpath

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/value"
)

func TestPathElementValueMap(t *testing.T) {
	m := PathElementValueMap{}

	if _, ok := m.Get(PathElement{FieldName: strptr("onion")}); ok {
		t.Fatal("Unexpected path-element found in empty map")
	}

	m.Insert(PathElement{FieldName: strptr("carrot")}, value.NewValueInterface("knife"))
	m.Insert(PathElement{FieldName: strptr("chive")}, value.NewValueInterface(2))

	if _, ok := m.Get(PathElement{FieldName: strptr("onion")}); ok {
		t.Fatal("Unexpected path-element in map")
	}

	if val, ok := m.Get(PathElement{FieldName: strptr("carrot")}); !ok {
		t.Fatal("Missing path-element in map")
	} else if !value.Equals(val, value.NewValueInterface("knife")) {
		t.Fatalf("Unexpected value found: %#v", val)
	}

	if val, ok := m.Get(PathElement{FieldName: strptr("chive")}); !ok {
		t.Fatal("Missing path-element in map")
	} else if !value.Equals(val, value.NewValueInterface(2)) {
		t.Fatalf("Unexpected value found: %#v", val)
	}
}
