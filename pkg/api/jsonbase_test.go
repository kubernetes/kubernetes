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
		APIVersion:      "a",
		Kind:            "b",
		ResourceVersion: 1,
	}
	g, err := newGenericJSONBase(reflect.ValueOf(&j).Elem())
	if err != nil {
		t.Fatalf("new err: %v", err)
	}
	// Proove g supports JSONBaseInterface.
	jbi := JSONBaseInterface(g)
	if e, a := "a", jbi.APIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := "b", jbi.Kind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := uint64(1), jbi.ResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	jbi.SetAPIVersion("c")
	jbi.SetKind("d")
	jbi.SetResourceVersion(2)

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
