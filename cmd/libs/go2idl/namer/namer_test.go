/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package namer

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

func TestNameStrategy(t *testing.T) {
	u := types.Universe{}

	// Add some types.
	base := u.Get(types.Name{"foo/bar", "Baz"})
	base.Kind = types.Struct

	tmp := u.Get(types.Name{"", "[]bar.Baz"})
	tmp.Kind = types.Slice
	tmp.Elem = base

	tmp = u.Get(types.Name{"", "map[string]bar.Baz"})
	tmp.Kind = types.Map
	tmp.Key = types.String
	tmp.Elem = base

	tmp = u.Get(types.Name{"foo/other", "Baz"})
	tmp.Kind = types.Struct
	tmp.Members = []types.Member{{
		Embedded: true,
		Type:     base,
	}}

	u.Get(types.Name{"", "string"})

	o := Orderer{NewPublicNamer(0)}
	order := o.Order(u)
	orderedNames := make([]string, len(order))
	for i, t := range order {
		orderedNames[i] = o.Name(t)
	}
	expect := []string{"Baz", "Baz", "MapStringToBaz", "SliceBaz", "String"}
	if e, a := expect, orderedNames; !reflect.DeepEqual(e, a) {
		t.Errorf("Wanted %#v, got %#v", e, a)
	}

	o = Orderer{NewRawNamer("my/package", nil)}
	order = o.Order(u)
	orderedNames = make([]string, len(order))
	for i, t := range order {
		orderedNames[i] = o.Name(t)
	}

	expect = []string{"[]bar.Baz", "bar.Baz", "map[string]bar.Baz", "other.Baz", "string"}
	if e, a := expect, orderedNames; !reflect.DeepEqual(e, a) {
		t.Errorf("Wanted %#v, got %#v", e, a)
	}

	o = Orderer{NewRawNamer("foo/bar", nil)}
	order = o.Order(u)
	orderedNames = make([]string, len(order))
	for i, t := range order {
		orderedNames[i] = o.Name(t)
	}

	expect = []string{"Baz", "[]Baz", "map[string]Baz", "other.Baz", "string"}
	if e, a := expect, orderedNames; !reflect.DeepEqual(e, a) {
		t.Errorf("Wanted %#v, got %#v", e, a)
	}

	o = Orderer{NewPublicNamer(1)}
	order = o.Order(u)
	orderedNames = make([]string, len(order))
	for i, t := range order {
		orderedNames[i] = o.Name(t)
	}
	expect = []string{"BarBaz", "MapStringToBarBaz", "OtherBaz", "SliceBarBaz", "String"}
	if e, a := expect, orderedNames; !reflect.DeepEqual(e, a) {
		t.Errorf("Wanted %#v, got %#v", e, a)
	}
}
