/*
Copyright 2015 The Kubernetes Authors.

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

package types

import (
	"reflect"
	"testing"
)

func TestFlatten(t *testing.T) {
	mapType := &Type{
		Name: Name{Package: "", Name: "map[string]string"},
		Kind: Map,
		Key:  String,
		Elem: String,
	}
	m := []Member{
		{
			Name:     "Baz",
			Embedded: true,
			Type: &Type{
				Name: Name{Package: "pkg", Name: "Baz"},
				Kind: Struct,
				Members: []Member{
					{Name: "Foo", Type: String},
					{
						Name:     "Qux",
						Embedded: true,
						Type: &Type{
							Name:    Name{Package: "pkg", Name: "Qux"},
							Kind:    Struct,
							Members: []Member{{Name: "Zot", Type: String}},
						},
					},
				},
			},
		},
		{Name: "Bar", Type: String},
		{
			Name:     "NotSureIfLegal",
			Embedded: true,
			Type:     mapType,
		},
	}
	e := []Member{
		{Name: "Bar", Type: String},
		{Name: "NotSureIfLegal", Type: mapType, Embedded: true},
		{Name: "Foo", Type: String},
		{Name: "Zot", Type: String},
	}
	if a := FlattenMembers(m); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected \n%#v\n, got \n%#v\n", e, a)
	}
}
