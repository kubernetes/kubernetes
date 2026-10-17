/*
Copyright 2026 The Kubernetes Authors.

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

package protobuf

import (
	"reflect"
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestInterfaceMembersToAny(t *testing.T) {
	localGo := types.Name{Package: "example.com/foo", Name: "foo"}
	localProto := types.Name{Package: "example.com.foo", Path: "example.com/foo"}

	emptyInterface := &types.Type{Name: types.Name{Name: "interface{}"}, Kind: types.Interface}
	namedInterface := &types.Type{
		Name:    types.Name{Package: "example.com/foo", Name: "Shape"},
		Kind:    types.Interface,
		Methods: map[string]*types.Type{"Area": {Kind: types.Func}},
	}
	anyAlias := &types.Type{Name: types.Name{Name: "any"}, Kind: types.Alias, Underlying: emptyInterface}

	parent := &types.Type{
		Name: types.Name{Package: "example.com/foo", Name: "Holder"},
		Kind: types.Struct,
		Members: []types.Member{
			{Name: "Empty", Type: emptyInterface, Tags: `json:"empty"`},
			{Name: "Named", Type: namedInterface, Tags: `json:"named"`},
			{Name: "Aliased", Type: anyAlias, Tags: `json:"aliased"`},
			{Name: "List", Type: &types.Type{Name: types.Name{Name: "[]interface{}"}, Kind: types.Slice, Elem: emptyInterface}, Tags: `json:"list"`},
		},
	}

	tracker := NewImportTracker(localProto)
	locator := &protobufLocator{
		namer:          NewProtobufNamer(),
		tracker:        tracker,
		universe:       types.Universe{},
		localGoPackage: localGo.Package,
	}

	fields, err := membersToFields(locator, parent, localProto, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(fields) != 4 {
		t.Fatalf("expected 4 fields, got %d", len(fields))
	}

	ln := localNamer{localPackage: localProto}
	for _, f := range fields {
		if got := ln.Name(f.Type); got != ".google.protobuf.Any" {
			t.Errorf("field %q: expected type .google.protobuf.Any, got %q", f.Name, got)
		}
		if !f.Nullable {
			t.Errorf("field %q: interfaces must be nullable", f.Name)
		}
		// casttype/nullable=false on a message field are rejected by gogo.
		for _, k := range []string{"(gogoproto.casttype)", "(gogoproto.nullable)"} {
			if v, ok := f.Extras[k]; ok {
				t.Errorf("field %q: unexpected extra %s = %s", f.Name, k, v)
			}
		}
	}
	if !fields[3].Repeated {
		t.Errorf("field %q: expected repeated", fields[3].Name)
	}

	if imports := tracker.ImportLines(); !reflect.DeepEqual(imports, []string{"google/protobuf/any.proto"}) {
		t.Errorf("expected only any.proto import, got %v", imports)
	}
}

func TestInterfaceDeclarationDoesNotImportAny(t *testing.T) {
	p := newProtobufPackage("example.com/foo", "", "example.com.foo", true, nil)
	p.Imports = NewImportTracker(p.ProtoTypeName())

	iface := &types.Type{Name: types.Name{Package: "example.com/foo", Name: "Shape"}, Kind: types.Interface}
	local, global := typeNameSet{}, typeNameSet{}
	assignGoTypeToProtoPackage(p, iface, local, global, map[types.Name]struct{}{})

	if imports := p.Imports.ImportLines(); len(imports) != 0 {
		t.Errorf("expected no imports for an unreferenced interface, got %v", imports)
	}
	if _, ok := local[iface.Name]; ok {
		t.Errorf("interface %v should not be assigned to the proto package", iface.Name)
	}
}
