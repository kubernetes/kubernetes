/*
Copyright 2024 The Kubernetes Authors.

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

package main

import (
	"testing"

	"k8s.io/gengo/v2/types"
)

// gengo has `PointerTo()` but not the rest, so keep this here for consistency.
func ptrTo(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "*" + t.Name.String(),
		},
		Kind: types.Pointer,
		Elem: t,
	}
}

func sliceOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "[]" + t.Name.String(),
		},
		Kind: types.Slice,
		Elem: t,
	}
}

func mapOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "map[string]" + t.Name.String(),
		},
		Kind: types.Map,
		Key:  types.String,
		Elem: t,
	}
}

func aliasOf(name string, t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "Alias_" + name,
		},
		Kind:       types.Alias,
		Underlying: t,
	}
}

func TestGetLeafTypeAndPrefixes(t *testing.T) {

	cases := []struct {
		in              *types.Type
		expectedType    *types.Type
		expectedTypePfx string
		expectedExprPfx string
	}{{
		// string
		in:              types.String,
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// *string
		in:              ptrTo(types.String),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **string
		in:              ptrTo(ptrTo(types.String)),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***string
		in:              ptrTo(ptrTo(ptrTo(types.String))),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []string
		in:              sliceOf(types.String),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]string
		in:              ptrTo(sliceOf(types.String)),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]string
		in:              ptrTo(ptrTo(sliceOf(types.String))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]string
		in:              mapOf(types.String),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]string
		in:              ptrTo(mapOf(types.String)),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]string
		in:              ptrTo(ptrTo(mapOf(types.String))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]string
		in:              ptrTo(ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// alias of string
		in:              aliasOf("s", types.String),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// alias of *string
		in:              aliasOf("ps", ptrTo(types.String)),
		expectedType:    aliasOf("ps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **string
		in:              aliasOf("pps", ptrTo(ptrTo(types.String))),
		expectedType:    aliasOf("pps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***string
		in:              aliasOf("ppps", ptrTo(ptrTo(ptrTo(types.String)))),
		expectedType:    aliasOf("ppps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of []string
		in:              aliasOf("ls", sliceOf(types.String)),
		expectedType:    aliasOf("ls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *[]string
		in:              aliasOf("pls", ptrTo(sliceOf(types.String))),
		expectedType:    aliasOf("pls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **[]string
		in:              aliasOf("ppls", ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    aliasOf("ppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***[]string
		in:              aliasOf("pppls", ptrTo(ptrTo(ptrTo(sliceOf(types.String))))),
		expectedType:    aliasOf("pppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of map[string]string
		in:              aliasOf("ms", mapOf(types.String)),
		expectedType:    aliasOf("ms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *map[string]string
		in:              aliasOf("pms", ptrTo(mapOf(types.String))),
		expectedType:    aliasOf("pms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **map[string]string
		in:              aliasOf("ppms", ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    aliasOf("ppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***map[string]string
		in:              aliasOf("pppms", ptrTo(ptrTo(ptrTo(mapOf(types.String))))),
		expectedType:    aliasOf("pppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *alias-of-string
		in:              ptrTo(aliasOf("s", types.String)),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **alias-of-string
		in:              ptrTo(ptrTo(aliasOf("s", types.String))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***alias-of-string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("s", types.String)))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []alias-of-string
		in:              sliceOf(aliasOf("s", types.String)),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-string
		in:              ptrTo(sliceOf(aliasOf("s", types.String))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("s", types.String)))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("s", types.String))))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-string
		in:              mapOf(aliasOf("s", types.String)),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-string
		in:              ptrTo(mapOf(aliasOf("s", types.String))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-string
		in:              ptrTo(ptrTo(mapOf(aliasOf("s", types.String)))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("s", types.String))))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// *alias-of-*string
		in:              ptrTo(aliasOf("ps", ptrTo(types.String))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **alias-of-*string
		in:              ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// []alias-of-*string
		in:              sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-*string
		in:              ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-*string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-*string
		in:              mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-*string
		in:              ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-*string
		in:              ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}}

	for _, tc := range cases {
		leafType, typePfx, exprPfx := getLeafTypeAndPrefixes(tc.in)
		if got, want := leafType.Name.String(), tc.expectedType.Name.String(); got != want {
			t.Errorf("%q: wrong leaf type: expected %q, got %q", tc.in, want, got)
		}
		if got, want := typePfx, tc.expectedTypePfx; got != want {
			t.Errorf("%q: wrong type prefix: expected %q, got %q", tc.in, want, got)
		}
		if got, want := exprPfx, tc.expectedExprPfx; got != want {
			t.Errorf("%q: wrong expr prefix: expected %q, got %q", tc.in, want, got)
		}
	}
}
