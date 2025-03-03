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

func TestGetLeafTypeAndPrefixes(t *testing.T) {
	stringType := &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "string",
		},
		Kind: types.Builtin,
	}

	ptrTo := func(t *types.Type) *types.Type {
		return &types.Type{
			Name: types.Name{
				Package: "",
				Name:    "*" + t.Name.String(),
			},
			Kind: types.Pointer,
			Elem: t,
		}
	}

	sliceOf := func(t *types.Type) *types.Type {
		return &types.Type{
			Name: types.Name{
				Package: "",
				Name:    "[]" + t.Name.String(),
			},
			Kind: types.Slice,
			Elem: t,
		}
	}

	mapOf := func(t *types.Type) *types.Type {
		return &types.Type{
			Name: types.Name{
				Package: "",
				Name:    "map[string]" + t.Name.String(),
			},
			Kind: types.Map,
			Key:  stringType,
			Elem: t,
		}
	}

	aliasOf := func(name string, t *types.Type) *types.Type {
		return &types.Type{
			Name: types.Name{
				Package: "",
				Name:    "Alias_" + name,
			},
			Kind:       types.Alias,
			Underlying: t,
		}
	}

	cases := []struct {
		in              *types.Type
		expectedType    *types.Type
		expectedTypePfx string
		expectedExprPfx string
	}{{
		// string
		in:              stringType,
		expectedType:    stringType,
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// *string
		in:              ptrTo(stringType),
		expectedType:    stringType,
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **string
		in:              ptrTo(ptrTo(stringType)),
		expectedType:    stringType,
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***string
		in:              ptrTo(ptrTo(ptrTo(stringType))),
		expectedType:    stringType,
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []string
		in:              sliceOf(stringType),
		expectedType:    sliceOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]string
		in:              ptrTo(sliceOf(stringType)),
		expectedType:    sliceOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]string
		in:              ptrTo(ptrTo(sliceOf(stringType))),
		expectedType:    sliceOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(stringType)))),
		expectedType:    sliceOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]string
		in:              mapOf(stringType),
		expectedType:    mapOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]string
		in:              ptrTo(mapOf(stringType)),
		expectedType:    mapOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]string
		in:              ptrTo(ptrTo(mapOf(stringType))),
		expectedType:    mapOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]string
		in:              ptrTo(ptrTo(ptrTo(mapOf(stringType)))),
		expectedType:    mapOf(stringType),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// alias of string
		in:              aliasOf("s", stringType),
		expectedType:    aliasOf("s", stringType),
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// alias of *string
		in:              aliasOf("ps", ptrTo(stringType)),
		expectedType:    aliasOf("ps", stringType),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **string
		in:              aliasOf("pps", ptrTo(ptrTo(stringType))),
		expectedType:    aliasOf("pps", stringType),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***string
		in:              aliasOf("ppps", ptrTo(ptrTo(ptrTo(stringType)))),
		expectedType:    aliasOf("ppps", stringType),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of []string
		in:              aliasOf("ls", sliceOf(stringType)),
		expectedType:    aliasOf("ls", sliceOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *[]string
		in:              aliasOf("pls", ptrTo(sliceOf(stringType))),
		expectedType:    aliasOf("pls", sliceOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **[]string
		in:              aliasOf("ppls", ptrTo(ptrTo(sliceOf(stringType)))),
		expectedType:    aliasOf("ppls", sliceOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***[]string
		in:              aliasOf("pppls", ptrTo(ptrTo(ptrTo(sliceOf(stringType))))),
		expectedType:    aliasOf("pppls", sliceOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of map[string]string
		in:              aliasOf("ms", mapOf(stringType)),
		expectedType:    aliasOf("ms", mapOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *map[string]string
		in:              aliasOf("pms", ptrTo(mapOf(stringType))),
		expectedType:    aliasOf("pms", mapOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **map[string]string
		in:              aliasOf("ppms", ptrTo(ptrTo(mapOf(stringType)))),
		expectedType:    aliasOf("ppms", mapOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***map[string]string
		in:              aliasOf("pppms", ptrTo(ptrTo(ptrTo(mapOf(stringType))))),
		expectedType:    aliasOf("pppms", mapOf(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *alias-of-string
		in:              ptrTo(aliasOf("s", stringType)),
		expectedType:    aliasOf("s", stringType),
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **alias-of-string
		in:              ptrTo(ptrTo(aliasOf("s", stringType))),
		expectedType:    aliasOf("s", stringType),
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***alias-of-string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("s", stringType)))),
		expectedType:    aliasOf("s", stringType),
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []alias-of-string
		in:              sliceOf(aliasOf("s", stringType)),
		expectedType:    sliceOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-string
		in:              ptrTo(sliceOf(aliasOf("s", stringType))),
		expectedType:    sliceOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("s", stringType)))),
		expectedType:    sliceOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("s", stringType))))),
		expectedType:    sliceOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-string
		in:              mapOf(aliasOf("s", stringType)),
		expectedType:    mapOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-string
		in:              ptrTo(mapOf(aliasOf("s", stringType))),
		expectedType:    mapOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-string
		in:              ptrTo(ptrTo(mapOf(aliasOf("s", stringType)))),
		expectedType:    mapOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("s", stringType))))),
		expectedType:    mapOf(aliasOf("s", stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// *alias-of-*string
		in:              ptrTo(aliasOf("ps", ptrTo(stringType))),
		expectedType:    aliasOf("ps", ptrTo(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **alias-of-*string
		in:              ptrTo(ptrTo(aliasOf("ps", ptrTo(stringType)))),
		expectedType:    aliasOf("ps", ptrTo(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("ps", ptrTo(stringType))))),
		expectedType:    aliasOf("ps", ptrTo(stringType)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// []alias-of-*string
		in:              sliceOf(aliasOf("ps", ptrTo(stringType))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-*string
		in:              ptrTo(sliceOf(aliasOf("ps", ptrTo(stringType)))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-*string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(stringType))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(stringType)))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-*string
		in:              mapOf(aliasOf("ps", ptrTo(stringType))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-*string
		in:              ptrTo(mapOf(aliasOf("ps", ptrTo(stringType)))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-*string
		in:              ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(stringType))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(stringType))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(stringType)))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(stringType))),
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
