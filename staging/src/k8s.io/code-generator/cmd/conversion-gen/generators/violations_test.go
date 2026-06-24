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

package generators

import (
	"slices"
	"strings"
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestDivergences(t *testing.T) {
	skippedA := object("a", "N", field("X", types.Int32))
	skippedB := object("b", "N", field("X", types.Int32))

	cases := []struct {
		name       string
		a, b       *types.Type
		skip       [2]*types.Type // conversion type pairs to skip, for manual conversion tests
		wantPaths  []string       // sorted
		wantDetail string         // contains check
	}{
		{
			name: "identical",
			a:    object("a", "T", field("X", types.Int32)),
			b:    object("b", "T", field("X", types.Int32)),
		},
		{
			name:      "builtin field change",
			a:         object("a", "T", field("X", types.Int64)),
			b:         object("b", "T", field("X", types.Int32)),
			wantPaths: []string{"X"},
		},
		{
			name:      "nested slice of struct field",
			a:         object("a", "T", field("Items", sliceOf(object("a", "I", field("Y", types.Int64))))),
			b:         object("b", "T", field("Items", sliceOf(object("b", "I", field("Y", types.Int32))))),
			wantPaths: []string{"Items[*].Y"},
		},
		{
			name:      "map value divergence",
			a:         object("a", "T", field("M", mapOf(types.String, types.Int64))),
			b:         object("b", "T", field("M", mapOf(types.String, types.Int32))),
			wantPaths: []string{"M[value]"},
		},
		{
			name:      "pointer is transparent",
			a:         object("a", "T", field("P", ptrTo(types.Int64))),
			b:         object("b", "T", field("P", ptrTo(types.Int32))),
			wantPaths: []string{"P"},
		},
		{
			name:      "extra and missing fields both reported",
			a:         object("a", "T", field("X", types.String), field("Extra", types.String)),
			b:         object("b", "T", field("X", types.String), field("Missing", types.String)),
			wantPaths: []string{"Extra", "Missing"},
		},
		{
			name:      "same field set, reordered, reports order mismatch only",
			a:         object("a", "T", field("X", types.Int32), field("Y", types.Int64)),
			b:         object("b", "T", field("Y", types.Int64), field("X", types.Int32)),
			wantPaths: []string{""},
		},
		{
			name:       "array is unsupported",
			a:          object("a", "T", field("IP", arrayOf(16, types.Byte))),
			b:          object("b", "T", field("IP", arrayOf(16, types.Byte))),
			wantPaths:  []string{"IP"},
			wantDetail: "not supported",
		},
		{
			name:       "manual conversion on nested type",
			a:          object("a", "T", field("N", skippedA)),
			b:          object("b", "T", field("N", skippedB)),
			skip:       [2]*types.Type{skippedA, skippedB},
			wantPaths:  []string{"N"},
			wantDetail: "manual conversion",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			eq := equalMemoryTypes{}
			if tc.skip[0] != nil {
				eq.Skip(tc.skip[0], tc.skip[1])
			}
			got := eq.Divergences(tc.a, tc.b)

			var gotPaths []string
			for _, d := range got {
				gotPaths = append(gotPaths, d.Path)
			}
			if strings.Join(gotPaths, "|") != strings.Join(tc.wantPaths, "|") {
				t.Errorf("paths = %v, want %v (full: %+v)", gotPaths, tc.wantPaths, got)
			}
			if tc.wantDetail != "" && !slices.ContainsFunc(got, func(d Divergence) bool { return strings.Contains(d.Detail, tc.wantDetail) }) {
				t.Errorf("no divergence Detail contains %q; got %+v", tc.wantDetail, got)
			}

			if (len(got) == 0) != eq.Equal(tc.a, tc.b) {
				t.Errorf("invariant broken: Divergences-empty=%v but Equal=%v", len(got) == 0, eq.Equal(tc.a, tc.b))
			}
		})
	}
}

func object(pkg, name string, members ...types.Member) *types.Type {
	return &types.Type{Name: types.Name{Package: pkg, Name: name}, Kind: types.Struct, Members: members}
}
func field(name string, t *types.Type) types.Member { return types.Member{Name: name, Type: t} }
func ptrTo(t *types.Type) *types.Type {
	return &types.Type{Name: types.Name{Name: "*" + t.Name.Name}, Kind: types.Pointer, Elem: t}
}
func sliceOf(t *types.Type) *types.Type {
	return &types.Type{Name: types.Name{Name: "[]" + t.Name.Name}, Kind: types.Slice, Elem: t}
}
func mapOf(k, v *types.Type) *types.Type {
	return &types.Type{Name: types.Name{Name: "map"}, Kind: types.Map, Key: k, Elem: v}
}
func arrayOf(n int64, t *types.Type) *types.Type {
	return &types.Type{Name: types.Name{Name: "array"}, Kind: types.Array, Len: n, Elem: t}
}
