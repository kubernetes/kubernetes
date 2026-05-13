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

package runtime

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestFmtRawDoc(t *testing.T) {
	tests := []struct {
		t, expected string
	}{
		{"aaa\n  --- asd\n TODO: tooooodo\n toooodoooooo\n", "aaa"},
		{"aaa\nasd\n TODO: tooooodo\nbbbb\n --- toooodoooooo\n", "aaa asd bbbb"},
		{" TODO: tooooodo\n", ""},
		{"Par1\n\nPar2\n\n", "Par1\\n\\nPar2"},
		{"", ""},
		{" ", ""},
		{" \n", ""},
		{" \n\n ", ""},
		{"Example:\n\tl1\n\t\tl2\n", "Example:\\n\\tl1\\n\\t\\tl2"},
	}

	for _, test := range tests {
		if o := fmtRawDoc(test.t); o != test.expected {
			t.Fatalf("Expected: %q, got %q", test.expected, o)
		}
	}
}

func TestFieldNameJSONInlineOption(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want string
	}{
		{
			name: "embedded inline field is skipped",
			src:  "TypeMeta `json:\",inline\"`",
			want: "-",
		},
		{
			name: "field name containing the substring inline is retained",
			src:  "InlineVolumeSpec *PersistentVolumeSpec `json:\"inlineVolumeSpec,omitempty\"`",
			want: "inlineVolumeSpec",
		},
		{
			name: "field whose JSON name is exactly inline is retained",
			src:  "Inline string `json:\"inline,omitempty\"`",
			want: "inline",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			field := mustParseStructField(t, tt.src)
			if got := fieldName(field); got != tt.want {
				t.Fatalf("fieldName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func mustParseStructField(t *testing.T, fieldSrc string) *ast.Field {
	t.Helper()

	src := "package p\n\ntype T struct {\n" + fieldSrc + "\n}\n"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("ParseFile() error = %v", err)
	}

	decl := f.Decls[0].(*ast.GenDecl)
	spec := decl.Specs[0].(*ast.TypeSpec)
	st := spec.Type.(*ast.StructType)
	return st.Fields.List[0]
}
