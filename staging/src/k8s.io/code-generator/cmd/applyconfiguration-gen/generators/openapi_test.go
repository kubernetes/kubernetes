/*
Copyright The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestNewTypeModels(t *testing.T) {
	schemaFile := filepath.Join(t.TempDir(), "schema.json")
	schema := `{"swagger": "2.0", "info": {"title": "test", "version": "v1"}, "paths": {}, "definitions": {
		"com.example.widgets.v1.Widget": {"type": "object"},
		"io.example.widgets.v1.Widget": {"type": "object"}}}`
	if err := os.WriteFile(schemaFile, []byte(schema), 0o644); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name         string
		comments     []string
		typeComments []string
		want         string
		wantErr      bool
	}{
		{name: "derived from Go package path", want: "com.example.widgets.v1.Widget"},
		{name: "model package tag", comments: []string{"+k8s:openapi-model-package=io.example.widgets.v1"}, want: "io.example.widgets.v1.Widget"},
		{name: "type model package tag", typeComments: []string{"+k8s:openapi-model-package=io.example.widgets.v1"}, want: "io.example.widgets.v1.Widget"},
		{name: "type model package overrides package", comments: []string{"+k8s:openapi-model-package=io.example.widgets.v1"}, typeComments: []string{"+k8s:openapi-model-package=com.example.widgets.v1"}, want: "com.example.widgets.v1.Widget"},
		{name: "empty model package uses Go package path", comments: []string{"+k8s:openapi-model-package="}, want: "com.example.widgets.v1.Widget"},
		{name: "no definition", comments: []string{"+k8s:openapi-model-package=io.example.other.v1"}, wantErr: true},
		{name: "type definition missing", comments: []string{"+k8s:openapi-model-package=io.example.widgets.v1"}, typeComments: []string{"+k8s:openapi-model-package=io.example.other.v1"}, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			widget := &types.Type{
				Name:         types.Name{Package: "example.com/widgets/v1", Name: "Widget"},
				Kind:         types.Struct,
				CommentLines: append([]string{"+genclient"}, tc.typeComments...),
			}
			pkg := &types.Package{
				Path:     "example.com/widgets/v1",
				Name:     "v1",
				Comments: tc.comments,
				Types:    map[string]*types.Type{"Widget": widget},
			}
			models, err := newTypeModels(schemaFile, map[string]*types.Package{pkg.Path: pkg})
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected an error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(models.gvkToOpenAPIType) != 1 {
				t.Fatalf("got %d root types, want 1", len(models.gvkToOpenAPIType))
			}
			for _, got := range models.gvkToOpenAPIType {
				if got != tc.want {
					t.Errorf("got %q, want %q", got, tc.want)
				}
			}
		})
	}
}
