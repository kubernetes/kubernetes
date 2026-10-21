/*
Copyright 2017 The Kubernetes Authors.

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

package explain

import (
	"bytes"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	tst "k8s.io/kubectl/pkg/util/openapi/testing"
)

func TestRecursiveFields(t *testing.T) {
	schema := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "OneKind",
	})
	if schema == nil {
		t.Fatal("Couldn't find schema v1.OneKind")
	}

	want := `field1	<Object>
   array	<[]integer>
   int	<integer>
   object	<map[string]string>
   primitive	<string>
   string	<string>
field2	<[]map[string]string>
`

	buf := bytes.Buffer{}
	f := Formatter{
		Writer: &buf,
		Wrap:   80,
	}
	s, err := LookupSchemaForField(schema, []string{})
	if err != nil {
		t.Fatalf("Invalid path %v: %v", []string{}, err)
	}
	if err := (fieldsPrinterBuilder{Recursive: true}).BuildFieldsPrinter(&f).PrintFields(s); err != nil {
		t.Fatalf("Failed to print fields: %v", err)
	}
	got := buf.String()
	if got != want {
		t.Errorf("Got:\n%v\nWant:\n%v\n", buf.String(), want)
	}
}

func TestRecursiveFieldsWithSelfReferenceObjects(t *testing.T) {
	var resources = tst.NewFakeResources("test-recursive-swagger.json")
	schema := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v2",
		Kind:    "OneKind",
	})
	if schema == nil {
		t.Fatal("Couldn't find schema v2.OneKind")
	}

	want := `field1	<Object>
   referencefield	<Object>
   referencesarray	<[]Object>
field2	<Object>
   reference	<Object>
      referencefield	<Object>
      referencesarray	<[]Object>
   string	<string>
`

	buf := bytes.Buffer{}
	f := Formatter{
		Writer: &buf,
		Wrap:   80,
	}
	s, err := LookupSchemaForField(schema, []string{})
	if err != nil {
		t.Fatalf("Invalid path %v: %v", []string{}, err)
	}
	if err := (fieldsPrinterBuilder{Recursive: true}).BuildFieldsPrinter(&f).PrintFields(s); err != nil {
		t.Fatalf("Failed to print fields: %v", err)
	}
	got := buf.String()
	if got != want {
		t.Errorf("Got:\n%v\nWant:\n%v\n", buf.String(), want)
	}
}

func TestRecursiveFieldsMaxDepth(t *testing.T) {
	resources := tst.NewFakeResources("test-recursive-swagger.json")
	s := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v2",
		Kind:    "OneKind",
	})
	if s == nil {
		t.Fatal("Couldn't find schema v2.OneKind")
	}

	unlimitedWant := `field1	<Object>
   referencefield	<Object>
   referencesarray	<[]Object>
field2	<Object>
   reference	<Object>
      referencefield	<Object>
      referencesarray	<[]Object>
   string	<string>
`

	tests := []struct {
		name     string
		maxDepth int
		want     string
	}{
		{
			name:     "depth 0 means unlimited",
			maxDepth: 0,
			want:     unlimitedWant,
		},
		{
			name:     "depth 1 prints only top-level fields",
			maxDepth: 1,
			want: `field1	<Object>
field2	<Object>
`,
		},
		{
			name:     "depth 2 prints two levels of nesting",
			maxDepth: 2,
			want: `field1	<Object>
   referencefield	<Object>
   referencesarray	<[]Object>
field2	<Object>
   reference	<Object>
   string	<string>
`,
		},
		{
			name:     "depth 3 reaches the cycle-detection bound",
			maxDepth: 3,
			want:     unlimitedWant,
		},
		{
			name:     "depth larger than schema is bounded by cycle detection",
			maxDepth: 100,
			want:     unlimitedWant,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schemaForField, err := LookupSchemaForField(s, []string{})
			if err != nil {
				t.Fatalf("Invalid path %v: %v", []string{}, err)
			}

			var buf bytes.Buffer
			f := Formatter{Writer: &buf, Wrap: 80}
			builder := fieldsPrinterBuilder{Recursive: true, MaxDepth: tt.maxDepth}
			if err := builder.BuildFieldsPrinter(&f).PrintFields(schemaForField); err != nil {
				t.Fatalf("Failed to print fields: %v", err)
			}
			if got := buf.String(); got != tt.want {
				t.Errorf("Got:\n%v\nWant:\n%v\n", got, tt.want)
			}
		})
	}
}

func TestRecursiveFieldsMaxDepthNonRecursiveSchema(t *testing.T) {
	s := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "OneKind",
	})
	if s == nil {
		t.Fatal("Couldn't find schema v1.OneKind")
	}

	tests := []struct {
		name     string
		maxDepth int
		want     string
	}{
		{
			name:     "depth 1 stops before object children",
			maxDepth: 1,
			want: `field1	<Object>
field2	<[]map[string]string>
`,
		},
		{
			name:     "depth 2 reaches the leaves",
			maxDepth: 2,
			want: `field1	<Object>
   array	<[]integer>
   int	<integer>
   object	<map[string]string>
   primitive	<string>
   string	<string>
field2	<[]map[string]string>
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schemaForField, err := LookupSchemaForField(s, []string{})
			if err != nil {
				t.Fatalf("Invalid path %v: %v", []string{}, err)
			}

			var buf bytes.Buffer
			f := Formatter{Writer: &buf, Wrap: 80}
			builder := fieldsPrinterBuilder{Recursive: true, MaxDepth: tt.maxDepth}
			if err := builder.BuildFieldsPrinter(&f).PrintFields(schemaForField); err != nil {
				t.Fatalf("Failed to print fields: %v", err)
			}
			if got := buf.String(); got != tt.want {
				t.Errorf("Got:\n%v\nWant:\n%v\n", got, tt.want)
			}
		})
	}
}
