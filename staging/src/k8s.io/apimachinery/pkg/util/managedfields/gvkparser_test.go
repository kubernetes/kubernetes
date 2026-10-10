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

package managedfields

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func gvkExtension(gvks ...schema.GroupVersionKind) spec.Extensions {
	list := make([]interface{}, 0, len(gvks))
	for _, gvk := range gvks {
		list = append(list, map[string]interface{}{"group": gvk.Group, "version": gvk.Version, "kind": gvk.Kind})
	}
	return spec.Extensions{groupVersionKindExtensionKey: list}
}

func objectSchema(ext spec.Extensions, properties map[string]spec.Schema) *spec.Schema {
	return &spec.Schema{
		SchemaProps:      spec.SchemaProps{Type: []string{"object"}, Properties: properties},
		VendorExtensible: spec.VendorExtensible{Extensions: ext},
	}
}

func TestNewGVKParserFromOpenAPIV3(t *testing.T) {
	widgetGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
	// DeleteOptions-style schema listed for several group-versions.
	optionsGVKs := []schema.GroupVersionKind{
		{Group: "", Version: "v1", Kind: "Options"},
		{Group: "example.com", Version: "v1", Kind: "Options"},
	}
	schemas := map[string]*spec.Schema{
		"com.example.v1.Widget": objectSchema(gvkExtension(widgetGVK), map[string]spec.Schema{
			"spec": {SchemaProps: spec.SchemaProps{Type: []string{"object"}, Properties: map[string]spec.Schema{
				"replicas": {SchemaProps: spec.SchemaProps{Type: []string{"integer"}}},
			}}},
		}),
		"com.example.v1.Options": objectSchema(gvkExtension(optionsGVKs...), nil),
		// A schema without the extension is a dependency, not a resolvable kind.
		"com.example.v1.Helper": objectSchema(nil, nil),
	}

	parser, err := NewGVKParserFromOpenAPIV3(schemas, false)
	if err != nil {
		t.Fatal(err)
	}

	for _, gvk := range append([]schema.GroupVersionKind{widgetGVK}, optionsGVKs...) {
		if parser.Type(gvk) == nil {
			t.Errorf("expected a type for %v", gvk)
		}
	}
	if got := parser.Type(schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Helper"}); got != nil {
		t.Errorf("expected no type for a schema without the %s extension, got %v", groupVersionKindExtensionKey, got)
	}
	if got := parser.Type(schema.GroupVersionKind{Group: "example.com", Version: "v2", Kind: "Widget"}); got != nil {
		t.Errorf("expected no type for an unknown group-version-kind, got %v", got)
	}

	// The type parses an object of that kind and rejects unknown fields,
	// which shows the schema reached structured-merge-diff.
	widget := map[string]interface{}{"spec": map[string]interface{}{"replicas": int64(3)}}
	if _, err := parser.Type(widgetGVK).FromUnstructured(widget); err != nil {
		t.Errorf("expected the Widget type to parse a Widget: %v", err)
	}
	widget["spec"].(map[string]interface{})["bogus"] = true
	if _, err := parser.Type(widgetGVK).FromUnstructured(widget); err == nil {
		t.Error("expected the Widget type to reject an unknown field")
	}
}

func TestNewGVKParserFromOpenAPIV3MalformedExtension(t *testing.T) {
	schemas := map[string]*spec.Schema{
		"com.example.v1.Widget": objectSchema(spec.Extensions{groupVersionKindExtensionKey: "not-a-list"}, nil),
	}
	if _, err := NewGVKParserFromOpenAPIV3(schemas, false); err == nil || !strings.Contains(err.Error(), groupVersionKindExtensionKey) {
		t.Errorf("expected an error naming the malformed extension, got %v", err)
	}
}
