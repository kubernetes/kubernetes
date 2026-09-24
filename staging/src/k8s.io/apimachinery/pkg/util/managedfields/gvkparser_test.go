/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

type fakeSchema struct {
	name       string
	extensions map[string]interface{}
}

func (s *fakeSchema) GetName() string                       { return s.name }
func (s *fakeSchema) GetPath() *proto.Path                  { return nil }
func (s *fakeSchema) GetDescription() string                { return "" }
func (s *fakeSchema) GetDefault() interface{}               { return nil }
func (s *fakeSchema) GetExtensions() map[string]interface{} { return s.extensions }
func (s *fakeSchema) Accept(v proto.SchemaVisitor)          {}

type fakeModels struct {
	models map[string]proto.Schema
	list   []string
}

func (m *fakeModels) LookupModel(name string) proto.Schema { return m.models[name] }
func (m *fakeModels) ListModels() []string                 { return m.list }

func fakeGVKExtension(group, version, kind string) map[string]interface{} {
	return map[string]interface{}{
		groupVersionKindExtensionKey: []interface{}{
			map[interface{}]interface{}{
				"group":   group,
				"version": version,
				"kind":    kind,
			},
		},
	}
}

func TestNewGVKParserDuplicateEntry(t *testing.T) {
	// Two different models claiming same GVK /v1, Kind=APIResourceList
	// This reproduces Calico duplicate where aggregated API server serves same type twice.
	models := &fakeModels{
		models: map[string]proto.Schema{
			"com.example.a": &fakeSchema{
				name:       "com.example.a",
				extensions: fakeGVKExtension("", "v1", "APIResourceList"),
			},
			"com.example.b": &fakeSchema{
				name:       "com.example.b",
				extensions: fakeGVKExtension("", "v1", "APIResourceList"),
			},
			"com.example.c": &fakeSchema{
				name:       "com.example.c",
				extensions: fakeGVKExtension("apps", "v1", "Deployment"),
			},
		},
		list: []string{"com.example.a", "com.example.b", "com.example.c"},
	}

	parser, err := NewGVKParser(models, false)
	if err != nil {
		t.Fatalf("NewGVKParser should not fail on duplicate GVK, got error: %v", err)
	}
	if parser == nil {
		t.Fatalf("parser is nil")
	}
	// The first model should win for duplicate GVK
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "APIResourceList"}
	typ := parser.Type(gvk)
	if typ == nil {
		t.Fatalf("Type for duplicate GVK should not be nil, should be first model")
	}
	// Verify non-duplicate still works
	gvk2 := schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	if parser.Type(gvk2) == nil {
		t.Fatalf("Type for non-duplicate GVK should not be nil")
	}
}

func TestNewGVKParserDuplicateSameModel(t *testing.T) {
	// Same GVK from same modelName duplicate is not possible via ListModels uniqueness,
	// but test that three models where two share GVK and one is different still succeeds
	models := &fakeModels{
		models: map[string]proto.Schema{
			"com.example.a": &fakeSchema{
				name:       "com.example.a",
				extensions: fakeGVKExtension("", "v1", "ConfigMap"),
			},
			"com.example.b": &fakeSchema{
				name:       "com.example.b",
				extensions: fakeGVKExtension("", "v1", "ConfigMap"),
			},
		},
		list: []string{"com.example.a", "com.example.b"},
	}
	_, err := NewGVKParser(models, false)
	if err != nil {
		t.Fatalf("expected no error for duplicate ConfigMap, got %v", err)
	}
}

func TestNewGVKParserNoDuplicate(t *testing.T) {
	models := &fakeModels{
		models: map[string]proto.Schema{
			"com.example.a": &fakeSchema{
				name:       "com.example.a",
				extensions: fakeGVKExtension("", "v1", "Pod"),
			},
		},
		list: []string{"com.example.a"},
	}
	parser, err := NewGVKParser(models, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if parser.Type(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}) == nil {
		t.Fatalf("expected Pod type")
	}
}
