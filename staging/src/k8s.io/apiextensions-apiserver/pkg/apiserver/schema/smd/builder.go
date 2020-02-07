/*
Copyright 2020 The Kubernetes Authors.

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

package smd

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"

	smdschema "sigs.k8s.io/structured-merge-diff/v3/schema"
	"sigs.k8s.io/structured-merge-diff/v3/typed"
)

// TypeConverterBuilder builds a fieldmanager.TypeConverter from structural schemas
type TypeConverterBuilder struct {
	schema          *smdschema.Schema
	preserveUnknown bool
}

// NewTypeConverterBuilder creates a builder for a fieldmanager.TypeConverter built from structural schemas
func NewTypeConverterBuilder(s *smdschema.Schema, preserveUnknown bool) *TypeConverterBuilder {
	return &TypeConverterBuilder{
		schema:          s,
		preserveUnknown: preserveUnknown,
	}
}

// AddStructural converts a structural schema to an smd typedef, and adds it to an smd schema
func (b *TypeConverterBuilder) AddStructural(v string, s *schema.Structural) error {
	if s == nil {
		return fmt.Errorf("structural schema must not be nil")
	}
	a, err := toAtom(s, b.preserveUnknown)
	if err != nil {
		return err
	}
	if a.Map != nil {
		a.Map.Fields = mergeStructFields(baseResourceFields, a.Map.Fields)
	}
	b.schema.Types = append(b.schema.Types, smdschema.TypeDef{
		Name: makeVersionNameUniqueForApply(v),
		Atom: *a,
	})
	return nil
}

// Build creates an implementation of fieldmanager.TypeConverter from the schema
func (b *TypeConverterBuilder) Build() fieldmanager.TypeConverter {
	return &typeConverter{
		Parser: &typed.Parser{
			Schema: *b.schema,
		},
	}
}

type typeConverter struct {
	*typed.Parser
}

// ObjectToTyped implements fieldmanager.TypeConverter
func (c *typeConverter) ObjectToTyped(obj runtime.Object) (*typed.TypedValue, error) {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	t := c.Parser.Type(makeVersionNameUniqueForApply(obj.GetObjectKind().GroupVersionKind().Version))
	if !t.IsValid() {
		return typed.DeducedParseableType.FromUnstructured(u)
	}
	return t.FromUnstructured(u)
}

// TypedToObject implements fieldmanager.TypeConverter
func (c *typeConverter) TypedToObject(val *typed.TypedValue) (runtime.Object, error) {
	vu := val.AsValue().Unstructured()
	switch o := vu.(type) {
	case map[string]interface{}:
		return &unstructured.Unstructured{Object: o}, nil
	default:
		return nil, fmt.Errorf("failed to convert value to unstructured for type %T", vu)
	}
}

// makeVersionNameUniqueForApply formats an api version (like '__version_v1_')
// so it doesn't collide with any other typenames from the static openapi spec
func makeVersionNameUniqueForApply(v string) string {
	return fmt.Sprintf("__version_%v_", v)
}
