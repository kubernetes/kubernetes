/*
Copyright 2022 The Kubernetes Authors.

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

package internal

import (
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/schemaconv"
	"k8s.io/kube-openapi/pkg/validation/spec"
	smdschema "sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// TypeConverter allows you to convert from runtime.Object to
// typed.TypedValue and the other way around.
type TypeConverter interface {
	ObjectToTyped(runtime.Object, ...typed.ValidationOptions) (*typed.TypedValue, error)
	TypedToObject(*typed.TypedValue) (runtime.Object, error)
}

type typeConverter struct {
	parser map[schema.GroupVersionKind]*typed.ParseableType
}

var _ TypeConverter = &typeConverter{}

func NewTypeConverter(openapiSpec map[string]*spec.Schema, preserveUnknownFields bool) (TypeConverter, error) {
	typeSchema, err := schemaconv.ToSchemaFromOpenAPI(openapiSpec, preserveUnknownFields)
	if err != nil {
		return nil, fmt.Errorf("failed to convert models to schema: %v", err)
	}

	typeParser := typed.Parser{Schema: smdschema.Schema{Types: typeSchema.Types}}
	tr := indexModels(&typeParser, openapiSpec)

	return &typeConverter{parser: tr}, nil
}

func (c *typeConverter) ObjectToTyped(obj runtime.Object, opts ...typed.ValidationOptions) (*typed.TypedValue, error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	t := c.parser[gvk]
	if t == nil {
		return nil, NewNoCorrespondingTypeError(gvk)
	}
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return t.FromUnstructured(o.UnstructuredContent(), opts...)
	default:
		return t.FromStructured(obj, opts...)
	}
}

func (c *typeConverter) TypedToObject(value *typed.TypedValue) (runtime.Object, error) {
	return valueToObject(value.AsValue())
}

type deducedTypeConverter struct{}

// DeducedTypeConverter is a TypeConverter for CRDs that don't have a
// schema. It does implement the same interface though (and create the
// same types of objects), so that everything can still work the same.
// CRDs are merged with all their fields being "atomic" (lists
// included).
func NewDeducedTypeConverter() TypeConverter {
	return deducedTypeConverter{}
}

// ObjectToTyped converts an object into a TypedValue with a "deduced type".
func (deducedTypeConverter) ObjectToTyped(obj runtime.Object, opts ...typed.ValidationOptions) (*typed.TypedValue, error) {
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return typed.DeducedParseableType.FromUnstructured(o.UnstructuredContent(), opts...)
	default:
		return typed.DeducedParseableType.FromStructured(obj, opts...)
	}
}

// TypedToObject transforms the typed value into a runtime.Object. That
// is not specific to deduced type.
func (deducedTypeConverter) TypedToObject(value *typed.TypedValue) (runtime.Object, error) {
	return valueToObject(value.AsValue())
}

func valueToObject(val value.Value) (runtime.Object, error) {
	vu := val.Unstructured()
	switch o := vu.(type) {
	case map[string]interface{}:
		return &unstructured.Unstructured{Object: o}, nil
	default:
		return nil, fmt.Errorf("failed to convert value to unstructured for type %T", vu)
	}
}

func indexModels(
	typeParser *typed.Parser,
	openAPISchemas map[string]*spec.Schema,
) map[schema.GroupVersionKind]*typed.ParseableType {
	tr := map[schema.GroupVersionKind]*typed.ParseableType{}
	for modelName, model := range openAPISchemas {
		gvkList := parseGroupVersionKind(model.Extensions)
		if len(gvkList) == 0 {
			continue
		}

		parsedType := typeParser.Type(modelName)
		for _, gvk := range gvkList {
			if len(gvk.Kind) > 0 {
				tr[schema.GroupVersionKind(gvk)] = &parsedType
			}
		}
	}
	return tr
}

// Get and parse GroupVersionKind from the extension. Returns empty if it doesn't have one.
func parseGroupVersionKind(extensions map[string]interface{}) []schema.GroupVersionKind {
	gvkListResult := []schema.GroupVersionKind{}

	// Get the extensions
	gvkExtension, ok := extensions["x-kubernetes-group-version-kind"]
	if !ok {
		return []schema.GroupVersionKind{}
	}

	// gvk extension must be a list of at least 1 element.
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return []schema.GroupVersionKind{}
	}

	for _, gvk := range gvkList {
		var group, version, kind string

		// gvk extension list must be a map with group, version, and
		// kind fields
		if gvkMap, ok := gvk.(map[interface{}]interface{}); ok {
			group, ok = gvkMap["group"].(string)
			if !ok {
				continue
			}
			version, ok = gvkMap["version"].(string)
			if !ok {
				continue
			}
			kind, ok = gvkMap["kind"].(string)
			if !ok {
				continue
			}

		} else if gvkMap, ok := gvk.(map[string]interface{}); ok {
			group, ok = gvkMap["group"].(string)
			if !ok {
				continue
			}
			version, ok = gvkMap["version"].(string)
			if !ok {
				continue
			}
			kind, ok = gvkMap["kind"].(string)
			if !ok {
				continue
			}
		} else {
			continue
		}

		gvkListResult = append(gvkListResult, schema.GroupVersionKind{
			Group:   group,
			Version: version,
			Kind:    kind,
		})
	}

	return gvkListResult
}
