/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/typed"
	"sigs.k8s.io/structured-merge-diff/value"
	"sigs.k8s.io/yaml"
)

// TypeConverter allows you to convert from runtime.Object to
// typed.TypedValue and the other way around.
type TypeConverter interface {
	ObjectToTyped(runtime.Object) (typed.TypedValue, error)
	YAMLToTyped([]byte) (typed.TypedValue, error)
	TypedToObject(typed.TypedValue) (runtime.Object, error)
}

// DeducedTypeConverter is a TypeConverter for CRDs that don't have a
// schema. It does implement the same interface though (and create the
// same types of objects), so that everything can still work the same.
// CRDs are merged with all their fields being "atomic" (lists
// included).
//
// Note that this is not going to be sufficient for converting to/from
// CRDs that have a schema defined (we don't support that schema yet).
// TODO(jennybuckley): Use the schema provided by a CRD if it exists.
type DeducedTypeConverter struct{}

var _ TypeConverter = DeducedTypeConverter{}

// ObjectToTyped converts an object into a TypedValue with a "deduced type".
func (DeducedTypeConverter) ObjectToTyped(obj runtime.Object) (typed.TypedValue, error) {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return typed.DeducedParseableType{}.FromUnstructured(u)
}

// YAMLToTyped parses a yaml object into a TypedValue with a "deduced type".
func (DeducedTypeConverter) YAMLToTyped(from []byte) (typed.TypedValue, error) {
	return typed.DeducedParseableType{}.FromYAML(typed.YAMLObject(from))
}

// TypedToObject transforms the typed value into a runtime.Object. That
// is not specific to deduced type.
func (DeducedTypeConverter) TypedToObject(value typed.TypedValue) (runtime.Object, error) {
	return valueToObject(value.AsValue())
}

type typeConverter struct {
	parser *gvkParser
}

var _ TypeConverter = &typeConverter{}

// NewTypeConverter builds a TypeConverter from a proto.Models. This
// will automatically find the proper version of the object, and the
// corresponding schema information.
func NewTypeConverter(models proto.Models) (TypeConverter, error) {
	parser, err := newGVKParser(models)
	if err != nil {
		return nil, err
	}
	return &typeConverter{parser: parser}, nil
}

func (c *typeConverter) ObjectToTyped(obj runtime.Object) (typed.TypedValue, error) {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	gvk := obj.GetObjectKind().GroupVersionKind()
	t := c.parser.Type(gvk)
	if t == nil {
		return nil, fmt.Errorf("no corresponding type for %v", gvk)
	}
	return t.FromUnstructured(u)
}

func (c *typeConverter) YAMLToTyped(from []byte) (typed.TypedValue, error) {
	unstructured := &unstructured.Unstructured{Object: map[string]interface{}{}}

	if err := yaml.Unmarshal(from, &unstructured.Object); err != nil {
		return nil, fmt.Errorf("error decoding YAML: %v", err)
	}

	gvk := unstructured.GetObjectKind().GroupVersionKind()
	t := c.parser.Type(gvk)
	if t == nil {
		return nil, fmt.Errorf("no corresponding type for %v", gvk)
	}
	return t.FromYAML(typed.YAMLObject(string(from)))
}

func (c *typeConverter) TypedToObject(value typed.TypedValue) (runtime.Object, error) {
	return valueToObject(value.AsValue())
}

func valueToObject(value *value.Value) (runtime.Object, error) {
	vu := value.ToUnstructured(false)
	u, ok := vu.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("failed to convert typed to unstructured: want map, got %T", vu)
	}
	return &unstructured.Unstructured{Object: u}, nil
}
