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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// TypeConverter allows you to convert from runtime.Object to
// typed.TypedValue and the other way around.
type TypeConverter interface {
	ObjectToTyped(runtime.Object) (*typed.TypedValue, error)
	TypedToObject(*typed.TypedValue) (runtime.Object, error)
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
func (DeducedTypeConverter) ObjectToTyped(obj runtime.Object) (*typed.TypedValue, error) {
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return typed.DeducedParseableType.FromUnstructured(o.UnstructuredContent())
	default:
		return typed.DeducedParseableType.FromStructured(obj)
	}
}

// TypedToObject transforms the typed value into a runtime.Object. That
// is not specific to deduced type.
func (DeducedTypeConverter) TypedToObject(value *typed.TypedValue) (runtime.Object, error) {
	return valueToObject(value.AsValue())
}

type typeConverter struct {
	parser *gvkParser
}

var _ TypeConverter = &typeConverter{}

// NewTypeConverter builds a TypeConverter from a proto.Models. This
// will automatically find the proper version of the object, and the
// corresponding schema information.
func NewTypeConverter(models proto.Models, preserveUnknownFields bool) (TypeConverter, error) {
	parser, err := newGVKParser(models, preserveUnknownFields)
	if err != nil {
		return nil, err
	}
	return &typeConverter{parser: parser}, nil
}

func (c *typeConverter) ObjectToTyped(obj runtime.Object) (*typed.TypedValue, error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	t := c.parser.Type(gvk)
	if t == nil {
		return nil, newNoCorrespondingTypeError(gvk)
	}
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return t.FromUnstructured(o.UnstructuredContent())
	default:
		return t.FromStructured(obj)
	}
}

func (c *typeConverter) TypedToObject(value *typed.TypedValue) (runtime.Object, error) {
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

type noCorrespondingTypeErr struct {
	gvk schema.GroupVersionKind
}

func newNoCorrespondingTypeError(gvk schema.GroupVersionKind) error {
	return &noCorrespondingTypeErr{gvk: gvk}
}

func (k *noCorrespondingTypeErr) Error() string {
	return fmt.Sprintf("no corresponding type for %v", k.gvk)
}

func isNoCorrespondingTypeError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*noCorrespondingTypeErr)
	return ok
}
