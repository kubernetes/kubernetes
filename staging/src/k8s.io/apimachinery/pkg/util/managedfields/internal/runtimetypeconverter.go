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

package internal

import (
	"fmt"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

type schemeTypeConverter struct {
	scheme *runtime.Scheme
	parser *typed.Parser
}

var _ TypeConverter = &schemeTypeConverter{}

// NewSchemeTypeConverter creates a TypeConverter that uses the provided scheme to
// convert between runtime.Objects and TypedValues.
func NewSchemeTypeConverter(scheme *runtime.Scheme, parser *typed.Parser) TypeConverter {
	return &schemeTypeConverter{scheme: scheme, parser: parser}
}

func (tc schemeTypeConverter) ObjectToTyped(obj runtime.Object, opts ...typed.ValidationOptions) (*typed.TypedValue, error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	name, err := tc.scheme.ToOpenAPIDefinitionName(gvk)
	if err != nil {
		return nil, err
	}
	t := tc.parser.Type(name)
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return t.FromUnstructured(o.UnstructuredContent(), opts...)
	default:
		return t.FromStructured(obj, opts...)
	}
}

func (tc schemeTypeConverter) TypedToObject(value *typed.TypedValue) (runtime.Object, error) {
	vu := value.AsValue().Unstructured()
	switch o := vu.(type) {
	case map[string]interface{}:
		return &unstructured.Unstructured{Object: o}, nil
	default:
		return nil, fmt.Errorf("failed to convert value to unstructured for type %T", vu)
	}
}
