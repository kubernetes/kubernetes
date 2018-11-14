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

package apply

import (
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/typed"
)

// TypeConverter allows you to convert from runtime.Object to
// typed.TypedValue and the other way around.
type TypeConverter interface {
	ObjectToTyped(runtime.Object) (typed.TypedValue, error)
	TypedToObject(typed.TypedValue) (runtime.Object, error)
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
		return typed.TypedValue{}, err
	}
	gvk := obj.GetObjectKind().GroupVersionKind()
	t := c.parser.Type(gvk)
	if t == nil {
		return typed.TypedValue{}, fmt.Errorf("no corresponding type for %v", gvk)
	}
	return t.FromUnstructured(u)
}

func (c *typeConverter) TypedToObject(value typed.TypedValue) (runtime.Object, error) {
	vu := value.AsValue().ToUnstructured(false)
	u, ok := vu.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("failed to convert typed to unstructured: want map, got %T", vu)
	}
	return &unstructured.Unstructured{Object: u}, nil
}
