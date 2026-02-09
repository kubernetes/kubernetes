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

// Package format is an extension of Gomega's format package which
// improves printing of objects that can be serialized well as YAML,
// like the structs in the Kubernetes API.
//
// Just importing it is enough to activate this special YAML support
// in Gomega.
package format

import (
	"reflect"
	"strings"

	"github.com/onsi/gomega/format"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"
)

func init() {
	format.RegisterCustomFormatter(handleYAML)
}

// Object makes Gomega's [format.Object] available without having to import that
// package.
func Object(object interface{}, indentation uint) string {
	return format.Object(object, indentation)
}

// handleYAML formats all values as YAML where the result
// is likely to look better as YAML:
//   - pointer to struct or struct where all fields
//     have `json` tags
//   - slices containing such a value
//   - maps where the key or value are such a value
func handleYAML(object interface{}) (string, bool) {
	value := reflect.ValueOf(object)
	if !useYAML(value.Type()) {
		return "", false
	}
	y, err := yaml.Marshal(object)
	if err != nil {
		return "", false
	}
	return "\n" + strings.TrimSpace(string(y)), true
}

var unstructuredObjectType = reflect.TypeOf(unstructured.Unstructured{})

func useYAML(t reflect.Type) bool {
	if t == unstructuredObjectType {
		// It looks nicer as YAML.
		//
		// unstructured.Unstructured is a map, but because
		// it's wrapped in a struct it does not get recognized
		// as one by the code below and thus needs a direct check.
		return true
	}

	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return useYAML(t.Elem())
	case reflect.Map:
		return useYAML(t.Key()) || useYAML(t.Elem())
	case reflect.Struct:
		// All fields must have a `json` tag.
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if _, ok := field.Tag.Lookup("json"); !ok {
				return false
			}
		}
		return true
	default:
		return false
	}
}
