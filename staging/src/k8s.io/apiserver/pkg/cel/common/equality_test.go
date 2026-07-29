/*
Copyright 2023 The Kubernetes Authors.

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

package common_test

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type TestCase struct {
	Name string

	// Expected old value after traversal. If nil, then the traversal should fail.
	OldValue interface{}

	// Expected value after traversal. If nil, then the traversal should fail.
	NewValue interface{}

	// Whether OldValue and NewValue are considered to be equal.
	// Defaults to reflect.DeepEqual comparison of the two. Can be overridden to
	// true here if the two values are not DeepEqual, but are considered equal
	// for instance due to map-list reordering.
	ExpectEqual bool

	// Schema to provide to the correlated object
	Schema common.Schema

	// Array of field names and indexes to traverse to get to the value
	KeyPath []interface{}

	// Root object to traverse from
	RootObject    interface{}
	RootOldObject interface{}
}

func (c TestCase) Run() error {
	// Create the correlated object
	correlatedObject := common.NewCorrelatedObject(c.RootObject, c.RootOldObject, c.Schema)

	// Traverse the correlated object
	var err error
	for _, key := range c.KeyPath {
		if correlatedObject == nil {
			break
		}

		switch k := key.(type) {
		case string:
			correlatedObject = correlatedObject.Key(k)
		case int:
			correlatedObject = correlatedObject.Index(k)
		default:
			return errors.New("key must be a string or int")
		}
		if err != nil {
			return err
		}
	}

	if correlatedObject == nil {
		if c.OldValue != nil || c.NewValue != nil {
			return fmt.Errorf("expected non-nil value, got nil")
		}
	} else {
		// Check that the correlated object has the expected values
		if !reflect.DeepEqual(correlatedObject.Value, c.NewValue) {
			return fmt.Errorf("expected value %v, got %v", c.NewValue, correlatedObject.Value)
		}
		if !reflect.DeepEqual(correlatedObject.OldValue, c.OldValue) {
			return fmt.Errorf("expected old value %v, got %v", c.OldValue, correlatedObject.OldValue)
		}

		// Check that the correlated object is considered equal to the expected value
		if (c.ExpectEqual || reflect.DeepEqual(correlatedObject.Value, correlatedObject.OldValue)) != correlatedObject.CachedDeepEqual() {
			return fmt.Errorf("expected equal, got not equal")
		}
	}

	return nil
}

// Creates a *spec.Schema Schema by decoding the given YAML. Panics on error
func mustSchema(source string) *openapi.Schema {
	d := yaml.NewYAMLOrJSONDecoder(strings.NewReader(source), 4096)
	res := &spec.Schema{}
	if err := d.Decode(res); err != nil {
		panic(err)
	}
	return &openapi.Schema{Schema: res}
}

// Creates an *unstructured by decoding the given YAML. Panics on error
func mustUnstructured(source string) interface{} {
	d := yaml.NewYAMLOrJSONDecoder(strings.NewReader(source), 4096)
	var res interface{}
	if err := d.Decode(&res); err != nil {
		panic(err)
	}
	return res
}

func TestCorrelation(t *testing.T) {
	// Tests ensure that the output of following keypath using the given
	// schema and root objects yields the provided new value and old value.
	// If new or old are nil, then ensures that the traversal failed due to
	// uncorrelatable field path.
	// Also confirms that CachedDeepEqual output is equal to expected result of
	// reflect.DeepEqual of the new and old values.
	cases := []TestCase{
		{
			Name:          "Basic Key",
			RootObject:    mustUnstructured(`a: b`),
			RootOldObject: mustUnstructured(`a: b`),
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{"a"},
			NewValue: "b",
			OldValue: "b",
		},
		{
			Name:          "Atomic Array not correlatable",
			RootObject:    mustUnstructured(`[a, b]`),
			RootOldObject: mustUnstructured(`[a, b]`),
			Schema: mustSchema(`
                items:
                  type: string
            `),
			KeyPath: []interface{}{1},
		},
		{
			Name: "Added Key Not In Old Object",
			RootObject: mustUnstructured(`
                a: b
                c: d
            `),
			RootOldObject: mustUnstructured(`
                a: b
            `),
			Schema: mustSchema(`
                properties:
                  a: { type: string }
                  c: { type: string }
            `),
			KeyPath: []interface{}{"c"},
		},
		{
			Name: "Added Index Not In Old Object",
			RootObject: mustUnstructured(`
                - a
                - b
                - c
            `),
			RootOldObject: mustUnstructured(`
                - a
                - b
            `),
			Schema: mustSchema(`
                items:
                    type: string
            `),
			KeyPath: []interface{}{2},
		},
		{
			Name: "Changed Index In Old Object not correlatable",
			RootObject: []interface{}{
				"a",
				"b",
			},
			RootOldObject: []interface{}{
				"a",
				"oldB",
			},
			Schema: mustSchema(`
                items:
                    type: string
            `),
			KeyPath: []interface{}{1},
		},
		{
			Name: "Changed Index In Nested Old Object",
			RootObject: []interface{}{
				"a",
				"b",
			},
			RootOldObject: []interface{}{
				"a",
				"oldB",
			},
			Schema: mustSchema(`
                items:
                    type: string
            `),
			KeyPath:  []interface{}{},
			NewValue: []interface{}{"a", "b"},
			OldValue: []interface{}{"a", "oldB"},
		},
		{
			Name: "Changed Key In Old Object",
			RootObject: map[string]interface{}{
				"a": "b",
			},
			RootOldObject: map[string]interface{}{
				"a": "oldB",
			},
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{"a"},
			NewValue: "b",
			OldValue: "oldB",
		},
		{
			Name: "Replaced Key In Old Object",
			RootObject: map[string]interface{}{
				"a": "b",
			},
			RootOldObject: map[string]interface{}{
				"b": "a",
			},
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{},
			NewValue: map[string]interface{}{"a": "b"},
			OldValue: map[string]interface{}{"b": "a"},
		},
		{
			Name: "Added Key In Old Object",
			RootObject: map[string]interface{}{
				"a": "b",
			},
			RootOldObject: map[string]interface{}{},
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{},
			NewValue: map[string]interface{}{"a": "b"},
			OldValue: map[string]interface{}{},
		},
		{
			Name: "Changed list to map",
			RootObject: map[string]interface{}{
				"a": "b",
			},
			RootOldObject: []interface{}{"a", "b"},
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{},
			NewValue: map[string]interface{}{"a": "b"},
			OldValue: []interface{}{"a", "b"},
		},
		{
			Name: "Changed string to map",
			RootObject: map[string]interface{}{
				"a": "b",
			},
			RootOldObject: "a string",
			Schema: mustSchema(`
                properties:
                  a: { type: string }
            `),
			KeyPath:  []interface{}{},
			NewValue: map[string]interface{}{"a": "b"},
			OldValue: "a string",
		},
		{
			Name: "Map list type",
			RootObject: mustUnstructured(`
                foo:
                - bar: baz
                  val: newBazValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - bar: fizz
                  val: fizzValue
                - bar: baz
                  val: bazValue
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: array
                    items:
                      type: object
                      properties:
                        bar:
                          type: string
                        val:
                          type: string
                    x-kubernetes-list-type: map
                    x-kubernetes-list-map-keys:
                      - bar
            `),
			KeyPath:  []interface{}{"foo", 0, "val"},
			NewValue: "newBazValue",
			OldValue: "bazValue",
		},
		{
			Name: "Atomic list item should not correlate",
			RootObject: mustUnstructured(`
                foo:
                - bar: baz
                  val: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - bar: fizz
                  val: fizzValue
                - bar: baz
                  val: barValue
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: array
                    items:
                      type: object
                      properties:
                        bar:
                          type: string
                        val:
                          type: string
                    x-kubernetes-list-type: atomic
            `),
			KeyPath: []interface{}{"foo", 0, "val"},
		},
		{
			Name: "Map used inside of map list type should correlate",
			RootObject: mustUnstructured(`
                foo:
                - key: keyValue
                  bar:
                    baz: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - key: otherKeyValue
                  bar:
                    baz: otherOldValue
                - key: altKeyValue
                  bar:
                    baz: altOldValue
                - key: keyValue
                  bar:
                    baz: oldValue
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: array
                    items:
                      type: object
                      properties:
                        key:
                          type: string
                        bar:
                          type: object
                          properties:
                            baz:
                              type: string
                    x-kubernetes-list-type: map
                    x-kubernetes-list-map-keys:
                      - key
            `),
			KeyPath:  []interface{}{"foo", 0, "bar", "baz"},
			NewValue: "newValue",
			OldValue: "oldValue",
		},
		{
			Name: "Map used inside another map should correlate",
			RootObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                    key: otherKeyValue
                    bar:
                        baz: otherOldValue
                altFoo:
                    key: altKeyValue
                    bar:
                        baz: altOldValue
                otherFoo:
                    key: keyValue
                    bar:
                        baz: oldValue
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: object
                    properties:
                      key:
                        type: string
                      bar:
                        type: object
                        properties:
                          baz:
                            type: string
            `),
			KeyPath:  []interface{}{"foo", "bar"},
			NewValue: map[string]interface{}{"baz": "newValue"},
			OldValue: map[string]interface{}{"baz": "otherOldValue"},
		},
		{
			Name: "Nested map equal to old",
			RootObject: mustUnstructured(`
                foo:
                    key: newKeyValue
                    bar:
                        baz: value
            `),
			RootOldObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: value
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: object
                    properties:
                      key:
                        type: string
                      bar:
                        type: object
                        properties:
                          baz:
                            type: string
            `),
			KeyPath:  []interface{}{"foo", "bar"},
			NewValue: map[string]interface{}{"baz": "value"},
			OldValue: map[string]interface{}{"baz": "value"},
		},
		{
			Name: "Re-ordered list considered equal to old value due to map keys",
			RootObject: mustUnstructured(`
                foo:
                - key: keyValue
                  bar:
                    baz: value
                - key: altKeyValue
                  bar:
                    baz: altValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - key: altKeyValue
                  bar:
                    baz: altValue
                - key: keyValue
                  bar:
                    baz: value
            `),
			Schema: mustSchema(`
                properties:
                  foo:
                    type: array
                    items:
                      type: object
                      properties:
                        key:
                          type: string
                        bar:
                          type: object
                          properties:
                            baz:
                              type: string
                    x-kubernetes-list-type: map
                    x-kubernetes-list-map-keys:
                      - key
            `),
			KeyPath: []interface{}{"foo"},
			NewValue: mustUnstructured(`
                - key: keyValue
                  bar:
                    baz: value
                - key: altKeyValue
                  bar:
                    baz: altValue
            `),
			OldValue: mustUnstructured(`
                - key: altKeyValue
                  bar:
                    baz: altValue
                - key: keyValue
                  bar:
                    baz: value
            `),
			ExpectEqual: true,
		},
		{
			Name: "Correlate unknown string key via additional properties",
			RootObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                    key: otherKeyValue
                    bar:
                        baz: otherOldValue
            `),
			Schema: mustSchema(`
                properties:
                    foo:
                        type: object
                        additionalProperties:
                            properties:
                                baz:
                                    type: string
            `),
			KeyPath:  []interface{}{"foo", "bar", "baz"},
			NewValue: "newValue",
			OldValue: "otherOldValue",
		},
		{
			Name: "Changed map value",
			RootObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: oldValue
            `),
			Schema: mustSchema(`
                properties:
                    foo:
                        type: object
                        properties:
                            key:
                                type: string
                            bar:
                                type: object
                                properties:
                                    baz:
                                        type: string
            `),
			KeyPath: []interface{}{"foo", "bar"},
			NewValue: mustUnstructured(`
                baz: newValue
            `),
			OldValue: mustUnstructured(`
                baz: oldValue
            `),
		},
		{
			Name: "Changed nested map value",
			RootObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: newValue
            `),
			RootOldObject: mustUnstructured(`
                foo:
                    key: keyValue
                    bar:
                        baz: oldValue
            `),
			Schema: mustSchema(`
                properties:
                    foo:
                        type: object
                        properties:
                            key:
                                type: string
                            bar:
                                type: object
                                properties:
                                    baz:
                                        type: string
            `),
			KeyPath: []interface{}{"foo"},
			NewValue: mustUnstructured(`
                key: keyValue    
                bar:
                  baz: newValue
            `),
			OldValue: mustUnstructured(`
                key: keyValue    
                bar:
                  baz: oldValue
            `),
		},
		{
			Name: "unchanged list type set with atomic map values",
			Schema: mustSchema(`
                properties:
                    foo:
                        type: array
                        items:
                            type: object
                            x-kubernetes-map-type: atomic
                            properties:
                                key:
                                    type: string
                                bar:
                                    type: string
                        x-kubernetes-list-type: set
            `),
			RootObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
			KeyPath: []interface{}{"foo"},
			NewValue: mustUnstructured(`
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
			OldValue: mustUnstructured(`
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
		},
		{
			Name: "changed list type set with atomic map values",
			Schema: mustSchema(`
                properties:
                    foo:
                        type: array
                        items:
                            type: object
                            x-kubernetes-map-type: atomic
                            properties:
                                key:
                                    type: string
                                bar:
                                    type: string
                        x-kubernetes-list-type: set
            `),
			RootObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: newValue2
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
			KeyPath: []interface{}{"foo"},
			NewValue: mustUnstructured(`
                - key: key1
                  bar: value1
                - key: key2
                  bar: newValue2
            `),
			OldValue: mustUnstructured(`
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
		},
		{
			Name: "elements of list type set with atomic map values are not correlated",
			Schema: mustSchema(`
                properties:
                    foo:
                        type: array
                        items:
                            type: object
                            x-kubernetes-map-type: atomic
                            properties:
                                key:
                                    type: string
                                bar:
                                    type: string
                        x-kubernetes-list-type: set
            `),
			RootObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: newValue2
            `),
			RootOldObject: mustUnstructured(`
                foo:
                - key: key1
                  bar: value1
                - key: key2
                  bar: value2
            `),
			KeyPath:  []interface{}{"foo", 0, "key"},
			NewValue: nil,
		},
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
