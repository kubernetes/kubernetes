/*
Copyright 2021 The Kubernetes Authors.

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

package common

import (
	"time"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	"k8s.io/apiserver/pkg/cel"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

// UnstructuredToVal converts a Kubernetes unstructured data element to a CEL Val.
func UnstructuredToVal(unstructured interface{}, schema Schema) ref.Val {
	if unstructured == nil {
		if schema != nil && schema.Nullable() {
			return types.NullValue
		}
		return types.NewErr("invalid data, got null for schema with nullable=false")
	}

	if schema == nil {
		return types.DefaultTypeAdapter.NativeToValue(unstructured)
	}

	if schema.IsXIntOrString() {
		switch v := unstructured.(type) {
		case string:
			return types.String(v)
		case int:
			return types.Int(v)
		case int32:
			return types.Int(v)
		case int64:
			return types.Int(v)
		}
		return types.NewErr("invalid data, expected XIntOrString value to be either a string or integer")
	}

	switch schema.Type() {
	case "object":
		m, ok := unstructured.(map[string]interface{})
		if !ok {
			return types.NewErr("invalid data, expected a map for the provided schema with type=object")
		}

		if schema.IsXEmbeddedResource() {
			schema = schema.WithTypeAndObjectMeta()
		}

		props := schema.Properties()
		addl := schema.AdditionalProperties()
		celMap := make(map[string]ref.Val, len(m))

		for k, v := range m {
			if v == nil && props != nil {
				continue // skip null fields if they are explicitly part of properties
			}

			var propSchema Schema
			if props != nil {
				if ps, ok := props[k]; ok {
					propSchema = ps
				}
			} else if addl != nil && addl.Schema() != nil {
				propSchema = addl.Schema()
			}

			escapedK := k
			if props != nil {
				if esc, ok := cel.Escape(k); ok {
					escapedK = esc
				}
			}

			celMap[escapedK] = UnstructuredToVal(v, propSchema)
		}
		return types.DefaultTypeAdapter.NativeToValue(celMap)

	case "array":
		l, ok := unstructured.([]interface{})
		if !ok {
			return types.NewErr("invalid data, expected an array for the provided schema with type=array")
		}

		itemSchema := schema.Items()
		if itemSchema == nil {
			return types.NewErr("invalid array type, expected Items with a non-empty Schema")
		}

		celItems := make([]ref.Val, len(l))
		for i, item := range l {
			celItems[i] = UnstructuredToVal(item, itemSchema)
		}

		nativeList := types.DefaultTypeAdapter.NativeToValue(celItems)

		listType := schema.XListType()
		if listType == "set" || listType == "map" {
			lister, ok := nativeList.(traits.Lister)
			if !ok {
				return types.NewErr("unexpected native list type")
			}
			return &celListWrapper{
				Lister:   lister,
				listType: listType,
				mapKeys:  schema.XListMapKeys(),
			}
		}
		return nativeList

	case "string":
		str, ok := unstructured.(string)
		if !ok {
			return types.NewErr("invalid data, expected string, got %T", unstructured)
		}
		switch schema.Format() {
		case "duration":
			d, err := strfmt.ParseDuration(str)
			if err != nil {
				return types.NewErr("Invalid duration %s: %v", str, err)
			}
			return types.Duration{Duration: d}
		case "date":
			d, err := time.Parse(strfmt.RFC3339FullDate, str)
			if err != nil {
				return types.NewErr("Invalid date formatted string %s: %v", str, err)
			}
			return types.Timestamp{Time: d}
		case "date-time":
			d, err := strfmt.ParseDateTime(str)
			if err != nil {
				return types.NewErr("Invalid date-time formatted string %s: %v", str, err)
			}
			return types.Timestamp{Time: time.Time(d)}
		case "byte":
			base64 := strfmt.Base64{}
			err := base64.UnmarshalText([]byte(str))
			if err != nil {
				return types.NewErr("Invalid byte formatted string %s: %v", str, err)
			}
			return types.Bytes(base64)
		}
		return types.String(str)

	case "number":
		switch v := unstructured.(type) {
		case int:
			return types.Double(v)
		case int32:
			return types.Double(v)
		case int64:
			return types.Double(v)
		case float32:
			return types.Double(v)
		case float64:
			return types.Double(v)
		default:
			return types.NewErr("invalid data, expected float, got %T", unstructured)
		}

	case "integer":
		switch v := unstructured.(type) {
		case int:
			return types.Int(v)
		case int32:
			return types.Int(v)
		case int64:
			return types.Int(v)
		default:
			return types.NewErr("invalid data, expected int, got %T", unstructured)
		}
	case "boolean":
		b, ok := unstructured.(bool)
		if !ok {
			return types.NewErr("invalid data, expected bool, got %T", unstructured)
		}
		return types.Bool(b)
	}

	if schema.IsXPreserveUnknownFields() {
		return types.DefaultTypeAdapter.NativeToValue(unstructured)
	}

	return types.NewErr("invalid type, expected object, array, number, integer, boolean or string, or no type with x-kubernetes-int-or-string or x-kubernetes-preserve-unknown-fields is true, got %s", schema.Type())
}

// celListWrapper overrides the Equal operator for set and map arrays.
type celListWrapper struct {
	traits.Lister
	listType string
	mapKeys  []string
}

func (w *celListWrapper) Add(other ref.Val) ref.Val {
	oList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}

	szVal := w.Size()
	sz, ok := szVal.(types.Int)
	if !ok {
		return types.NewErr("list size is not an int")
	}

	elements := make([]ref.Val, int(sz))
	for i := 0; i < int(sz); i++ {
		elements[i] = w.Get(types.Int(i))
	}

	switch w.listType {
	case "set":
		return addToSetList(elements, oList)
	case "map":
		escapedMapKeys := escapeKeyProps(w.mapKeys)
		return addToMapList(elements, oList, escapedMapKeys)
	default:
		return w.Lister.Add(other)
	}
}

func (w *celListWrapper) Equal(other ref.Val) ref.Val {
	oList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}

	sz1Val := w.Size()
	sz2Val := oList.Size()
	sz1, ok1 := sz1Val.(types.Int)
	sz2, ok2 := sz2Val.(types.Int)
	if !ok1 || !ok2 || sz1 != sz2 {
		return types.False
	}

	switch w.listType {
	case "set":
		return setListEqual(w, oList, sz1, sz2)
	case "map":
		return mapListEqual(w, oList, w.mapKeys, sz1, sz2)
	default:
		return w.Lister.Equal(other)
	}
}

// escapeKeyProps returns identifiers with Escape applied to each.
func escapeKeyProps(idents []string) []string {
	result := make([]string, len(idents))
	for i, prop := range idents {
		if escaped, ok := cel.Escape(prop); ok {
			result[i] = escaped
		} else {
			result[i] = prop
		}
	}
	return result
}

func setListEqual(lister, other traits.Lister, sz1, sz2 types.Int) ref.Val {
	tSet := make(map[interface{}]struct{}, int(sz1))
	for i := types.Int(0); i < sz1; i++ {
		v := lister.Get(i)
		k, err := setElementKey(v)
		if err != nil {
			return err
		}
		tSet[k] = struct{}{}
	}

	seen := make(map[interface{}]struct{}, len(tSet))
	for j := types.Int(0); j < sz2; j++ {
		v := other.Get(j)
		k, err := setElementKey(v)
		if err != nil {
			return err
		}
		if _, exists := tSet[k]; !exists {
			return types.False
		}
		seen[k] = struct{}{}
	}
	return types.Bool(len(seen) == len(tSet))
}

func mapListEqual(lister, other traits.Lister, mapKeys []string, sz1, sz2 types.Int) ref.Val {
	escapedMapKeys := escapeKeyProps(mapKeys)
	tMap := make(map[interface{}]interface{}, int(sz1))
	for i := types.Int(0); i < sz1; i++ {
		v := lister.Get(i)
		k := refValMapKey(v, escapedMapKeys)
		tMap[k] = v
	}

	seen := make(map[interface{}]struct{}, len(tMap))
	for j := types.Int(0); j < sz2; j++ {
		v := other.Get(j)
		k := refValMapKey(v, escapedMapKeys)
		tVal, exists := tMap[k]
		if !exists {
			return types.False
		}
		tRefVal, ok := tVal.(ref.Val)
		if !ok {
			return types.False
		}
		if tRefVal.Equal(v) != types.True {
			return types.False
		}
		seen[k] = struct{}{}
	}
	return types.Bool(len(seen) == len(tMap))
}
