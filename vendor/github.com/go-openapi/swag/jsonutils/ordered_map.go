// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonutils

import (
	"iter"

	"github.com/go-openapi/swag/jsonutils/adapters"
	"github.com/go-openapi/swag/typeutils"
)

// JSONMapSlice represents a JSON object, with the order of keys maintained.
//
// It behaves like an ordered map, but keys can't be accessed in constant time.
type JSONMapSlice []JSONMapItem

// OrderedItems iterates over all (key,value) pairs with the order of keys maintained.
//
// This implements the [ifaces.Ordered] interface, so that [ifaces.Adapter] s know how to marshal
// keys in the desired order.
func (s JSONMapSlice) OrderedItems() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for _, item := range s {
			if !yield(item.Key, item.Value) {
				return
			}
		}
	}
}

// SetOrderedItems sets keys in the [JSONMapSlice] objects, as presented by
// the provided iterator.
//
// As a special case, if items is nil, this sets to receiver to a nil slice.
//
// This implements the [ifaces.SetOrdered] interface, so that [ifaces.Adapter] s know how to unmarshal
// keys in the desired order.
func (s *JSONMapSlice) SetOrderedItems(items iter.Seq2[string, any]) {
	if items == nil {
		// force receiver to be a nil slice
		*s = nil

		return
	}

	m := *s
	if len(m) > 0 {
		// update mode: short-circuited when unmarshaling fresh data structures
		idx := make(map[string]int, len(m))

		for i, item := range m {
			idx[item.Key] = i
		}

		for k, v := range items {
			idx, ok := idx[k]
			if ok {
				m[idx].Value = v

				continue
			}

			m = append(m, JSONMapItem{Key: k, Value: v})
		}

		*s = m

		return
	}

	for k, v := range items {
		m = append(m, JSONMapItem{Key: k, Value: v})
	}

	*s = m
}

// MarshalJSON renders a [JSONMapSlice] as JSON bytes, preserving the order of keys.
//
// It will pick the JSON library currently configured by the [adapters.Registry] (defaults to the standard library).
func (s JSONMapSlice) MarshalJSON() ([]byte, error) {
	orderedMarshaler := adapters.OrderedMarshalAdapterFor(s)
	defer orderedMarshaler.Redeem()

	return orderedMarshaler.OrderedMarshal(s)
}

// UnmarshalJSON builds a [JSONMapSlice] from JSON bytes, preserving the order of keys.
//
// Inner objects are unmarshaled as ordered [JSONMapSlice] slices and not map[string]any.
//
// It will pick the JSON library currently configured by the [adapters.Registry] (defaults to the standard library).
func (s *JSONMapSlice) UnmarshalJSON(data []byte) error {
	if typeutils.IsNil(*s) {
		// allow to unmarshal with a simple var declaration (nil slice)
		*s = JSONMapSlice{}
	}

	orderedUnmarshaler := adapters.OrderedUnmarshalAdapterFor(s)
	defer orderedUnmarshaler.Redeem()

	return orderedUnmarshaler.OrderedUnmarshal(data, s)
}

// JSONMapItem represents the value of a key in a JSON object held by [JSONMapSlice].
//
// Notice that JSONMapItem should not be marshaled to or unmarshaled from JSON directly.
//
// Use this type as part of a [JSONMapSlice] when dealing with JSON bytes.
type JSONMapItem struct {
	Key   string
	Value any
}
