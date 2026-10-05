// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonutils

import (
	"bytes"
	"encoding/json"

	"github.com/go-openapi/swag/jsonutils/adapters"
	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
)

// WriteJSON marshals a data structure as JSON.
//
// The difference with [json.Marshal] is that it may check among several alternatives
// to do so.
//
// See [adapters.Registrar] for more details about how to configure
// multiple serialization alternatives.
//
// NOTE: to allow types that are [easyjson.Marshaler] s to use that route to process JSON,
// you now need to register the adapter for easyjson at runtime.
func WriteJSON(value any) ([]byte, error) {
	if orderedMap, isOrdered := value.(ifaces.Ordered); isOrdered {
		orderedMarshaler := adapters.OrderedMarshalAdapterFor(orderedMap)

		if orderedMarshaler != nil {
			defer orderedMarshaler.Redeem()

			return orderedMarshaler.OrderedMarshal(orderedMap)
		}

		// no support found in registered adapters, fallback to the default (unordered) case
	}

	marshaler := adapters.MarshalAdapterFor(value)
	if marshaler != nil {
		defer marshaler.Redeem()

		return marshaler.Marshal(value)
	}

	// no support found in registered adapters, fallback to the default standard library.
	//
	// This only happens when tinkering with the global registry of adapters, since the default handles all the above cases.
	return json.Marshal(value) // Codecov ignore // this is a safeguard not easily simulated in tests
}

// ReadJSON unmarshals JSON data into a data structure.
//
// The difference with [json.Unmarshal] is that it may check among several alternatives
// to do so.
//
// See [adapters.Registrar] for more details about how to configure
// multiple serialization alternatives.
//
// NOTE: value must be a pointer.
//
// If the provided value implements [ifaces.SetOrdered], it is a considered an "ordered map" and [ReadJSON]
// will favor an adapter that supports the [ifaces.OrderedUnmarshal] feature, or fallback to
// an unordered behavior if none is found.
//
// NOTE: to allow types that are [easyjson.Unmarshaler] s to use that route to process JSON,
// you now need to register the adapter for easyjson at runtime.
func ReadJSON(data []byte, value any) error {
	trimmedData := bytes.Trim(data, "\x00")

	if orderedMap, isOrdered := value.(ifaces.SetOrdered); isOrdered {
		// if the value is an ordered map, favors support for OrderedUnmarshal.

		orderedUnmarshaler := adapters.OrderedUnmarshalAdapterFor(orderedMap)

		if orderedUnmarshaler != nil {
			defer orderedUnmarshaler.Redeem()

			return orderedUnmarshaler.OrderedUnmarshal(trimmedData, orderedMap)
		}

		// no support found in registered adapters, fallback to the default (unordered) case
	}

	unmarshaler := adapters.UnmarshalAdapterFor(value)
	if unmarshaler != nil {
		defer unmarshaler.Redeem()

		return unmarshaler.Unmarshal(trimmedData, value)
	}

	// no support found in registered adapters, fallback to the default standard library.
	//
	// This only happens when tinkering with the global registry of adapters, since the default handles all the above cases.
	return json.Unmarshal(trimmedData, value) // Codecov ignore // this is a safeguard not easily simulated in tests
}

// FromDynamicJSON turns a go value into a properly JSON typed structure.
//
// "Dynamic JSON" refers to what you get when unmarshaling JSON into an untyped any,
// i.e. objects are represented by map[string]any, arrays by []any, and
// all numbers are represented as float64.
//
// NOTE: target must be a pointer.
//
// # Maintaining the order of keys in objects
//
// If source and target implement [ifaces.Ordered] and [ifaces.SetOrdered] respectively,
// they are considered "ordered maps" and the order of keys is maintained in the
// "jsonification" process. In that case, map[string]any values are replaced by (ordered) [JSONMapSlice] ones.
func FromDynamicJSON(source, target any) error {
	b, err := WriteJSON(source)
	if err != nil {
		return err
	}

	return ReadJSON(b, target)
}
