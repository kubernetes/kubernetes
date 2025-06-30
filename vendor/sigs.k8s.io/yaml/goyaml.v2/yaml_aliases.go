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

package yaml

import (
	gopkg_yaml "go.yaml.in/yaml/v2"
)

// Type aliases for public types from go.yaml.in/yaml/v2
type (
	// MapSlice encodes and decodes as a YAML map.
	// The order of keys is preserved when encoding and decoding.
	// Deprecated: Use go.yaml.in/yaml/v2.MapSlice directly.
	MapSlice = gopkg_yaml.MapSlice

	// MapItem is an item in a MapSlice.
	// Deprecated: Use go.yaml.in/yaml/v2.MapItem directly.
	MapItem = gopkg_yaml.MapItem

	// Unmarshaler is implemented by types to customize their behavior when being unmarshaled from a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v2.Unmarshaler directly.
	Unmarshaler = gopkg_yaml.Unmarshaler

	// Marshaler is implemented by types to customize their behavior when being marshaled into a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v2.Marshaler directly.
	Marshaler = gopkg_yaml.Marshaler

	// IsZeroer is used to check whether an object is zero to determine whether it should be omitted when
	// marshaling with the omitempty flag. One notable implementation is time.Time.
	// Deprecated: Use go.yaml.in/yaml/v2.IsZeroer directly.
	IsZeroer = gopkg_yaml.IsZeroer

	// Decoder reads and decodes YAML values from an input stream.
	// Deprecated: Use go.yaml.in/yaml/v2.Decoder directly.
	Decoder = gopkg_yaml.Decoder

	// Encoder writes YAML values to an output stream.
	// Deprecated: Use go.yaml.in/yaml/v2.Encoder directly.
	Encoder = gopkg_yaml.Encoder

	// TypeError is returned by Unmarshal when one or more fields in the YAML document cannot be properly decoded.
	// Deprecated: Use go.yaml.in/yaml/v2.TypeError directly.
	TypeError = gopkg_yaml.TypeError
)

// Function aliases for public functions from go.yaml.in/yaml/v2
var (
	// Unmarshal decodes the first document found within the in byte slice and assigns decoded values into the out value.
	// Deprecated: Use go.yaml.in/yaml/v2.Unmarshal directly.
	Unmarshal = gopkg_yaml.Unmarshal

	// UnmarshalStrict is like Unmarshal except that any fields that are found in the data that do not have corresponding struct members will result in an error.
	// Deprecated: Use go.yaml.in/yaml/v2.UnmarshalStrict directly.
	UnmarshalStrict = gopkg_yaml.UnmarshalStrict

	// Marshal serializes the value provided into a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v2.Marshal directly.
	Marshal = gopkg_yaml.Marshal

	// NewDecoder returns a new decoder that reads from r.
	// Deprecated: Use go.yaml.in/yaml/v2.NewDecoder directly.
	NewDecoder = gopkg_yaml.NewDecoder

	// NewEncoder returns a new encoder that writes to w.
	// Deprecated: Use go.yaml.in/yaml/v2.NewEncoder directly.
	NewEncoder = gopkg_yaml.NewEncoder

	// FutureLineWrap globally disables line wrapping when encoding long strings.
	// Deprecated: Use go.yaml.in/yaml/v2.FutureLineWrap directly.
	FutureLineWrap = gopkg_yaml.FutureLineWrap
)
