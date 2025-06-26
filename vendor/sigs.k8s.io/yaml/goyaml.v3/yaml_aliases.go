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
	gopkg_yaml "go.yaml.in/yaml/v3"
)

// Type aliases for public types from go.yaml.in/yaml/v3
type (
	// Unmarshaler is implemented by types to customize their behavior when being unmarshaled from a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v3.Unmarshaler directly.
	Unmarshaler = gopkg_yaml.Unmarshaler

	// Marshaler is implemented by types to customize their behavior when being marshaled into a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v3.Marshaler directly.
	Marshaler = gopkg_yaml.Marshaler

	// IsZeroer is used to check whether an object is zero to determine whether it should be omitted when
	// marshaling with the omitempty flag. One notable implementation is time.Time.
	// Deprecated: Use go.yaml.in/yaml/v3.IsZeroer directly.
	IsZeroer = gopkg_yaml.IsZeroer

	// Decoder reads and decodes YAML values from an input stream.
	// Deprecated: Use go.yaml.in/yaml/v3.Decoder directly.
	Decoder = gopkg_yaml.Decoder

	// Encoder writes YAML values to an output stream.
	// Deprecated: Use go.yaml.in/yaml/v3.Encoder directly.
	Encoder = gopkg_yaml.Encoder

	// TypeError is returned by Unmarshal when one or more fields in the YAML document cannot be properly decoded.
	// Deprecated: Use go.yaml.in/yaml/v3.TypeError directly.
	TypeError = gopkg_yaml.TypeError

	// Node represents a YAML node in the document.
	// Deprecated: Use go.yaml.in/yaml/v3.Node directly.
	Node = gopkg_yaml.Node

	// Kind represents the kind of a YAML node.
	// Deprecated: Use go.yaml.in/yaml/v3.Kind directly.
	Kind = gopkg_yaml.Kind

	// Style represents the style of a YAML node.
	// Deprecated: Use go.yaml.in/yaml/v3.Style directly.
	Style = gopkg_yaml.Style
)

// Constants for Kind type from go.yaml.in/yaml/v3
const (
	// DocumentNode represents a YAML document node.
	// Deprecated: Use go.yaml.in/yaml/v3.DocumentNode directly.
	DocumentNode = gopkg_yaml.DocumentNode

	// SequenceNode represents a YAML sequence node.
	// Deprecated: Use go.yaml.in/yaml/v3.SequenceNode directly.
	SequenceNode = gopkg_yaml.SequenceNode

	// MappingNode represents a YAML mapping node.
	// Deprecated: Use go.yaml.in/yaml/v3.MappingNode directly.
	MappingNode = gopkg_yaml.MappingNode

	// ScalarNode represents a YAML scalar node.
	// Deprecated: Use go.yaml.in/yaml/v3.ScalarNode directly.
	ScalarNode = gopkg_yaml.ScalarNode

	// AliasNode represents a YAML alias node.
	// Deprecated: Use go.yaml.in/yaml/v3.AliasNode directly.
	AliasNode = gopkg_yaml.AliasNode
)

// Constants for Style type from go.yaml.in/yaml/v3
const (
	// TaggedStyle represents a tagged YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.TaggedStyle directly.
	TaggedStyle = gopkg_yaml.TaggedStyle

	// DoubleQuotedStyle represents a double-quoted YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.DoubleQuotedStyle directly.
	DoubleQuotedStyle = gopkg_yaml.DoubleQuotedStyle

	// SingleQuotedStyle represents a single-quoted YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.SingleQuotedStyle directly.
	SingleQuotedStyle = gopkg_yaml.SingleQuotedStyle

	// LiteralStyle represents a literal YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.LiteralStyle directly.
	LiteralStyle = gopkg_yaml.LiteralStyle

	// FoldedStyle represents a folded YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.FoldedStyle directly.
	FoldedStyle = gopkg_yaml.FoldedStyle

	// FlowStyle represents a flow YAML style.
	// Deprecated: Use go.yaml.in/yaml/v3.FlowStyle directly.
	FlowStyle = gopkg_yaml.FlowStyle
)

// Function aliases for public functions from go.yaml.in/yaml/v3
var (
	// Unmarshal decodes the first document found within the in byte slice and assigns decoded values into the out value.
	// Deprecated: Use go.yaml.in/yaml/v3.Unmarshal directly.
	Unmarshal = gopkg_yaml.Unmarshal

	// Marshal serializes the value provided into a YAML document.
	// Deprecated: Use go.yaml.in/yaml/v3.Marshal directly.
	Marshal = gopkg_yaml.Marshal

	// NewDecoder returns a new decoder that reads from r.
	// Deprecated: Use go.yaml.in/yaml/v3.NewDecoder directly.
	NewDecoder = gopkg_yaml.NewDecoder

	// NewEncoder returns a new encoder that writes to w.
	// Deprecated: Use go.yaml.in/yaml/v3.NewEncoder directly.
	NewEncoder = gopkg_yaml.NewEncoder
)
