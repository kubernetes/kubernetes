// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"bytes"
	"io"

	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/go-yaml/yaml"
)

const (
	WideSequenceStyle    SequenceIndentStyle = "wide"
	CompactSequenceStyle SequenceIndentStyle = "compact"
	DefaultIndent                            = 2
	// BareSeqNodeWrappingKey kyaml uses reader annotations to track resources, it is not possible to
	// add them to bare sequence nodes, this key is used to wrap such bare
	// sequence nodes into map node, byteio_writer unwraps it while writing back
	BareSeqNodeWrappingKey = "bareSeqNodeWrappingKey"
)

// SeqIndentType holds the indentation style for sequence nodes
type SequenceIndentStyle string

// EncoderOptions are options that can be used to configure the encoder,
// do not expose new options without considerable justification
type EncoderOptions struct {
	// SeqIndent is the indentation style for YAML Sequence nodes
	SeqIndent SequenceIndentStyle
}

// Expose the yaml.v3 functions so this package can be used as a replacement

type (
	Decoder     = yaml.Decoder
	Encoder     = yaml.Encoder
	IsZeroer    = yaml.IsZeroer
	Kind        = yaml.Kind
	Marshaler   = yaml.Marshaler
	Node        = yaml.Node
	Style       = yaml.Style
	TypeError   = yaml.TypeError
	Unmarshaler = yaml.Unmarshaler
)

var Marshal = func(in interface{}) ([]byte, error) {
	var buf bytes.Buffer
	err := NewEncoder(&buf).Encode(in)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var (
	Unmarshal  = yaml.Unmarshal
	NewDecoder = yaml.NewDecoder
	NewEncoder = func(w io.Writer) *yaml.Encoder {
		e := yaml.NewEncoder(w)
		e.SetIndent(DefaultIndent)
		e.CompactSeqIndent()
		return e
	}
)

// MarshalWithOptions marshals the input interface with provided options
func MarshalWithOptions(in interface{}, opts *EncoderOptions) ([]byte, error) {
	var buf bytes.Buffer
	err := NewEncoderWithOptions(&buf, opts).Encode(in)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// NewEncoderWithOptions returns the encoder with provided options
func NewEncoderWithOptions(w io.Writer, opts *EncoderOptions) *yaml.Encoder {
	encoder := NewEncoder(w)
	encoder.SetIndent(DefaultIndent)
	if opts.SeqIndent == WideSequenceStyle {
		encoder.DefaultSeqIndent()
	} else {
		encoder.CompactSeqIndent()
	}
	return encoder
}

var (
	AliasNode    yaml.Kind = yaml.AliasNode
	DocumentNode yaml.Kind = yaml.DocumentNode
	MappingNode  yaml.Kind = yaml.MappingNode
	ScalarNode   yaml.Kind = yaml.ScalarNode
	SequenceNode yaml.Kind = yaml.SequenceNode
)

var (
	DoubleQuotedStyle yaml.Style = yaml.DoubleQuotedStyle
	FlowStyle         yaml.Style = yaml.FlowStyle
	FoldedStyle       yaml.Style = yaml.FoldedStyle
	LiteralStyle      yaml.Style = yaml.LiteralStyle
	SingleQuotedStyle yaml.Style = yaml.SingleQuotedStyle
	TaggedStyle       yaml.Style = yaml.TaggedStyle
)
