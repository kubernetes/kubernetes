// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"bytes"
	"io"

	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/go-yaml/yaml"
)

const CompactSequenceStyle = "compact"
const WideSequenceStyle = "wide"

const DefaultIndent = 2
const DefaultSequenceStyle = CompactSequenceStyle

var sequenceIndentationStyle = DefaultSequenceStyle
var indent = DefaultIndent

// Expose the yaml.v3 functions so this package can be used as a replacement

type Decoder = yaml.Decoder
type Encoder = yaml.Encoder
type IsZeroer = yaml.IsZeroer
type Kind = yaml.Kind
type Marshaler = yaml.Marshaler
type Node = yaml.Node
type Style = yaml.Style
type TypeError = yaml.TypeError
type Unmarshaler = yaml.Unmarshaler

var Marshal = func(in interface{}) ([]byte, error) {
	var buf bytes.Buffer
	err := NewEncoder(&buf).Encode(in)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
var Unmarshal = yaml.Unmarshal
var NewDecoder = yaml.NewDecoder
var NewEncoder = func(w io.Writer) *yaml.Encoder {
	e := yaml.NewEncoder(w)
	e.SetIndent(indent)
	if sequenceIndentationStyle == CompactSequenceStyle {
		e.CompactSeqIndent()
	}
	return e
}

var AliasNode yaml.Kind = yaml.AliasNode
var DocumentNode yaml.Kind = yaml.DocumentNode
var MappingNode yaml.Kind = yaml.MappingNode
var ScalarNode yaml.Kind = yaml.ScalarNode
var SequenceNode yaml.Kind = yaml.SequenceNode

var DoubleQuotedStyle yaml.Style = yaml.DoubleQuotedStyle
var FlowStyle yaml.Style = yaml.FlowStyle
var FoldedStyle yaml.Style = yaml.FoldedStyle
var LiteralStyle yaml.Style = yaml.LiteralStyle
var SingleQuotedStyle yaml.Style = yaml.SingleQuotedStyle
var TaggedStyle yaml.Style = yaml.TaggedStyle
