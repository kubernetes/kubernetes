// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

const (
	// NodeTagNull is the tag set for a yaml.Document that contains no data;
	// e.g. it isn't a Map, Slice, Document, etc
	NodeTagNull   = "!!null"
	NodeTagFloat  = "!!float"
	NodeTagString = "!!str"
	NodeTagBool   = "!!bool"
	NodeTagInt    = "!!int"
	NodeTagMap    = "!!map"
	NodeTagSeq    = "!!seq"
	NodeTagEmpty  = ""
)

// Field names
const (
	AnnotationsField = "annotations"
	APIVersionField  = "apiVersion"
	KindField        = "kind"
	MetadataField    = "metadata"
	DataField        = "data"
	BinaryDataField  = "binaryData"
	NameField        = "name"
	NamespaceField   = "namespace"
	LabelsField      = "labels"
)
