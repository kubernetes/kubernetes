// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

const DefaultReplacementFieldPath = "metadata.name"

// Replacement defines how to perform a substitution
// where it is from and where it is to.
type Replacement struct {
	// The source of the value.
	Source *SourceSelector `json:"source" yaml:"source"`

	// The N fields to write the value to.
	Targets []*TargetSelector `json:"targets" yaml:"targets"`
}

// SourceSelector is the source of the replacement transformer.
type SourceSelector struct {
	// A specific object to read it from.
	KrmId `json:",inline,omitempty" yaml:",inline,omitempty"`

	// Structured field path expected in the allowed object.
	FieldPath string `json:"fieldPath" yaml:"fieldPath"`

	// Used to refine the interpretation of the field.
	Options *FieldOptions `json:"options" yaml:"options"`
}

// TargetSelector specifies fields in one or more objects.
type TargetSelector struct {
	// Include objects that match this.
	Select *Selector `json:"select" yaml:"select"`

	// From the allowed set, remove objects that match this.
	Reject []*Selector `json:"reject" yaml:"reject"`

	// Structured field paths expected in each allowed object.
	FieldPaths []string `json:"fieldPaths" yaml:"fieldPaths"`

	// Used to refine the interpretation of the field.
	Options *FieldOptions `json:"options" yaml:"options"`
}

// FieldOptions refine the interpretation of FieldPaths.
type FieldOptions struct {
	// Used to split/join the field.
	Delimiter string `json:"delimiter" yaml:"delimiter"`

	// Which position in the split to consider.
	Index int `json:"index" yaml:"index"`

	// TODO (#3492): Implement use of this option
	// None, Base64, URL, Hex, etc
	Encoding string `json:"encoding" yaml:"encoding"`

	// If field missing, add it.
	Create bool `json:"create" yaml:"create"`
}
