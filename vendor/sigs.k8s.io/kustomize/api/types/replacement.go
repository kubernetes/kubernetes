// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/resid"
)

const DefaultReplacementFieldPath = "metadata.name"

// Replacement defines how to perform a substitution
// where it is from and where it is to.
type Replacement struct {
	// The source of the value.
	Source *SourceSelector `json:"source,omitempty" yaml:"source,omitempty"`

	// The N fields to write the value to.
	Targets []*TargetSelector `json:"targets,omitempty" yaml:"targets,omitempty"`

	// Used to define an static value
	SourceValue *string `json:"sourceValue,omitempty" yaml:"sourceValue,omitempty"`
}

// SourceSelector is the source of the replacement transformer.
type SourceSelector struct {
	// A specific object to read it from.
	resid.ResId `json:",inline,omitempty" yaml:",inline,omitempty"`

	// Structured field path expected in the allowed object.
	FieldPath string `json:"fieldPath,omitempty" yaml:"fieldPath,omitempty"`

	// Used to refine the interpretation of the field.
	Options *FieldOptions `json:"options,omitempty" yaml:"options,omitempty"`
}

func (s *SourceSelector) String() string {
	if s == nil {
		return ""
	}
	result := []string{s.ResId.String()}
	if s.FieldPath != "" {
		result = append(result, s.FieldPath)
	}
	if opts := s.Options.String(); opts != "" {
		result = append(result, opts)
	}
	return strings.Join(result, ":")
}

// TargetSelector specifies fields in one or more objects.
type TargetSelector struct {
	// Include objects that match this.
	Select *Selector `json:"select" yaml:"select"`

	// From the allowed set, remove objects that match this.
	Reject []*Selector `json:"reject,omitempty" yaml:"reject,omitempty"`

	// Structured field paths expected in each allowed object.
	FieldPaths []string `json:"fieldPaths,omitempty" yaml:"fieldPaths,omitempty"`

	// Used to refine the interpretation of the field.
	Options *FieldOptions `json:"options,omitempty" yaml:"options,omitempty"`
}

// FieldOptions refine the interpretation of FieldPaths.
type FieldOptions struct {
	// Used to split/join the field.
	Delimiter string `json:"delimiter,omitempty" yaml:"delimiter,omitempty"`

	// Which position in the split to consider.
	Index int `json:"index,omitempty" yaml:"index,omitempty"`

	// TODO (#3492): Implement use of this option
	// None, Base64, URL, Hex, etc
	Encoding string `json:"encoding,omitempty" yaml:"encoding,omitempty"`

	// If field missing, add it.
	Create bool `json:"create,omitempty" yaml:"create,omitempty"`
}

func (fo *FieldOptions) String() string {
	if fo == nil || (fo.Delimiter == "" && !fo.Create) {
		return ""
	}
	return fmt.Sprintf("%s(%d), create=%t", fo.Delimiter, fo.Index, fo.Create)
}
