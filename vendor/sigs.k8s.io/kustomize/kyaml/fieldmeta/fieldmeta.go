// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package fieldmeta

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// FieldMeta contains metadata that may be attached to fields as comments
type FieldMeta struct {
	Schema spec.Schema

	Extensions XKustomize

	SettersSchema *spec.Schema
}

type XKustomize struct {
	SetBy               string               `yaml:"setBy,omitempty" json:"setBy,omitempty"`
	PartialFieldSetters []PartialFieldSetter `yaml:"partialSetters,omitempty" json:"partialSetters,omitempty"`
	FieldSetter         *PartialFieldSetter  `yaml:"setter,omitempty" json:"setter,omitempty"`
}

// PartialFieldSetter defines how to set part of a field rather than the full field
// value.  e.g. the tag part of an image field
type PartialFieldSetter struct {
	// Name is the name of this setter.
	Name string `yaml:"name" json:"name"`

	// Value is the current value that has been set.
	Value string `yaml:"value" json:"value"`
}

// IsEmpty returns true if the FieldMeta has any empty Schema
func (fm *FieldMeta) IsEmpty() bool {
	if fm == nil {
		return true
	}
	return reflect.DeepEqual(fm.Schema, spec.Schema{})
}

// Read reads the FieldMeta from a node
func (fm *FieldMeta) Read(n *yaml.RNode) error {
	// check for metadata on head and line comments
	comments := []string{n.YNode().LineComment, n.YNode().HeadComment}
	for _, c := range comments {
		if c == "" {
			continue
		}
		c := strings.TrimLeft(c, "#")

		// check for new short hand notation or fall back to openAPI ref format
		if !fm.processShortHand(c) {
			// if it doesn't Unmarshal that is fine, it means there is no metadata
			// other comments are valid, they just don't parse
			// TODO: consider more sophisticated parsing techniques similar to what is used
			// for go struct tags.
			if err := fm.Schema.UnmarshalJSON([]byte(c)); err != nil {
				// note: don't return an error if the comment isn't a fieldmeta struct
				return nil
			}
		}
		fe := fm.Schema.VendorExtensible.Extensions["x-kustomize"]
		if fe == nil {
			return nil
		}
		b, err := json.Marshal(fe)
		if err != nil {
			return errors.Wrap(err)
		}
		return json.Unmarshal(b, &fm.Extensions)
	}
	return nil
}

// processShortHand parses the comment for short hand ref, loads schema to fm
// and returns true if successful, returns false for any other cases and not throw
// error, as the comment might not be a setter ref
func (fm *FieldMeta) processShortHand(comment string) bool {
	input := map[string]string{}
	err := json.Unmarshal([]byte(comment), &input)
	if err != nil {
		return false
	}
	name := input[shortHandRef]
	if name == "" {
		return false
	}

	// check if setter with the name exists, else check for a substitution
	// setter and substitution can't have same name in shorthand

	setterRef, err := spec.NewRef(DefinitionsPrefix + SetterDefinitionPrefix + name)
	if err != nil {
		return false
	}

	setterRefBytes, err := setterRef.MarshalJSON()
	if err != nil {
		return false
	}

	if _, err := openapi.Resolve(&setterRef, fm.SettersSchema); err == nil {
		setterErr := fm.Schema.UnmarshalJSON(setterRefBytes)
		return setterErr == nil
	}

	substRef, err := spec.NewRef(DefinitionsPrefix + SubstitutionDefinitionPrefix + name)
	if err != nil {
		return false
	}

	substRefBytes, err := substRef.MarshalJSON()
	if err != nil {
		return false
	}

	if _, err := openapi.Resolve(&substRef, fm.SettersSchema); err == nil {
		substErr := fm.Schema.UnmarshalJSON(substRefBytes)
		return substErr == nil
	}
	return false
}

func isExtensionEmpty(x XKustomize) bool {
	if x.FieldSetter != nil {
		return false
	}
	if x.SetBy != "" {
		return false
	}
	if len(x.PartialFieldSetters) > 0 {
		return false
	}
	return true
}

// Write writes the FieldMeta to a node
func (fm *FieldMeta) Write(n *yaml.RNode) error {
	if !isExtensionEmpty(fm.Extensions) {
		return fm.WriteV1Setters(n)
	}

	// Ref is removed when a setter is deleted, so the Ref string could be empty.
	if fm.Schema.Ref.String() != "" {
		// Ex: {"$ref":"#/definitions/io.k8s.cli.setters.replicas"} should be converted to
		// {"$openAPI":"replicas"} and added to the line comment
		ref := fm.Schema.Ref.String()
		var shortHandRefValue string
		switch {
		case strings.HasPrefix(ref, DefinitionsPrefix+SetterDefinitionPrefix):
			shortHandRefValue = strings.TrimPrefix(ref, DefinitionsPrefix+SetterDefinitionPrefix)
		case strings.HasPrefix(ref, DefinitionsPrefix+SubstitutionDefinitionPrefix):
			shortHandRefValue = strings.TrimPrefix(ref, DefinitionsPrefix+SubstitutionDefinitionPrefix)
		default:
			return fmt.Errorf("unexpected ref format: %s", ref)
		}
		n.YNode().LineComment = fmt.Sprintf(`{"%s":"%s"}`, shortHandRef,
			shortHandRefValue)
	} else {
		n.YNode().LineComment = ""
	}

	return nil
}

// WriteV1Setters is the v1 setters way of writing setter definitions
// TODO: pmarupaka - remove this method after migration
func (fm *FieldMeta) WriteV1Setters(n *yaml.RNode) error {
	fm.Schema.VendorExtensible.AddExtension("x-kustomize", fm.Extensions)
	b, err := json.Marshal(fm.Schema)
	if err != nil {
		return errors.Wrap(err)
	}
	n.YNode().LineComment = string(b)
	return nil
}

// FieldValueType defines the type of input to register
type FieldValueType string

const (
	// String defines a string flag
	String FieldValueType = "string"
	// Bool defines a bool flag
	Bool = "boolean"
	// Int defines an int flag
	Int = "integer"
)

func (it FieldValueType) String() string {
	if it == "" {
		return "string"
	}
	return string(it)
}

func (it FieldValueType) Validate(value string) error {
	switch it {
	case Int:
		if _, err := strconv.Atoi(value); err != nil {
			return errors.WrapPrefixf(err, "value must be an int")
		}
	case Bool:
		if _, err := strconv.ParseBool(value); err != nil {
			return errors.WrapPrefixf(err, "value must be a bool")
		}
	}
	return nil
}

func (it FieldValueType) Tag() string {
	switch it {
	case String:
		return yaml.NodeTagString
	case Bool:
		return yaml.NodeTagBool
	case Int:
		return yaml.NodeTagInt
	}
	return ""
}

func (it FieldValueType) TagForValue(value string) string {
	switch it {
	case String:
		return yaml.NodeTagString
	case Bool:
		if _, err := strconv.ParseBool(string(it)); err != nil {
			return ""
		}
		return yaml.NodeTagBool
	case Int:
		if _, err := strconv.ParseInt(string(it), 0, 32); err != nil {
			return ""
		}
		return yaml.NodeTagInt
	}
	return ""
}

const (
	// CLIDefinitionsPrefix is the prefix for cli definition keys.
	CLIDefinitionsPrefix = "io.k8s.cli."

	// SetterDefinitionPrefix is the prefix for setter definition keys.
	SetterDefinitionPrefix = CLIDefinitionsPrefix + "setters."

	// SubstitutionDefinitionPrefix is the prefix for substitution definition keys.
	SubstitutionDefinitionPrefix = CLIDefinitionsPrefix + "substitutions."

	// DefinitionsPrefix is the prefix used to reference definitions in the OpenAPI
	DefinitionsPrefix = "#/definitions/"
)

// shortHandRef is the shorthand reference to setters and substitutions
var shortHandRef = "$openapi"

func SetShortHandRef(ref string) {
	shortHandRef = ref
}

func ShortHandRef() string {
	return shortHandRef
}
