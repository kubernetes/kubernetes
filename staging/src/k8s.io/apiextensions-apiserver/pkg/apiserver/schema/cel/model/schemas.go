/*
Copyright 2022 The Kubernetes Authors.

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

package model

import (
	"maps"

	apimachineryvalidation "k8s.io/apimachinery/pkg/util/validation"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// SchemaDeclType converts the structural schema to a CEL declaration, or returns nil if the
// structural schema should not be exposed in CEL expressions.
// Set isResourceRoot to true for the root of a custom resource or embedded resource.
//
// Schemas with XPreserveUnknownFields not exposed unless they are objects. Array and "maps" schemas
// are not exposed if their items or additionalProperties schemas are not exposed. Object Properties are not exposed
// if their schema is not exposed.
//
// The CEL declaration for objects with XPreserveUnknownFields does not expose unknown fields.
func SchemaDeclType(s *schema.Structural, isResourceRoot bool) *apiservercel.DeclType {
	return common.SchemaDeclType(&Structural{Structural: foldAllOfBounds(s)}, isResourceRoot)
}

// foldAllOfBounds returns a schema where, on every node, the maxLength, maxItems and
// maxProperties bounds declared inside allOf members (including allOf members nested in other
// allOf members) are folded into the bounds declared directly on the node, keeping the tightest
// bound of each kind. Tools like controller-gen declare bounds inside allOf when multiple
// validation markers apply to a single field, and the CEL cost estimator only consults the
// bounds attached directly to a node, so without folding such fields are cost estimated as
// unbounded (https://github.com/kubernetes/kubernetes/issues/134029).
//
// Bounds are only folded for the node types they constrain: maxLength for strings and
// x-kubernetes-int-or-string nodes, maxItems for arrays and maxProperties for objects.
// Bounds declared for descendant paths inside allOf members (items, properties and
// additionalProperties of a member) are not folded.
//
// Only allOf members are folded: because every allOf member must hold, the tightest member
// bound is guaranteed to bound the value, so folding it can never under-estimate cost.
// anyOf, oneOf and not members provide no such guarantee for any individual member and are
// deliberately ignored.
//
// The input schema is shared and must not be mutated, so folding is copy on write: the
// original pointer is returned whenever no folding is needed.
func foldAllOfBounds(s *schema.Structural) *schema.Structural {
	if s == nil {
		return nil
	}

	out := s
	copyOnWrite := func() {
		if out == s {
			shallow := *s
			out = &shallow
		}
	}

	if items := foldAllOfBounds(s.Items); items != s.Items {
		copyOnWrite()
		out.Items = items
	}
	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		if folded := foldAllOfBounds(s.AdditionalProperties.Structural); folded != s.AdditionalProperties.Structural {
			copyOnWrite()
			out.AdditionalProperties = &schema.StructuralOrBool{Structural: folded, Bool: s.AdditionalProperties.Bool}
		}
	}
	// foldedProps is cloned from the full Properties map when the first property needs
	// folding, then the folded properties overwrite their clones; it stays nil if no
	// property needs folding.
	var foldedProps map[string]schema.Structural
	for name := range s.Properties {
		prop := s.Properties[name]
		if folded := foldAllOfBounds(&prop); folded != &prop {
			if foldedProps == nil {
				foldedProps = maps.Clone(s.Properties)
			}
			foldedProps[name] = *folded
		}
	}
	if foldedProps != nil {
		copyOnWrite()
		out.Properties = foldedProps
	}

	if s.ValueValidation == nil || len(s.ValueValidation.AllOf) == 0 {
		return out
	}

	var get func(*schema.ValueValidation) *int64
	var set func(*schema.ValueValidation, *int64)
	switch {
	case s.Type == "string" || s.XIntOrString:
		get = func(vv *schema.ValueValidation) *int64 { return vv.MaxLength }
		set = func(vv *schema.ValueValidation, bound *int64) { vv.MaxLength = bound }
	case s.Type == "array":
		get = func(vv *schema.ValueValidation) *int64 { return vv.MaxItems }
		set = func(vv *schema.ValueValidation, bound *int64) { vv.MaxItems = bound }
	case s.Type == "object":
		get = func(vv *schema.ValueValidation) *int64 { return vv.MaxProperties }
		set = func(vv *schema.ValueValidation, bound *int64) { vv.MaxProperties = bound }
	default:
		return out
	}

	direct := get(s.ValueValidation)
	tightest := tightestAllOfBound(get, s.ValueValidation.AllOf)
	if tightest == nil || (direct != nil && *direct <= *tightest) {
		return out
	}

	copyOnWrite()
	vv := *out.ValueValidation
	set(&vv, new(*tightest))
	out.ValueValidation = &vv
	return out
}

// tightestAllOfBound returns the smallest bound selected by get that is declared across the
// given allOf members, recursing into allOf members nested inside other allOf members, or nil
// if no member declares one.
func tightestAllOfBound(get func(*schema.ValueValidation) *int64, allOf []schema.NestedValueValidation) *int64 {
	var tightest *int64
	for i := range allOf {
		member := &allOf[i]
		if bound := get(&member.ValueValidation); bound != nil && (tightest == nil || *bound < *tightest) {
			tightest = bound
		}
		if bound := tightestAllOfBound(get, member.AllOf); bound != nil && (tightest == nil || *bound < *tightest) {
			tightest = bound
		}
	}
	return tightest
}

// WithTypeAndObjectMeta ensures the kind, apiVersion and
// metadata.name and metadata.generateName properties are specified, making a shallow copy of the provided schema if needed.
func WithTypeAndObjectMeta(s *schema.Structural) *schema.Structural {
	if s.Properties != nil &&
		s.Properties["kind"].Type == "string" &&
		s.Properties["apiVersion"].Type == "string" &&
		s.Properties["metadata"].Type == "object" &&
		s.Properties["metadata"].Properties != nil &&
		s.Properties["metadata"].Properties["name"].Type == "string" &&
		s.Properties["metadata"].Properties["generateName"].Type == "string" {
		return s
	}
	result := &schema.Structural{
		AdditionalProperties: s.AdditionalProperties,
		Generic:              s.Generic,
		Extensions:           s.Extensions,
		ValueValidation:      s.ValueValidation,
		ValidationExtensions: s.ValidationExtensions,
	}
	props := make(map[string]schema.Structural, len(s.Properties))
	for k, prop := range s.Properties {
		props[k] = prop
	}
	stringType := schema.Structural{Generic: schema.Generic{Type: "string"}}
	props["kind"] = stringType
	props["apiVersion"] = stringType
	metadataProps := s.Properties["metadata"].Properties
	props["metadata"] = schema.Structural{
		Generic: schema.Generic{Type: "object"},
		Properties: map[string]schema.Structural{
			"name":         cappedLengthStringType(metadataProps["name"], nameMaxLength),
			"generateName": cappedLengthStringType(metadataProps["generateName"], nameMaxLength-1), // The generated suffix is at least 1 byte
		},
	}
	result.Properties = props

	return result
}

// nameMaxLength is the default maxLength applied to metadata.name.
const nameMaxLength = int64(apimachineryvalidation.DNS1123SubdomainMaxLength)

// cappedLengthStringType returns a string type whose max length is capped at maxLength,
// using the existing schema's maxLength instead when it is shorter.
func cappedLengthStringType(existing schema.Structural, maxLength int64) schema.Structural {
	if existing.ValueValidation != nil && existing.ValueValidation.MaxLength != nil {
		if *existing.ValueValidation.MaxLength < maxLength {
			maxLength = *existing.ValueValidation.MaxLength
		}
	}
	return schema.Structural{
		Generic:         schema.Generic{Type: "string"},
		ValueValidation: &schema.ValueValidation{MaxLength: &maxLength},
	}
}
