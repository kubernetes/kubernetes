/*
Copyright 2023 The Kubernetes Authors.

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
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiserver/pkg/cel/common"
)

var _ common.Schema = (*Structural)(nil)
var _ common.SchemaOrBool = (*StructuralOrBool)(nil)

type Structural struct {
	Structural *schema.Structural
}

type StructuralOrBool struct {
	StructuralOrBool *schema.StructuralOrBool
}

func (sb *StructuralOrBool) Schema() common.Schema {
	if sb.StructuralOrBool.Structural == nil {
		return nil
	}
	return &Structural{Structural: sb.StructuralOrBool.Structural}
}

func (sb *StructuralOrBool) Allows() bool {
	return sb.StructuralOrBool.Bool
}

func (s *Structural) Type() string {
	return s.Structural.Type
}

func (s *Structural) Format() string {
	if s.Structural.ValueValidation == nil {
		return ""
	}
	return s.Structural.ValueValidation.Format
}

func (s *Structural) Pattern() string {
	if s.Structural.ValueValidation == nil {
		return ""
	}
	return s.Structural.ValueValidation.Pattern
}

func (s *Structural) Items() common.Schema {
	return &Structural{Structural: s.Structural.Items}
}

func (s *Structural) Properties() map[string]common.Schema {
	if s.Structural.Properties == nil {
		return nil
	}
	res := make(map[string]common.Schema, len(s.Structural.Properties))
	for n, prop := range s.Structural.Properties {
		s := prop
		res[n] = &Structural{Structural: &s}
	}
	return res
}

func (s *Structural) AdditionalProperties() common.SchemaOrBool {
	if s.Structural.AdditionalProperties == nil {
		return nil
	}
	return &StructuralOrBool{StructuralOrBool: s.Structural.AdditionalProperties}
}

func (s *Structural) Default() any {
	return s.Structural.Default.Object
}

func (s *Structural) Minimum() *float64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.Minimum
}

func (s *Structural) IsExclusiveMinimum() bool {
	if s.Structural.ValueValidation == nil {
		return false
	}
	return s.Structural.ValueValidation.ExclusiveMinimum
}

func (s *Structural) Maximum() *float64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.Maximum
}

func (s *Structural) IsExclusiveMaximum() bool {
	if s.Structural.ValueValidation == nil {
		return false
	}
	return s.Structural.ValueValidation.ExclusiveMaximum
}

func (s *Structural) MultipleOf() *float64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MultipleOf
}

func (s *Structural) MinItems() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MinItems
}

func (s *Structural) MaxItems() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MaxItems
}

func (s *Structural) MinLength() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MinLength
}

func (s *Structural) MaxLength() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MaxLength
}

func (s *Structural) MinProperties() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MinProperties
}

func (s *Structural) MaxProperties() *int64 {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.MaxProperties
}

func (s *Structural) Required() []string {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	return s.Structural.ValueValidation.Required
}

func (s *Structural) UniqueItems() bool {
	// This field is forbidden in structural schema.
	// but you can just you x-kubernetes-list-type:set to get around it :)
	return false
}

func (s *Structural) Enum() []any {
	if s.Structural.ValueValidation == nil {
		return nil
	}
	ret := make([]any, 0, len(s.Structural.ValueValidation.Enum))
	for _, e := range s.Structural.ValueValidation.Enum {
		ret = append(ret, e.Object)
	}
	return ret
}

func (s *Structural) Nullable() bool {
	return s.Structural.Nullable
}

func (s *Structural) IsXIntOrString() bool {
	return s.Structural.XIntOrString
}

func (s *Structural) IsXEmbeddedResource() bool {
	return s.Structural.XEmbeddedResource
}

func (s *Structural) IsXPreserveUnknownFields() bool {
	return s.Structural.XPreserveUnknownFields
}

func (s *Structural) XListType() string {
	if s.Structural.XListType == nil {
		return ""
	}
	return *s.Structural.XListType
}

func (s *Structural) XMapType() string {
	if s.Structural.XMapType == nil {
		return ""
	}
	return *s.Structural.XMapType
}

func (s *Structural) XListMapKeys() []string {
	return s.Structural.XListMapKeys
}

func (s *Structural) AllOf() []common.Schema {
	var res []common.Schema
	for _, subSchema := range s.Structural.ValueValidation.AllOf {
		subSchema := subSchema
		res = append(res, nestedValueValidationToStructural(&subSchema))
	}
	return res
}

func (s *Structural) AnyOf() []common.Schema {
	var res []common.Schema
	for _, subSchema := range s.Structural.ValueValidation.AnyOf {
		subSchema := subSchema
		res = append(res, nestedValueValidationToStructural(&subSchema))
	}
	return res
}

func (s *Structural) OneOf() []common.Schema {
	var res []common.Schema
	for _, subSchema := range s.Structural.ValueValidation.OneOf {
		subSchema := subSchema
		res = append(res, nestedValueValidationToStructural(&subSchema))
	}
	return res
}

func (s *Structural) Not() common.Schema {
	if s.Structural.ValueValidation.Not == nil {
		return nil
	}
	return nestedValueValidationToStructural(s.Structural.ValueValidation.Not)
}

// nestedValueValidationToStructural converts a nested value validation to
// an equivalent structural schema instance.
//
// This lets us avoid needing a separate adaptor for the nested value
// validations, and doesn't cost too much since since we are usually exploring the
// entire schema anyway.
func nestedValueValidationToStructural(nvv *schema.NestedValueValidation) *Structural {
	var newItems *schema.Structural
	if nvv.Items != nil {
		newItems = nestedValueValidationToStructural(nvv.Items).Structural
	}

	var newProperties map[string]schema.Structural
	for k, v := range nvv.Properties {
		if newProperties == nil {
			newProperties = make(map[string]schema.Structural)
		}

		v := v
		newProperties[k] = *nestedValueValidationToStructural(&v).Structural
	}

	var newAdditionalProperties *schema.StructuralOrBool
	if nvv.AdditionalProperties != nil {
		newAdditionalProperties = &schema.StructuralOrBool{Structural: nestedValueValidationToStructural(nvv.AdditionalProperties).Structural}
	}

	return &Structural{
		Structural: &schema.Structural{
			Items:                newItems,
			Properties:           newProperties,
			AdditionalProperties: newAdditionalProperties,
			ValueValidation:      &nvv.ValueValidation,
			ValidationExtensions: nvv.ValidationExtensions,
		},
	}
}

type StructuralValidationRule struct {
	rule, message, messageExpression, fieldPath string
}

func (s *StructuralValidationRule) Rule() string {
	return s.rule
}
func (s *StructuralValidationRule) Message() string {
	return s.message
}
func (s *StructuralValidationRule) FieldPath() string {
	return s.fieldPath
}
func (s *StructuralValidationRule) MessageExpression() string {
	return s.messageExpression
}

func (s *Structural) XValidations() []common.ValidationRule {
	if len(s.Structural.XValidations) == 0 {
		return nil
	}
	result := make([]common.ValidationRule, len(s.Structural.XValidations))
	for i, v := range s.Structural.XValidations {
		result[i] = &StructuralValidationRule{rule: v.Rule, message: v.Message, messageExpression: v.MessageExpression, fieldPath: v.FieldPath}
	}
	return result
}

func (s *Structural) WithTypeAndObjectMeta() common.Schema {
	return &Structural{Structural: WithTypeAndObjectMeta(s.Structural)}
}
