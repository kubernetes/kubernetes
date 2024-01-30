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

package openapi

import (
	"github.com/google/cel-go/common/types/ref"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var _ common.Schema = (*Schema)(nil)
var _ common.SchemaOrBool = (*SchemaOrBool)(nil)

type Schema struct {
	Schema *spec.Schema
}

type SchemaOrBool struct {
	SchemaOrBool *spec.SchemaOrBool
}

func (sb *SchemaOrBool) Schema() common.Schema {
	return &Schema{Schema: sb.SchemaOrBool.Schema}
}

func (sb *SchemaOrBool) Allows() bool {
	return sb.SchemaOrBool.Allows
}

func (s *Schema) Type() string {
	if len(s.Schema.Type) == 0 {
		return ""
	}
	return s.Schema.Type[0]
}

func (s *Schema) Format() string {
	return s.Schema.Format
}

func (s *Schema) Pattern() string {
	return s.Schema.Pattern
}

func (s *Schema) Items() common.Schema {
	if s.Schema.Items == nil || s.Schema.Items.Schema == nil {
		return nil
	}
	return &Schema{Schema: s.Schema.Items.Schema}
}

func (s *Schema) Properties() map[string]common.Schema {
	if s.Schema.Properties == nil {
		return nil
	}
	res := make(map[string]common.Schema, len(s.Schema.Properties))
	for n, prop := range s.Schema.Properties {
		// map value is unaddressable, create a shallow copy
		// this is a shallow non-recursive copy
		s := prop
		res[n] = &Schema{Schema: &s}
	}
	return res
}

func (s *Schema) AdditionalProperties() common.SchemaOrBool {
	if s.Schema.AdditionalProperties == nil {
		return nil
	}
	return &SchemaOrBool{SchemaOrBool: s.Schema.AdditionalProperties}
}

func (s *Schema) Default() any {
	return s.Schema.Default
}

func (s *Schema) Minimum() *float64 {
	return s.Schema.Minimum
}

func (s *Schema) IsExclusiveMinimum() bool {
	return s.Schema.ExclusiveMinimum
}

func (s *Schema) Maximum() *float64 {
	return s.Schema.Maximum
}

func (s *Schema) IsExclusiveMaximum() bool {
	return s.Schema.ExclusiveMaximum
}

func (s *Schema) MultipleOf() *float64 {
	return s.Schema.MultipleOf
}

func (s *Schema) UniqueItems() bool {
	return s.Schema.UniqueItems
}

func (s *Schema) MinItems() *int64 {
	return s.Schema.MinItems
}

func (s *Schema) MaxItems() *int64 {
	return s.Schema.MaxItems
}

func (s *Schema) MinLength() *int64 {
	return s.Schema.MinLength
}

func (s *Schema) MaxLength() *int64 {
	return s.Schema.MaxLength
}

func (s *Schema) MinProperties() *int64 {
	return s.Schema.MinProperties
}

func (s *Schema) MaxProperties() *int64 {
	return s.Schema.MaxProperties
}

func (s *Schema) Required() []string {
	return s.Schema.Required
}

func (s *Schema) Enum() []any {
	return s.Schema.Enum
}

func (s *Schema) Nullable() bool {
	return s.Schema.Nullable
}

func (s *Schema) AllOf() []common.Schema {
	var res []common.Schema
	for _, nestedSchema := range s.Schema.AllOf {
		nestedSchema := nestedSchema
		res = append(res, &Schema{&nestedSchema})
	}
	return res
}

func (s *Schema) AnyOf() []common.Schema {
	var res []common.Schema
	for _, nestedSchema := range s.Schema.AnyOf {
		nestedSchema := nestedSchema
		res = append(res, &Schema{&nestedSchema})
	}
	return res
}

func (s *Schema) OneOf() []common.Schema {
	var res []common.Schema
	for _, nestedSchema := range s.Schema.OneOf {
		nestedSchema := nestedSchema
		res = append(res, &Schema{&nestedSchema})
	}
	return res
}

func (s *Schema) Not() common.Schema {
	if s.Schema.Not == nil {
		return nil
	}
	return &Schema{s.Schema.Not}
}

func (s *Schema) IsXIntOrString() bool {
	return isXIntOrString(s.Schema)
}

func (s *Schema) IsXEmbeddedResource() bool {
	return isXEmbeddedResource(s.Schema)
}

func (s *Schema) IsXPreserveUnknownFields() bool {
	return isXPreserveUnknownFields(s.Schema)
}

func (s *Schema) XListType() string {
	return getXListType(s.Schema)
}

func (s *Schema) XMapType() string {
	return getXMapType(s.Schema)
}

func (s *Schema) XListMapKeys() []string {
	return getXListMapKeys(s.Schema)
}

func (s *Schema) XValidations() []common.ValidationRule {
	return getXValidations(s.Schema)
}

func (s *Schema) WithTypeAndObjectMeta() common.Schema {
	return &Schema{common.WithTypeAndObjectMeta(s.Schema)}
}

func UnstructuredToVal(unstructured any, schema *spec.Schema) ref.Val {
	return common.UnstructuredToVal(unstructured, &Schema{schema})
}

func SchemaDeclType(s *spec.Schema, isResourceRoot bool) *apiservercel.DeclType {
	return common.SchemaDeclType(&Schema{Schema: s}, isResourceRoot)
}

func MakeMapList(sts *spec.Schema, items []interface{}) (rv common.MapList) {
	return common.MakeMapList(&Schema{Schema: sts}, items)
}
