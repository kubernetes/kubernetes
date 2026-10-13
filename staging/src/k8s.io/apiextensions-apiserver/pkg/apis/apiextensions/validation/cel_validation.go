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

package validation

import (
	"fmt"
	"math"
	"sort"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/utils/ptr"
)

// unbounded uses nil to represent an unbounded cardinality value.
var unbounded *uint64 = nil

// CELSchemaContext keeps track of data used by x-kubernetes-validations rules for a specific schema node.
type CELSchemaContext struct {
	// withinValidationRuleScope is true if the schema at the current level or above have x-kubernetes-validations rules. typeInfo
	// should only be populated for schema nodes where this is true.
	withinValidationRuleScope bool

	// typeInfo is lazily loaded for schema nodes withinValidationRuleScope and may be
	// populated one of two possible ways:
	//   1. Using a typeInfoAccessor to access it from the parent's type info. This is a cheap operation and should be
	//      used when a schema at a higher level already has type info.
	//   2. Using a converter to construct type info from the jsonSchema. This is an expensive operation.
	typeInfo *CELTypeInfo
	// typeInfoErr is any cached error resulting from an attempt to lazily load typeInfo.
	typeInfoErr error

	// parent is the context of the parent schema node, or nil if this is the context for the root schema node.
	parent *CELSchemaContext
	// typeInfoAccessor provides a way to access the type info of this schema node from the parent CELSchemaContext.
	// nil if not extraction is possible, or the parent is nil.
	typeInfoAccessor typeInfoAccessor

	// jsonSchema is the schema for this CELSchemaContext node. It must be non-nil.
	jsonSchema *apiextensions.JSONSchemaProps
	// converter converts a JSONSchemaProps to CELTypeInfo.
	// Tests that check how many conversions are performed during CRD validation wrap DefaultConverter
	// with a converter that counts how many conversion operations.
	converter converter

	// MaxCardinality represents a limit to the number of data elements that can exist for the current
	// schema based on MaxProperties or MaxItems limits present on parent schemas, If all parent
	// map and array schemas have MaxProperties or MaxItems limits declared MaxCardinality is
	// an int pointer representing the product of these limits.  If least one parent map or list schema
	// does not have a MaxProperties or MaxItems limits set, the MaxCardinality is nil, indicating
	// that the parent schemas offer no bound to the number of times a data element for the current
	// schema can exist.
	MaxCardinality *uint64
	// TotalCost accumulates the x-kubernetes-validators estimated rule cost total for an entire custom resource
	// definition. A single TotalCost is allocated for each CustomResourceDefinition and passed through the stack as the
	// CustomResourceDefinition's OpenAPIv3 schema is recursively validated.
	TotalCost *TotalCost
}

// TypeInfo returns the CELTypeInfo for this CELSchemaContext node.  Returns nil, nil if this CELSchemaContext is nil,
// or if current level or above does not have x-kubernetes-validations rules. The returned type info is shared and
// should not be modified by the caller.
func (c *CELSchemaContext) TypeInfo() (*CELTypeInfo, error) {
	if c == nil || !c.withinValidationRuleScope {
		return nil, nil
	}
	if c.typeInfo != nil || c.typeInfoErr != nil {
		return c.typeInfo, c.typeInfoErr // return already computed result if available
	}

	// If able to get the type info from the parent's type info, prefer this approach
	// since it is more efficient.
	if c.parent != nil {
		parentTypeInfo, parentErr := c.parent.TypeInfo()
		if parentErr != nil {
			c.typeInfoErr = parentErr
			return nil, parentErr
		}
		if parentTypeInfo != nil && c.typeInfoAccessor != nil {
			c.typeInfo = c.typeInfoAccessor.accessTypeInfo(parentTypeInfo)
			if c.typeInfo != nil {
				return c.typeInfo, nil
			}
		}
	}
	// If unable to get the type info from the parent, convert the jsonSchema to type info.
	// This is expensive for large schemas.
	c.typeInfo, c.typeInfoErr = c.converter(c.jsonSchema, c.parent == nil || c.jsonSchema.XEmbeddedResource)
	return c.typeInfo, c.typeInfoErr
}

// CELTypeInfo represents all the typeInfo needed by CEL to compile x-kubernetes-validations rules for a schema node.
type CELTypeInfo struct {
	// Schema is a structural schema for this CELSchemaContext node. It must be non-nil.
	Schema *structuralschema.Structural
	// DeclType is a CEL declaration representation of Schema of this CELSchemaContext node. It must be non-nil.
	DeclType *cel.DeclType
}

// converter converts from JSON schema to a structural schema and a CEL declType, or returns an error if the conversion
// fails. This should be defaultConverter except in tests where it is useful to wrap it with a converter that tracks
// how many conversions have been performed.
type converter func(schema *apiextensions.JSONSchemaProps, isRoot bool) (*CELTypeInfo, error)

func defaultConverter(schema *apiextensions.JSONSchemaProps, isRoot bool) (*CELTypeInfo, error) {
	structural, err := structuralschema.NewStructural(schema)
	if err != nil {
		return nil, err
	}
	declType := model.SchemaDeclType(structural, isRoot)
	if declType == nil {
		return nil, fmt.Errorf("unable to convert structural schema to CEL declarations")
	}
	return &CELTypeInfo{structural, declType}, nil
}

// RootCELContext constructs CELSchemaContext for the given root schema.
func RootCELContext(schema *apiextensions.JSONSchemaProps) *CELSchemaContext {
	rootCardinality := uint64(1)
	r := &CELSchemaContext{
		jsonSchema:                schema,
		withinValidationRuleScope: len(schema.XValidations) > 0,
		MaxCardinality:            &rootCardinality,
		TotalCost:                 &TotalCost{},
		converter:                 defaultConverter,
	}
	return r
}

// ChildPropertyContext returns nil, nil if this CELSchemaContext is nil, otherwise constructs and returns a
// CELSchemaContext for propertyName.
func (c *CELSchemaContext) ChildPropertyContext(propSchema *apiextensions.JSONSchemaProps, propertyName string) *CELSchemaContext {
	if c == nil {
		return nil
	}
	return c.childContext(propSchema, propertyTypeInfoAccessor{propertyName: propertyName})
}

// ChildAdditionalPropertiesContext returns nil, nil if this CELSchemaContext is nil, otherwise it constructs and returns
// a CELSchemaContext for the properties of an object if this CELSchemaContext is an object.
// schema must be non-nil and have a non-nil schema.AdditionalProperties.
func (c *CELSchemaContext) ChildAdditionalPropertiesContext(propsSchema *apiextensions.JSONSchemaProps) *CELSchemaContext {
	if c == nil {
		return nil
	}
	return c.childContext(propsSchema, additionalItemsTypeInfoAccessor{})
}

// ChildItemsContext returns nil, nil if this CELSchemaContext is nil, otherwise it constructs and returns a CELSchemaContext
// for the items of an array if this CELSchemaContext is an array.
func (c *CELSchemaContext) ChildItemsContext(itemsSchema *apiextensions.JSONSchemaProps) *CELSchemaContext {
	if c == nil {
		return nil
	}
	return c.childContext(itemsSchema, itemsTypeInfoAccessor{})
}

// childContext returns nil, nil if this CELSchemaContext is nil, otherwise it constructs a new CELSchemaContext for the
// given child schema of the current schema context.
// accessor optionally provides a way to access CELTypeInfo of the child from the current schema context's CELTypeInfo.
// childContext returns a CELSchemaContext where the MaxCardinality is multiplied by the
// factor that the schema increases the cardinality of its children. If the CELSchemaContext's
// MaxCardinality is unbounded (nil) or the factor that the schema increase the cardinality
// is unbounded, the resulting CELSchemaContext's MaxCardinality is also unbounded.
func (c *CELSchemaContext) childContext(child *apiextensions.JSONSchemaProps, accessor typeInfoAccessor) *CELSchemaContext {
	result := &CELSchemaContext{
		parent:                    c,
		typeInfoAccessor:          accessor,
		withinValidationRuleScope: c.withinValidationRuleScope,
		TotalCost:                 c.TotalCost,
		MaxCardinality:            unbounded,
		converter:                 c.converter,
	}
	if child != nil {
		result.jsonSchema = child
		if len(child.XValidations) > 0 {
			result.withinValidationRuleScope = true
		}
	}
	if c.jsonSchema == nil {
		// nil schemas can be passed since we call ChildSchemaContext
		// before ValidateCustomResourceDefinitionOpenAPISchema performs its nil check
		return result
	}
	if c.MaxCardinality == unbounded {
		return result
	}
	maxElements := extractMaxElements(c.jsonSchema)
	if maxElements == unbounded {
		return result
	}
	result.MaxCardinality = ptr.To[uint64](multiplyWithOverflowGuard(*c.MaxCardinality, *maxElements))
	return result
}

type typeInfoAccessor interface {
	// accessTypeInfo looks up type information for a child schema from a non-nil parentTypeInfo and returns it,
	// or returns nil if the child schema information is not accessible. For example, a nil
	// return value is expected when a property name is unescapable in CEL.
	// The caller MUST ensure the provided parentTypeInfo is non-nil.
	accessTypeInfo(parentTypeInfo *CELTypeInfo) *CELTypeInfo
}

type propertyTypeInfoAccessor struct {
	// propertyName is the property name in the parent schema that this schema is declared at.
	propertyName string
}

func (c propertyTypeInfoAccessor) accessTypeInfo(parentTypeInfo *CELTypeInfo) *CELTypeInfo {
	if parentTypeInfo.Schema.Properties != nil {
		propSchema := parentTypeInfo.Schema.Properties[c.propertyName]
		if escapedPropName, ok := cel.Escape(c.propertyName); ok {
			if fieldDeclType, ok := parentTypeInfo.DeclType.Fields[escapedPropName]; ok {
				return &CELTypeInfo{Schema: &propSchema, DeclType: fieldDeclType.Type}
			} // else fields with unknown types are omitted from CEL validation entirely
		} // fields with unescapable names are expected to be absent
	}
	return nil
}

type itemsTypeInfoAccessor struct{}

func (c itemsTypeInfoAccessor) accessTypeInfo(parentTypeInfo *CELTypeInfo) *CELTypeInfo {
	if parentTypeInfo.Schema.Items != nil {
		itemsSchema := parentTypeInfo.Schema.Items
		itemsDeclType := parentTypeInfo.DeclType.ElemType
		return &CELTypeInfo{Schema: itemsSchema, DeclType: itemsDeclType}
	}
	return nil
}

type additionalItemsTypeInfoAccessor struct{}

func (c additionalItemsTypeInfoAccessor) accessTypeInfo(parentTypeInfo *CELTypeInfo) *CELTypeInfo {
	if parentTypeInfo.Schema.AdditionalProperties != nil {
		propsSchema := parentTypeInfo.Schema.AdditionalProperties.Structural
		valuesDeclType := parentTypeInfo.DeclType.ElemType
		return &CELTypeInfo{Schema: propsSchema, DeclType: valuesDeclType}
	}
	return nil
}

// TotalCost tracks the total cost of evaluating all the x-kubernetes-validations rules of a CustomResourceDefinition.
type TotalCost struct {
	// Total accumulates the x-kubernetes-validations estimated rule cost total.
	Total uint64
	// MostExpensive accumulates the top 4 most expensive rules contributing to the Total. Only rules
	// that accumulate at least 1% of total cost limit are included.
	MostExpensive []RuleCost
}

// ObserveExpressionCost accumulates the cost of evaluating a -kubernetes-validations rule.
func (c *TotalCost) ObserveExpressionCost(path *field.Path, cost uint64) {
	if math.MaxUint64-c.Total < cost {
		c.Total = math.MaxUint64
	} else {
		c.Total += cost
	}

	if cost < StaticEstimatedCRDCostLimit/100 { // ignore rules that contribute < 1% of total cost limit
		return
	}
	c.MostExpensive = append(c.MostExpensive, RuleCost{Path: path, Cost: cost})
	sort.Slice(c.MostExpensive, func(i, j int) bool {
		// sort in descending order so the most expensive rule is first
		return c.MostExpensive[i].Cost > c.MostExpensive[j].Cost
	})
	if len(c.MostExpensive) > 4 {
		c.MostExpensive = c.MostExpensive[:4]
	}
}

// RuleCost represents the cost of evaluating a single x-kubernetes-validations rule.
type RuleCost struct {
	Path *field.Path
	Cost uint64
}

// extractMaxElements returns the factor by which the schema increases the cardinality
// (number of possible data elements) of its children.  If schema is a map and has
// MaxProperties or an array has MaxItems, the int pointer of the max value is returned.
// If schema is a map or array and does not have MaxProperties or MaxItems,
// unbounded (nil) is returned to indicate that there is no limit to the possible
// number of data elements imposed by the current schema.  If the schema is an object, 1 is
// returned to indicate that there is no increase to the number of possible data elements
// for its children.  Primitives do not have children, but 1 is returned for simplicity.
func extractMaxElements(schema *apiextensions.JSONSchemaProps) *uint64 {
	switch schema.Type {
	case "object":
		if schema.AdditionalProperties != nil {
			if schema.MaxProperties != nil {
				maxProps := uint64(zeroIfNegative(*schema.MaxProperties))
				return &maxProps
			}
			return unbounded
		}
		// return 1 to indicate that all fields of an object exist at most one for
		// each occurrence of the object they are fields of
		return ptr.To[uint64](1)
	case "array":
		if schema.MaxItems != nil {
			maxItems := uint64(zeroIfNegative(*schema.MaxItems))
			return &maxItems
		}
		return unbounded
	default:
		return ptr.To[uint64](1)
	}
}

func zeroIfNegative(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}
