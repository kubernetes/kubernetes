// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package model contains abstract representations of policy template and instance config objects.
package model

import (
	"strings"
)

// NewInstance returns an empty policy instance.
func NewInstance(info SourceMetadata) *Instance {
	return &Instance{
		Metadata:  &InstanceMetadata{},
		Selectors: []Selector{},
		Rules:     []Rule{},
		Meta:      info,
	}
}

// Instance represents the compiled, type-checked, and validated policy instance.
type Instance struct {
	APIVersion  string
	Kind        string
	Metadata    *InstanceMetadata
	Description string

	// Selectors determine whether the instance applies to the current evaluation context.
	// All Selector values must return true for the policy instance to be included in policy
	// evaluation step.
	Selectors []Selector

	// Rules represent reference data to be used in evaluation policy decisions.
	// Depending on the nature of the decisions being emitted, some or all Rules may be evaluated
	// and the results aggregated according to the decision types being emitted.
	Rules []Rule

	// Meta represents the source metadata from the input instance.
	Meta SourceMetadata
}

// MetadataMap returns the metadata name to value map, which can be used in evaluation.
// Only "name" field is supported for now.
func (i *Instance) MetadataMap() map[string]interface{} {
	return map[string]interface{}{
		"name": i.Metadata.Name,
	}
}

// InstanceMetadata contains standard metadata which may be associated with an instance.
type InstanceMetadata struct {
	UID       string
	Name      string
	Namespace string
}

// Selector interface indicates a pre-formatted instance selection condition.
//
// The implementations of such conditions are expected to be platform specific.
//
// Note, if there is a clear need to tailor selection more heavily, then the schema definition
// for a selector should be moved into the Template schema.
type Selector interface {
	isSelector()
}

// LabelSelector matches key, value pairs of labels associated with the evaluation context.
//
// In Kubernetes, the such labels are provided as 'resource.labels'.
type LabelSelector struct {
	// LabelValues provides a map of the string keys and values expected.
	LabelValues map[string]string
}

func (*LabelSelector) isSelector() {}

// ExpressionSelector matches a label against an existence condition.
type ExpressionSelector struct {
	// Label name being matched.
	Label string

	// Operator determines the evaluation behavior. Must be one of Exists, NotExists, In, or NotIn.
	Operator string

	// Values set, optional, to be used in the NotIn, In set membership tests.
	Values []interface{}
}

func (*ExpressionSelector) isSelector() {}

// Rule interface indicates the value types that may be used as Rule instances.
//
// Note, the code within the main repo deals exclusively with custom, yaml-based rules, but it
// is entirely possible to use a protobuf message as the rule container.
type Rule interface {
	isRule()
	GetID() int64
	GetFieldID(field string) int64
}

// CustomRule embeds the DynValue and represents rules whose type definition is provided in the
// policy template.
type CustomRule struct {
	*DynValue
}

func (*CustomRule) isRule() {}

// GetID returns the parse-time generated ID of the rule node.
func (c *CustomRule) GetID() int64 {
	return c.ID
}

// GetFieldID returns the parse-time generated ID pointing to the rule field. If field is not
// specified or is not found, falls back to the ID of the rule node.
func (c *CustomRule) GetFieldID(field string) int64 {
	if field == "" {
		return c.GetID()
	}
	paths := strings.Split(field, ".")
	val := c.DynValue
	for _, path := range paths {
		var f *Field
		var ok bool
		switch v := val.Value().(type) {
		case *ObjectValue:
			f, ok = v.GetField(path)
		case *MapValue:
			f, ok = v.GetField(path)
		}
		if !ok {
			return c.GetID()
		}
		val = f.Ref
	}
	return val.ID
}
