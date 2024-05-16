// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package jsonschema

import (
	"fmt"
	"strings"
)

//
// DISPLAY
// The following methods display Schemas.
//

// Description returns a string representation of a string or string array.
func (s *StringOrStringArray) Description() string {
	if s.String != nil {
		return *s.String
	}
	if s.StringArray != nil {
		return strings.Join(*s.StringArray, ", ")
	}
	return ""
}

// Returns a string representation of a Schema.
func (schema *Schema) String() string {
	return schema.describeSchema("")
}

// Helper: Returns a string representation of a Schema indented by a specified string.
func (schema *Schema) describeSchema(indent string) string {
	result := ""
	if schema.Schema != nil {
		result += indent + "$schema: " + *(schema.Schema) + "\n"
	}
	if schema.ID != nil {
		result += indent + "id: " + *(schema.ID) + "\n"
	}
	if schema.MultipleOf != nil {
		result += indent + fmt.Sprintf("multipleOf: %+v\n", *(schema.MultipleOf))
	}
	if schema.Maximum != nil {
		result += indent + fmt.Sprintf("maximum: %+v\n", *(schema.Maximum))
	}
	if schema.ExclusiveMaximum != nil {
		result += indent + fmt.Sprintf("exclusiveMaximum: %+v\n", *(schema.ExclusiveMaximum))
	}
	if schema.Minimum != nil {
		result += indent + fmt.Sprintf("minimum: %+v\n", *(schema.Minimum))
	}
	if schema.ExclusiveMinimum != nil {
		result += indent + fmt.Sprintf("exclusiveMinimum: %+v\n", *(schema.ExclusiveMinimum))
	}
	if schema.MaxLength != nil {
		result += indent + fmt.Sprintf("maxLength: %+v\n", *(schema.MaxLength))
	}
	if schema.MinLength != nil {
		result += indent + fmt.Sprintf("minLength: %+v\n", *(schema.MinLength))
	}
	if schema.Pattern != nil {
		result += indent + fmt.Sprintf("pattern: %+v\n", *(schema.Pattern))
	}
	if schema.AdditionalItems != nil {
		s := schema.AdditionalItems.Schema
		if s != nil {
			result += indent + "additionalItems:\n"
			result += s.describeSchema(indent + "  ")
		} else {
			b := *(schema.AdditionalItems.Boolean)
			result += indent + fmt.Sprintf("additionalItems: %+v\n", b)
		}
	}
	if schema.Items != nil {
		result += indent + "items:\n"
		items := schema.Items
		if items.SchemaArray != nil {
			for i, s := range *(items.SchemaArray) {
				result += indent + "  " + fmt.Sprintf("%d", i) + ":\n"
				result += s.describeSchema(indent + "  " + "  ")
			}
		} else if items.Schema != nil {
			result += items.Schema.describeSchema(indent + "  " + "  ")
		}
	}
	if schema.MaxItems != nil {
		result += indent + fmt.Sprintf("maxItems: %+v\n", *(schema.MaxItems))
	}
	if schema.MinItems != nil {
		result += indent + fmt.Sprintf("minItems: %+v\n", *(schema.MinItems))
	}
	if schema.UniqueItems != nil {
		result += indent + fmt.Sprintf("uniqueItems: %+v\n", *(schema.UniqueItems))
	}
	if schema.MaxProperties != nil {
		result += indent + fmt.Sprintf("maxProperties: %+v\n", *(schema.MaxProperties))
	}
	if schema.MinProperties != nil {
		result += indent + fmt.Sprintf("minProperties: %+v\n", *(schema.MinProperties))
	}
	if schema.Required != nil {
		result += indent + fmt.Sprintf("required: %+v\n", *(schema.Required))
	}
	if schema.AdditionalProperties != nil {
		s := schema.AdditionalProperties.Schema
		if s != nil {
			result += indent + "additionalProperties:\n"
			result += s.describeSchema(indent + "  ")
		} else {
			b := *(schema.AdditionalProperties.Boolean)
			result += indent + fmt.Sprintf("additionalProperties: %+v\n", b)
		}
	}
	if schema.Properties != nil {
		result += indent + "properties:\n"
		for _, pair := range *(schema.Properties) {
			name := pair.Name
			s := pair.Value
			result += indent + "  " + name + ":\n"
			result += s.describeSchema(indent + "  " + "  ")
		}
	}
	if schema.PatternProperties != nil {
		result += indent + "patternProperties:\n"
		for _, pair := range *(schema.PatternProperties) {
			name := pair.Name
			s := pair.Value
			result += indent + "  " + name + ":\n"
			result += s.describeSchema(indent + "  " + "  ")
		}
	}
	if schema.Dependencies != nil {
		result += indent + "dependencies:\n"
		for _, pair := range *(schema.Dependencies) {
			name := pair.Name
			schemaOrStringArray := pair.Value
			s := schemaOrStringArray.Schema
			if s != nil {
				result += indent + "  " + name + ":\n"
				result += s.describeSchema(indent + "  " + "  ")
			} else {
				a := schemaOrStringArray.StringArray
				if a != nil {
					result += indent + "  " + name + ":\n"
					for _, s2 := range *a {
						result += indent + "  " + "  " + s2 + "\n"
					}
				}
			}

		}
	}
	if schema.Enumeration != nil {
		result += indent + "enumeration:\n"
		for _, value := range *(schema.Enumeration) {
			if value.String != nil {
				result += indent + "  " + fmt.Sprintf("%+v\n", *value.String)
			} else {
				result += indent + "  " + fmt.Sprintf("%+v\n", *value.Bool)
			}
		}
	}
	if schema.Type != nil {
		result += indent + fmt.Sprintf("type: %+v\n", schema.Type.Description())
	}
	if schema.AllOf != nil {
		result += indent + "allOf:\n"
		for _, s := range *(schema.AllOf) {
			result += s.describeSchema(indent + "  ")
			result += indent + "-\n"
		}
	}
	if schema.AnyOf != nil {
		result += indent + "anyOf:\n"
		for _, s := range *(schema.AnyOf) {
			result += s.describeSchema(indent + "  ")
			result += indent + "-\n"
		}
	}
	if schema.OneOf != nil {
		result += indent + "oneOf:\n"
		for _, s := range *(schema.OneOf) {
			result += s.describeSchema(indent + "  ")
			result += indent + "-\n"
		}
	}
	if schema.Not != nil {
		result += indent + "not:\n"
		result += schema.Not.describeSchema(indent + "  ")
	}
	if schema.Definitions != nil {
		result += indent + "definitions:\n"
		for _, pair := range *(schema.Definitions) {
			name := pair.Name
			s := pair.Value
			result += indent + "  " + name + ":\n"
			result += s.describeSchema(indent + "  " + "  ")
		}
	}
	if schema.Title != nil {
		result += indent + "title: " + *(schema.Title) + "\n"
	}
	if schema.Description != nil {
		result += indent + "description: " + *(schema.Description) + "\n"
	}
	if schema.Default != nil {
		result += indent + "default:\n"
		result += indent + fmt.Sprintf("  %+v\n", *(schema.Default))
	}
	if schema.Format != nil {
		result += indent + "format: " + *(schema.Format) + "\n"
	}
	if schema.Ref != nil {
		result += indent + "$ref: " + *(schema.Ref) + "\n"
	}
	return result
}
