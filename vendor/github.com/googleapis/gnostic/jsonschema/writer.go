// Copyright 2017 Google Inc. All Rights Reserved.
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
	"gopkg.in/yaml.v2"
)

const indentation = "  "

func renderMap(info interface{}, indent string) (result string) {
	result = "{\n"
	innerIndent := indent + indentation
	switch pairs := info.(type) {
	case yaml.MapSlice:
		for i, pair := range pairs {
			// first print the key
			result += fmt.Sprintf("%s\"%+v\": ", innerIndent, pair.Key)
			// then the value
			switch value := pair.Value.(type) {
			case string:
				result += "\"" + value + "\""
			case bool:
				if value {
					result += "true"
				} else {
					result += "false"
				}
			case []interface{}:
				result += renderArray(value, innerIndent)
			case yaml.MapSlice:
				result += renderMap(value, innerIndent)
			case int:
				result += fmt.Sprintf("%d", value)
			case int64:
				result += fmt.Sprintf("%d", value)
			case []string:
				result += renderStringArray(value, innerIndent)
			default:
				result += fmt.Sprintf("???MapItem(Key:%+v, Value:%T)", value, value)
			}
			if i < len(pairs)-1 {
				result += ","
			}
			result += "\n"
		}
	default:
		// t is some other type that we didn't name.
	}

	result += indent + "}"
	return result
}

func renderArray(array []interface{}, indent string) (result string) {
	result = "[\n"
	innerIndent := indent + indentation
	for i, item := range array {
		switch item := item.(type) {
		case string:
			result += innerIndent + "\"" + item + "\""
		case bool:
			if item {
				result += innerIndent + "true"
			} else {
				result += innerIndent + "false"
			}
		case yaml.MapSlice:
			result += innerIndent + renderMap(item, innerIndent) + ""
		default:
			result += innerIndent + fmt.Sprintf("???ArrayItem(%+v)", item)
		}
		if i < len(array)-1 {
			result += ","
		}
		result += "\n"
	}
	result += indent + "]"
	return result
}

func renderStringArray(array []string, indent string) (result string) {
	result = "[\n"
	innerIndent := indent + indentation
	for i, item := range array {
		result += innerIndent + "\"" + item + "\""
		if i < len(array)-1 {
			result += ","
		}
		result += "\n"
	}
	result += indent + "]"
	return result
}

func Render(info yaml.MapSlice) string {
	return renderMap(info, "") + "\n"
}

func (object *SchemaNumber) jsonValue() interface{} {
	if object.Integer != nil {
		return object.Integer
	} else if object.Float != nil {
		return object.Float
	} else {
		return nil
	}
}

func (object *SchemaOrBoolean) jsonValue() interface{} {
	if object.Schema != nil {
		return object.Schema.jsonValue()
	} else if object.Boolean != nil {
		return *object.Boolean
	} else {
		return nil
	}
}

func (object *StringOrStringArray) jsonValue() interface{} {
	if object.String != nil {
		return *object.String
	} else if object.StringArray != nil {
		array := make([]interface{}, 0)
		for _, item := range *(object.StringArray) {
			array = append(array, item)
		}
		return array
	} else {
		return nil
	}
}

func (object *SchemaOrStringArray) jsonValue() interface{} {
	if object.Schema != nil {
		return object.Schema.jsonValue()
	} else if object.StringArray != nil {
		array := make([]interface{}, 0)
		for _, item := range *(object.StringArray) {
			array = append(array, item)
		}
		return array
	} else {
		return nil
	}
}

func (object *SchemaOrSchemaArray) jsonValue() interface{} {
	if object.Schema != nil {
		return object.Schema.jsonValue()
	} else if object.SchemaArray != nil {
		array := make([]interface{}, 0)
		for _, item := range *(object.SchemaArray) {
			array = append(array, item.jsonValue())
		}
		return array
	} else {
		return nil
	}
}

func (object *SchemaEnumValue) jsonValue() interface{} {
	if object.String != nil {
		return *object.String
	} else if object.Bool != nil {
		return *object.Bool
	} else {
		return nil
	}
}

func namedSchemaArrayValue(array *[]*NamedSchema) interface{} {
	m2 := yaml.MapSlice{}
	for _, pair := range *(array) {
		var item2 yaml.MapItem
		item2.Key = pair.Name
		item2.Value = pair.Value.jsonValue()
		m2 = append(m2, item2)
	}
	return m2
}

func namedSchemaOrStringArrayValue(array *[]*NamedSchemaOrStringArray) interface{} {
	m2 := yaml.MapSlice{}
	for _, pair := range *(array) {
		var item2 yaml.MapItem
		item2.Key = pair.Name
		item2.Value = pair.Value.jsonValue()
		m2 = append(m2, item2)
	}
	return m2
}

func schemaEnumArrayValue(array *[]SchemaEnumValue) []interface{} {
	a := make([]interface{}, 0)
	for _, item := range *array {
		a = append(a, item.jsonValue())
	}
	return a
}

func schemaArrayValue(array *[]*Schema) []interface{} {
	a := make([]interface{}, 0)
	for _, item := range *array {
		a = append(a, item.jsonValue())
	}
	return a
}

func (schema *Schema) jsonValue() yaml.MapSlice {
	m := yaml.MapSlice{}
	if schema.Title != nil {
		m = append(m, yaml.MapItem{Key: "title", Value: *schema.Title})
	}
	if schema.ID != nil {
		m = append(m, yaml.MapItem{Key: "id", Value: *schema.ID})
	}
	if schema.Schema != nil {
		m = append(m, yaml.MapItem{Key: "$schema", Value: *schema.Schema})
	}
	if schema.Type != nil {
		m = append(m, yaml.MapItem{Key: "type", Value: schema.Type.jsonValue()})
	}
	if schema.Items != nil {
		m = append(m, yaml.MapItem{Key: "items", Value: schema.Items.jsonValue()})
	}
	if schema.Description != nil {
		m = append(m, yaml.MapItem{Key: "description", Value: *schema.Description})
	}
	if schema.Required != nil {
		m = append(m, yaml.MapItem{Key: "required", Value: *schema.Required})
	}
	if schema.AdditionalProperties != nil {
		m = append(m, yaml.MapItem{Key: "additionalProperties", Value: schema.AdditionalProperties.jsonValue()})
	}
	if schema.PatternProperties != nil {
		m = append(m, yaml.MapItem{Key: "patternProperties", Value: namedSchemaArrayValue(schema.PatternProperties)})
	}
	if schema.Properties != nil {
		m = append(m, yaml.MapItem{Key: "properties", Value: namedSchemaArrayValue(schema.Properties)})
	}
	if schema.Dependencies != nil {
		m = append(m, yaml.MapItem{Key: "dependencies", Value: namedSchemaOrStringArrayValue(schema.Dependencies)})
	}
	if schema.Ref != nil {
		m = append(m, yaml.MapItem{Key: "$ref", Value: *schema.Ref})
	}
	if schema.MultipleOf != nil {
		m = append(m, yaml.MapItem{Key: "multipleOf", Value: schema.MultipleOf.jsonValue()})
	}
	if schema.Maximum != nil {
		m = append(m, yaml.MapItem{Key: "maximum", Value: schema.Maximum.jsonValue()})
	}
	if schema.ExclusiveMaximum != nil {
		m = append(m, yaml.MapItem{Key: "exclusiveMaximum", Value: schema.ExclusiveMaximum})
	}
	if schema.Minimum != nil {
		m = append(m, yaml.MapItem{Key: "minimum", Value: schema.Minimum.jsonValue()})
	}
	if schema.ExclusiveMinimum != nil {
		m = append(m, yaml.MapItem{Key: "exclusiveMinimum", Value: schema.ExclusiveMinimum})
	}
	if schema.MaxLength != nil {
		m = append(m, yaml.MapItem{Key: "maxLength", Value: *schema.MaxLength})
	}
	if schema.MinLength != nil {
		m = append(m, yaml.MapItem{Key: "minLength", Value: *schema.MinLength})
	}
	if schema.Pattern != nil {
		m = append(m, yaml.MapItem{Key: "pattern", Value: *schema.Pattern})
	}
	if schema.AdditionalItems != nil {
		m = append(m, yaml.MapItem{Key: "additionalItems", Value: schema.AdditionalItems.jsonValue()})
	}
	if schema.MaxItems != nil {
		m = append(m, yaml.MapItem{Key: "maxItems", Value: *schema.MaxItems})
	}
	if schema.MinItems != nil {
		m = append(m, yaml.MapItem{Key: "minItems", Value: *schema.MinItems})
	}
	if schema.UniqueItems != nil {
		m = append(m, yaml.MapItem{Key: "uniqueItems", Value: *schema.UniqueItems})
	}
	if schema.MaxProperties != nil {
		m = append(m, yaml.MapItem{Key: "maxProperties", Value: *schema.MaxProperties})
	}
	if schema.MinProperties != nil {
		m = append(m, yaml.MapItem{Key: "minProperties", Value: *schema.MinProperties})
	}
	if schema.Enumeration != nil {
		m = append(m, yaml.MapItem{Key: "enum", Value: schemaEnumArrayValue(schema.Enumeration)})
	}
	if schema.AllOf != nil {
		m = append(m, yaml.MapItem{Key: "allOf", Value: schemaArrayValue(schema.AllOf)})
	}
	if schema.AnyOf != nil {
		m = append(m, yaml.MapItem{Key: "anyOf", Value: schemaArrayValue(schema.AnyOf)})
	}
	if schema.OneOf != nil {
		m = append(m, yaml.MapItem{Key: "oneOf", Value: schemaArrayValue(schema.OneOf)})
	}
	if schema.Not != nil {
		m = append(m, yaml.MapItem{Key: "not", Value: schema.Not.jsonValue()})
	}
	if schema.Definitions != nil {
		m = append(m, yaml.MapItem{Key: "definitions", Value: namedSchemaArrayValue(schema.Definitions)})
	}
	if schema.Default != nil {
		m = append(m, yaml.MapItem{Key: "default", Value: *schema.Default})
	}
	if schema.Format != nil {
		m = append(m, yaml.MapItem{Key: "format", Value: *schema.Format})
	}
	return m
}

// JSONString returns a json representation of a schema.
func (schema *Schema) JSONString() string {
	info := schema.jsonValue()
	return Render(info)
}
