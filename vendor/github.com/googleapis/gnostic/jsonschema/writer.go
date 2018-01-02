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
				result += fmt.Sprintf("???MapItem(%+v, %T)", value, value)
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

func render(info yaml.MapSlice) string {
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
		m = append(m, yaml.MapItem{"title", *schema.Title})
	}
	if schema.ID != nil {
		m = append(m, yaml.MapItem{"id", *schema.ID})
	}
	if schema.Schema != nil {
		m = append(m, yaml.MapItem{"$schema", *schema.Schema})
	}
	if schema.Type != nil {
		m = append(m, yaml.MapItem{"type", schema.Type.jsonValue()})
	}
	if schema.Items != nil {
		m = append(m, yaml.MapItem{"items", schema.Items.jsonValue()})
	}
	if schema.Description != nil {
		m = append(m, yaml.MapItem{"description", *schema.Description})
	}
	if schema.Required != nil {
		m = append(m, yaml.MapItem{"required", *schema.Required})
	}
	if schema.AdditionalProperties != nil {
		m = append(m, yaml.MapItem{"additionalProperties", schema.AdditionalProperties.jsonValue()})
	}
	if schema.PatternProperties != nil {
		m = append(m, yaml.MapItem{"patternProperties", namedSchemaArrayValue(schema.PatternProperties)})
	}
	if schema.Properties != nil {
		m = append(m, yaml.MapItem{"properties", namedSchemaArrayValue(schema.Properties)})
	}
	if schema.Dependencies != nil {
		m = append(m, yaml.MapItem{"dependencies", namedSchemaOrStringArrayValue(schema.Dependencies)})
	}
	if schema.Ref != nil {
		m = append(m, yaml.MapItem{"$ref", *schema.Ref})
	}
	if schema.MultipleOf != nil {
		m = append(m, yaml.MapItem{"multipleOf", schema.MultipleOf.jsonValue()})
	}
	if schema.Maximum != nil {
		m = append(m, yaml.MapItem{"maximum", schema.Maximum.jsonValue()})
	}
	if schema.ExclusiveMaximum != nil {
		m = append(m, yaml.MapItem{"exclusiveMaximum", *schema.ExclusiveMaximum})
	}
	if schema.Minimum != nil {
		m = append(m, yaml.MapItem{"minimum", schema.Minimum.jsonValue()})
	}
	if schema.ExclusiveMinimum != nil {
		m = append(m, yaml.MapItem{"exclusiveMinimum", *schema.ExclusiveMinimum})
	}
	if schema.MaxLength != nil {
		m = append(m, yaml.MapItem{"maxLength", *schema.MaxLength})
	}
	if schema.MinLength != nil {
		m = append(m, yaml.MapItem{"minLength", *schema.MinLength})
	}
	if schema.Pattern != nil {
		m = append(m, yaml.MapItem{"pattern", *schema.Pattern})
	}
	if schema.AdditionalItems != nil {
		m = append(m, yaml.MapItem{"additionalItems", schema.AdditionalItems.jsonValue()})
	}
	if schema.MaxItems != nil {
		m = append(m, yaml.MapItem{"maxItems", *schema.MaxItems})
	}
	if schema.MinItems != nil {
		m = append(m, yaml.MapItem{"minItems", *schema.MinItems})
	}
	if schema.UniqueItems != nil {
		m = append(m, yaml.MapItem{"uniqueItems", *schema.UniqueItems})
	}
	if schema.MaxProperties != nil {
		m = append(m, yaml.MapItem{"maxProperties", *schema.MaxProperties})
	}
	if schema.MinProperties != nil {
		m = append(m, yaml.MapItem{"minProperties", *schema.MinProperties})
	}
	if schema.Enumeration != nil {
		m = append(m, yaml.MapItem{"enum", schemaEnumArrayValue(schema.Enumeration)})
	}
	if schema.AllOf != nil {
		m = append(m, yaml.MapItem{"allOf", schemaArrayValue(schema.AllOf)})
	}
	if schema.AnyOf != nil {
		m = append(m, yaml.MapItem{"anyOf", schemaArrayValue(schema.AnyOf)})
	}
	if schema.OneOf != nil {
		m = append(m, yaml.MapItem{"oneOf", schemaArrayValue(schema.OneOf)})
	}
	if schema.Not != nil {
		m = append(m, yaml.MapItem{"not", schema.Not.jsonValue()})
	}
	if schema.Definitions != nil {
		m = append(m, yaml.MapItem{"definitions", namedSchemaArrayValue(schema.Definitions)})
	}
	if schema.Default != nil {
		m = append(m, yaml.MapItem{"default", *schema.Default})
	}
	if schema.Format != nil {
		m = append(m, yaml.MapItem{"format", *schema.Format})
	}
	return m
}

// JSONString returns a json representation of a schema.
func (schema *Schema) JSONString() string {
	info := schema.jsonValue()
	return render(info)
}
