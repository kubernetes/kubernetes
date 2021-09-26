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
	"io/ioutil"

	"gopkg.in/yaml.v2"
)

// This is a global map of all known Schemas.
// It is initialized when the first Schema is created and inserted.
var schemas map[string]*Schema

// NewSchemaFromFile reads a schema from a file.
// Currently this assumes that schemas are stored in the source distribution of this project.
func NewSchemaFromFile(filename string) (schema *Schema, err error) {
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var info yaml.MapSlice
	err = yaml.Unmarshal(file, &info)
	if err != nil {
		return nil, err
	}
	return NewSchemaFromObject(info), nil
}

// NewSchemaFromObject constructs a schema from a parsed JSON object.
// Due to the complexity of the schema representation, this is a
// custom reader and not the standard Go JSON reader (encoding/json).
func NewSchemaFromObject(jsonData interface{}) *Schema {
	switch t := jsonData.(type) {
	default:
		fmt.Printf("schemaValue: unexpected type %T\n", t)
		return nil
	case yaml.MapSlice:
		schema := &Schema{}
		for _, mapItem := range t {
			k := mapItem.Key.(string)
			v := mapItem.Value

			switch k {
			case "$schema":
				schema.Schema = schema.stringValue(v)
			case "id":
				schema.ID = schema.stringValue(v)

			case "multipleOf":
				schema.MultipleOf = schema.numberValue(v)
			case "maximum":
				schema.Maximum = schema.numberValue(v)
			case "exclusiveMaximum":
				schema.ExclusiveMaximum = schema.boolValue(v)
			case "minimum":
				schema.Minimum = schema.numberValue(v)
			case "exclusiveMinimum":
				schema.ExclusiveMinimum = schema.boolValue(v)

			case "maxLength":
				schema.MaxLength = schema.intValue(v)
			case "minLength":
				schema.MinLength = schema.intValue(v)
			case "pattern":
				schema.Pattern = schema.stringValue(v)

			case "additionalItems":
				schema.AdditionalItems = schema.schemaOrBooleanValue(v)
			case "items":
				schema.Items = schema.schemaOrSchemaArrayValue(v)
			case "maxItems":
				schema.MaxItems = schema.intValue(v)
			case "minItems":
				schema.MinItems = schema.intValue(v)
			case "uniqueItems":
				schema.UniqueItems = schema.boolValue(v)

			case "maxProperties":
				schema.MaxProperties = schema.intValue(v)
			case "minProperties":
				schema.MinProperties = schema.intValue(v)
			case "required":
				schema.Required = schema.arrayOfStringsValue(v)
			case "additionalProperties":
				schema.AdditionalProperties = schema.schemaOrBooleanValue(v)
			case "properties":
				schema.Properties = schema.mapOfSchemasValue(v)
			case "patternProperties":
				schema.PatternProperties = schema.mapOfSchemasValue(v)
			case "dependencies":
				schema.Dependencies = schema.mapOfSchemasOrStringArraysValue(v)

			case "enum":
				schema.Enumeration = schema.arrayOfEnumValuesValue(v)

			case "type":
				schema.Type = schema.stringOrStringArrayValue(v)
			case "allOf":
				schema.AllOf = schema.arrayOfSchemasValue(v)
			case "anyOf":
				schema.AnyOf = schema.arrayOfSchemasValue(v)
			case "oneOf":
				schema.OneOf = schema.arrayOfSchemasValue(v)
			case "not":
				schema.Not = NewSchemaFromObject(v)
			case "definitions":
				schema.Definitions = schema.mapOfSchemasValue(v)

			case "title":
				schema.Title = schema.stringValue(v)
			case "description":
				schema.Description = schema.stringValue(v)

			case "default":
				schema.Default = &v

			case "format":
				schema.Format = schema.stringValue(v)
			case "$ref":
				schema.Ref = schema.stringValue(v)
			default:
				fmt.Printf("UNSUPPORTED (%s)\n", k)
			}
		}

		// insert schema in global map
		if schema.ID != nil {
			if schemas == nil {
				schemas = make(map[string]*Schema, 0)
			}
			schemas[*(schema.ID)] = schema
		}
		return schema
	}
	return nil
}

//
// BUILDERS
// The following methods build elements of Schemas from interface{} values.
// Each returns nil if it is unable to build the desired element.
//

// Gets the string value of an interface{} value if possible.
func (schema *Schema) stringValue(v interface{}) *string {
	switch v := v.(type) {
	default:
		fmt.Printf("stringValue: unexpected type %T\n", v)
	case string:
		return &v
	}
	return nil
}

// Gets the numeric value of an interface{} value if possible.
func (schema *Schema) numberValue(v interface{}) *SchemaNumber {
	number := &SchemaNumber{}
	switch v := v.(type) {
	default:
		fmt.Printf("numberValue: unexpected type %T\n", v)
	case float64:
		v2 := float64(v)
		number.Float = &v2
		return number
	case float32:
		v2 := float64(v)
		number.Float = &v2
		return number
	case int:
		v2 := int64(v)
		number.Integer = &v2
	}
	return nil
}

// Gets the integer value of an interface{} value if possible.
func (schema *Schema) intValue(v interface{}) *int64 {
	switch v := v.(type) {
	default:
		fmt.Printf("intValue: unexpected type %T\n", v)
	case float64:
		v2 := int64(v)
		return &v2
	case int64:
		return &v
	case int:
		v2 := int64(v)
		return &v2
	}
	return nil
}

// Gets the bool value of an interface{} value if possible.
func (schema *Schema) boolValue(v interface{}) *bool {
	switch v := v.(type) {
	default:
		fmt.Printf("boolValue: unexpected type %T\n", v)
	case bool:
		return &v
	}
	return nil
}

// Gets a map of Schemas from an interface{} value if possible.
func (schema *Schema) mapOfSchemasValue(v interface{}) *[]*NamedSchema {
	switch v := v.(type) {
	default:
		fmt.Printf("mapOfSchemasValue: unexpected type %T\n", v)
	case yaml.MapSlice:
		m := make([]*NamedSchema, 0)
		for _, mapItem := range v {
			k2 := mapItem.Key.(string)
			v2 := mapItem.Value
			pair := &NamedSchema{Name: k2, Value: NewSchemaFromObject(v2)}
			m = append(m, pair)
		}
		return &m
	}
	return nil
}

// Gets an array of Schemas from an interface{} value if possible.
func (schema *Schema) arrayOfSchemasValue(v interface{}) *[]*Schema {
	switch v := v.(type) {
	default:
		fmt.Printf("arrayOfSchemasValue: unexpected type %T\n", v)
	case []interface{}:
		m := make([]*Schema, 0)
		for _, v2 := range v {
			switch v2 := v2.(type) {
			default:
				fmt.Printf("arrayOfSchemasValue: unexpected type %T\n", v2)
			case yaml.MapSlice:
				s := NewSchemaFromObject(v2)
				m = append(m, s)
			}
		}
		return &m
	case yaml.MapSlice:
		m := make([]*Schema, 0)
		s := NewSchemaFromObject(v)
		m = append(m, s)
		return &m
	}
	return nil
}

// Gets a Schema or an array of Schemas from an interface{} value if possible.
func (schema *Schema) schemaOrSchemaArrayValue(v interface{}) *SchemaOrSchemaArray {
	switch v := v.(type) {
	default:
		fmt.Printf("schemaOrSchemaArrayValue: unexpected type %T\n", v)
	case []interface{}:
		m := make([]*Schema, 0)
		for _, v2 := range v {
			switch v2 := v2.(type) {
			default:
				fmt.Printf("schemaOrSchemaArrayValue: unexpected type %T\n", v2)
			case map[string]interface{}:
				s := NewSchemaFromObject(v2)
				m = append(m, s)
			}
		}
		return &SchemaOrSchemaArray{SchemaArray: &m}
	case yaml.MapSlice:
		s := NewSchemaFromObject(v)
		return &SchemaOrSchemaArray{Schema: s}
	}
	return nil
}

// Gets an array of strings from an interface{} value if possible.
func (schema *Schema) arrayOfStringsValue(v interface{}) *[]string {
	switch v := v.(type) {
	default:
		fmt.Printf("arrayOfStringsValue: unexpected type %T\n", v)
	case []string:
		return &v
	case string:
		a := []string{v}
		return &a
	case []interface{}:
		a := make([]string, 0)
		for _, v2 := range v {
			switch v2 := v2.(type) {
			default:
				fmt.Printf("arrayOfStringsValue: unexpected type %T\n", v2)
			case string:
				a = append(a, v2)
			}
		}
		return &a
	}
	return nil
}

// Gets a string or an array of strings from an interface{} value if possible.
func (schema *Schema) stringOrStringArrayValue(v interface{}) *StringOrStringArray {
	switch v := v.(type) {
	default:
		fmt.Printf("arrayOfStringsValue: unexpected type %T\n", v)
	case []string:
		s := &StringOrStringArray{}
		s.StringArray = &v
		return s
	case string:
		s := &StringOrStringArray{}
		s.String = &v
		return s
	case []interface{}:
		a := make([]string, 0)
		for _, v2 := range v {
			switch v2 := v2.(type) {
			default:
				fmt.Printf("arrayOfStringsValue: unexpected type %T\n", v2)
			case string:
				a = append(a, v2)
			}
		}
		s := &StringOrStringArray{}
		s.StringArray = &a
		return s
	}
	return nil
}

// Gets an array of enum values from an interface{} value if possible.
func (schema *Schema) arrayOfEnumValuesValue(v interface{}) *[]SchemaEnumValue {
	a := make([]SchemaEnumValue, 0)
	switch v := v.(type) {
	default:
		fmt.Printf("arrayOfEnumValuesValue: unexpected type %T\n", v)
	case []interface{}:
		for _, v2 := range v {
			switch v2 := v2.(type) {
			default:
				fmt.Printf("arrayOfEnumValuesValue: unexpected type %T\n", v2)
			case string:
				a = append(a, SchemaEnumValue{String: &v2})
			case bool:
				a = append(a, SchemaEnumValue{Bool: &v2})
			}
		}
	}
	return &a
}

// Gets a map of schemas or string arrays from an interface{} value if possible.
func (schema *Schema) mapOfSchemasOrStringArraysValue(v interface{}) *[]*NamedSchemaOrStringArray {
	m := make([]*NamedSchemaOrStringArray, 0)
	switch v := v.(type) {
	default:
		fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected type %T %+v\n", v, v)
	case yaml.MapSlice:
		for _, mapItem := range v {
			k2 := mapItem.Key.(string)
			v2 := mapItem.Value
			switch v2 := v2.(type) {
			default:
				fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected type %T %+v\n", v2, v2)
			case []interface{}:
				a := make([]string, 0)
				for _, v3 := range v2 {
					switch v3 := v3.(type) {
					default:
						fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected type %T %+v\n", v3, v3)
					case string:
						a = append(a, v3)
					}
				}
				s := &SchemaOrStringArray{}
				s.StringArray = &a
				pair := &NamedSchemaOrStringArray{Name: k2, Value: s}
				m = append(m, pair)
			}
		}
	}
	return &m
}

// Gets a schema or a boolean value from an interface{} value if possible.
func (schema *Schema) schemaOrBooleanValue(v interface{}) *SchemaOrBoolean {
	schemaOrBoolean := &SchemaOrBoolean{}
	switch v := v.(type) {
	case bool:
		schemaOrBoolean.Boolean = &v
	case yaml.MapSlice:
		schemaOrBoolean.Schema = NewSchemaFromObject(v)
	default:
		fmt.Printf("schemaOrBooleanValue: unexpected type %T\n", v)
	case []map[string]interface{}:

	}
	return schemaOrBoolean
}
