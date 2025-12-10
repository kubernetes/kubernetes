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

//go:generate go run generate-base.go

package jsonschema

import (
	"fmt"
	"io/ioutil"
	"strconv"

	yaml "go.yaml.in/yaml/v3"
)

// This is a global map of all known Schemas.
// It is initialized when the first Schema is created and inserted.
var schemas map[string]*Schema

// NewBaseSchema builds a schema object from an embedded json representation.
func NewBaseSchema() (schema *Schema, err error) {
	b, err := baseSchemaBytes()
	if err != nil {
		return nil, err
	}
	var node yaml.Node
	err = yaml.Unmarshal(b, &node)
	if err != nil {
		return nil, err
	}
	return NewSchemaFromObject(&node), nil
}

// NewSchemaFromFile reads a schema from a file.
// Currently this assumes that schemas are stored in the source distribution of this project.
func NewSchemaFromFile(filename string) (schema *Schema, err error) {
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var node yaml.Node
	err = yaml.Unmarshal(file, &node)
	if err != nil {
		return nil, err
	}
	return NewSchemaFromObject(&node), nil
}

// NewSchemaFromObject constructs a schema from a parsed JSON object.
// Due to the complexity of the schema representation, this is a
// custom reader and not the standard Go JSON reader (encoding/json).
func NewSchemaFromObject(jsonData *yaml.Node) *Schema {
	switch jsonData.Kind {
	case yaml.DocumentNode:
		return NewSchemaFromObject(jsonData.Content[0])
	case yaml.MappingNode:
		schema := &Schema{}

		for i := 0; i < len(jsonData.Content); i += 2 {
			k := jsonData.Content[i].Value
			v := jsonData.Content[i+1]

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
				schema.Default = v

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

	default:
		fmt.Printf("schemaValue: unexpected node %+v\n", jsonData)
		return nil
	}

	return nil
}

//
// BUILDERS
// The following methods build elements of Schemas from interface{} values.
// Each returns nil if it is unable to build the desired element.
//

// Gets the string value of an interface{} value if possible.
func (schema *Schema) stringValue(v *yaml.Node) *string {
	switch v.Kind {
	case yaml.ScalarNode:
		return &v.Value
	default:
		fmt.Printf("stringValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets the numeric value of an interface{} value if possible.
func (schema *Schema) numberValue(v *yaml.Node) *SchemaNumber {
	number := &SchemaNumber{}
	switch v.Kind {
	case yaml.ScalarNode:
		switch v.Tag {
		case "!!float":
			v2, _ := strconv.ParseFloat(v.Value, 64)
			number.Float = &v2
			return number
		case "!!int":
			v2, _ := strconv.ParseInt(v.Value, 10, 64)
			number.Integer = &v2
			return number
		default:
			fmt.Printf("stringValue: unexpected node %+v\n", v)
		}
	default:
		fmt.Printf("stringValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets the integer value of an interface{} value if possible.
func (schema *Schema) intValue(v *yaml.Node) *int64 {
	switch v.Kind {
	case yaml.ScalarNode:
		switch v.Tag {
		case "!!float":
			v2, _ := strconv.ParseFloat(v.Value, 64)
			v3 := int64(v2)
			return &v3
		case "!!int":
			v2, _ := strconv.ParseInt(v.Value, 10, 64)
			return &v2
		default:
			fmt.Printf("intValue: unexpected node %+v\n", v)
		}
	default:
		fmt.Printf("intValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets the bool value of an interface{} value if possible.
func (schema *Schema) boolValue(v *yaml.Node) *bool {
	switch v.Kind {
	case yaml.ScalarNode:
		switch v.Tag {
		case "!!bool":
			v2, _ := strconv.ParseBool(v.Value)
			return &v2
		default:
			fmt.Printf("boolValue: unexpected node %+v\n", v)
		}
	default:
		fmt.Printf("boolValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets a map of Schemas from an interface{} value if possible.
func (schema *Schema) mapOfSchemasValue(v *yaml.Node) *[]*NamedSchema {
	switch v.Kind {
	case yaml.MappingNode:
		m := make([]*NamedSchema, 0)
		for i := 0; i < len(v.Content); i += 2 {
			k2 := v.Content[i].Value
			v2 := v.Content[i+1]
			pair := &NamedSchema{Name: k2, Value: NewSchemaFromObject(v2)}
			m = append(m, pair)
		}
		return &m
	default:
		fmt.Printf("mapOfSchemasValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets an array of Schemas from an interface{} value if possible.
func (schema *Schema) arrayOfSchemasValue(v *yaml.Node) *[]*Schema {
	switch v.Kind {
	case yaml.SequenceNode:
		m := make([]*Schema, 0)
		for _, v2 := range v.Content {
			switch v2.Kind {
			case yaml.MappingNode:
				s := NewSchemaFromObject(v2)
				m = append(m, s)
			default:
				fmt.Printf("arrayOfSchemasValue: unexpected node %+v\n", v2)
			}
		}
		return &m
	case yaml.MappingNode:
		m := make([]*Schema, 0)
		s := NewSchemaFromObject(v)
		m = append(m, s)
		return &m
	default:
		fmt.Printf("arrayOfSchemasValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets a Schema or an array of Schemas from an interface{} value if possible.
func (schema *Schema) schemaOrSchemaArrayValue(v *yaml.Node) *SchemaOrSchemaArray {
	switch v.Kind {
	case yaml.SequenceNode:
		m := make([]*Schema, 0)
		for _, v2 := range v.Content {
			switch v2.Kind {
			case yaml.MappingNode:
				s := NewSchemaFromObject(v2)
				m = append(m, s)
			default:
				fmt.Printf("schemaOrSchemaArrayValue: unexpected node %+v\n", v2)
			}
		}
		return &SchemaOrSchemaArray{SchemaArray: &m}
	case yaml.MappingNode:
		s := NewSchemaFromObject(v)
		return &SchemaOrSchemaArray{Schema: s}
	default:
		fmt.Printf("schemaOrSchemaArrayValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets an array of strings from an interface{} value if possible.
func (schema *Schema) arrayOfStringsValue(v *yaml.Node) *[]string {
	switch v.Kind {
	case yaml.ScalarNode:
		a := []string{v.Value}
		return &a
	case yaml.SequenceNode:
		a := make([]string, 0)
		for _, v2 := range v.Content {
			switch v2.Kind {
			case yaml.ScalarNode:
				a = append(a, v2.Value)
			default:
				fmt.Printf("arrayOfStringsValue: unexpected node %+v\n", v2)
			}
		}
		return &a
	default:
		fmt.Printf("arrayOfStringsValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets a string or an array of strings from an interface{} value if possible.
func (schema *Schema) stringOrStringArrayValue(v *yaml.Node) *StringOrStringArray {
	switch v.Kind {
	case yaml.ScalarNode:
		s := &StringOrStringArray{}
		s.String = &v.Value
		return s
	case yaml.SequenceNode:
		a := make([]string, 0)
		for _, v2 := range v.Content {
			switch v2.Kind {
			case yaml.ScalarNode:
				a = append(a, v2.Value)
			default:
				fmt.Printf("arrayOfStringsValue: unexpected node %+v\n", v2)
			}
		}
		s := &StringOrStringArray{}
		s.StringArray = &a
		return s
	default:
		fmt.Printf("arrayOfStringsValue: unexpected node %+v\n", v)
	}
	return nil
}

// Gets an array of enum values from an interface{} value if possible.
func (schema *Schema) arrayOfEnumValuesValue(v *yaml.Node) *[]SchemaEnumValue {
	a := make([]SchemaEnumValue, 0)
	switch v.Kind {
	case yaml.SequenceNode:
		for _, v2 := range v.Content {
			switch v2.Kind {
			case yaml.ScalarNode:
				switch v2.Tag {
				case "!!str":
					a = append(a, SchemaEnumValue{String: &v2.Value})
				case "!!bool":
					v3, _ := strconv.ParseBool(v2.Value)
					a = append(a, SchemaEnumValue{Bool: &v3})
				default:
					fmt.Printf("arrayOfEnumValuesValue: unexpected type %s\n", v2.Tag)
				}
			default:
				fmt.Printf("arrayOfEnumValuesValue: unexpected node %+v\n", v2)
			}
		}
	default:
		fmt.Printf("arrayOfEnumValuesValue: unexpected node %+v\n", v)
	}
	return &a
}

// Gets a map of schemas or string arrays from an interface{} value if possible.
func (schema *Schema) mapOfSchemasOrStringArraysValue(v *yaml.Node) *[]*NamedSchemaOrStringArray {
	m := make([]*NamedSchemaOrStringArray, 0)
	switch v.Kind {
	case yaml.MappingNode:
		for i := 0; i < len(v.Content); i += 2 {
			k2 := v.Content[i].Value
			v2 := v.Content[i+1]
			switch v2.Kind {
			case yaml.SequenceNode:
				a := make([]string, 0)
				for _, v3 := range v2.Content {
					switch v3.Kind {
					case yaml.ScalarNode:
						a = append(a, v3.Value)
					default:
						fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected node %+v\n", v3)
					}
				}
				s := &SchemaOrStringArray{}
				s.StringArray = &a
				pair := &NamedSchemaOrStringArray{Name: k2, Value: s}
				m = append(m, pair)
			default:
				fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected node %+v\n", v2)
			}
		}
	default:
		fmt.Printf("mapOfSchemasOrStringArraysValue: unexpected node %+v\n", v)
	}
	return &m
}

// Gets a schema or a boolean value from an interface{} value if possible.
func (schema *Schema) schemaOrBooleanValue(v *yaml.Node) *SchemaOrBoolean {
	schemaOrBoolean := &SchemaOrBoolean{}
	switch v.Kind {
	case yaml.ScalarNode:
		v2, _ := strconv.ParseBool(v.Value)
		schemaOrBoolean.Boolean = &v2
	case yaml.MappingNode:
		schemaOrBoolean.Schema = NewSchemaFromObject(v)
	default:
		fmt.Printf("schemaOrBooleanValue: unexpected node %+v\n", v)
	}
	return schemaOrBoolean
}
