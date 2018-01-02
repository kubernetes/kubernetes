// Copyright 2015 go-swagger maintainers
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

package spec

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

var schema = Schema{
	VendorExtensible: VendorExtensible{Extensions: map[string]interface{}{"x-framework": "go-swagger"}},
	SchemaProps: SchemaProps{
		Ref:              MustCreateRef("Cat"),
		Type:             []string{"string"},
		Format:           "date",
		Description:      "the description of this schema",
		Title:            "the title",
		Default:          "blah",
		Maximum:          float64Ptr(100),
		ExclusiveMaximum: true,
		ExclusiveMinimum: true,
		Minimum:          float64Ptr(5),
		MaxLength:        int64Ptr(100),
		MinLength:        int64Ptr(5),
		Pattern:          "\\w{1,5}\\w+",
		MaxItems:         int64Ptr(100),
		MinItems:         int64Ptr(5),
		UniqueItems:      true,
		MultipleOf:       float64Ptr(5),
		Enum:             []interface{}{"hello", "world"},
		MaxProperties:    int64Ptr(5),
		MinProperties:    int64Ptr(1),
		Required:         []string{"id", "name"},
		Items:            &SchemaOrArray{Schema: &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}},
		AllOf:            []Schema{Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}},
		Properties: map[string]Schema{
			"id":   Schema{SchemaProps: SchemaProps{Type: []string{"integer"}, Format: "int64"}},
			"name": Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
		},
		AdditionalProperties: &SchemaOrBool{Allows: true, Schema: &Schema{SchemaProps: SchemaProps{
			Type:   []string{"integer"},
			Format: "int32",
		}}},
	},
	SwaggerSchemaProps: SwaggerSchemaProps{
		Discriminator: "not this",
		ReadOnly:      true,
		XML:           &XMLObject{"sch", "io", "sw", true, true},
		ExternalDocs: &ExternalDocumentation{
			Description: "the documentation etc",
			URL:         "http://readthedocs.org/swagger",
		},
		Example: []interface{}{
			map[string]interface{}{
				"id":   1,
				"name": "a book",
			},
			map[string]interface{}{
				"id":   2,
				"name": "the thing",
			},
		},
	},
}

var schemaJSON = `{
	"x-framework": "go-swagger",
  "$ref": "Cat",
  "description": "the description of this schema",
  "maximum": 100,
  "minimum": 5,
  "exclusiveMaximum": true,
  "exclusiveMinimum": true,
  "maxLength": 100,
  "minLength": 5,
  "pattern": "\\w{1,5}\\w+",
  "maxItems": 100,
  "minItems": 5,
  "uniqueItems": true,
  "multipleOf": 5,
  "enum": ["hello", "world"],
  "type": "string",
  "format": "date",
  "title": "the title",
  "default": "blah",
  "maxProperties": 5,
  "minProperties": 1,
  "required": ["id", "name"],
  "items": {
    "type": "string"
  },
  "allOf": [
    {
      "type": "string"
    }
  ],
  "properties": {
    "id": {
      "type": "integer",
      "format": "int64"
    },
    "name": {
      "type": "string"
    }
  },
  "discriminator": "not this",
  "readOnly": true,
  "xml": {
    "name": "sch",
    "namespace": "io",
    "prefix": "sw",
    "wrapped": true,
    "attribute": true
  },
  "externalDocs": {
    "description": "the documentation etc",
    "url": "http://readthedocs.org/swagger"
  },
  "example": [
    {
      "id": 1,
      "name": "a book"
    },
    {
      "id": 2,
      "name": "the thing"
    }
  ],
  "additionalProperties": {
    "type": "integer",
    "format": "int32"
  }
}
`

func TestSchema(t *testing.T) {

	expected := map[string]interface{}{}
	json.Unmarshal([]byte(schemaJSON), &expected)
	b, err := json.Marshal(schema)
	if assert.NoError(t, err) {
		var actual map[string]interface{}
		json.Unmarshal(b, &actual)
		assert.Equal(t, expected, actual)
	}

	actual2 := Schema{}
	if assert.NoError(t, json.Unmarshal([]byte(schemaJSON), &actual2)) {
		assert.Equal(t, schema.Ref, actual2.Ref)
		assert.Equal(t, schema.Description, actual2.Description)
		assert.Equal(t, schema.Maximum, actual2.Maximum)
		assert.Equal(t, schema.Minimum, actual2.Minimum)
		assert.Equal(t, schema.ExclusiveMinimum, actual2.ExclusiveMinimum)
		assert.Equal(t, schema.ExclusiveMaximum, actual2.ExclusiveMaximum)
		assert.Equal(t, schema.MaxLength, actual2.MaxLength)
		assert.Equal(t, schema.MinLength, actual2.MinLength)
		assert.Equal(t, schema.Pattern, actual2.Pattern)
		assert.Equal(t, schema.MaxItems, actual2.MaxItems)
		assert.Equal(t, schema.MinItems, actual2.MinItems)
		assert.True(t, actual2.UniqueItems)
		assert.Equal(t, schema.MultipleOf, actual2.MultipleOf)
		assert.Equal(t, schema.Enum, actual2.Enum)
		assert.Equal(t, schema.Type, actual2.Type)
		assert.Equal(t, schema.Format, actual2.Format)
		assert.Equal(t, schema.Title, actual2.Title)
		assert.Equal(t, schema.MaxProperties, actual2.MaxProperties)
		assert.Equal(t, schema.MinProperties, actual2.MinProperties)
		assert.Equal(t, schema.Required, actual2.Required)
		assert.Equal(t, schema.Items, actual2.Items)
		assert.Equal(t, schema.AllOf, actual2.AllOf)
		assert.Equal(t, schema.Properties, actual2.Properties)
		assert.Equal(t, schema.Discriminator, actual2.Discriminator)
		assert.Equal(t, schema.ReadOnly, actual2.ReadOnly)
		assert.Equal(t, schema.XML, actual2.XML)
		assert.Equal(t, schema.ExternalDocs, actual2.ExternalDocs)
		assert.Equal(t, schema.AdditionalProperties, actual2.AdditionalProperties)
		assert.Equal(t, schema.Extensions, actual2.Extensions)
		examples := actual2.Example.([]interface{})
		expEx := schema.Example.([]interface{})
		ex1 := examples[0].(map[string]interface{})
		ex2 := examples[1].(map[string]interface{})
		exp1 := expEx[0].(map[string]interface{})
		exp2 := expEx[1].(map[string]interface{})

		assert.EqualValues(t, exp1["id"], ex1["id"])
		assert.Equal(t, exp1["name"], ex1["name"])
		assert.EqualValues(t, exp2["id"], ex2["id"])
		assert.Equal(t, exp2["name"], ex2["name"])
	}

}
