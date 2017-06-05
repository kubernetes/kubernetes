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

var parameter = Parameter{
	VendorExtensible: VendorExtensible{Extensions: map[string]interface{}{
		"x-framework": "swagger-go",
	}},
	Refable: Refable{Ref: MustCreateRef("Dog")},
	CommonValidations: CommonValidations{
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
	},
	SimpleSchema: SimpleSchema{
		Type:             "string",
		Format:           "date",
		CollectionFormat: "csv",
		Items: &Items{
			Refable: Refable{Ref: MustCreateRef("Cat")},
		},
		Default: "8",
	},
	ParamProps: ParamProps{
		Name:        "param-name",
		In:          "header",
		Required:    true,
		Schema:      &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
		Description: "the description of this parameter",
	},
}

var parameterJSON = `{
	"items": {
		"$ref": "Cat"
	},
	"x-framework": "swagger-go",
  "$ref": "Dog",
  "description": "the description of this parameter",
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
	"name": "param-name",
	"in": "header",
	"required": true,
	"schema": {
		"type": "string"
	},
	"collectionFormat": "csv",
	"default": "8"
}`

func TestIntegrationParameter(t *testing.T) {
	var actual Parameter
	if assert.NoError(t, json.Unmarshal([]byte(parameterJSON), &actual)) {
		assert.EqualValues(t, actual, parameter)
	}

	assertParsesJSON(t, parameterJSON, parameter)
}

func TestParameterSerialization(t *testing.T) {
	items := &Items{
		SimpleSchema: SimpleSchema{Type: "string"},
	}

	intItems := &Items{
		SimpleSchema: SimpleSchema{Type: "int", Format: "int32"},
	}

	assertSerializeJSON(t, QueryParam("").Typed("string", ""), `{"type":"string","in":"query"}`)

	assertSerializeJSON(t,
		QueryParam("").CollectionOf(items, "multi"),
		`{"type":"array","items":{"type":"string"},"collectionFormat":"multi","in":"query"}`)

	assertSerializeJSON(t, PathParam("").Typed("string", ""), `{"type":"string","in":"path","required":true}`)

	assertSerializeJSON(t,
		PathParam("").CollectionOf(items, "multi"),
		`{"type":"array","items":{"type":"string"},"collectionFormat":"multi","in":"path","required":true}`)

	assertSerializeJSON(t,
		PathParam("").CollectionOf(intItems, "multi"),
		`{"type":"array","items":{"type":"int","format":"int32"},"collectionFormat":"multi","in":"path","required":true}`)

	assertSerializeJSON(t, HeaderParam("").Typed("string", ""), `{"type":"string","in":"header","required":true}`)

	assertSerializeJSON(t,
		HeaderParam("").CollectionOf(items, "multi"),
		`{"type":"array","items":{"type":"string"},"collectionFormat":"multi","in":"header","required":true}`)
	schema := &Schema{SchemaProps: SchemaProps{
		Properties: map[string]Schema{
			"name": Schema{SchemaProps: SchemaProps{
				Type: []string{"string"},
			}},
		},
	}}

	refSchema := &Schema{
		SchemaProps: SchemaProps{Ref: MustCreateRef("Cat")},
	}

	assertSerializeJSON(t,
		BodyParam("", schema),
		`{"type":"object","in":"body","schema":{"properties":{"name":{"type":"string"}}}}`)

	assertSerializeJSON(t,
		BodyParam("", refSchema),
		`{"type":"object","in":"body","schema":{"$ref":"Cat"}}`)

	// array body param
	assertSerializeJSON(t,
		BodyParam("", ArrayProperty(RefProperty("Cat"))),
		`{"type":"object","in":"body","schema":{"type":"array","items":{"$ref":"Cat"}}}`)

}
