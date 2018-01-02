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
	"testing"
)

func TestPropertySerialization(t *testing.T) {
	strProp := StringProperty()
	strProp.Enum = append(strProp.Enum, "a", "b")

	prop := &Schema{SchemaProps: SchemaProps{
		Items: &SchemaOrArray{Schemas: []Schema{
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
		}},
	}}

	var propSerData = []struct {
		Schema *Schema
		JSON   string
	}{
		{BooleanProperty(), `{"type":"boolean"}`},
		{DateProperty(), `{"type":"string","format":"date"}`},
		{DateTimeProperty(), `{"type":"string","format":"date-time"}`},
		{Float64Property(), `{"type":"number","format":"double"}`},
		{Float32Property(), `{"type":"number","format":"float"}`},
		{Int32Property(), `{"type":"integer","format":"int32"}`},
		{Int64Property(), `{"type":"integer","format":"int64"}`},
		{MapProperty(StringProperty()), `{"type":"object","additionalProperties":{"type":"string"}}`},
		{MapProperty(Int32Property()), `{"type":"object","additionalProperties":{"type":"integer","format":"int32"}}`},
		{RefProperty("Dog"), `{"$ref":"Dog"}`},
		{StringProperty(), `{"type":"string"}`},
		{strProp, `{"type":"string","enum":["a","b"]}`},
		{ArrayProperty(StringProperty()), `{"type":"array","items":{"type":"string"}}`},
		{prop, `{"items":[{"type":"string"},{"type":"string"}]}`},
	}

	for _, v := range propSerData {
		t.Log("roundtripping for", v.JSON)
		assertSerializeJSON(t, v.Schema, v.JSON)
		assertParsesJSON(t, v.JSON, v.Schema)
	}

}
