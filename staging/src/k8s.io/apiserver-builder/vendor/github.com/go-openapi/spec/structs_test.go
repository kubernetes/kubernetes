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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v2"
)

func assertSerializeJSON(t testing.TB, actual interface{}, expected string) bool {
	ser, err := json.Marshal(actual)
	if err != nil {
		return assert.Fail(t, "unable to marshal to json (%s): %#v", err, actual)
	}
	return assert.Equal(t, string(ser), expected)
}

func assertParsesJSON(t testing.TB, actual string, expected interface{}) bool {
	tpe := reflect.TypeOf(expected)
	var pointed bool
	if tpe.Kind() == reflect.Ptr {
		tpe = tpe.Elem()
		pointed = true
	}

	parsed := reflect.New(tpe)
	err := json.Unmarshal([]byte(actual), parsed.Interface())
	if err != nil {
		return assert.Fail(t, "unable to unmarshal from json (%s): %s", err, actual)
	}
	act := parsed.Interface()
	if !pointed {
		act = reflect.Indirect(parsed).Interface()
	}
	return assert.Equal(t, act, expected)
}

func assertSerializeYAML(t testing.TB, actual interface{}, expected string) bool {
	ser, err := yaml.Marshal(actual)
	if err != nil {
		return assert.Fail(t, "unable to marshal to yaml (%s): %#v", err, actual)
	}
	return assert.Equal(t, string(ser), expected)
}

func assertParsesYAML(t testing.TB, actual string, expected interface{}) bool {
	tpe := reflect.TypeOf(expected)
	var pointed bool
	if tpe.Kind() == reflect.Ptr {
		tpe = tpe.Elem()
		pointed = true
	}
	parsed := reflect.New(tpe)
	err := yaml.Unmarshal([]byte(actual), parsed.Interface())
	if err != nil {
		return assert.Fail(t, "unable to unmarshal from yaml (%s): %s", err, actual)
	}
	act := parsed.Interface()
	if !pointed {
		act = reflect.Indirect(parsed).Interface()
	}
	return assert.EqualValues(t, act, expected)
}

func TestSerialization_SerializeJSON(t *testing.T) {
	assertSerializeJSON(t, []string{"hello"}, "[\"hello\"]")
	assertSerializeJSON(t, []string{"hello", "world", "and", "stuff"}, "[\"hello\",\"world\",\"and\",\"stuff\"]")
	assertSerializeJSON(t, StringOrArray(nil), "null")
	assertSerializeJSON(t, SchemaOrArray{Schemas: []Schema{Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}}}, "[{\"type\":\"string\"}]")
	assertSerializeJSON(t, SchemaOrArray{
		Schemas: []Schema{
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
		}}, "[{\"type\":\"string\"},{\"type\":\"string\"}]")
	assertSerializeJSON(t, SchemaOrArray{}, "null")
}

func TestSerialization_DeserializeJSON(t *testing.T) {
	// String
	assertParsesJSON(t, "\"hello\"", StringOrArray([]string{"hello"}))
	assertParsesJSON(t, "[\"hello\",\"world\",\"and\",\"stuff\"]", StringOrArray([]string{"hello", "world", "and", "stuff"}))
	assertParsesJSON(t, "[\"hello\",\"world\",null,\"stuff\"]", StringOrArray([]string{"hello", "world", "", "stuff"}))
	assertParsesJSON(t, "null", StringOrArray(nil))

	// Schema
	assertParsesJSON(t, "{\"type\":\"string\"}", SchemaOrArray{Schema: &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}})
	assertParsesJSON(t, "[{\"type\":\"string\"},{\"type\":\"string\"}]", &SchemaOrArray{
		Schemas: []Schema{
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
			Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
		},
	})
	assertParsesJSON(t, "null", SchemaOrArray{})
}
