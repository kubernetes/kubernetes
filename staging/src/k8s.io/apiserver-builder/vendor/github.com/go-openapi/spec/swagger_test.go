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
	"fmt"
	"reflect"
	"testing"

	"github.com/go-openapi/swag"
	"github.com/stretchr/testify/assert"
)

var spec = Swagger{
	SwaggerProps: SwaggerProps{
		ID:          "http://localhost:3849/api-docs",
		Swagger:     "2.0",
		Consumes:    []string{"application/json", "application/x-yaml"},
		Produces:    []string{"application/json"},
		Schemes:     []string{"http", "https"},
		Info:        &info,
		Host:        "some.api.out.there",
		BasePath:    "/",
		Paths:       &paths,
		Definitions: map[string]Schema{"Category": {SchemaProps: SchemaProps{Type: []string{"string"}}}},
		Parameters: map[string]Parameter{
			"categoryParam": {ParamProps: ParamProps{Name: "category", In: "query"}, SimpleSchema: SimpleSchema{Type: "string"}},
		},
		Responses: map[string]Response{
			"EmptyAnswer": {
				ResponseProps: ResponseProps{
					Description: "no data to return for this operation",
				},
			},
		},
		SecurityDefinitions: map[string]*SecurityScheme{
			"internalApiKey": APIKeyAuth("api_key", "header"),
		},
		Security: []map[string][]string{
			{"internalApiKey": {}},
		},
		Tags:         []Tag{NewTag("pets", "", nil)},
		ExternalDocs: &ExternalDocumentation{"the name", "the url"},
	},
	VendorExtensible: VendorExtensible{map[string]interface{}{
		"x-some-extension": "vendor",
		"x-schemes":        []interface{}{"unix", "amqp"},
	}},
}

var specJSON = `{
	"id": "http://localhost:3849/api-docs",
	"consumes": ["application/json", "application/x-yaml"],
	"produces": ["application/json"],
	"schemes": ["http", "https"],
	"swagger": "2.0",
	"info": {
		"contact": {
			"name": "wordnik api team",
			"url": "http://developer.wordnik.com"
		},
		"description": "A sample API that uses a petstore as an example to demonstrate features in the swagger-2.0 specification",
		"license": {
			"name": "Creative Commons 4.0 International",
			"url": "http://creativecommons.org/licenses/by/4.0/"
		},
		"termsOfService": "http://helloreverb.com/terms/",
		"title": "Swagger Sample API",
		"version": "1.0.9-abcd",
		"x-framework": "go-swagger"
	},
	"host": "some.api.out.there",
	"basePath": "/",
	"paths": {"x-framework":"go-swagger","/":{"$ref":"cats"}},
	"definitions": { "Category": { "type": "string"} },
	"parameters": {
		"categoryParam": {
			"name": "category",
			"in": "query",
			"type": "string"
		}
	},
	"responses": { "EmptyAnswer": { "description": "no data to return for this operation" } },
	"securityDefinitions": {
		"internalApiKey": {
			"type": "apiKey",
			"in": "header",
			"name": "api_key"
		}
	},
	"security": [{"internalApiKey":[]}],
	"tags": [{"name":"pets"}],
	"externalDocs": {"description":"the name","url":"the url"},
	"x-some-extension": "vendor",
	"x-schemes": ["unix","amqp"]
}`

//
// func verifySpecSerialize(specJSON []byte, spec Swagger) {
// 	expected := map[string]interface{}{}
// 	json.Unmarshal(specJSON, &expected)
// 	b, err := json.MarshalIndent(spec, "", "  ")
// 	So(err, ShouldBeNil)
// 	var actual map[string]interface{}
// 	err = json.Unmarshal(b, &actual)
// 	So(err, ShouldBeNil)
// 	compareSpecMaps(actual, expected)
// }

func assertEquivalent(t testing.TB, actual, expected interface{}) bool {
	if actual == nil || expected == nil || reflect.DeepEqual(actual, expected) {
		return true
	}

	actualType := reflect.TypeOf(actual)
	expectedType := reflect.TypeOf(expected)
	if reflect.TypeOf(actual).ConvertibleTo(expectedType) {
		expectedValue := reflect.ValueOf(expected)
		if swag.IsZero(expectedValue) && swag.IsZero(reflect.ValueOf(actual)) {
			return true
		}

		// Attempt comparison after type conversion
		if reflect.DeepEqual(actual, expectedValue.Convert(actualType).Interface()) {
			return true
		}
	}

	// Last ditch effort
	if fmt.Sprintf("%#v", expected) == fmt.Sprintf("%#v", actual) {
		return true
	}
	errFmt := "Expected: '%T(%#v)'\nActual:   '%T(%#v)'\n(Should be equivalent)!"
	return assert.Fail(t, errFmt, expected, expected, actual, actual)
}

func ShouldBeEquivalentTo(actual interface{}, expecteds ...interface{}) string {
	expected := expecteds[0]
	if actual == nil || expected == nil {
		return ""
	}

	if reflect.DeepEqual(expected, actual) {
		return ""
	}

	actualType := reflect.TypeOf(actual)
	expectedType := reflect.TypeOf(expected)
	if reflect.TypeOf(actual).ConvertibleTo(expectedType) {
		expectedValue := reflect.ValueOf(expected)
		if swag.IsZero(expectedValue) && swag.IsZero(reflect.ValueOf(actual)) {
			return ""
		}

		// Attempt comparison after type conversion
		if reflect.DeepEqual(actual, expectedValue.Convert(actualType).Interface()) {
			return ""
		}
	}

	// Last ditch effort
	if fmt.Sprintf("%#v", expected) == fmt.Sprintf("%#v", actual) {
		return ""
	}
	errFmt := "Expected: '%T(%#v)'\nActual:   '%T(%#v)'\n(Should be equivalent)!"
	return fmt.Sprintf(errFmt, expected, expected, actual, actual)

}

func assertSpecMaps(t testing.TB, actual, expected map[string]interface{}) bool {
	res := true
	if id, ok := expected["id"]; ok {
		res = assert.Equal(t, id, actual["id"])
	}
	res = res && assert.Equal(t, expected["consumes"], actual["consumes"])
	res = res && assert.Equal(t, expected["produces"], actual["produces"])
	res = res && assert.Equal(t, expected["schemes"], actual["schemes"])
	res = res && assert.Equal(t, expected["swagger"], actual["swagger"])
	res = res && assert.Equal(t, expected["info"], actual["info"])
	res = res && assert.Equal(t, expected["host"], actual["host"])
	res = res && assert.Equal(t, expected["basePath"], actual["basePath"])
	res = res && assert.Equal(t, expected["paths"], actual["paths"])
	res = res && assert.Equal(t, expected["definitions"], actual["definitions"])
	res = res && assert.Equal(t, expected["responses"], actual["responses"])
	res = res && assert.Equal(t, expected["securityDefinitions"], actual["securityDefinitions"])
	res = res && assert.Equal(t, expected["tags"], actual["tags"])
	res = res && assert.Equal(t, expected["externalDocs"], actual["externalDocs"])
	res = res && assert.Equal(t, expected["x-some-extension"], actual["x-some-extension"])
	res = res && assert.Equal(t, expected["x-schemes"], actual["x-schemes"])

	return res
}

//
// func compareSpecMaps(actual, expected map[string]interface{}) {
// 	if id, ok := expected["id"]; ok {
// 		So(actual["id"], ShouldEqual, id)
// 	}
// 	//So(actual["$schema"], ShouldEqual, SwaggerSchemaURL)
// 	So(actual["consumes"], ShouldResemble, expected["consumes"])
// 	So(actual["produces"], ShouldResemble, expected["produces"])
// 	So(actual["schemes"], ShouldResemble, expected["schemes"])
// 	So(actual["swagger"], ShouldEqual, expected["swagger"])
// 	So(actual["info"], ShouldResemble, expected["info"])
// 	So(actual["host"], ShouldEqual, expected["host"])
// 	So(actual["basePath"], ShouldEqual, expected["basePath"])
// 	So(actual["paths"], ShouldBeEquivalentTo, expected["paths"])
// 	So(actual["definitions"], ShouldBeEquivalentTo, expected["definitions"])
// 	So(actual["responses"], ShouldBeEquivalentTo, expected["responses"])
// 	So(actual["securityDefinitions"], ShouldResemble, expected["securityDefinitions"])
// 	So(actual["tags"], ShouldResemble, expected["tags"])
// 	So(actual["externalDocs"], ShouldResemble, expected["externalDocs"])
// 	So(actual["x-some-extension"], ShouldResemble, expected["x-some-extension"])
// 	So(actual["x-schemes"], ShouldResemble, expected["x-schemes"])
// }

func assertSpecs(t testing.TB, actual, expected Swagger) bool {
	expected.Swagger = "2.0"
	return assert.Equal(t, actual, expected)
}

//
// func compareSpecs(actual Swagger, spec Swagger) {
// 	spec.Swagger = "2.0"
// 	So(actual, ShouldBeEquivalentTo, spec)
// }

func assertSpecJSON(t testing.TB, specJSON []byte) bool {
	var expected map[string]interface{}
	if !assert.NoError(t, json.Unmarshal(specJSON, &expected)) {
		return false
	}

	obj := Swagger{}
	if !assert.NoError(t, json.Unmarshal(specJSON, &obj)) {
		return false
	}

	cb, err := json.MarshalIndent(obj, "", "  ")
	if assert.NoError(t, err) {
		return false
	}
	var actual map[string]interface{}
	if !assert.NoError(t, json.Unmarshal(cb, &actual)) {
		return false
	}
	return assertSpecMaps(t, actual, expected)
}

// func verifySpecJSON(specJSON []byte) {
// 	//Println()
// 	//Println("json to verify", string(specJson))
// 	var expected map[string]interface{}
// 	err := json.Unmarshal(specJSON, &expected)
// 	So(err, ShouldBeNil)
//
// 	obj := Swagger{}
// 	err = json.Unmarshal(specJSON, &obj)
// 	So(err, ShouldBeNil)
//
// 	//spew.Dump(obj)
//
// 	cb, err := json.MarshalIndent(obj, "", "  ")
// 	So(err, ShouldBeNil)
// 	//Println()
// 	//Println("Marshalling to json returned", string(cb))
//
// 	var actual map[string]interface{}
// 	err = json.Unmarshal(cb, &actual)
// 	So(err, ShouldBeNil)
// 	//Println()
// 	//spew.Dump(expected)
// 	//spew.Dump(actual)
// 	//fmt.Printf("comparing %s\n\t%#v\nto\n\t%#+v\n", fileName, expected, actual)
// 	compareSpecMaps(actual, expected)
// }

func TestSwaggerSpec_Serialize(t *testing.T) {
	expected := make(map[string]interface{})
	json.Unmarshal([]byte(specJSON), &expected)
	b, err := json.MarshalIndent(spec, "", "  ")
	if assert.NoError(t, err) {
		var actual map[string]interface{}
		err := json.Unmarshal(b, &actual)
		if assert.NoError(t, err) {
			assert.EqualValues(t, actual, expected)
		}
	}
}

func TestSwaggerSpec_Deserialize(t *testing.T) {
	var actual Swagger
	err := json.Unmarshal([]byte(specJSON), &actual)
	if assert.NoError(t, err) {
		assert.EqualValues(t, actual, spec)
	}
}

func TestVendorExtensionStringSlice(t *testing.T) {
	var actual Swagger
	err := json.Unmarshal([]byte(specJSON), &actual)
	if assert.NoError(t, err) {
		schemes, ok := actual.Extensions.GetStringSlice("x-schemes")
		if assert.True(t, ok) {
			assert.EqualValues(t, []string{"unix", "amqp"}, schemes)
		}
	}
}
