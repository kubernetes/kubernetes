// Copyright 2017 go-swagger maintainers
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

var response = Response{
	Refable: Refable{Ref: MustCreateRef("Dog")},
	VendorExtensible: VendorExtensible{
		Extensions: map[string]interface{}{
			"x-go-name": "PutDogExists",
		},
	},
	ResponseProps: ResponseProps{
		Description: "Dog exists",
		Schema:      &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}},
	},
}

var responseJSON = `{
	"$ref": "Dog",
	"x-go-name": "PutDogExists",
	"description": "Dog exists",
	"schema": {
		"type": "string"
	}
}`

func TestIntegrationResponse(t *testing.T) {
	var actual Response
	if assert.NoError(t, json.Unmarshal([]byte(responseJSON), &actual)) {
		assert.EqualValues(t, actual, response)
	}

	assertParsesJSON(t, responseJSON, response)
}
