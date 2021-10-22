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

package validate_test

import (
	"encoding/json"
	"fmt"
	"testing"

	// Spec loading
	"github.com/stretchr/testify/require"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"   // OpenAPI format extensions
	"k8s.io/kube-openapi/pkg/validation/validate" // This package
)

func ExampleAgainstSchema() {
	// Example using encoding/json as unmarshaller
	var schemaJSON = `
{
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[A-Za-z]+$",
            "minLength": 1
        }
	},
    "patternProperties": {
	  "address-[0-9]+": {
         "type": "string",
         "pattern": "^[\\s|a-z]+$"
	  }
    },
    "required": [
        "name"
    ],
	"additionalProperties": false
}`

	schema := new(spec.Schema)
	_ = json.Unmarshal([]byte(schemaJSON), schema)

	input := map[string]interface{}{}

	// JSON data to validate
	inputJSON := `{"name": "Ivan","address-1": "sesame street"}`
	_ = json.Unmarshal([]byte(inputJSON), &input)

	// strfmt.Default is the registry of recognized formats
	err := validate.AgainstSchema(schema, input, strfmt.Default)
	if err != nil {
		fmt.Printf("JSON does not validate against schema: %v", err)
	} else {
		fmt.Printf("OK")
	}
	// Output:
	// OK
}

func TestValidate_Issue112(t *testing.T) {
	t.Run("returns no error on body includes `items` key", func(t *testing.T) {
		body := map[string]interface{}{"items1": nil}
		err := validate.AgainstSchema(getSimpleSchema(), body, strfmt.Default)
		require.NoError(t, err)
	})

	t.Run("returns no error when body includes `items` key", func(t *testing.T) {
		body := map[string]interface{}{"items": nil}
		err := validate.AgainstSchema(getSimpleSchema(), body, strfmt.Default)
		require.NoError(t, err)
	})
}

func getSimpleSchema() *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type: spec.StringOrArray{"object"},
		},
	}
}
