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

func TestXmlObject_Serialize(t *testing.T) {
	obj1 := XMLObject{}
	actual, err := json.Marshal(obj1)
	if assert.NoError(t, err) {
		assert.Equal(t, "{}", string(actual))
	}

	obj2 := XMLObject{
		Name:      "the name",
		Namespace: "the namespace",
		Prefix:    "the prefix",
		Attribute: true,
		Wrapped:   true,
	}

	actual, err = json.Marshal(obj2)
	if assert.NoError(t, err) {
		var ad map[string]interface{}
		if assert.NoError(t, json.Unmarshal(actual, &ad)) {
			assert.Equal(t, obj2.Name, ad["name"])
			assert.Equal(t, obj2.Namespace, ad["namespace"])
			assert.Equal(t, obj2.Prefix, ad["prefix"])
			assert.True(t, ad["attribute"].(bool))
			assert.True(t, ad["wrapped"].(bool))
		}
	}
}

func TestXmlObject_Deserialize(t *testing.T) {
	expected := XMLObject{}
	actual := XMLObject{}
	if assert.NoError(t, json.Unmarshal([]byte("{}"), &actual)) {
		assert.Equal(t, expected, actual)
	}

	completed := `{"name":"the name","namespace":"the namespace","prefix":"the prefix","attribute":true,"wrapped":true}`
	expected = XMLObject{"the name", "the namespace", "the prefix", true, true}
	actual = XMLObject{}
	if assert.NoError(t, json.Unmarshal([]byte(completed), &actual)) {
		assert.Equal(t, expected, actual)
	}
}
