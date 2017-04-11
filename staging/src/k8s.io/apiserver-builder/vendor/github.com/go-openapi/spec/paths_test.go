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

var paths = Paths{
	VendorExtensible: VendorExtensible{Extensions: map[string]interface{}{"x-framework": "go-swagger"}},
	Paths: map[string]PathItem{
		"/": PathItem{
			Refable: Refable{Ref: MustCreateRef("cats")},
		},
	},
}

var pathsJSON = `{"x-framework":"go-swagger","/":{"$ref":"cats"}}`

func TestIntegrationPaths(t *testing.T) {
	var actual Paths
	if assert.NoError(t, json.Unmarshal([]byte(pathsJSON), &actual)) {
		assert.EqualValues(t, actual, paths)
	}

	assertParsesJSON(t, pathsJSON, paths)

}
