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

package validate

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test edge cases in schema_props_validator which are difficult
// to simulate with specs
// (this one is a trivial, just to check all methods are filled)
func TestSchemaPropsValidator_EdgeCases(t *testing.T) {
	s := schemaPropsValidator{}
	s.SetPath("path")
	assert.Equal(t, "path", s.Path)
}
