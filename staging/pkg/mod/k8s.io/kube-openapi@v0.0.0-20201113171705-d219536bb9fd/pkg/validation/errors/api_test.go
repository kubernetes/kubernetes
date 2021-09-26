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

package errors

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateName(t *testing.T) {
	v := &Validation{Name: "myValidation", message: "myMessage"}

	// unchanged
	vv := v.ValidateName("")
	assert.EqualValues(t, "myValidation", vv.Name)
	assert.EqualValues(t, "myMessage", vv.message)

	// unchanged
	vv = v.ValidateName("myNewName")
	assert.EqualValues(t, "myValidation", vv.Name)
	assert.EqualValues(t, "myMessage", vv.message)

	v.Name = ""

	// unchanged
	vv = v.ValidateName("")
	assert.EqualValues(t, "", vv.Name)
	assert.EqualValues(t, "myMessage", vv.message)

	// forced
	vv = v.ValidateName("myNewName")
	assert.EqualValues(t, "myNewName", vv.Name)
	assert.EqualValues(t, "myNewNamemyMessage", vv.message)
}
