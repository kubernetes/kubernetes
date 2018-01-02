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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseError(t *testing.T) {
	err := NewParseError("Content-Type", "header", "application(", errors.New("unable to parse"))
	assert.EqualValues(t, 400, err.Code())
	assert.Equal(t, "parsing Content-Type header from \"application(\" failed, because unable to parse", err.Error())

	err = NewParseError("Content-Type", "", "application(", errors.New("unable to parse"))
	assert.EqualValues(t, 400, err.Code())
	assert.Equal(t, "parsing Content-Type from \"application(\" failed, because unable to parse", err.Error())
}
