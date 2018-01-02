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

func TestAPIVerificationFailed(t *testing.T) {
	err := &APIVerificationFailed{
		Section:              "consumer",
		MissingSpecification: []string{"application/json", "application/x-yaml"},
		MissingRegistration:  []string{"text/html", "application/xml"},
	}

	expected := `missing [text/html, application/xml] consumer registrations
missing from spec file [application/json, application/x-yaml] consumer`
	assert.Equal(t, expected, err.Error())
}
