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

package runtime

import (
	"testing"

	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
)

func TestAuthInfoWriter(t *testing.T) {
	hand := ClientAuthInfoWriterFunc(func(r ClientRequest, _ strfmt.Registry) error {
		r.SetHeaderParam("authorization", "Bearer the-token-goes-here")
		return nil
	})

	tr := new(trw)
	hand.AuthenticateRequest(tr, nil)
	assert.Equal(t, "Bearer the-token-goes-here", tr.Headers.Get("Authorization"))
}
