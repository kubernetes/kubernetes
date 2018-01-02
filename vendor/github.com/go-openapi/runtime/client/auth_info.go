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

package client

import (
	"encoding/base64"

	"github.com/go-openapi/runtime"
	"github.com/go-openapi/strfmt"
)

// BasicAuth provides a basic auth info writer
func BasicAuth(username, password string) runtime.ClientAuthInfoWriter {
	return runtime.ClientAuthInfoWriterFunc(func(r runtime.ClientRequest, _ strfmt.Registry) error {
		encoded := base64.StdEncoding.EncodeToString([]byte(username + ":" + password))
		r.SetHeaderParam("Authorization", "Basic "+encoded)
		return nil
	})
}

// APIKeyAuth provides an API key auth info writer
func APIKeyAuth(name, in, value string) runtime.ClientAuthInfoWriter {
	if in == "query" {
		return runtime.ClientAuthInfoWriterFunc(func(r runtime.ClientRequest, _ strfmt.Registry) error {
			r.SetQueryParam(name, value)
			return nil
		})
	}

	if in == "header" {
		return runtime.ClientAuthInfoWriterFunc(func(r runtime.ClientRequest, _ strfmt.Registry) error {
			r.SetHeaderParam(name, value)
			return nil
		})
	}
	return nil
}

// BearerToken provides a header based oauth2 bearer access token auth info writer
func BearerToken(token string) runtime.ClientAuthInfoWriter {
	return runtime.ClientAuthInfoWriterFunc(func(r runtime.ClientRequest, _ strfmt.Registry) error {
		r.SetHeaderParam("Authorization", "Bearer "+token)
		return nil
	})
}
