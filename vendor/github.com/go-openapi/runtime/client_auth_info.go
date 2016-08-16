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

import "github.com/go-openapi/strfmt"

// A ClientAuthInfoWriterFunc converts a function to a request writer interface
type ClientAuthInfoWriterFunc func(ClientRequest, strfmt.Registry) error

// AuthenticateRequest adds authentication data to the request
func (fn ClientAuthInfoWriterFunc) AuthenticateRequest(req ClientRequest, reg strfmt.Registry) error {
	return fn(req, reg)
}

// A ClientAuthInfoWriter implementor knows how to write authentication info to a request
type ClientAuthInfoWriter interface {
	AuthenticateRequest(ClientRequest, strfmt.Registry) error
}
