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
	"io"
	"net/http"

	"github.com/go-openapi/runtime"
)

var _ runtime.ClientResponse = response{}

type response struct {
	resp *http.Response
}

func (r response) Code() int {
	return r.resp.StatusCode
}

func (r response) Message() string {
	return r.resp.Status
}

func (r response) GetHeader(name string) string {
	return r.resp.Header.Get(name)
}

func (r response) Body() io.ReadCloser {
	return r.resp.Body
}
