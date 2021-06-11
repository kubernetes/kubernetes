/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package rest

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
)

const (
	Path = "/rest"
)

// Resource wraps url.URL with helpers
type Resource struct {
	u *url.URL
}

func (r *Resource) String() string {
	return r.u.String()
}

// WithID appends id to the URL.Path
func (r *Resource) WithID(id string) *Resource {
	r.u.Path += "/id:" + id
	return r
}

// WithAction sets adds action to the URL.RawQuery
func (r *Resource) WithAction(action string) *Resource {
	return r.WithParam("~action", action)
}

// WithParam adds one parameter on the URL.RawQuery
func (r *Resource) WithParam(name string, value string) *Resource {
	// ParseQuery handles empty case, and we control access to query string so shouldn't encounter an error case
	params, _ := url.ParseQuery(r.u.RawQuery)
	params[name] = []string{value}
	r.u.RawQuery = params.Encode()
	return r
}

// Request returns a new http.Request for the given method.
// An optional body can be provided for POST and PATCH methods.
func (r *Resource) Request(method string, body ...interface{}) *http.Request {
	rdr := io.MultiReader() // empty body by default
	if len(body) != 0 {
		rdr = encode(body[0])
	}
	req, err := http.NewRequest(method, r.u.String(), rdr)
	if err != nil {
		panic(err)
	}
	return req
}

type errorReader struct {
	e error
}

func (e errorReader) Read([]byte) (int, error) {
	return -1, e.e
}

// encode body as JSON, deferring any errors until io.Reader is used.
func encode(body interface{}) io.Reader {
	var b bytes.Buffer
	err := json.NewEncoder(&b).Encode(body)
	if err != nil {
		return errorReader{err}
	}
	return &b
}
