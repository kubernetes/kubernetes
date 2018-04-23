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
	"io"
	"io/ioutil"
	"net/url"
	"time"

	"github.com/go-openapi/strfmt"
)

// ClientRequestWriterFunc converts a function to a request writer interface
type ClientRequestWriterFunc func(ClientRequest, strfmt.Registry) error

// WriteToRequest adds data to the request
func (fn ClientRequestWriterFunc) WriteToRequest(req ClientRequest, reg strfmt.Registry) error {
	return fn(req, reg)
}

// ClientRequestWriter is an interface for things that know how to write to a request
type ClientRequestWriter interface {
	WriteToRequest(ClientRequest, strfmt.Registry) error
}

// ClientRequest is an interface for things that know how to
// add information to a swagger client request
type ClientRequest interface {
	SetHeaderParam(string, ...string) error

	SetQueryParam(string, ...string) error

	SetFormParam(string, ...string) error

	SetPathParam(string, string) error

	GetQueryParams() url.Values

	SetFileParam(string, ...NamedReadCloser) error

	SetBodyParam(interface{}) error

	SetTimeout(time.Duration) error

	GetMethod() string

	GetPath() string

	GetBody() []byte
}

// NamedReadCloser represents a named ReadCloser interface
type NamedReadCloser interface {
	io.ReadCloser
	Name() string
}

// NamedReader creates a NamedReadCloser for use as file upload
func NamedReader(name string, rdr io.Reader) NamedReadCloser {
	rc, ok := rdr.(io.ReadCloser)
	if !ok {
		rc = ioutil.NopCloser(rdr)
	}
	return &namedReadCloser{
		name: name,
		cr:   rc,
	}
}

type namedReadCloser struct {
	name string
	cr   io.ReadCloser
}

func (n *namedReadCloser) Close() error {
	return n.cr.Close()
}
func (n *namedReadCloser) Read(p []byte) (int, error) {
	return n.cr.Read(p)
}
func (n *namedReadCloser) Name() string {
	return n.name
}
