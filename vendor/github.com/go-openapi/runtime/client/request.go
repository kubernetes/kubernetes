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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-openapi/runtime"
	"github.com/go-openapi/strfmt"
)

// NewRequest creates a new swagger http client request
func newRequest(method, pathPattern string, writer runtime.ClientRequestWriter) (*request, error) {
	return &request{
		pathPattern: pathPattern,
		method:      method,
		writer:      writer,
		header:      make(http.Header),
		query:       make(url.Values),
		timeout:     DefaultTimeout,
	}, nil
}

// Request represents a swagger client request.
//
// This Request struct converts to a HTTP request.
// There might be others that convert to other transports.
// There is no error checking here, it is assumed to be used after a spec has been validated.
// so impossible combinations should not arise (hopefully).
//
// The main purpose of this struct is to hide the machinery of adding params to a transport request.
// The generated code only implements what is necessary to turn a param into a valid value for these methods.
type request struct {
	pathPattern string
	method      string
	writer      runtime.ClientRequestWriter

	pathParams map[string]string
	header     http.Header
	query      url.Values
	formFields url.Values
	fileFields map[string]*os.File
	payload    interface{}
	timeout    time.Duration
}

var (
	// ensure interface compliance
	_ runtime.ClientRequest = new(request)
)

// BuildHTTP creates a new http request based on the data from the params
func (r *request) BuildHTTP(mediaType string, producers map[string]runtime.Producer, registry strfmt.Registry) (*http.Request, error) {
	// build the data
	if err := r.writer.WriteToRequest(r, registry); err != nil {
		return nil, err
	}

	// create http request
	path := r.pathPattern
	for k, v := range r.pathParams {
		path = strings.Replace(path, "{"+k+"}", v, -1)
	}

	var body io.ReadCloser
	var pr *io.PipeReader
	var pw *io.PipeWriter
	buf := bytes.NewBuffer(nil)
	body = ioutil.NopCloser(buf)
	if r.fileFields != nil {
		pr, pw = io.Pipe()
		body = pr
	}
	req, err := http.NewRequest(r.method, path, body)
	if err != nil {
		return nil, err
	}
	req.URL.RawQuery = r.query.Encode()
	req.Header = r.header

	// check if this is a form type request
	if len(r.formFields) > 0 || len(r.fileFields) > 0 {
		// check if this is multipart
		if len(r.fileFields) > 0 {
			mp := multipart.NewWriter(pw)
			req.Header.Set(runtime.HeaderContentType, mp.FormDataContentType())

			go func() {
				defer func() {
					mp.Close()
					pw.Close()
				}()

				for fn, v := range r.formFields {
					if len(v) > 0 {
						if err := mp.WriteField(fn, v[0]); err != nil {
							pw.CloseWithError(err)
							log.Fatal(err)
						}
					}
				}

				for fn, f := range r.fileFields {
					wrtr, err := mp.CreateFormFile(fn, filepath.Base(f.Name()))
					if err != nil {
						pw.CloseWithError(err)
						log.Fatal(err)
					}
					defer func() {
						for _, ff := range r.fileFields {
							ff.Close()
						}

					}()
					if _, err := io.Copy(wrtr, f); err != nil {
						pw.CloseWithError(err)
						log.Fatal(err)
					}
				}

			}()
			return req, nil
		} else {
			req.Header.Set(runtime.HeaderContentType, mediaType)
			// write the form values as the body
			buf.WriteString(r.formFields.Encode())
			return req, nil
		}
	}

	// if there is payload, use the producer to write the payload, and then
	// set the header to the content-type appropriate for the payload produced
	if r.payload != nil {
		// TODO: infer most appropriate content type based on the producer used,
		// and the `consumers` section of the spec/operation
		req.Header.Set(runtime.HeaderContentType, mediaType)
		if rdr, ok := r.payload.(io.ReadCloser); ok {
			req.Body = rdr
			return req, nil
		}

		if rdr, ok := r.payload.(io.Reader); ok {
			req.Body = ioutil.NopCloser(rdr)
			return req, nil
		}

		// set the content length of the request or else a chunked transfer is
		// declared, and this corrupts outgoing JSON payloads. the content's
		// length must be set prior to the body being written per the spec at
		// https://golang.org/pkg/net/http
		//
		//     If Body is present, Content-Length is <= 0 and TransferEncoding
		//     hasn't been set to "identity", Write adds
		//     "Transfer-Encoding: chunked" to the header. Body is closed
		//     after it is sent.
		//
		// to that end a temporary buffer, b, is created to produce the payload
		// body, and then its size is used to set the request's content length
		var b bytes.Buffer
		producer := producers[mediaType]
		if err := producer.Produce(&b, r.payload); err != nil {
			return nil, err
		}
		req.ContentLength = int64(b.Len())
		if _, err := buf.Write(b.Bytes()); err != nil {
			return nil, err
		}
	}
	return req, nil
}

// SetHeaderParam adds a header param to the request
// when there is only 1 value provided for the varargs, it will set it.
// when there are several values provided for the varargs it will add it (no overriding)
func (r *request) SetHeaderParam(name string, values ...string) error {
	if r.header == nil {
		r.header = make(http.Header)
	}
	r.header[http.CanonicalHeaderKey(name)] = values
	return nil
}

// SetQueryParam adds a query param to the request
// when there is only 1 value provided for the varargs, it will set it.
// when there are several values provided for the varargs it will add it (no overriding)
func (r *request) SetQueryParam(name string, values ...string) error {
	if r.header == nil {
		r.query = make(url.Values)
	}
	r.query[name] = values
	return nil
}

// SetFormParam adds a forn param to the request
// when there is only 1 value provided for the varargs, it will set it.
// when there are several values provided for the varargs it will add it (no overriding)
func (r *request) SetFormParam(name string, values ...string) error {
	if r.formFields == nil {
		r.formFields = make(url.Values)
	}
	r.formFields[name] = values
	return nil
}

// SetPathParam adds a path param to the request
func (r *request) SetPathParam(name string, value string) error {
	if r.pathParams == nil {
		r.pathParams = make(map[string]string)
	}

	r.pathParams[name] = value
	return nil
}

// SetFileParam adds a file param to the request
func (r *request) SetFileParam(name string, file *os.File) error {
	fi, err := os.Stat(file.Name())
	if err != nil {
		return err
	}
	if fi.IsDir() {
		return fmt.Errorf("%q is a directory, only files are supported", file.Name())
	}

	if r.fileFields == nil {
		r.fileFields = make(map[string]*os.File)
	}
	if r.formFields == nil {
		r.formFields = make(url.Values)
	}

	r.fileFields[name] = file
	return nil
}

// SetBodyParam sets a body parameter on the request.
// This does not yet serialze the object, this happens as late as possible.
func (r *request) SetBodyParam(payload interface{}) error {
	r.payload = payload
	return nil
}

// SetTimeout sets the timeout for a request
func (r *request) SetTimeout(timeout time.Duration) error {
	r.timeout = timeout
	return nil
}
