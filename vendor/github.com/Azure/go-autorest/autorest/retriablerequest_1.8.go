//go:build go1.8
// +build go1.8

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package autorest

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
)

// RetriableRequest provides facilities for retrying an HTTP request.
type RetriableRequest struct {
	req *http.Request
	rc  io.ReadCloser
	br  *bytes.Reader
}

// Prepare signals that the request is about to be sent.
func (rr *RetriableRequest) Prepare() (err error) {
	// preserve the request body; this is to support retry logic as
	// the underlying transport will always close the reqeust body
	if rr.req.Body != nil {
		if rr.rc != nil {
			rr.req.Body = rr.rc
		} else if rr.br != nil {
			_, err = rr.br.Seek(0, io.SeekStart)
			rr.req.Body = ioutil.NopCloser(rr.br)
		}
		if err != nil {
			return err
		}
		if rr.req.GetBody != nil {
			// this will allow us to preserve the body without having to
			// make a copy.  note we need to do this on each iteration
			rr.rc, err = rr.req.GetBody()
			if err != nil {
				return err
			}
		} else if rr.br == nil {
			// fall back to making a copy (only do this once)
			err = rr.prepareFromByteReader()
		}
	}
	return err
}

func removeRequestBody(req *http.Request) {
	req.Body = nil
	req.GetBody = nil
	req.ContentLength = 0
}
