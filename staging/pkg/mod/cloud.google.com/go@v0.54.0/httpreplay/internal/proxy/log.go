// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"sync"

	"github.com/google/martian"
)

// Replacement for the HAR logging that comes with martian. HAR is not designed for
// replay. In particular, response bodies are interpreted (e.g. decompressed), and we
// just want them to be stored literally. This isn't something we can fix in martian: it
// is required in the HAR spec (http://www.softwareishard.com/blog/har-12-spec/#content).

// LogVersion is the current version of the log format. It can be used to
// support changes to the format over time, so newer code can read older files.
const LogVersion = "0.2"

// A Log is a record of HTTP interactions, suitable for replay. It can be serialized to JSON.
type Log struct {
	Initial   []byte // initial data for replay
	Version   string // version of this log format
	Converter *Converter
	Entries   []*Entry
}

// An Entry  single request-response pair.
type Entry struct {
	ID       string // unique ID
	Request  *Request
	Response *Response
}

// A Request represents an http.Request in the log.
type Request struct {
	Method string      // http.Request.Method
	URL    string      // http.Request.URL, as a string
	Header http.Header // http.Request.Header
	// We need to understand multipart bodies because the boundaries are
	// generated randomly, so we can't just compare the entire bodies for equality.
	MediaType string      // the media type part of the Content-Type header
	BodyParts [][]byte    // http.Request.Body, read to completion and split for multipart
	Trailer   http.Header `json:",omitempty"` // http.Request.Trailer
}

// A Response represents an http.Response in the log.
type Response struct {
	StatusCode int         // http.Response.StatusCode
	Proto      string      // http.Response.Proto
	ProtoMajor int         // http.Response.ProtoMajor
	ProtoMinor int         // http.Response.ProtoMinor
	Header     http.Header // http.Response.Header
	Body       []byte      // http.Response.Body, read to completion
	Trailer    http.Header `json:",omitempty"` // http.Response.Trailer
}

// A Logger maintains a request-response log.
type Logger struct {
	mu      sync.Mutex
	entries map[string]*Entry // from ID
	log     *Log
}

// newLogger creates a new logger.
func newLogger() *Logger {
	return &Logger{
		log: &Log{
			Version:   LogVersion,
			Converter: defaultConverter(),
		},
		entries: map[string]*Entry{},
	}
}

// ModifyRequest logs requests.
func (l *Logger) ModifyRequest(req *http.Request) error {
	if req.Method == "CONNECT" {
		return nil
	}
	ctx := martian.NewContext(req)
	if ctx.SkippingLogging() {
		return nil
	}
	lreq, err := l.log.Converter.convertRequest(req)
	if err != nil {
		return err
	}
	id := ctx.ID()
	entry := &Entry{ID: id, Request: lreq}

	l.mu.Lock()
	defer l.mu.Unlock()

	if _, ok := l.entries[id]; ok {
		panic(fmt.Sprintf("proxy: duplicate request ID: %s", id))
	}
	l.entries[id] = entry
	l.log.Entries = append(l.log.Entries, entry)
	return nil
}

// ModifyResponse logs responses.
func (l *Logger) ModifyResponse(res *http.Response) error {
	ctx := martian.NewContext(res.Request)
	if ctx.SkippingLogging() {
		return nil
	}
	id := ctx.ID()
	lres, err := l.log.Converter.convertResponse(res)
	if err != nil {
		return err
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if e, ok := l.entries[id]; ok {
		e.Response = lres
	}
	// Ignore the response if we haven't seen the request.
	return nil
}

// Extract returns the Log and removes it. The Logger is not usable
// after this call.
func (l *Logger) Extract() *Log {
	l.mu.Lock()
	defer l.mu.Unlock()
	r := l.log
	l.log = nil
	l.entries = nil
	return r
}

func toHTTPResponse(lr *Response, req *http.Request) *http.Response {
	res := &http.Response{
		StatusCode:    lr.StatusCode,
		Proto:         lr.Proto,
		ProtoMajor:    lr.ProtoMajor,
		ProtoMinor:    lr.ProtoMinor,
		Header:        lr.Header,
		Body:          ioutil.NopCloser(bytes.NewReader(lr.Body)),
		ContentLength: int64(len(lr.Body)),
	}
	res.Request = req
	// For HEAD, set ContentLength to the value of the Content-Length header, or -1
	// if there isn't one.
	if req.Method == "HEAD" {
		res.ContentLength = -1
		if c := res.Header["Content-Length"]; len(c) == 1 {
			if c64, err := strconv.ParseInt(c[0], 10, 64); err == nil {
				res.ContentLength = c64
			}
		}
	}
	return res
}
