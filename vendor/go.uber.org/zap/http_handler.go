// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zap

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"go.uber.org/zap/zapcore"
)

// ServeHTTP is a simple JSON endpoint that can report on or change the current
// logging level.
//
// # GET
//
// The GET request returns a JSON description of the current logging level like:
//
//	{"level":"info"}
//
// # PUT
//
// The PUT request changes the logging level. It is perfectly safe to change the
// logging level while a program is running. Two content types are supported:
//
//	Content-Type: application/x-www-form-urlencoded
//
// With this content type, the level can be provided through the request body or
// a query parameter. The log level is URL encoded like:
//
//	level=debug
//
// The request body takes precedence over the query parameter, if both are
// specified.
//
// This content type is the default for a curl PUT request. Following are two
// example curl requests that both set the logging level to debug.
//
//	curl -X PUT localhost:8080/log/level?level=debug
//	curl -X PUT localhost:8080/log/level -d level=debug
//
// For any other content type, the payload is expected to be JSON encoded and
// look like:
//
//	{"level":"info"}
//
// An example curl request could look like this:
//
//	curl -X PUT localhost:8080/log/level -H "Content-Type: application/json" -d '{"level":"debug"}'
func (lvl AtomicLevel) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := lvl.serveHTTP(w, r); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprintf(w, "internal error: %v", err)
	}
}

func (lvl AtomicLevel) serveHTTP(w http.ResponseWriter, r *http.Request) error {
	type errorResponse struct {
		Error string `json:"error"`
	}
	type payload struct {
		Level zapcore.Level `json:"level"`
	}

	enc := json.NewEncoder(w)

	switch r.Method {
	case http.MethodGet:
		return enc.Encode(payload{Level: lvl.Level()})

	case http.MethodPut:
		requestedLvl, err := decodePutRequest(r.Header.Get("Content-Type"), r)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return enc.Encode(errorResponse{Error: err.Error()})
		}
		lvl.SetLevel(requestedLvl)
		return enc.Encode(payload{Level: lvl.Level()})

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return enc.Encode(errorResponse{
			Error: "Only GET and PUT are supported.",
		})
	}
}

// Decodes incoming PUT requests and returns the requested logging level.
func decodePutRequest(contentType string, r *http.Request) (zapcore.Level, error) {
	if contentType == "application/x-www-form-urlencoded" {
		return decodePutURL(r)
	}
	return decodePutJSON(r.Body)
}

func decodePutURL(r *http.Request) (zapcore.Level, error) {
	lvl := r.FormValue("level")
	if lvl == "" {
		return 0, errors.New("must specify logging level")
	}
	var l zapcore.Level
	if err := l.UnmarshalText([]byte(lvl)); err != nil {
		return 0, err
	}
	return l, nil
}

func decodePutJSON(body io.Reader) (zapcore.Level, error) {
	var pld struct {
		Level *zapcore.Level `json:"level"`
	}
	if err := json.NewDecoder(body).Decode(&pld); err != nil {
		return 0, fmt.Errorf("malformed request body: %v", err)
	}
	if pld.Level == nil {
		return 0, errors.New("must specify logging level")
	}
	return *pld.Level, nil
}
