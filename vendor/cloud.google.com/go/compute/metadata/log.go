// Copyright 2024 Google LLC
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

package metadata

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
)

// Code below this point is copied from github.com/googleapis/gax-go/v2/internallog
// to avoid the dependency. The compute/metadata module is used by too many
// non-client library modules that can't justify the dependency.

// The handler returned if logging is not enabled.
type noOpHandler struct{}

func (h noOpHandler) Enabled(_ context.Context, _ slog.Level) bool {
	return false
}

func (h noOpHandler) Handle(_ context.Context, _ slog.Record) error {
	return nil
}

func (h noOpHandler) WithAttrs(_ []slog.Attr) slog.Handler {
	return h
}

func (h noOpHandler) WithGroup(_ string) slog.Handler {
	return h
}

// httpRequest returns a lazily evaluated [slog.LogValuer] for a
// [http.Request] and the associated body.
func httpRequest(req *http.Request, body []byte) slog.LogValuer {
	return &request{
		req:     req,
		payload: body,
	}
}

type request struct {
	req     *http.Request
	payload []byte
}

func (r *request) LogValue() slog.Value {
	if r == nil || r.req == nil {
		return slog.Value{}
	}
	var groupValueAttrs []slog.Attr
	groupValueAttrs = append(groupValueAttrs, slog.String("method", r.req.Method))
	groupValueAttrs = append(groupValueAttrs, slog.String("url", r.req.URL.String()))

	var headerAttr []slog.Attr
	for k, val := range r.req.Header {
		headerAttr = append(headerAttr, slog.String(k, strings.Join(val, ",")))
	}
	if len(headerAttr) > 0 {
		groupValueAttrs = append(groupValueAttrs, slog.Any("headers", headerAttr))
	}

	if len(r.payload) > 0 {
		if attr, ok := processPayload(r.payload); ok {
			groupValueAttrs = append(groupValueAttrs, attr)
		}
	}
	return slog.GroupValue(groupValueAttrs...)
}

// httpResponse returns a lazily evaluated [slog.LogValuer] for a
// [http.Response] and the associated body.
func httpResponse(resp *http.Response, body []byte) slog.LogValuer {
	return &response{
		resp:    resp,
		payload: body,
	}
}

type response struct {
	resp    *http.Response
	payload []byte
}

func (r *response) LogValue() slog.Value {
	if r == nil {
		return slog.Value{}
	}
	var groupValueAttrs []slog.Attr
	groupValueAttrs = append(groupValueAttrs, slog.String("status", fmt.Sprint(r.resp.StatusCode)))

	var headerAttr []slog.Attr
	for k, val := range r.resp.Header {
		headerAttr = append(headerAttr, slog.String(k, strings.Join(val, ",")))
	}
	if len(headerAttr) > 0 {
		groupValueAttrs = append(groupValueAttrs, slog.Any("headers", headerAttr))
	}

	if len(r.payload) > 0 {
		if attr, ok := processPayload(r.payload); ok {
			groupValueAttrs = append(groupValueAttrs, attr)
		}
	}
	return slog.GroupValue(groupValueAttrs...)
}

func processPayload(payload []byte) (slog.Attr, bool) {
	peekChar := payload[0]
	if peekChar == '{' {
		// JSON object
		var m map[string]any
		if err := json.Unmarshal(payload, &m); err == nil {
			return slog.Any("payload", m), true
		}
	} else if peekChar == '[' {
		// JSON array
		var m []any
		if err := json.Unmarshal(payload, &m); err == nil {
			return slog.Any("payload", m), true
		}
	} else {
		// Everything else
		buf := &bytes.Buffer{}
		if err := json.Compact(buf, payload); err != nil {
			// Write raw payload incase of error
			buf.Write(payload)
		}
		return slog.String("payload", buf.String()), true
	}
	return slog.Attr{}, false
}
