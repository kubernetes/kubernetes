// Copyright 2023, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Package callctx provides helpers for storing and retrieving values out of
// [context.Context]. These values are used by our client libraries in various
// ways across the stack.
package callctx

import (
	"context"
	"fmt"
)

const (
	headerKey = contextKey("header")
)

// contextKey is a private type used to store/retrieve context values.
type contextKey string

// HeadersFromContext retrieves headers set from [SetHeaders]. These headers
// can then be cast to http.Header or metadata.MD to send along on requests.
func HeadersFromContext(ctx context.Context) map[string][]string {
	m, ok := ctx.Value(headerKey).(map[string][]string)
	if !ok {
		return nil
	}
	return m
}

// SetHeaders stores key value pairs in the returned context that can later
// be retrieved by [HeadersFromContext]. Values stored in this manner will
// automatically be retrieved by client libraries and sent as outgoing headers
// on all requests. keyvals should have a corresponding value for every key
// provided. If there is an odd number of keyvals this method will panic.
func SetHeaders(ctx context.Context, keyvals ...string) context.Context {
	if len(keyvals)%2 != 0 {
		panic(fmt.Sprintf("callctx: an even number of key value pairs must be provided, got %d", len(keyvals)))
	}
	h, ok := ctx.Value(headerKey).(map[string][]string)
	if !ok {
		h = make(map[string][]string)
	}
	for i := 0; i < len(keyvals); i = i + 2 {
		h[keyvals[i]] = append(h[keyvals[i]], keyvals[i+1])
	}
	return context.WithValue(ctx, headerKey, h)
}
