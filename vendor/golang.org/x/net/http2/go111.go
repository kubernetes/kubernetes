// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11

package http2

import (
	"net/http/httptrace"
	"net/textproto"
)

func traceHasWroteHeaderField(trace *httptrace.ClientTrace) bool {
	return trace != nil && trace.WroteHeaderField != nil
}

func traceWroteHeaderField(trace *httptrace.ClientTrace, k, v string) {
	if trace != nil && trace.WroteHeaderField != nil {
		trace.WroteHeaderField(k, []string{v})
	}
}

func traceGot1xxResponseFunc(trace *httptrace.ClientTrace) func(int, textproto.MIMEHeader) error {
	if trace != nil {
		return trace.Got1xxResponse
	}
	return nil
}
