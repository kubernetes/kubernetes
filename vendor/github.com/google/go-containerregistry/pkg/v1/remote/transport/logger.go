// Copyright 2020 Google LLC All Rights Reserved.
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

package transport

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"time"

	"github.com/google/go-containerregistry/internal/redact"
	"github.com/google/go-containerregistry/pkg/logs"
)

type logTransport struct {
	inner http.RoundTripper
}

// NewLogger returns a transport that logs requests and responses to
// github.com/google/go-containerregistry/pkg/logs.Debug.
func NewLogger(inner http.RoundTripper) http.RoundTripper {
	return &logTransport{inner}
}

func (t *logTransport) RoundTrip(in *http.Request) (out *http.Response, err error) {
	// Inspired by: github.com/motemen/go-loghttp

	// We redact token responses and binary blobs in response/request.
	omitBody, reason := redact.FromContext(in.Context())
	if omitBody {
		logs.Debug.Printf("--> %s %s [body redacted: %s]", in.Method, in.URL, reason)
	} else {
		logs.Debug.Printf("--> %s %s", in.Method, in.URL)
	}

	// Save these headers so we can redact Authorization.
	savedHeaders := in.Header.Clone()
	if in.Header != nil && in.Header.Get("authorization") != "" {
		in.Header.Set("authorization", "<redacted>")
	}

	b, err := httputil.DumpRequestOut(in, !omitBody)
	if err == nil {
		logs.Debug.Println(string(b))
	} else {
		logs.Debug.Printf("Failed to dump request %s %s: %v", in.Method, in.URL, err)
	}

	// Restore the non-redacted headers.
	in.Header = savedHeaders

	start := time.Now()
	out, err = t.inner.RoundTrip(in)
	duration := time.Since(start)
	if err != nil {
		logs.Debug.Printf("<-- %v %s %s (%s)", err, in.Method, in.URL, duration)
	}
	if out != nil {
		msg := fmt.Sprintf("<-- %d", out.StatusCode)
		if out.Request != nil {
			msg = fmt.Sprintf("%s %s", msg, out.Request.URL)
		}
		msg = fmt.Sprintf("%s (%s)", msg, duration)

		if omitBody {
			msg = fmt.Sprintf("%s [body redacted: %s]", msg, reason)
		}

		logs.Debug.Print(msg)

		b, err := httputil.DumpResponse(out, !omitBody)
		if err == nil {
			logs.Debug.Println(string(b))
		} else {
			logs.Debug.Printf("Failed to dump response %s %s: %v", in.Method, in.URL, err)
		}
	}
	return
}
