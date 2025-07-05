// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"bufio"
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"os"
	"time"
)

// Response represents a cached HTTP response entry.
type Response struct {
	ID          string         // unique identifier for the response entry
	Data        *http.Response // the actual HTTP response data
	RequestedAt time.Time      // time when the request was made used for determining cache freshness
	ReceivedAt  time.Time      // time when the response was received, used for determining cache freshness
}

var _ slog.LogValuer = (*Response)(nil)

func (r Response) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("id", r.ID),
		slog.Time("requested_at", r.RequestedAt),
		slog.Time("received_at", r.ReceivedAt),
		slog.String("status", r.Data.Status),
		slog.Int("status_code", r.Data.StatusCode),
	)
}

// DateHeader returns the parsed value of the "Date" header from the response.
//
// NOTE: It assumes a valid "Date" header has been set by [FixDateHeader].
func (r *Response) DateHeader() time.Time {
	date, _ := RawTime(r.Data.Header.Get("Date")).Value()
	return date
}

// Deprecated: This function is a workaround for Kubernetes' handling of "UTC" Expires headers.
func parseHTTPDateCompat(dateStr string) (t time.Time, err error) {
	if os.Getenv("HTTPCACHE_ALLOW_UTC_DATETIMEFORMAT") == "1" {
		// TODO(bartventer): PR Kubernetes to emit "GMT" per RFC 9110 §5.6.7.
		// See k8s.io/kube-openapi/pkg/handler3/handler.go for "UTC" usage.
		return time.Parse(time.RFC1123, dateStr)
	}
	return
}

func (r *Response) ExpiresHeader() (t time.Time, found bool, valid bool) {
	expiresStr := r.Data.Header.Get("Expires")
	if expiresStr == "" {
		return
	}
	found = true
	if t, valid = RawTime(expiresStr).Value(); valid {
		return
	}
	expires, err := parseHTTPDateCompat(expiresStr)
	if err != nil || expires.IsZero() {
		return
	}
	return expires, true, true
}

func (r *Response) WriteTo(w io.Writer) (int64, error) {
	n, err := fmt.Fprintf(
		w,
		"%s\t%s\t%s\n",
		r.ID,
		r.RequestedAt.Format(time.RFC3339Nano),
		r.ReceivedAt.Format(time.RFC3339Nano),
	)
	return int64(n), err
}

var _ encoding.BinaryMarshaler = (*Response)(nil)

func (r Response) MarshalBinary() ([]byte, error) {
	respBytes, err := httputil.DumpResponse(r.Data, true)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal response: %w", err)
	}

	var buf bytes.Buffer
	_, err = r.WriteTo(&buf)
	if err != nil {
		return nil, fmt.Errorf("failed to write metadata: %w", err)
	}
	buf.Write(respBytes)
	return buf.Bytes(), nil
}

var (
	errReadBytes       = errors.New("failed to read bytes")
	errInvalidMetaLine = errors.New("invalid metadata line format")
	errInvalidResponse = errors.New("invalid response")
)

// ParseResponse parses a cached HTTP response entry from binary data and reconstructs
// a [Response] using the provided request for context. Returns an error if parsing fails.
func ParseResponse(data []byte, req *http.Request) (resp *Response, err error) {
	reader := bufio.NewReader(bytes.NewReader(data))
	metaLine, err := reader.ReadBytes('\n')
	if err != nil {
		return nil, errors.Join(errReadBytes, fmt.Errorf("failed to read metadata line: %w", err))
	}
	metaLine = bytes.TrimSpace(metaLine)
	parts := bytes.Split(metaLine, []byte("\t"))
	if len(parts) != 3 {
		return nil, fmt.Errorf("%w: expected 3 parts, got %d", errInvalidMetaLine, len(parts))
	}
	resp = new(Response)
	resp.ID = string(parts[0])
	resp.RequestedAt, _ = time.Parse(time.RFC3339Nano, string(parts[1]))
	resp.ReceivedAt, _ = time.Parse(time.RFC3339Nano, string(parts[2]))
	//nolint:bodyclose // The response body is not closed here, as it may be reused later.
	r, err := http.ReadResponse(reader, req)
	if err != nil {
		return nil, errors.Join(errInvalidResponse, fmt.Errorf("failed to read response: %w", err))
	}
	resp.Data = r
	return resp, nil
}

// ResponseRef represents a reference to a cached HTTP response.
type ResponseRef struct {
	ResponseID   string            `json:"id"`                   // unique identifier for the response entry.
	Vary         string            `json:"vary"`                 // value of the Vary response header.
	VaryResolved map[string]string `json:"vary_resolved"`        // resolved varying request headers, keys are canonicalized.
	ReceivedAt   time.Time         `json:"received_at,omitzero"` // when the response was generated.
}

var _ slog.LogValuer = (*ResponseRef)(nil)

func (r ResponseRef) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("response_id", r.ResponseID),
		slog.String("vary", r.Vary),
		slog.Any("vary_resolved", r.VaryResolved),
		slog.Time("received_at", r.ReceivedAt),
	)
}

type ResponseRefs []*ResponseRef

func (he ResponseRefs) ResponseIDs() iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, entry := range he {
			if !yield(entry.ResponseID) {
				return
			}
		}
	}
}
