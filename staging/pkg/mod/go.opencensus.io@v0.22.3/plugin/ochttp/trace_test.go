// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ochttp

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"go.opencensus.io/plugin/ochttp/propagation/b3"
	"go.opencensus.io/plugin/ochttp/propagation/tracecontext"
	"go.opencensus.io/trace"
)

type testExporter struct {
	spans []*trace.SpanData
}

func (t *testExporter) ExportSpan(s *trace.SpanData) {
	t.spans = append(t.spans, s)
}

type testTransport struct {
	ch chan *http.Request
}

func (t *testTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.ch <- req
	return nil, errors.New("noop")
}

type testPropagator struct{}

func (t testPropagator) SpanContextFromRequest(req *http.Request) (sc trace.SpanContext, ok bool) {
	header := req.Header.Get("trace")
	buf, err := hex.DecodeString(header)
	if err != nil {
		log.Fatalf("Cannot decode trace header: %q", header)
	}
	r := bytes.NewReader(buf)
	r.Read(sc.TraceID[:])
	r.Read(sc.SpanID[:])
	opts, err := r.ReadByte()
	if err != nil {
		log.Fatalf("Cannot read trace options from trace header: %q", header)
	}
	sc.TraceOptions = trace.TraceOptions(opts)
	return sc, true
}

func (t testPropagator) SpanContextToRequest(sc trace.SpanContext, req *http.Request) {
	var buf bytes.Buffer
	buf.Write(sc.TraceID[:])
	buf.Write(sc.SpanID[:])
	buf.WriteByte(byte(sc.TraceOptions))
	req.Header.Set("trace", hex.EncodeToString(buf.Bytes()))
}

func TestTransport_RoundTrip_Race(t *testing.T) {
	// This tests that we don't modify the request in accordance with the
	// specification for http.RoundTripper.
	// We attempt to trigger a race by reading the request from a separate
	// goroutine. If the request is modified by Transport, this should trigger
	// the race detector.

	transport := &testTransport{ch: make(chan *http.Request, 1)}
	rt := &Transport{
		Propagation: &testPropagator{},
		Base:        transport,
	}
	req, _ := http.NewRequest("GET", "http://foo.com", nil)
	go func() {
		fmt.Println(*req)
	}()
	rt.RoundTrip(req)
	_ = <-transport.ch
}

func TestTransport_RoundTrip(t *testing.T) {
	_, parent := trace.StartSpan(context.Background(), "parent")
	tests := []struct {
		name   string
		parent *trace.Span
	}{
		{
			name:   "no parent",
			parent: nil,
		},
		{
			name:   "parent",
			parent: parent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transport := &testTransport{ch: make(chan *http.Request, 1)}

			rt := &Transport{
				Propagation: &testPropagator{},
				Base:        transport,
			}

			req, _ := http.NewRequest("GET", "http://foo.com", nil)
			if tt.parent != nil {
				req = req.WithContext(trace.NewContext(req.Context(), tt.parent))
			}
			rt.RoundTrip(req)

			req = <-transport.ch
			span := trace.FromContext(req.Context())

			if header := req.Header.Get("trace"); header == "" {
				t.Fatalf("Trace header = empty; want valid trace header")
			}
			if span == nil {
				t.Fatalf("Got no spans in req context; want one")
			}
			if tt.parent != nil {
				if got, want := span.SpanContext().TraceID, tt.parent.SpanContext().TraceID; got != want {
					t.Errorf("span.SpanContext().TraceID=%v; want %v", got, want)
				}
			}
		})
	}
}

func TestHandler(t *testing.T) {
	traceID := [16]byte{16, 84, 69, 170, 120, 67, 188, 139, 242, 6, 177, 32, 0, 16, 0, 0}
	tests := []struct {
		header           string
		wantTraceID      trace.TraceID
		wantTraceOptions trace.TraceOptions
	}{
		{
			header:           "105445aa7843bc8bf206b12000100000000000000000000000",
			wantTraceID:      traceID,
			wantTraceOptions: trace.TraceOptions(0),
		},
		{
			header:           "105445aa7843bc8bf206b12000100000000000000000000001",
			wantTraceID:      traceID,
			wantTraceOptions: trace.TraceOptions(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.header, func(t *testing.T) {
			handler := &Handler{
				Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					span := trace.FromContext(r.Context())
					sc := span.SpanContext()
					if got, want := sc.TraceID, tt.wantTraceID; got != want {
						t.Errorf("TraceID = %q; want %q", got, want)
					}
					if got, want := sc.TraceOptions, tt.wantTraceOptions; got != want {
						t.Errorf("TraceOptions = %v; want %v", got, want)
					}
				}),
				StartOptions: trace.StartOptions{Sampler: trace.ProbabilitySampler(0.0)},
				Propagation:  &testPropagator{},
			}
			req, _ := http.NewRequest("GET", "http://foo.com", nil)
			req.Header.Add("trace", tt.header)
			handler.ServeHTTP(nil, req)
		})
	}
}

var _ http.RoundTripper = (*traceTransport)(nil)

type collector []*trace.SpanData

func (c *collector) ExportSpan(s *trace.SpanData) {
	*c = append(*c, s)
}

func TestEndToEnd(t *testing.T) {
	tc := []struct {
		name            string
		handler         *Handler
		transport       *Transport
		wantSameTraceID bool
		wantLinks       bool // expect a link between client and server span
	}{
		{
			name:            "internal default propagation",
			handler:         &Handler{},
			transport:       &Transport{},
			wantSameTraceID: true,
		},
		{
			name:            "external default propagation",
			handler:         &Handler{IsPublicEndpoint: true},
			transport:       &Transport{},
			wantSameTraceID: false,
			wantLinks:       true,
		},
		{
			name:            "internal TraceContext propagation",
			handler:         &Handler{Propagation: &tracecontext.HTTPFormat{}},
			transport:       &Transport{Propagation: &tracecontext.HTTPFormat{}},
			wantSameTraceID: true,
		},
		{
			name:            "misconfigured propagation",
			handler:         &Handler{IsPublicEndpoint: true, Propagation: &tracecontext.HTTPFormat{}},
			transport:       &Transport{Propagation: &b3.HTTPFormat{}},
			wantSameTraceID: false,
			wantLinks:       false,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			var spans collector
			trace.RegisterExporter(&spans)
			defer trace.UnregisterExporter(&spans)

			// Start the server.
			serverDone := make(chan struct{})
			serverReturn := make(chan time.Time)
			tt.handler.StartOptions.Sampler = trace.AlwaysSample()
			url := serveHTTP(tt.handler, serverDone, serverReturn, 200)

			ctx := context.Background()
			// Make the request.
			req, err := http.NewRequest(
				http.MethodPost,
				fmt.Sprintf("%s/example/url/path?qparam=val", url),
				strings.NewReader("expected-request-body"))
			if err != nil {
				t.Fatal(err)
			}
			req = req.WithContext(ctx)
			tt.transport.StartOptions.Sampler = trace.AlwaysSample()
			c := &http.Client{
				Transport: tt.transport,
			}
			resp, err := c.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("resp.StatusCode = %d", resp.StatusCode)
			}

			// Tell the server to return from request handling.
			serverReturn <- time.Now().Add(time.Millisecond)

			respBody, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := string(respBody), "expected-response"; got != want {
				t.Fatalf("respBody = %q; want %q", got, want)
			}

			resp.Body.Close()

			<-serverDone
			trace.UnregisterExporter(&spans)

			if got, want := len(spans), 2; got != want {
				t.Fatalf("len(spans) = %d; want %d", got, want)
			}

			var client, server *trace.SpanData
			for _, sp := range spans {
				switch sp.SpanKind {
				case trace.SpanKindClient:
					client = sp
					if got, want := client.Name, "/example/url/path"; got != want {
						t.Errorf("Span name: %q; want %q", got, want)
					}
				case trace.SpanKindServer:
					server = sp
					if got, want := server.Name, "/example/url/path"; got != want {
						t.Errorf("Span name: %q; want %q", got, want)
					}
				default:
					t.Fatalf("server or client span missing; kind = %v", sp.SpanKind)
				}
			}

			if tt.wantSameTraceID {
				if server.TraceID != client.TraceID {
					t.Errorf("TraceID does not match: server.TraceID=%q client.TraceID=%q", server.TraceID, client.TraceID)
				}
				if !server.HasRemoteParent {
					t.Errorf("server span should have remote parent")
				}
				if server.ParentSpanID != client.SpanID {
					t.Errorf("server span should have client span as parent")
				}
			}
			if !tt.wantSameTraceID {
				if server.TraceID == client.TraceID {
					t.Errorf("TraceID should not be trusted")
				}
			}
			if tt.wantLinks {
				if got, want := len(server.Links), 1; got != want {
					t.Errorf("len(server.Links) = %d; want %d", got, want)
				} else {
					link := server.Links[0]
					if got, want := link.Type, trace.LinkTypeParent; got != want {
						t.Errorf("link.Type = %v; want %v", got, want)
					}
				}
			}
			if server.StartTime.Before(client.StartTime) {
				t.Errorf("server span starts before client span")
			}
			if server.EndTime.After(client.EndTime) {
				t.Errorf("client span ends before server span")
			}
		})
	}
}

func serveHTTP(handler *Handler, done chan struct{}, wait chan time.Time, statusCode int) string {
	handler.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(statusCode)
		w.(http.Flusher).Flush()

		// Simulate a slow-responding server.
		sleepUntil := <-wait
		for time.Now().Before(sleepUntil) {
			time.Sleep(time.Until(sleepUntil))
		}

		io.WriteString(w, "expected-response")
		close(done)
	})
	server := httptest.NewServer(handler)
	go func() {
		<-done
		server.Close()
	}()
	return server.URL
}

func TestSpanNameFromURL(t *testing.T) {
	tests := []struct {
		u    string
		want string
	}{
		{
			u:    "http://localhost:80/hello?q=a",
			want: "/hello",
		},
		{
			u:    "/a/b?q=c",
			want: "/a/b",
		},
	}
	for _, tt := range tests {
		t.Run(tt.u, func(t *testing.T) {
			req, err := http.NewRequest("GET", tt.u, nil)
			if err != nil {
				t.Errorf("url issue = %v", err)
			}
			if got := spanNameFromURL(req); got != tt.want {
				t.Errorf("spanNameFromURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFormatSpanName(t *testing.T) {
	formatSpanName := func(r *http.Request) string {
		return r.Method + " " + r.URL.Path
	}

	handler := &Handler{
		Handler: http.HandlerFunc(func(resp http.ResponseWriter, req *http.Request) {
			resp.Write([]byte("Hello, world!"))
		}),
		FormatSpanName: formatSpanName,
	}

	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{
		Transport: &Transport{
			FormatSpanName: formatSpanName,
			StartOptions: trace.StartOptions{
				Sampler: trace.AlwaysSample(),
			},
		},
	}

	tests := []struct {
		u    string
		want string
	}{
		{
			u:    "/hello?q=a",
			want: "GET /hello",
		},
		{
			u:    "/a/b?q=c",
			want: "GET /a/b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.u, func(t *testing.T) {
			var te testExporter
			trace.RegisterExporter(&te)
			res, err := client.Get(server.URL + tt.u)
			if err != nil {
				t.Fatalf("error creating request: %v", err)
			}
			res.Body.Close()
			trace.UnregisterExporter(&te)
			if want, got := 2, len(te.spans); want != got {
				t.Fatalf("got exported spans %#v, wanted two spans", te.spans)
			}
			if got := te.spans[0].Name; got != tt.want {
				t.Errorf("spanNameFromURL() = %v, want %v", got, tt.want)
			}
			if got := te.spans[1].Name; got != tt.want {
				t.Errorf("spanNameFromURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRequestAttributes(t *testing.T) {
	tests := []struct {
		name      string
		makeReq   func() *http.Request
		wantAttrs []trace.Attribute
	}{
		{
			name: "GET example.com/hello",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com:779/hello", nil)
				req.Header.Add("User-Agent", "ua")
				return req
			},
			wantAttrs: []trace.Attribute{
				trace.StringAttribute("http.path", "/hello"),
				trace.StringAttribute("http.url", "http://example.com:779/hello"),
				trace.StringAttribute("http.host", "example.com:779"),
				trace.StringAttribute("http.method", "GET"),
				trace.StringAttribute("http.user_agent", "ua"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := tt.makeReq()
			attrs := requestAttrs(req)

			if got, want := attrs, tt.wantAttrs; !reflect.DeepEqual(got, want) {
				t.Errorf("Request attributes = %#v; want %#v", got, want)
			}
		})
	}
}

func TestResponseAttributes(t *testing.T) {
	tests := []struct {
		name      string
		resp      *http.Response
		wantAttrs []trace.Attribute
	}{
		{
			name: "non-zero HTTP 200 response",
			resp: &http.Response{StatusCode: 200},
			wantAttrs: []trace.Attribute{
				trace.Int64Attribute("http.status_code", 200),
			},
		},
		{
			name: "zero HTTP 500 response",
			resp: &http.Response{StatusCode: 500},
			wantAttrs: []trace.Attribute{
				trace.Int64Attribute("http.status_code", 500),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attrs := responseAttrs(tt.resp)
			if got, want := attrs, tt.wantAttrs; !reflect.DeepEqual(got, want) {
				t.Errorf("Response attributes = %#v; want %#v", got, want)
			}
		})
	}
}

type TestCase struct {
	Name           string
	Method         string
	URL            string
	Headers        map[string]string
	ResponseCode   int
	SpanName       string
	SpanStatus     string
	SpanKind       string
	SpanAttributes map[string]string
}

func TestAgainstSpecs(t *testing.T) {

	fmt.Println("start")

	dat, err := ioutil.ReadFile("testdata/http-out-test-cases.json")
	if err != nil {
		t.Fatalf("error reading file: %v", err)
	}

	tests := make([]TestCase, 0)
	err = json.Unmarshal(dat, &tests)
	if err != nil {
		t.Fatalf("error parsing json: %v", err)
	}

	trace.ApplyConfig(trace.Config{DefaultSampler: trace.AlwaysSample()})

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			var spans collector
			trace.RegisterExporter(&spans)
			defer trace.UnregisterExporter(&spans)

			handler := &Handler{}
			transport := &Transport{}

			serverDone := make(chan struct{})
			serverReturn := make(chan time.Time)
			host := ""
			port := ""
			serverRequired := strings.Contains(tt.URL, "{")
			if serverRequired {
				// Start the server.
				localServerURL := serveHTTP(handler, serverDone, serverReturn, tt.ResponseCode)
				u, _ := url.Parse(localServerURL)
				host, port, _ = net.SplitHostPort(u.Host)

				tt.URL = strings.Replace(tt.URL, "{host}", host, 1)
				tt.URL = strings.Replace(tt.URL, "{port}", port, 1)
			}

			// Start a root Span in the client.
			ctx, _ := trace.StartSpan(
				context.Background(),
				"top-level")
			// Make the request.
			req, err := http.NewRequest(
				tt.Method,
				tt.URL,
				nil)
			for headerName, headerValue := range tt.Headers {
				req.Header.Add(headerName, headerValue)
			}
			if err != nil {
				t.Fatal(err)
			}
			req = req.WithContext(ctx)
			resp, err := transport.RoundTrip(req)
			if err != nil {
				// do not fail. We want to validate DNS issues
				//t.Fatal(err)
			}

			if serverRequired {
				// Tell the server to return from request handling.
				serverReturn <- time.Now().Add(time.Millisecond)
			}

			if resp != nil {
				// If it simply closes body without reading
				// synchronization problem may happen for spans slice.
				// Server span and client span will write themselves
				// at the same time
				ioutil.ReadAll(resp.Body)
				resp.Body.Close()
				if serverRequired {
					<-serverDone
				}
			}
			trace.UnregisterExporter(&spans)

			var client *trace.SpanData
			for _, sp := range spans {
				if sp.SpanKind == trace.SpanKindClient {
					client = sp
				}
			}

			if client.Name != tt.SpanName {
				t.Errorf("span names don't match: expected: %s, actual: %s", tt.SpanName, client.Name)
			}

			spanKindToStr := map[int]string{
				trace.SpanKindClient: "Client",
				trace.SpanKindServer: "Server",
			}

			if !strings.EqualFold(codeToStr[client.Status.Code], tt.SpanStatus) {
				t.Errorf("span status don't match: expected: %s, actual: %d (%s)", tt.SpanStatus, client.Status.Code, codeToStr[client.Status.Code])
			}

			if !strings.EqualFold(spanKindToStr[client.SpanKind], tt.SpanKind) {
				t.Errorf("span kind don't match: expected: %s, actual: %d (%s)", tt.SpanKind, client.SpanKind, spanKindToStr[client.SpanKind])
			}

			normalizedActualAttributes := map[string]string{}
			for k, v := range client.Attributes {
				normalizedActualAttributes[k] = fmt.Sprintf("%v", v)
			}

			normalizedExpectedAttributes := map[string]string{}
			for k, v := range tt.SpanAttributes {
				normalizedValue := v
				normalizedValue = strings.Replace(normalizedValue, "{host}", host, 1)
				normalizedValue = strings.Replace(normalizedValue, "{port}", port, 1)

				normalizedExpectedAttributes[k] = normalizedValue
			}

			if got, want := normalizedActualAttributes, normalizedExpectedAttributes; !reflect.DeepEqual(got, want) {
				t.Errorf("Request attributes = %#v; want %#v", got, want)
			}
		})
	}
}

func TestStatusUnitTest(t *testing.T) {
	tests := []struct {
		in   int
		want trace.Status
	}{
		{200, trace.Status{Code: trace.StatusCodeOK, Message: `OK`}},
		{204, trace.Status{Code: trace.StatusCodeOK, Message: `OK`}},
		{100, trace.Status{Code: trace.StatusCodeUnknown, Message: `UNKNOWN`}},
		{500, trace.Status{Code: trace.StatusCodeUnknown, Message: `UNKNOWN`}},
		{400, trace.Status{Code: trace.StatusCodeInvalidArgument, Message: `INVALID_ARGUMENT`}},
		{422, trace.Status{Code: trace.StatusCodeInvalidArgument, Message: `INVALID_ARGUMENT`}},
		{499, trace.Status{Code: trace.StatusCodeCancelled, Message: `CANCELLED`}},
		{404, trace.Status{Code: trace.StatusCodeNotFound, Message: `NOT_FOUND`}},
		{600, trace.Status{Code: trace.StatusCodeUnknown, Message: `UNKNOWN`}},
		{401, trace.Status{Code: trace.StatusCodeUnauthenticated, Message: `UNAUTHENTICATED`}},
		{403, trace.Status{Code: trace.StatusCodePermissionDenied, Message: `PERMISSION_DENIED`}},
		{301, trace.Status{Code: trace.StatusCodeOK, Message: `OK`}},
		{501, trace.Status{Code: trace.StatusCodeUnimplemented, Message: `UNIMPLEMENTED`}},
		{409, trace.Status{Code: trace.StatusCodeAlreadyExists, Message: `ALREADY_EXISTS`}},
		{429, trace.Status{Code: trace.StatusCodeResourceExhausted, Message: `RESOURCE_EXHAUSTED`}},
		{503, trace.Status{Code: trace.StatusCodeUnavailable, Message: `UNAVAILABLE`}},
		{504, trace.Status{Code: trace.StatusCodeDeadlineExceeded, Message: `DEADLINE_EXCEEDED`}},
	}

	for _, tt := range tests {
		got, want := TraceStatus(tt.in, ""), tt.want
		if got != want {
			t.Errorf("status(%d) got = (%#v) want = (%#v)", tt.in, got, want)
		}
	}
}
