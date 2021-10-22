package ochttp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/http2"

	"go.opencensus.io/stats/view"
	"go.opencensus.io/trace"
)

func httpHandler(statusCode, respSize int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(statusCode)
		body := make([]byte, respSize)
		w.Write(body)
	})
}

func updateMean(mean float64, sample, count int) float64 {
	if count == 1 {
		return float64(sample)
	}
	return mean + (float64(sample)-mean)/float64(count)
}

func TestHandlerStatsCollection(t *testing.T) {
	if err := view.Register(DefaultServerViews...); err != nil {
		t.Fatalf("Failed to register ochttp.DefaultServerViews error: %v", err)
	}

	views := []string{
		"opencensus.io/http/server/request_count",
		"opencensus.io/http/server/latency",
		"opencensus.io/http/server/request_bytes",
		"opencensus.io/http/server/response_bytes",
	}

	// TODO: test latency measurements?
	tests := []struct {
		name, method, target                 string
		count, statusCode, reqSize, respSize int
	}{
		{"get 200", "GET", "http://opencensus.io/request/one", 10, 200, 512, 512},
		{"post 503", "POST", "http://opencensus.io/request/two", 5, 503, 1024, 16384},
		{"no body 302", "GET", "http://opencensus.io/request/three", 2, 302, 0, 0},
	}
	totalCount, meanReqSize, meanRespSize := 0, 0.0, 0.0

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := bytes.NewBuffer(make([]byte, test.reqSize))
			r := httptest.NewRequest(test.method, test.target, body)
			w := httptest.NewRecorder()
			mux := http.NewServeMux()
			mux.Handle("/request/", httpHandler(test.statusCode, test.respSize))
			h := &Handler{
				Handler: mux,
				StartOptions: trace.StartOptions{
					Sampler: trace.NeverSample(),
				},
			}
			for i := 0; i < test.count; i++ {
				h.ServeHTTP(w, r)
				totalCount++
				// Distributions do not track sum directly, we must
				// mimic their behaviour to avoid rounding failures.
				meanReqSize = updateMean(meanReqSize, test.reqSize, totalCount)
				meanRespSize = updateMean(meanRespSize, test.respSize, totalCount)
			}
		})
	}

	for _, viewName := range views {
		v := view.Find(viewName)
		if v == nil {
			t.Errorf("view not found %q", viewName)
			continue
		}
		rows, err := view.RetrieveData(viewName)
		if err != nil {
			t.Error(err)
			continue
		}
		if got, want := len(rows), 1; got != want {
			t.Errorf("len(%q) = %d; want %d", viewName, got, want)
			continue
		}
		data := rows[0].Data

		var count int
		var sum float64
		switch data := data.(type) {
		case *view.CountData:
			count = int(data.Value)
		case *view.DistributionData:
			count = int(data.Count)
			sum = data.Sum()
		default:
			t.Errorf("Unknown data type: %v", data)
			continue
		}

		if got, want := count, totalCount; got != want {
			t.Fatalf("%s = %d; want %d", viewName, got, want)
		}

		// We can only check sum for distribution views.
		switch viewName {
		case "opencensus.io/http/server/request_bytes":
			if got, want := sum, meanReqSize*float64(totalCount); got != want {
				t.Fatalf("%s = %g; want %g", viewName, got, want)
			}
		case "opencensus.io/http/server/response_bytes":
			if got, want := sum, meanRespSize*float64(totalCount); got != want {
				t.Fatalf("%s = %g; want %g", viewName, got, want)
			}
		}
	}
}

type testResponseWriterHijacker struct {
	httptest.ResponseRecorder
}

func (trw *testResponseWriterHijacker) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, nil
}

func TestUnitTestHandlerProxiesHijack(t *testing.T) {
	tests := []struct {
		w         http.ResponseWriter
		hasHijack bool
	}{
		{httptest.NewRecorder(), false},
		{nil, false},
		{new(testResponseWriterHijacker), true},
	}

	for i, tt := range tests {
		tw := &trackingResponseWriter{writer: tt.w}
		w := tw.wrappedResponseWriter()
		_, ttHijacker := w.(http.Hijacker)
		if want, have := tt.hasHijack, ttHijacker; want != have {
			t.Errorf("#%d Hijack got %t, want %t", i, have, want)
		}
	}
}

// Integration test with net/http to ensure that our Handler proxies to its
// response the call to (http.Hijack).Hijacker() and that that successfully
// passes with HTTP/1.1 connections. See Issue #642
func TestHandlerProxiesHijack_HTTP1(t *testing.T) {
	cst := httptest.NewServer(&Handler{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var writeMsg func(string)
			defer func() {
				err := recover()
				writeMsg(fmt.Sprintf("Proto=%s\npanic=%v", r.Proto, err != nil))
			}()
			conn, _, _ := w.(http.Hijacker).Hijack()
			writeMsg = func(msg string) {
				fmt.Fprintf(conn, "%s 200\nContentLength: %d", r.Proto, len(msg))
				fmt.Fprintf(conn, "\r\n\r\n%s", msg)
				conn.Close()
			}
		}),
	})
	defer cst.Close()

	testCases := []struct {
		name string
		tr   *http.Transport
		want string
	}{
		{
			name: "http1-transport",
			tr:   new(http.Transport),
			want: "Proto=HTTP/1.1\npanic=false",
		},
		{
			name: "http2-transport",
			tr: func() *http.Transport {
				tr := new(http.Transport)
				http2.ConfigureTransport(tr)
				return tr
			}(),
			want: "Proto=HTTP/1.1\npanic=false",
		},
	}

	for _, tc := range testCases {
		c := &http.Client{Transport: &Transport{Base: tc.tr}}
		res, err := c.Get(cst.URL)
		if err != nil {
			t.Errorf("(%s) unexpected error %v", tc.name, err)
			continue
		}
		blob, _ := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if g, w := string(blob), tc.want; g != w {
			t.Errorf("(%s) got = %q; want = %q", tc.name, g, w)
		}
	}
}

// Integration test with net/http, x/net/http2 to ensure that our Handler proxies
// to its response the call to (http.Hijack).Hijacker() and that that crashes
// since http.Hijacker and HTTP/2.0 connections are incompatible, but the
// detection is only at runtime and ensure that we can stream and flush to the
// connection even after invoking Hijack(). See Issue #642.
func TestHandlerProxiesHijack_HTTP2(t *testing.T) {
	cst := httptest.NewUnstartedServer(&Handler{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if _, ok := w.(http.Hijacker); ok {
				conn, _, err := w.(http.Hijacker).Hijack()
				if conn != nil {
					data := fmt.Sprintf("Surprisingly got the Hijacker() Proto: %s", r.Proto)
					fmt.Fprintf(conn, "%s 200\nContent-Length:%d\r\n\r\n%s", r.Proto, len(data), data)
					conn.Close()
					return
				}

				switch {
				case err == nil:
					fmt.Fprintf(w, "Unexpectedly did not encounter an error!")
				default:
					fmt.Fprintf(w, "Unexpected error: %v", err)
				case strings.Contains(err.(error).Error(), "Hijack"):
					// Confirmed HTTP/2.0, let's stream to it
					for i := 0; i < 5; i++ {
						fmt.Fprintf(w, "%d\n", i)
						w.(http.Flusher).Flush()
					}
				}
			} else {
				// Confirmed HTTP/2.0, let's stream to it
				for i := 0; i < 5; i++ {
					fmt.Fprintf(w, "%d\n", i)
					w.(http.Flusher).Flush()
				}
			}
		}),
	})
	cst.TLS = &tls.Config{NextProtos: []string{"h2"}}
	cst.StartTLS()
	defer cst.Close()

	if wantPrefix := "https://"; !strings.HasPrefix(cst.URL, wantPrefix) {
		t.Fatalf("URL got = %q wantPrefix = %q", cst.URL, wantPrefix)
	}

	tr := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	http2.ConfigureTransport(tr)
	c := &http.Client{Transport: tr}
	res, err := c.Get(cst.URL)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	blob, _ := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if g, w := string(blob), "0\n1\n2\n3\n4\n"; g != w {
		t.Errorf("got = %q; want = %q", g, w)
	}
}

func TestEnsureTrackingResponseWriterSetsStatusCode(t *testing.T) {
	// Ensure that the trackingResponseWriter always sets the spanStatus on ending the span.
	// Because we can only examine the Status after exporting, this test roundtrips a
	// couple of requests and then later examines the exported spans.
	// See Issue #700.
	exporter := &spanExporter{cur: make(chan *trace.SpanData, 1)}
	trace.RegisterExporter(exporter)
	defer trace.UnregisterExporter(exporter)

	tests := []struct {
		res  *http.Response
		want trace.Status
	}{
		{res: &http.Response{StatusCode: 200}, want: trace.Status{Code: trace.StatusCodeOK, Message: `OK`}},
		{res: &http.Response{StatusCode: 500}, want: trace.Status{Code: trace.StatusCodeUnknown, Message: `UNKNOWN`}},
		{res: &http.Response{StatusCode: 403}, want: trace.Status{Code: trace.StatusCodePermissionDenied, Message: `PERMISSION_DENIED`}},
		{res: &http.Response{StatusCode: 401}, want: trace.Status{Code: trace.StatusCodeUnauthenticated, Message: `UNAUTHENTICATED`}},
		{res: &http.Response{StatusCode: 429}, want: trace.Status{Code: trace.StatusCodeResourceExhausted, Message: `RESOURCE_EXHAUSTED`}},
	}

	for _, tt := range tests {
		t.Run(tt.want.Message, func(t *testing.T) {
			ctx := context.Background()
			prc, pwc := io.Pipe()
			go func() {
				pwc.Write([]byte("Foo"))
				pwc.Close()
			}()
			inRes := tt.res
			inRes.Body = prc
			tr := &traceTransport{
				base:           &testResponseTransport{res: inRes},
				formatSpanName: spanNameFromURL,
				startOptions: trace.StartOptions{
					Sampler: trace.AlwaysSample(),
				},
			}
			req, err := http.NewRequest("POST", "https://example.org", bytes.NewReader([]byte("testing")))
			if err != nil {
				t.Fatalf("NewRequest error: %v", err)
			}
			req = req.WithContext(ctx)
			res, err := tr.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip error: %v", err)
			}
			_, _ = ioutil.ReadAll(res.Body)
			res.Body.Close()

			cur := <-exporter.cur
			if got, want := cur.Status, tt.want; got != want {
				t.Fatalf("SpanData:\ngot =  (%#v)\nwant = (%#v)", got, want)
			}
		})
	}
}

type spanExporter struct {
	sync.Mutex
	cur chan *trace.SpanData
}

var _ trace.Exporter = (*spanExporter)(nil)

func (se *spanExporter) ExportSpan(sd *trace.SpanData) {
	se.Lock()
	se.cur <- sd
	se.Unlock()
}

type testResponseTransport struct {
	res *http.Response
}

var _ http.RoundTripper = (*testResponseTransport)(nil)

func (rb *testResponseTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return rb.res, nil
}

func TestHandlerImplementsHTTPPusher(t *testing.T) {
	cst := setupAndStartServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pusher, ok := w.(http.Pusher)
		if !ok {
			w.Write([]byte("false"))
			return
		}
		err := pusher.Push("/static.css", &http.PushOptions{
			Method: "GET",
			Header: http.Header{"Accept-Encoding": r.Header["Accept-Encoding"]},
		})
		if err != nil && false {
			// TODO: (@odeke-em) consult with Go stdlib for why trying
			// to configure even an HTTP/2 server and HTTP/2 transport
			// still return http.ErrNotSupported even without using ochttp.Handler.
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Write([]byte("true"))
	}), asHTTP2)
	defer cst.Close()

	tests := []struct {
		rt       http.RoundTripper
		wantBody string
	}{
		{
			rt:       h1Transport(),
			wantBody: "false",
		},
		{
			rt:       h2Transport(),
			wantBody: "true",
		},
		{
			rt:       &Transport{Base: h1Transport()},
			wantBody: "false",
		},
		{
			rt:       &Transport{Base: h2Transport()},
			wantBody: "true",
		},
	}

	for i, tt := range tests {
		c := &http.Client{Transport: &Transport{Base: tt.rt}}
		res, err := c.Get(cst.URL)
		if err != nil {
			t.Errorf("#%d: Unexpected error %v", i, err)
			continue
		}
		body, _ := ioutil.ReadAll(res.Body)
		_ = res.Body.Close()
		if g, w := string(body), tt.wantBody; g != w {
			t.Errorf("#%d: got = %q; want = %q", i, g, w)
		}
	}
}

const (
	isNil       = "isNil"
	hang        = "hang"
	ended       = "ended"
	nonNotifier = "nonNotifier"

	asHTTP1 = false
	asHTTP2 = true
)

func setupAndStartServer(hf func(http.ResponseWriter, *http.Request), isHTTP2 bool) *httptest.Server {
	cst := httptest.NewUnstartedServer(&Handler{
		Handler: http.HandlerFunc(hf),
	})
	if isHTTP2 {
		http2.ConfigureServer(cst.Config, new(http2.Server))
		cst.TLS = cst.Config.TLSConfig
		cst.StartTLS()
	} else {
		cst.Start()
	}

	return cst
}

func insecureTLS() *tls.Config     { return &tls.Config{InsecureSkipVerify: true} }
func h1Transport() *http.Transport { return &http.Transport{TLSClientConfig: insecureTLS()} }
func h2Transport() *http.Transport {
	tr := &http.Transport{TLSClientConfig: insecureTLS()}
	http2.ConfigureTransport(tr)
	return tr
}

type concurrentBuffer struct {
	sync.RWMutex
	bw *bytes.Buffer
}

func (cw *concurrentBuffer) Write(b []byte) (int, error) {
	cw.Lock()
	defer cw.Unlock()

	return cw.bw.Write(b)
}

func (cw *concurrentBuffer) String() string {
	cw.Lock()
	defer cw.Unlock()

	return cw.bw.String()
}

func handleCloseNotify(outLog io.Writer) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cn, ok := w.(http.CloseNotifier)
		if !ok {
			fmt.Fprintln(outLog, nonNotifier)
			return
		}
		ch := cn.CloseNotify()
		if ch == nil {
			fmt.Fprintln(outLog, isNil)
			return
		}

		<-ch
		fmt.Fprintln(outLog, ended)
	})
}

func TestHandlerImplementsHTTPCloseNotify(t *testing.T) {
	http1Log := &concurrentBuffer{bw: new(bytes.Buffer)}
	http1Server := setupAndStartServer(handleCloseNotify(http1Log), asHTTP1)
	http2Log := &concurrentBuffer{bw: new(bytes.Buffer)}
	http2Server := setupAndStartServer(handleCloseNotify(http2Log), asHTTP2)

	defer http1Server.Close()
	defer http2Server.Close()

	tests := []struct {
		url  string
		want string
	}{
		{url: http1Server.URL, want: nonNotifier},
		{url: http2Server.URL, want: ended},
	}

	transports := []struct {
		name string
		rt   http.RoundTripper
	}{
		{name: "http2+ochttp", rt: &Transport{Base: h2Transport()}},
		{name: "http1+ochttp", rt: &Transport{Base: h1Transport()}},
		{name: "http1-ochttp", rt: h1Transport()},
		{name: "http2-ochttp", rt: h2Transport()},
	}

	// Each transport invokes one of two server types, either HTTP/1 or HTTP/2
	for _, trc := range transports {
		// Try out all the transport combinations
		for i, tt := range tests {
			req, err := http.NewRequest("GET", tt.url, nil)
			if err != nil {
				t.Errorf("#%d: Unexpected error making request: %v", i, err)
				continue
			}

			// Using a timeout to ensure that the request is cancelled and the server
			// if its handler implements CloseNotify will see this as the client leaving.
			ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
			defer cancel()
			req = req.WithContext(ctx)

			client := &http.Client{Transport: trc.rt}
			res, err := client.Do(req)
			if err != nil && !strings.Contains(err.Error(), "context deadline exceeded") {
				t.Errorf("#%d: %sClient Unexpected error %v", i, trc.name, err)
				continue
			}
			if res != nil && res.Body != nil {
				io.CopyN(ioutil.Discard, res.Body, 5)
				_ = res.Body.Close()
			}
		}
	}

	// Wait for a couple of milliseconds for the GoAway frames to be properly propagated
	<-time.After(200 * time.Millisecond)

	wantHTTP1Log := strings.Repeat("ended\n", len(transports))
	wantHTTP2Log := strings.Repeat("ended\n", len(transports))
	if g, w := http1Log.String(), wantHTTP1Log; g != w {
		t.Errorf("HTTP1Log got\n\t%q\nwant\n\t%q", g, w)
	}
	if g, w := http2Log.String(), wantHTTP2Log; g != w {
		t.Errorf("HTTP2Log got\n\t%q\nwant\n\t%q", g, w)
	}
}

func testHealthEndpointSkipArray(r *http.Request) bool {
	for _, toSkip := range []string{"/health", "/metrics"} {
		if r.URL.Path == toSkip {
			return true
		}
	}
	return false
}

func TestIgnoreHealthEndpoints(t *testing.T) {
	var spans int

	client := &http.Client{}
	tests := []struct {
		path               string
		healthEndpointFunc func(*http.Request) bool
	}{
		{"/healthz", nil},
		{"/_ah/health", nil},
		{"/healthz", testHealthEndpointSkipArray},
		{"/_ah/health", testHealthEndpointSkipArray},
		{"/health", testHealthEndpointSkipArray},
		{"/metrics", testHealthEndpointSkipArray},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			ts := httptest.NewServer(&Handler{
				Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					span := trace.FromContext(r.Context())
					if span != nil {
						spans++
					}
					fmt.Fprint(w, "ok")
				}),
				StartOptions: trace.StartOptions{
					Sampler: trace.AlwaysSample(),
				},
				IsHealthEndpoint: tt.healthEndpointFunc,
			})
			defer ts.Close()

			resp, err := client.Get(ts.URL + tt.path)
			if err != nil {
				t.Fatalf("Cannot GET %q: %v", tt.path, err)
			}
			b, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("Cannot read body for %q: %v", tt.path, err)
			}

			if got, want := string(b), "ok"; got != want {
				t.Fatalf("Body for %q = %q; want %q", tt.path, got, want)
			}
			resp.Body.Close()
		})
	}

	if spans > 0 {
		t.Errorf("Got %v spans; want no spans", spans)
	}
}
