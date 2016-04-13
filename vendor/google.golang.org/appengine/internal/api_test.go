// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package internal

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	netcontext "golang.org/x/net/context"

	basepb "google.golang.org/appengine/internal/base"
	remotepb "google.golang.org/appengine/internal/remote_api"
)

const testTicketHeader = "X-Magic-Ticket-Header"

func init() {
	ticketHeader = testTicketHeader
}

type fakeAPIHandler struct {
	hang chan int // used for RunSlowly RPC

	LogFlushes int32 // atomic
}

func (f *fakeAPIHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	writeResponse := func(res *remotepb.Response) {
		hresBody, err := proto.Marshal(res)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed encoding API response: %v", err), 500)
			return
		}
		w.Write(hresBody)
	}

	if r.URL.Path != "/rpc_http" {
		http.NotFound(w, r)
		return
	}
	hreqBody, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Bad body: %v", err), 500)
		return
	}
	apiReq := &remotepb.Request{}
	if err := proto.Unmarshal(hreqBody, apiReq); err != nil {
		http.Error(w, fmt.Sprintf("Bad encoded API request: %v", err), 500)
		return
	}
	if *apiReq.RequestId != "s3cr3t" {
		writeResponse(&remotepb.Response{
			RpcError: &remotepb.RpcError{
				Code:   proto.Int32(int32(remotepb.RpcError_SECURITY_VIOLATION)),
				Detail: proto.String("bad security ticket"),
			},
		})
		return
	}
	if got, want := r.Header.Get(dapperHeader), "trace-001"; got != want {
		writeResponse(&remotepb.Response{
			RpcError: &remotepb.RpcError{
				Code:   proto.Int32(int32(remotepb.RpcError_BAD_REQUEST)),
				Detail: proto.String(fmt.Sprintf("trace info = %q, want %q", got, want)),
			},
		})
		return
	}

	service, method := *apiReq.ServiceName, *apiReq.Method
	var resOut proto.Message
	if service == "actordb" && method == "LookupActor" {
		req := &basepb.StringProto{}
		res := &basepb.StringProto{}
		if err := proto.Unmarshal(apiReq.Request, req); err != nil {
			http.Error(w, fmt.Sprintf("Bad encoded request: %v", err), 500)
			return
		}
		if *req.Value == "Doctor Who" {
			res.Value = proto.String("David Tennant")
		}
		resOut = res
	}
	if service == "errors" {
		switch method {
		case "Non200":
			http.Error(w, "I'm a little teapot.", 418)
			return
		case "ShortResponse":
			w.Header().Set("Content-Length", "100")
			w.Write([]byte("way too short"))
			return
		case "OverQuota":
			writeResponse(&remotepb.Response{
				RpcError: &remotepb.RpcError{
					Code:   proto.Int32(int32(remotepb.RpcError_OVER_QUOTA)),
					Detail: proto.String("you are hogging the resources!"),
				},
			})
			return
		case "RunSlowly":
			// TestAPICallRPCFailure creates f.hang, but does not strobe it
			// until Call returns with remotepb.RpcError_CANCELLED.
			// This is here to force a happens-before relationship between
			// the httptest server handler and shutdown.
			<-f.hang
			resOut = &basepb.VoidProto{}
		}
	}
	if service == "logservice" && method == "Flush" {
		// Pretend log flushing is slow.
		time.Sleep(50 * time.Millisecond)
		atomic.AddInt32(&f.LogFlushes, 1)
		resOut = &basepb.VoidProto{}
	}

	encOut, err := proto.Marshal(resOut)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed encoding response: %v", err), 500)
		return
	}
	writeResponse(&remotepb.Response{
		Response: encOut,
	})
}

func setup() (f *fakeAPIHandler, c *context, cleanup func()) {
	f = &fakeAPIHandler{}
	srv := httptest.NewServer(f)
	u, err := url.Parse(srv.URL + apiPath)
	if err != nil {
		panic(fmt.Sprintf("url.Parse(%q): %v", srv.URL+apiPath, err))
	}
	return f, &context{
		req: &http.Request{
			Header: http.Header{
				ticketHeader: []string{"s3cr3t"},
				dapperHeader: []string{"trace-001"},
			},
		},
		apiURL: u,
	}, srv.Close
}

func TestAPICall(t *testing.T) {
	_, c, cleanup := setup()
	defer cleanup()

	req := &basepb.StringProto{
		Value: proto.String("Doctor Who"),
	}
	res := &basepb.StringProto{}
	err := Call(toContext(c), "actordb", "LookupActor", req, res)
	if err != nil {
		t.Fatalf("API call failed: %v", err)
	}
	if got, want := *res.Value, "David Tennant"; got != want {
		t.Errorf("Response is %q, want %q", got, want)
	}
}

func TestAPICallRPCFailure(t *testing.T) {
	f, c, cleanup := setup()
	defer cleanup()

	testCases := []struct {
		method string
		code   remotepb.RpcError_ErrorCode
	}{
		{"Non200", remotepb.RpcError_UNKNOWN},
		{"ShortResponse", remotepb.RpcError_UNKNOWN},
		{"OverQuota", remotepb.RpcError_OVER_QUOTA},
		{"RunSlowly", remotepb.RpcError_CANCELLED},
	}
	f.hang = make(chan int) // only for RunSlowly
	for _, tc := range testCases {
		ctx, _ := netcontext.WithTimeout(toContext(c), 100*time.Millisecond)
		err := Call(ctx, "errors", tc.method, &basepb.VoidProto{}, &basepb.VoidProto{})
		ce, ok := err.(*CallError)
		if !ok {
			t.Errorf("%s: API call error is %T (%v), want *CallError", tc.method, err, err)
			continue
		}
		if ce.Code != int32(tc.code) {
			t.Errorf("%s: ce.Code = %d, want %d", tc.method, ce.Code, tc.code)
		}
		if tc.method == "RunSlowly" {
			f.hang <- 1 // release the HTTP handler
		}
	}
}

func TestAPICallDialFailure(t *testing.T) {
	// See what happens if the API host is unresponsive.
	// This should time out quickly, not hang forever.
	_, c, cleanup := setup()
	defer cleanup()
	// Reset the URL to the production address so that dialing fails.
	c.apiURL = apiURL()

	start := time.Now()
	err := Call(toContext(c), "foo", "bar", &basepb.VoidProto{}, &basepb.VoidProto{})
	const max = 1 * time.Second
	if taken := time.Since(start); taken > max {
		t.Errorf("Dial hang took too long: %v > %v", taken, max)
	}
	if err == nil {
		t.Error("Call did not fail")
	}
}

func TestDelayedLogFlushing(t *testing.T) {
	f, c, cleanup := setup()
	defer cleanup()

	http.HandleFunc("/quick_log", func(w http.ResponseWriter, r *http.Request) {
		logC := WithContext(netcontext.Background(), r)
		fromContext(logC).apiURL = c.apiURL // Otherwise it will try to use the default URL.
		Logf(logC, 1, "It's a lovely day.")
		w.WriteHeader(200)
		w.Write(make([]byte, 100<<10)) // write 100 KB to force HTTP flush
	})

	r := &http.Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Path:   "/quick_log",
		},
		Header: c.req.Header,
		Body:   ioutil.NopCloser(bytes.NewReader(nil)),
	}
	w := httptest.NewRecorder()

	// Check that log flushing does not hold up the HTTP response.
	start := time.Now()
	handleHTTP(w, r)
	if d := time.Since(start); d > 10*time.Millisecond {
		t.Errorf("handleHTTP took %v, want under 10ms", d)
	}
	const hdr = "X-AppEngine-Log-Flush-Count"
	if h := w.HeaderMap.Get(hdr); h != "1" {
		t.Errorf("%s header = %q, want %q", hdr, h, "1")
	}
	if f := atomic.LoadInt32(&f.LogFlushes); f != 0 {
		t.Errorf("After HTTP response: f.LogFlushes = %d, want 0", f)
	}

	// Check that the log flush eventually comes in.
	time.Sleep(100 * time.Millisecond)
	if f := atomic.LoadInt32(&f.LogFlushes); f != 1 {
		t.Errorf("After 100ms: f.LogFlushes = %d, want 1", f)
	}
}

func TestRemoteAddr(t *testing.T) {
	var addr string
	http.HandleFunc("/remote_addr", func(w http.ResponseWriter, r *http.Request) {
		addr = r.RemoteAddr
	})

	testCases := []struct {
		headers http.Header
		addr    string
	}{
		{http.Header{"X-Appengine-User-Ip": []string{"10.5.2.1"}}, "10.5.2.1:80"},
		{http.Header{"X-Appengine-Remote-Addr": []string{"1.2.3.4"}}, "1.2.3.4:80"},
		{http.Header{"X-Appengine-Remote-Addr": []string{"1.2.3.4:8080"}}, "1.2.3.4:8080"},
		{
			http.Header{"X-Appengine-Remote-Addr": []string{"2401:fa00:9:1:7646:a0ff:fe90:ca66"}},
			"[2401:fa00:9:1:7646:a0ff:fe90:ca66]:80",
		},
		{
			http.Header{"X-Appengine-Remote-Addr": []string{"[::1]:http"}},
			"[::1]:http",
		},
		{http.Header{}, "127.0.0.1:80"},
	}

	for _, tc := range testCases {
		r := &http.Request{
			Method: "GET",
			URL:    &url.URL{Scheme: "http", Path: "/remote_addr"},
			Header: tc.headers,
			Body:   ioutil.NopCloser(bytes.NewReader(nil)),
		}
		handleHTTP(httptest.NewRecorder(), r)
		if addr != tc.addr {
			t.Errorf("Header %v, got %q, want %q", tc.headers, addr, tc.addr)
		}
	}
}

func TestPanickingHandler(t *testing.T) {
	http.HandleFunc("/panic", func(http.ResponseWriter, *http.Request) {
		panic("whoops!")
	})
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Scheme: "http", Path: "/panic"},
		Body:   ioutil.NopCloser(bytes.NewReader(nil)),
	}
	rec := httptest.NewRecorder()
	handleHTTP(rec, r)
	if rec.Code != 500 {
		t.Errorf("Panicking handler returned HTTP %d, want HTTP %d", rec.Code, 500)
	}
}

var raceDetector = false

func TestAPICallAllocations(t *testing.T) {
	if raceDetector {
		t.Skip("not running under race detector")
	}

	// Run the test API server in a subprocess so we aren't counting its allocations.
	u, cleanup := launchHelperProcess(t)
	defer cleanup()
	c := &context{
		req: &http.Request{
			Header: http.Header{
				ticketHeader: []string{"s3cr3t"},
				dapperHeader: []string{"trace-001"},
			},
		},
		apiURL: u,
	}

	req := &basepb.StringProto{
		Value: proto.String("Doctor Who"),
	}
	res := &basepb.StringProto{}
	var apiErr error
	avg := testing.AllocsPerRun(100, func() {
		ctx, _ := netcontext.WithTimeout(toContext(c), 100*time.Millisecond)
		if err := Call(ctx, "actordb", "LookupActor", req, res); err != nil && apiErr == nil {
			apiErr = err // get the first error only
		}
	})
	if apiErr != nil {
		t.Errorf("API call failed: %v", apiErr)
	}

	// Lots of room for improvement...
	// TODO(djd): Reduce maximum to 85 once the App Engine SDK is based on 1.6.
	const min, max float64 = 70, 90
	if avg < min || max < avg {
		t.Errorf("Allocations per API call = %g, want in [%g,%g]", avg, min, max)
	}
}

func launchHelperProcess(t *testing.T) (apiURL *url.URL, cleanup func()) {
	cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcess")
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Starting helper process: %v", err)
	}

	scan := bufio.NewScanner(stdout)
	var u *url.URL
	for scan.Scan() {
		line := scan.Text()
		if hp := strings.TrimPrefix(line, helperProcessMagic); hp != line {
			var err error
			u, err = url.Parse(hp)
			if err != nil {
				t.Fatalf("Failed to parse %q: %v", hp, err)
			}
			break
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatalf("Scanning helper process stdout: %v", err)
	}
	if u == nil {
		t.Fatal("Helper process never reported")
	}

	return u, func() {
		stdin.Close()
		if err := cmd.Wait(); err != nil {
			t.Errorf("Helper process did not exit cleanly: %v", err)
		}
	}
}

const helperProcessMagic = "A lovely helper process is listening at "

// This isn't a real test. It's used as a helper process.
func TestHelperProcess(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	f := &fakeAPIHandler{}
	srv := httptest.NewServer(f)
	defer srv.Close()
	fmt.Println(helperProcessMagic + srv.URL + apiPath)

	// Wait for stdin to be closed.
	io.Copy(ioutil.Discard, os.Stdin)
}

func TestBackgroundContext(t *testing.T) {
	environ := []struct {
		key, value string
	}{
		{"GAE_LONG_APP_ID", "my-app-id"},
		{"GAE_MINOR_VERSION", "067924799508853122"},
		{"GAE_MODULE_INSTANCE", "0"},
		{"GAE_MODULE_NAME", "default"},
		{"GAE_MODULE_VERSION", "20150612t184001"},
	}
	for _, v := range environ {
		old := os.Getenv(v.key)
		os.Setenv(v.key, v.value)
		v.value = old
	}
	defer func() { // Restore old environment after the test completes.
		for _, v := range environ {
			if v.value == "" {
				os.Unsetenv(v.key)
				continue
			}
			os.Setenv(v.key, v.value)
		}
	}()

	ctx, key := fromContext(BackgroundContext()), "X-Magic-Ticket-Header"
	if g, w := ctx.req.Header.Get(key), "my-app-id/default.20150612t184001.0"; g != w {
		t.Errorf("%v = %q, want %q", key, g, w)
	}

	// Check that using the background context doesn't panic.
	req := &basepb.StringProto{
		Value: proto.String("Doctor Who"),
	}
	res := &basepb.StringProto{}
	Call(BackgroundContext(), "actordb", "LookupActor", req, res) // expected to fail
}
