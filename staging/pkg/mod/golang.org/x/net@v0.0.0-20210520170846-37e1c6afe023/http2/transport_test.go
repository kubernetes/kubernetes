// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/textproto"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/http2/hpack"
)

var (
	extNet        = flag.Bool("extnet", false, "do external network tests")
	transportHost = flag.String("transporthost", "http2.golang.org", "hostname to use for TestTransport")
	insecure      = flag.Bool("insecure", false, "insecure TLS dials") // TODO: dead code. remove?
)

var tlsConfigInsecure = &tls.Config{InsecureSkipVerify: true}

var canceledCtx context.Context

func init() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	canceledCtx = ctx
}

func TestTransportExternal(t *testing.T) {
	if !*extNet {
		t.Skip("skipping external network test")
	}
	req, _ := http.NewRequest("GET", "https://"+*transportHost+"/", nil)
	rt := &Transport{TLSClientConfig: tlsConfigInsecure}
	res, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("%v", err)
	}
	res.Write(os.Stdout)
}

type fakeTLSConn struct {
	net.Conn
}

func (c *fakeTLSConn) ConnectionState() tls.ConnectionState {
	return tls.ConnectionState{
		Version:     tls.VersionTLS12,
		CipherSuite: cipher_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
	}
}

func startH2cServer(t *testing.T) net.Listener {
	h2Server := &Server{}
	l := newLocalListener(t)
	go func() {
		conn, err := l.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		h2Server.ServeConn(&fakeTLSConn{conn}, &ServeConnOpts{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "Hello, %v, http: %v", r.URL.Path, r.TLS == nil)
		})})
	}()
	return l
}

func TestTransportH2c(t *testing.T) {
	l := startH2cServer(t)
	defer l.Close()
	req, err := http.NewRequest("GET", "http://"+l.Addr().String()+"/foobar", nil)
	if err != nil {
		t.Fatal(err)
	}
	var gotConnCnt int32
	trace := &httptrace.ClientTrace{
		GotConn: func(connInfo httptrace.GotConnInfo) {
			if !connInfo.Reused {
				atomic.AddInt32(&gotConnCnt, 1)
			}
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	tr := &Transport{
		AllowHTTP: true,
		DialTLS: func(network, addr string, cfg *tls.Config) (net.Conn, error) {
			return net.Dial(network, addr)
		},
	}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.ProtoMajor != 2 {
		t.Fatal("proto not h2c")
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(body), "Hello, /foobar, http: true"; got != want {
		t.Fatalf("response got %v, want %v", got, want)
	}
	if got, want := gotConnCnt, int32(1); got != want {
		t.Errorf("Too many got connections: %d", gotConnCnt)
	}
}

func TestTransport(t *testing.T) {
	const body = "sup"
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, body)
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	u, err := url.Parse(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	for i, m := range []string{"GET", ""} {
		req := &http.Request{
			Method: m,
			URL:    u,
		}
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatalf("%d: %s", i, err)
		}

		t.Logf("%d: Got res: %+v", i, res)
		if g, w := res.StatusCode, 200; g != w {
			t.Errorf("%d: StatusCode = %v; want %v", i, g, w)
		}
		if g, w := res.Status, "200 OK"; g != w {
			t.Errorf("%d: Status = %q; want %q", i, g, w)
		}
		wantHeader := http.Header{
			"Content-Length": []string{"3"},
			"Content-Type":   []string{"text/plain; charset=utf-8"},
			"Date":           []string{"XXX"}, // see cleanDate
		}
		cleanDate(res)
		if !reflect.DeepEqual(res.Header, wantHeader) {
			t.Errorf("%d: res Header = %v; want %v", i, res.Header, wantHeader)
		}
		if res.Request != req {
			t.Errorf("%d: Response.Request = %p; want %p", i, res.Request, req)
		}
		if res.TLS == nil {
			t.Errorf("%d: Response.TLS = nil; want non-nil", i)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Errorf("%d: Body read: %v", i, err)
		} else if string(slurp) != body {
			t.Errorf("%d: Body = %q; want %q", i, slurp, body)
		}
		res.Body.Close()
	}
}

func onSameConn(t *testing.T, modReq func(*http.Request)) bool {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.RemoteAddr)
	}, optOnlyServer, func(c net.Conn, st http.ConnState) {
		t.Logf("conn %v is now state %v", c.RemoteAddr(), st)
	})
	defer st.Close()
	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	get := func() string {
		req, err := http.NewRequest("GET", st.ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		modReq(req)
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("Body read: %v", err)
		}
		addr := strings.TrimSpace(string(slurp))
		if addr == "" {
			t.Fatalf("didn't get an addr in response")
		}
		return addr
	}
	first := get()
	second := get()
	return first == second
}

func TestTransportReusesConns(t *testing.T) {
	if !onSameConn(t, func(*http.Request) {}) {
		t.Errorf("first and second responses were on different connections")
	}
}

func TestTransportReusesConn_RequestClose(t *testing.T) {
	if onSameConn(t, func(r *http.Request) { r.Close = true }) {
		t.Errorf("first and second responses were not on different connections")
	}
}

func TestTransportReusesConn_ConnClose(t *testing.T) {
	if onSameConn(t, func(r *http.Request) { r.Header.Set("Connection", "close") }) {
		t.Errorf("first and second responses were not on different connections")
	}
}

// Tests that the Transport only keeps one pending dial open per destination address.
// https://golang.org/issue/13397
func TestTransportGroupsPendingDials(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.RemoteAddr)
	}, optOnlyServer)
	defer st.Close()
	tr := &Transport{
		TLSClientConfig: tlsConfigInsecure,
	}
	defer tr.CloseIdleConnections()
	var (
		mu    sync.Mutex
		dials = map[string]int{}
	)
	var gotConnCnt int32
	trace := &httptrace.ClientTrace{
		GotConn: func(connInfo httptrace.GotConnInfo) {
			if !connInfo.Reused {
				atomic.AddInt32(&gotConnCnt, 1)
			}
		},
	}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req, err := http.NewRequest("GET", st.ts.URL, nil)
			if err != nil {
				t.Error(err)
				return
			}
			req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
			res, err := tr.RoundTrip(req)
			if err != nil {
				t.Error(err)
				return
			}
			defer res.Body.Close()
			slurp, err := ioutil.ReadAll(res.Body)
			if err != nil {
				t.Errorf("Body read: %v", err)
			}
			addr := strings.TrimSpace(string(slurp))
			if addr == "" {
				t.Errorf("didn't get an addr in response")
			}
			mu.Lock()
			dials[addr]++
			mu.Unlock()
		}()
	}
	wg.Wait()
	if len(dials) != 1 {
		t.Errorf("saw %d dials; want 1: %v", len(dials), dials)
	}
	tr.CloseIdleConnections()
	if err := retry(50, 10*time.Millisecond, func() error {
		cp, ok := tr.connPool().(*clientConnPool)
		if !ok {
			return fmt.Errorf("Conn pool is %T; want *clientConnPool", tr.connPool())
		}
		cp.mu.Lock()
		defer cp.mu.Unlock()
		if len(cp.dialing) != 0 {
			return fmt.Errorf("dialing map = %v; want empty", cp.dialing)
		}
		if len(cp.conns) != 0 {
			return fmt.Errorf("conns = %v; want empty", cp.conns)
		}
		if len(cp.keys) != 0 {
			return fmt.Errorf("keys = %v; want empty", cp.keys)
		}
		return nil
	}); err != nil {
		t.Errorf("State of pool after CloseIdleConnections: %v", err)
	}
	if got, want := gotConnCnt, int32(1); got != want {
		t.Errorf("Too many got connections: %d", gotConnCnt)
	}
}

func retry(tries int, delay time.Duration, fn func() error) error {
	var err error
	for i := 0; i < tries; i++ {
		err = fn()
		if err == nil {
			return nil
		}
		time.Sleep(delay)
	}
	return err
}

func TestTransportAbortClosesPipes(t *testing.T) {
	shutdown := make(chan struct{})
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			w.(http.Flusher).Flush()
			<-shutdown
		},
		optOnlyServer,
	)
	defer st.Close()
	defer close(shutdown) // we must shutdown before st.Close() to avoid hanging

	errCh := make(chan error)
	go func() {
		defer close(errCh)
		tr := &Transport{TLSClientConfig: tlsConfigInsecure}
		req, err := http.NewRequest("GET", st.ts.URL, nil)
		if err != nil {
			errCh <- err
			return
		}
		res, err := tr.RoundTrip(req)
		if err != nil {
			errCh <- err
			return
		}
		defer res.Body.Close()
		st.closeConn()
		_, err = ioutil.ReadAll(res.Body)
		if err == nil {
			errCh <- errors.New("expected error from res.Body.Read")
			return
		}
	}()

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatal(err)
		}
	// deadlock? that's a bug.
	case <-time.After(3 * time.Second):
		t.Fatal("timeout")
	}
}

// TODO: merge this with TestTransportBody to make TestTransportRequest? This
// could be a table-driven test with extra goodies.
func TestTransportPath(t *testing.T) {
	gotc := make(chan *url.URL, 1)
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			gotc <- r.URL
		},
		optOnlyServer,
	)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	const (
		path  = "/testpath"
		query = "q=1"
	)
	surl := st.ts.URL + path + "?" + query
	req, err := http.NewRequest("POST", surl, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := &http.Client{Transport: tr}
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	got := <-gotc
	if got.Path != path {
		t.Errorf("Read Path = %q; want %q", got.Path, path)
	}
	if got.RawQuery != query {
		t.Errorf("Read RawQuery = %q; want %q", got.RawQuery, query)
	}
}

func randString(n int) string {
	rnd := rand.New(rand.NewSource(int64(n)))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rnd.Intn(256))
	}
	return string(b)
}

type panicReader struct{}

func (panicReader) Read([]byte) (int, error) { panic("unexpected Read") }
func (panicReader) Close() error             { panic("unexpected Close") }

func TestActualContentLength(t *testing.T) {
	tests := []struct {
		req  *http.Request
		want int64
	}{
		// Verify we don't read from Body:
		0: {
			req:  &http.Request{Body: panicReader{}},
			want: -1,
		},
		// nil Body means 0, regardless of ContentLength:
		1: {
			req:  &http.Request{Body: nil, ContentLength: 5},
			want: 0,
		},
		// ContentLength is used if set.
		2: {
			req:  &http.Request{Body: panicReader{}, ContentLength: 5},
			want: 5,
		},
		// http.NoBody means 0, not -1.
		3: {
			req:  &http.Request{Body: http.NoBody},
			want: 0,
		},
	}
	for i, tt := range tests {
		got := actualContentLength(tt.req)
		if got != tt.want {
			t.Errorf("test[%d]: got %d; want %d", i, got, tt.want)
		}
	}
}

func TestTransportBody(t *testing.T) {
	bodyTests := []struct {
		body         string
		noContentLen bool
	}{
		{body: "some message"},
		{body: "some message", noContentLen: true},
		{body: strings.Repeat("a", 1<<20), noContentLen: true},
		{body: strings.Repeat("a", 1<<20)},
		{body: randString(16<<10 - 1)},
		{body: randString(16 << 10)},
		{body: randString(16<<10 + 1)},
		{body: randString(512<<10 - 1)},
		{body: randString(512 << 10)},
		{body: randString(512<<10 + 1)},
		{body: randString(1<<20 - 1)},
		{body: randString(1 << 20)},
		{body: randString(1<<20 + 2)},
	}

	type reqInfo struct {
		req   *http.Request
		slurp []byte
		err   error
	}
	gotc := make(chan reqInfo, 1)
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			slurp, err := ioutil.ReadAll(r.Body)
			if err != nil {
				gotc <- reqInfo{err: err}
			} else {
				gotc <- reqInfo{req: r, slurp: slurp}
			}
		},
		optOnlyServer,
	)
	defer st.Close()

	for i, tt := range bodyTests {
		tr := &Transport{TLSClientConfig: tlsConfigInsecure}
		defer tr.CloseIdleConnections()

		var body io.Reader = strings.NewReader(tt.body)
		if tt.noContentLen {
			body = struct{ io.Reader }{body} // just a Reader, hiding concrete type and other methods
		}
		req, err := http.NewRequest("POST", st.ts.URL, body)
		if err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		c := &http.Client{Transport: tr}
		res, err := c.Do(req)
		if err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		defer res.Body.Close()
		ri := <-gotc
		if ri.err != nil {
			t.Errorf("#%d: read error: %v", i, ri.err)
			continue
		}
		if got := string(ri.slurp); got != tt.body {
			t.Errorf("#%d: Read body mismatch.\n got: %q (len %d)\nwant: %q (len %d)", i, shortString(got), len(got), shortString(tt.body), len(tt.body))
		}
		wantLen := int64(len(tt.body))
		if tt.noContentLen && tt.body != "" {
			wantLen = -1
		}
		if ri.req.ContentLength != wantLen {
			t.Errorf("#%d. handler got ContentLength = %v; want %v", i, ri.req.ContentLength, wantLen)
		}
	}
}

func shortString(v string) string {
	const maxLen = 100
	if len(v) <= maxLen {
		return v
	}
	return fmt.Sprintf("%v[...%d bytes omitted...]%v", v[:maxLen/2], len(v)-maxLen, v[len(v)-maxLen/2:])
}

func TestTransportDialTLS(t *testing.T) {
	var mu sync.Mutex // guards following
	var gotReq, didDial bool

	ts := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			gotReq = true
			mu.Unlock()
		},
		optOnlyServer,
	)
	defer ts.Close()
	tr := &Transport{
		DialTLS: func(netw, addr string, cfg *tls.Config) (net.Conn, error) {
			mu.Lock()
			didDial = true
			mu.Unlock()
			cfg.InsecureSkipVerify = true
			c, err := tls.Dial(netw, addr, cfg)
			if err != nil {
				return nil, err
			}
			return c, c.Handshake()
		},
	}
	defer tr.CloseIdleConnections()
	client := &http.Client{Transport: tr}
	res, err := client.Get(ts.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	mu.Lock()
	if !gotReq {
		t.Error("didn't get request")
	}
	if !didDial {
		t.Error("didn't use dial hook")
	}
}

func TestConfigureTransport(t *testing.T) {
	t1 := &http.Transport{}
	err := ConfigureTransport(t1)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%#v", t1); !strings.Contains(got, `"h2"`) {
		// Laziness, to avoid buildtags.
		t.Errorf("stringification of HTTP/1 transport didn't contain \"h2\": %v", got)
	}
	wantNextProtos := []string{"h2", "http/1.1"}
	if t1.TLSClientConfig == nil {
		t.Errorf("nil t1.TLSClientConfig")
	} else if !reflect.DeepEqual(t1.TLSClientConfig.NextProtos, wantNextProtos) {
		t.Errorf("TLSClientConfig.NextProtos = %q; want %q", t1.TLSClientConfig.NextProtos, wantNextProtos)
	}
	if err := ConfigureTransport(t1); err == nil {
		t.Error("unexpected success on second call to ConfigureTransport")
	}

	// And does it work?
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.Proto)
	}, optOnlyServer)
	defer st.Close()

	t1.TLSClientConfig.InsecureSkipVerify = true
	c := &http.Client{Transport: t1}
	res, err := c.Get(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(slurp), "HTTP/2.0"; got != want {
		t.Errorf("body = %q; want %q", got, want)
	}
}

type capitalizeReader struct {
	r io.Reader
}

func (cr capitalizeReader) Read(p []byte) (n int, err error) {
	n, err = cr.r.Read(p)
	for i, b := range p[:n] {
		if b >= 'a' && b <= 'z' {
			p[i] = b - ('a' - 'A')
		}
	}
	return
}

type flushWriter struct {
	w io.Writer
}

func (fw flushWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if f, ok := fw.w.(http.Flusher); ok {
		f.Flush()
	}
	return
}

type clientTester struct {
	t      *testing.T
	tr     *Transport
	sc, cc net.Conn // server and client conn
	fr     *Framer  // server's framer
	client func() error
	server func() error
}

func newClientTester(t *testing.T) *clientTester {
	var dialOnce struct {
		sync.Mutex
		dialed bool
	}
	ct := &clientTester{
		t: t,
	}
	ct.tr = &Transport{
		TLSClientConfig: tlsConfigInsecure,
		DialTLS: func(network, addr string, cfg *tls.Config) (net.Conn, error) {
			dialOnce.Lock()
			defer dialOnce.Unlock()
			if dialOnce.dialed {
				return nil, errors.New("only one dial allowed in test mode")
			}
			dialOnce.dialed = true
			return ct.cc, nil
		},
	}

	ln := newLocalListener(t)
	cc, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)

	}
	sc, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
	ct.cc = cc
	ct.sc = sc
	ct.fr = NewFramer(sc, sc)
	return ct
}

func newLocalListener(t *testing.T) net.Listener {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err == nil {
		return ln
	}
	ln, err = net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func (ct *clientTester) greet(settings ...Setting) {
	buf := make([]byte, len(ClientPreface))
	_, err := io.ReadFull(ct.sc, buf)
	if err != nil {
		ct.t.Fatalf("reading client preface: %v", err)
	}
	f, err := ct.fr.ReadFrame()
	if err != nil {
		ct.t.Fatalf("Reading client settings frame: %v", err)
	}
	if sf, ok := f.(*SettingsFrame); !ok {
		ct.t.Fatalf("Wanted client settings frame; got %v", f)
		_ = sf // stash it away?
	}
	if err := ct.fr.WriteSettings(settings...); err != nil {
		ct.t.Fatal(err)
	}
	if err := ct.fr.WriteSettingsAck(); err != nil {
		ct.t.Fatal(err)
	}
}

func (ct *clientTester) readNonSettingsFrame() (Frame, error) {
	for {
		f, err := ct.fr.ReadFrame()
		if err != nil {
			return nil, err
		}
		if _, ok := f.(*SettingsFrame); ok {
			continue
		}
		return f, nil
	}
}

func (ct *clientTester) cleanup() {
	ct.tr.CloseIdleConnections()

	// close both connections, ignore the error if its already closed
	ct.sc.Close()
	ct.cc.Close()
}

func (ct *clientTester) run() {
	var errOnce sync.Once
	var wg sync.WaitGroup

	run := func(which string, fn func() error) {
		defer wg.Done()
		if err := fn(); err != nil {
			errOnce.Do(func() {
				ct.t.Errorf("%s: %v", which, err)
				ct.cleanup()
			})
		}
	}

	wg.Add(2)
	go run("client", ct.client)
	go run("server", ct.server)
	wg.Wait()

	errOnce.Do(ct.cleanup) // clean up if no error
}

func (ct *clientTester) readFrame() (Frame, error) {
	return readFrameTimeout(ct.fr, 2*time.Second)
}

func (ct *clientTester) firstHeaders() (*HeadersFrame, error) {
	for {
		f, err := ct.readFrame()
		if err != nil {
			return nil, fmt.Errorf("ReadFrame while waiting for Headers: %v", err)
		}
		switch f.(type) {
		case *WindowUpdateFrame, *SettingsFrame:
			continue
		}
		hf, ok := f.(*HeadersFrame)
		if !ok {
			return nil, fmt.Errorf("Got %T; want HeadersFrame", f)
		}
		return hf, nil
	}
}

type countingReader struct {
	n *int64
}

func (r countingReader) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(i)
	}
	atomic.AddInt64(r.n, int64(len(p)))
	return len(p), err
}

func TestTransportReqBodyAfterResponse_200(t *testing.T) { testTransportReqBodyAfterResponse(t, 200) }
func TestTransportReqBodyAfterResponse_403(t *testing.T) { testTransportReqBodyAfterResponse(t, 403) }

func testTransportReqBodyAfterResponse(t *testing.T, status int) {
	const bodySize = 10 << 20
	clientDone := make(chan struct{})
	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		defer close(clientDone)

		var n int64 // atomic
		req, err := http.NewRequest("PUT", "https://dummy.tld/", io.LimitReader(countingReader{&n}, bodySize))
		if err != nil {
			return err
		}
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != status {
			return fmt.Errorf("status code = %v; want %v", res.StatusCode, status)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("Slurp: %v", err)
		}
		if len(slurp) > 0 {
			return fmt.Errorf("unexpected body: %q", slurp)
		}
		if status == 200 {
			if got := atomic.LoadInt64(&n); got != bodySize {
				return fmt.Errorf("For 200 response, Transport wrote %d bytes; want %d", got, bodySize)
			}
		} else {
			if got := atomic.LoadInt64(&n); got == 0 || got >= bodySize {
				return fmt.Errorf("For %d response, Transport wrote %d bytes; want (0,%d) exclusive", status, got, bodySize)
			}
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		var dataRecv int64
		var closed bool
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-clientDone:
					// If the client's done, it
					// will have reported any
					// errors on its side.
					return nil
				default:
					return err
				}
			}
			//println(fmt.Sprintf("server got frame: %v", f))
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				if !f.HeadersEnded() {
					return fmt.Errorf("headers should have END_HEADERS be ended: %v", f)
				}
				if f.StreamEnded() {
					return fmt.Errorf("headers contains END_STREAM unexpectedly: %v", f)
				}
			case *DataFrame:
				dataLen := len(f.Data())
				if dataLen > 0 {
					if dataRecv == 0 {
						enc.WriteField(hpack.HeaderField{Name: ":status", Value: strconv.Itoa(status)})
						ct.fr.WriteHeaders(HeadersFrameParam{
							StreamID:      f.StreamID,
							EndHeaders:    true,
							EndStream:     false,
							BlockFragment: buf.Bytes(),
						})
					}
					if err := ct.fr.WriteWindowUpdate(0, uint32(dataLen)); err != nil {
						return err
					}
					if err := ct.fr.WriteWindowUpdate(f.StreamID, uint32(dataLen)); err != nil {
						return err
					}
				}
				dataRecv += int64(dataLen)

				if !closed && ((status != 200 && dataRecv > 0) ||
					(status == 200 && dataRecv == bodySize)) {
					closed = true
					if err := ct.fr.WriteData(f.StreamID, true, nil); err != nil {
						return err
					}
				}
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

// See golang.org/issue/13444
func TestTransportFullDuplex(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200) // redundant but for clarity
		w.(http.Flusher).Flush()
		io.Copy(flushWriter{w}, capitalizeReader{r.Body})
		fmt.Fprintf(w, "bye.\n")
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}

	pr, pw := io.Pipe()
	req, err := http.NewRequest("PUT", st.ts.URL, ioutil.NopCloser(pr))
	if err != nil {
		t.Fatal(err)
	}
	req.ContentLength = -1
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("StatusCode = %v; want %v", res.StatusCode, 200)
	}
	bs := bufio.NewScanner(res.Body)
	want := func(v string) {
		if !bs.Scan() {
			t.Fatalf("wanted to read %q but Scan() = false, err = %v", v, bs.Err())
		}
	}
	write := func(v string) {
		_, err := io.WriteString(pw, v)
		if err != nil {
			t.Fatalf("pipe write: %v", err)
		}
	}
	write("foo\n")
	want("FOO")
	write("bar\n")
	want("BAR")
	pw.Close()
	want("bye.")
	if err := bs.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestTransportConnectRequest(t *testing.T) {
	gotc := make(chan *http.Request, 1)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		gotc <- r
	}, optOnlyServer)
	defer st.Close()

	u, err := url.Parse(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}

	tests := []struct {
		req  *http.Request
		want string
	}{
		{
			req: &http.Request{
				Method: "CONNECT",
				Header: http.Header{},
				URL:    u,
			},
			want: u.Host,
		},
		{
			req: &http.Request{
				Method: "CONNECT",
				Header: http.Header{},
				URL:    u,
				Host:   "example.com:123",
			},
			want: "example.com:123",
		},
	}

	for i, tt := range tests {
		res, err := c.Do(tt.req)
		if err != nil {
			t.Errorf("%d. RoundTrip = %v", i, err)
			continue
		}
		res.Body.Close()
		req := <-gotc
		if req.Method != "CONNECT" {
			t.Errorf("method = %q; want CONNECT", req.Method)
		}
		if req.Host != tt.want {
			t.Errorf("Host = %q; want %q", req.Host, tt.want)
		}
		if req.URL.Host != tt.want {
			t.Errorf("URL.Host = %q; want %q", req.URL.Host, tt.want)
		}
	}
}

type headerType int

const (
	noHeader headerType = iota // omitted
	oneHeader
	splitHeader // broken into continuation on purpose
)

const (
	f0 = noHeader
	f1 = oneHeader
	f2 = splitHeader
	d0 = false
	d1 = true
)

// Test all 36 combinations of response frame orders:
//    (3 ways of 100-continue) * (2 ways of headers) * (2 ways of data) * (3 ways of trailers):func TestTransportResponsePattern_00f0(t *testing.T) { testTransportResponsePattern(h0, h1, false, h0) }
// Generated by http://play.golang.org/p/SScqYKJYXd
func TestTransportResPattern_c0h1d0t0(t *testing.T) { testTransportResPattern(t, f0, f1, d0, f0) }
func TestTransportResPattern_c0h1d0t1(t *testing.T) { testTransportResPattern(t, f0, f1, d0, f1) }
func TestTransportResPattern_c0h1d0t2(t *testing.T) { testTransportResPattern(t, f0, f1, d0, f2) }
func TestTransportResPattern_c0h1d1t0(t *testing.T) { testTransportResPattern(t, f0, f1, d1, f0) }
func TestTransportResPattern_c0h1d1t1(t *testing.T) { testTransportResPattern(t, f0, f1, d1, f1) }
func TestTransportResPattern_c0h1d1t2(t *testing.T) { testTransportResPattern(t, f0, f1, d1, f2) }
func TestTransportResPattern_c0h2d0t0(t *testing.T) { testTransportResPattern(t, f0, f2, d0, f0) }
func TestTransportResPattern_c0h2d0t1(t *testing.T) { testTransportResPattern(t, f0, f2, d0, f1) }
func TestTransportResPattern_c0h2d0t2(t *testing.T) { testTransportResPattern(t, f0, f2, d0, f2) }
func TestTransportResPattern_c0h2d1t0(t *testing.T) { testTransportResPattern(t, f0, f2, d1, f0) }
func TestTransportResPattern_c0h2d1t1(t *testing.T) { testTransportResPattern(t, f0, f2, d1, f1) }
func TestTransportResPattern_c0h2d1t2(t *testing.T) { testTransportResPattern(t, f0, f2, d1, f2) }
func TestTransportResPattern_c1h1d0t0(t *testing.T) { testTransportResPattern(t, f1, f1, d0, f0) }
func TestTransportResPattern_c1h1d0t1(t *testing.T) { testTransportResPattern(t, f1, f1, d0, f1) }
func TestTransportResPattern_c1h1d0t2(t *testing.T) { testTransportResPattern(t, f1, f1, d0, f2) }
func TestTransportResPattern_c1h1d1t0(t *testing.T) { testTransportResPattern(t, f1, f1, d1, f0) }
func TestTransportResPattern_c1h1d1t1(t *testing.T) { testTransportResPattern(t, f1, f1, d1, f1) }
func TestTransportResPattern_c1h1d1t2(t *testing.T) { testTransportResPattern(t, f1, f1, d1, f2) }
func TestTransportResPattern_c1h2d0t0(t *testing.T) { testTransportResPattern(t, f1, f2, d0, f0) }
func TestTransportResPattern_c1h2d0t1(t *testing.T) { testTransportResPattern(t, f1, f2, d0, f1) }
func TestTransportResPattern_c1h2d0t2(t *testing.T) { testTransportResPattern(t, f1, f2, d0, f2) }
func TestTransportResPattern_c1h2d1t0(t *testing.T) { testTransportResPattern(t, f1, f2, d1, f0) }
func TestTransportResPattern_c1h2d1t1(t *testing.T) { testTransportResPattern(t, f1, f2, d1, f1) }
func TestTransportResPattern_c1h2d1t2(t *testing.T) { testTransportResPattern(t, f1, f2, d1, f2) }
func TestTransportResPattern_c2h1d0t0(t *testing.T) { testTransportResPattern(t, f2, f1, d0, f0) }
func TestTransportResPattern_c2h1d0t1(t *testing.T) { testTransportResPattern(t, f2, f1, d0, f1) }
func TestTransportResPattern_c2h1d0t2(t *testing.T) { testTransportResPattern(t, f2, f1, d0, f2) }
func TestTransportResPattern_c2h1d1t0(t *testing.T) { testTransportResPattern(t, f2, f1, d1, f0) }
func TestTransportResPattern_c2h1d1t1(t *testing.T) { testTransportResPattern(t, f2, f1, d1, f1) }
func TestTransportResPattern_c2h1d1t2(t *testing.T) { testTransportResPattern(t, f2, f1, d1, f2) }
func TestTransportResPattern_c2h2d0t0(t *testing.T) { testTransportResPattern(t, f2, f2, d0, f0) }
func TestTransportResPattern_c2h2d0t1(t *testing.T) { testTransportResPattern(t, f2, f2, d0, f1) }
func TestTransportResPattern_c2h2d0t2(t *testing.T) { testTransportResPattern(t, f2, f2, d0, f2) }
func TestTransportResPattern_c2h2d1t0(t *testing.T) { testTransportResPattern(t, f2, f2, d1, f0) }
func TestTransportResPattern_c2h2d1t1(t *testing.T) { testTransportResPattern(t, f2, f2, d1, f1) }
func TestTransportResPattern_c2h2d1t2(t *testing.T) { testTransportResPattern(t, f2, f2, d1, f2) }

func testTransportResPattern(t *testing.T, expect100Continue, resHeader headerType, withData bool, trailers headerType) {
	const reqBody = "some request body"
	const resBody = "some response body"

	if resHeader == noHeader {
		// TODO: test 100-continue followed by immediate
		// server stream reset, without headers in the middle?
		panic("invalid combination")
	}

	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("POST", "https://dummy.tld/", strings.NewReader(reqBody))
		if expect100Continue != noHeader {
			req.Header.Set("Expect", "100-continue")
		}
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("status code = %v; want 200", res.StatusCode)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("Slurp: %v", err)
		}
		wantBody := resBody
		if !withData {
			wantBody = ""
		}
		if string(slurp) != wantBody {
			return fmt.Errorf("body = %q; want %q", slurp, wantBody)
		}
		if trailers == noHeader {
			if len(res.Trailer) > 0 {
				t.Errorf("Trailer = %v; want none", res.Trailer)
			}
		} else {
			want := http.Header{"Some-Trailer": {"some-value"}}
			if !reflect.DeepEqual(res.Trailer, want) {
				t.Errorf("Trailer = %v; want %v", res.Trailer, want)
			}
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			endStream := false
			send := func(mode headerType) {
				hbf := buf.Bytes()
				switch mode {
				case oneHeader:
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.Header().StreamID,
						EndHeaders:    true,
						EndStream:     endStream,
						BlockFragment: hbf,
					})
				case splitHeader:
					if len(hbf) < 2 {
						panic("too small")
					}
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.Header().StreamID,
						EndHeaders:    false,
						EndStream:     endStream,
						BlockFragment: hbf[:1],
					})
					ct.fr.WriteContinuation(f.Header().StreamID, true, hbf[1:])
				default:
					panic("bogus mode")
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *DataFrame:
				if !f.StreamEnded() {
					// No need to send flow control tokens. The test request body is tiny.
					continue
				}
				// Response headers (1+ frames; 1 or 2 in this test, but never 0)
				{
					buf.Reset()
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
					enc.WriteField(hpack.HeaderField{Name: "x-foo", Value: "blah"})
					enc.WriteField(hpack.HeaderField{Name: "x-bar", Value: "more"})
					if trailers != noHeader {
						enc.WriteField(hpack.HeaderField{Name: "trailer", Value: "some-trailer"})
					}
					endStream = withData == false && trailers == noHeader
					send(resHeader)
				}
				if withData {
					endStream = trailers == noHeader
					ct.fr.WriteData(f.StreamID, endStream, []byte(resBody))
				}
				if trailers != noHeader {
					endStream = true
					buf.Reset()
					enc.WriteField(hpack.HeaderField{Name: "some-trailer", Value: "some-value"})
					send(trailers)
				}
				if endStream {
					return nil
				}
			case *HeadersFrame:
				if expect100Continue != noHeader {
					buf.Reset()
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "100"})
					send(expect100Continue)
				}
			}
		}
	}
	ct.run()
}

// Issue 26189, Issue 17739: ignore unknown 1xx responses
func TestTransportUnknown1xx(t *testing.T) {
	var buf bytes.Buffer
	defer func() { got1xxFuncForTests = nil }()
	got1xxFuncForTests = func(code int, header textproto.MIMEHeader) error {
		fmt.Fprintf(&buf, "code=%d header=%v\n", code, header)
		return nil
	}

	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != 204 {
			return fmt.Errorf("status code = %v; want 204", res.StatusCode)
		}
		want := `code=110 header=map[Foo-Bar:[110]]
code=111 header=map[Foo-Bar:[111]]
code=112 header=map[Foo-Bar:[112]]
code=113 header=map[Foo-Bar:[113]]
code=114 header=map[Foo-Bar:[114]]
`
		if got := buf.String(); got != want {
			t.Errorf("Got trace:\n%s\nWant:\n%s", got, want)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				for i := 110; i <= 114; i++ {
					buf.Reset()
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: fmt.Sprint(i)})
					enc.WriteField(hpack.HeaderField{Name: "foo-bar", Value: fmt.Sprint(i)})
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.StreamID,
						EndHeaders:    true,
						EndStream:     false,
						BlockFragment: buf.Bytes(),
					})
				}
				buf.Reset()
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "204"})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.StreamID,
					EndHeaders:    true,
					EndStream:     false,
					BlockFragment: buf.Bytes(),
				})
				return nil
			}
		}
	}
	ct.run()

}

func TestTransportReceiveUndeclaredTrailer(t *testing.T) {
	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("status code = %v; want 200", res.StatusCode)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("res.Body ReadAll error = %q, %v; want %v", slurp, err, nil)
		}
		if len(slurp) > 0 {
			return fmt.Errorf("body = %q; want nothing", slurp)
		}
		if _, ok := res.Trailer["Some-Trailer"]; !ok {
			return fmt.Errorf("expected Some-Trailer")
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()

		var n int
		var hf *HeadersFrame
		for hf == nil && n < 10 {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			hf, _ = f.(*HeadersFrame)
			n++
		}

		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		// send headers without Trailer header
		enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     false,
			BlockFragment: buf.Bytes(),
		})

		// send trailers
		buf.Reset()
		enc.WriteField(hpack.HeaderField{Name: "some-trailer", Value: "I'm an undeclared Trailer!"})
		ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     true,
			BlockFragment: buf.Bytes(),
		})
		return nil
	}
	ct.run()
}

func TestTransportInvalidTrailer_Pseudo1(t *testing.T) {
	testTransportInvalidTrailer_Pseudo(t, oneHeader)
}
func TestTransportInvalidTrailer_Pseudo2(t *testing.T) {
	testTransportInvalidTrailer_Pseudo(t, splitHeader)
}
func testTransportInvalidTrailer_Pseudo(t *testing.T, trailers headerType) {
	testInvalidTrailer(t, trailers, pseudoHeaderError(":colon"), func(enc *hpack.Encoder) {
		enc.WriteField(hpack.HeaderField{Name: ":colon", Value: "foo"})
		enc.WriteField(hpack.HeaderField{Name: "foo", Value: "bar"})
	})
}

func TestTransportInvalidTrailer_Capital1(t *testing.T) {
	testTransportInvalidTrailer_Capital(t, oneHeader)
}
func TestTransportInvalidTrailer_Capital2(t *testing.T) {
	testTransportInvalidTrailer_Capital(t, splitHeader)
}
func testTransportInvalidTrailer_Capital(t *testing.T, trailers headerType) {
	testInvalidTrailer(t, trailers, headerFieldNameError("Capital"), func(enc *hpack.Encoder) {
		enc.WriteField(hpack.HeaderField{Name: "foo", Value: "bar"})
		enc.WriteField(hpack.HeaderField{Name: "Capital", Value: "bad"})
	})
}
func TestTransportInvalidTrailer_EmptyFieldName(t *testing.T) {
	testInvalidTrailer(t, oneHeader, headerFieldNameError(""), func(enc *hpack.Encoder) {
		enc.WriteField(hpack.HeaderField{Name: "", Value: "bad"})
	})
}
func TestTransportInvalidTrailer_BinaryFieldValue(t *testing.T) {
	testInvalidTrailer(t, oneHeader, headerFieldValueError("has\nnewline"), func(enc *hpack.Encoder) {
		enc.WriteField(hpack.HeaderField{Name: "x", Value: "has\nnewline"})
	})
}

func testInvalidTrailer(t *testing.T, trailers headerType, wantErr error, writeTrailer func(*hpack.Encoder)) {
	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("status code = %v; want 200", res.StatusCode)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		se, ok := err.(StreamError)
		if !ok || se.Cause != wantErr {
			return fmt.Errorf("res.Body ReadAll error = %q, %#v; want StreamError with cause %T, %#v", slurp, err, wantErr, wantErr)
		}
		if len(slurp) > 0 {
			return fmt.Errorf("body = %q; want nothing", slurp)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			switch f := f.(type) {
			case *HeadersFrame:
				var endStream bool
				send := func(mode headerType) {
					hbf := buf.Bytes()
					switch mode {
					case oneHeader:
						ct.fr.WriteHeaders(HeadersFrameParam{
							StreamID:      f.StreamID,
							EndHeaders:    true,
							EndStream:     endStream,
							BlockFragment: hbf,
						})
					case splitHeader:
						if len(hbf) < 2 {
							panic("too small")
						}
						ct.fr.WriteHeaders(HeadersFrameParam{
							StreamID:      f.StreamID,
							EndHeaders:    false,
							EndStream:     endStream,
							BlockFragment: hbf[:1],
						})
						ct.fr.WriteContinuation(f.StreamID, true, hbf[1:])
					default:
						panic("bogus mode")
					}
				}
				// Response headers (1+ frames; 1 or 2 in this test, but never 0)
				{
					buf.Reset()
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
					enc.WriteField(hpack.HeaderField{Name: "trailer", Value: "declared"})
					endStream = false
					send(oneHeader)
				}
				// Trailers:
				{
					endStream = true
					buf.Reset()
					writeTrailer(enc)
					send(trailers)
				}
				return nil
			}
		}
	}
	ct.run()
}

// headerListSize returns the HTTP2 header list size of h.
//   http://httpwg.org/specs/rfc7540.html#SETTINGS_MAX_HEADER_LIST_SIZE
//   http://httpwg.org/specs/rfc7540.html#MaxHeaderBlock
func headerListSize(h http.Header) (size uint32) {
	for k, vv := range h {
		for _, v := range vv {
			hf := hpack.HeaderField{Name: k, Value: v}
			size += hf.Size()
		}
	}
	return size
}

// padHeaders adds data to an http.Header until headerListSize(h) ==
// limit. Due to the way header list sizes are calculated, padHeaders
// cannot add fewer than len("Pad-Headers") + 32 bytes to h, and will
// call t.Fatal if asked to do so. PadHeaders first reserves enough
// space for an empty "Pad-Headers" key, then adds as many copies of
// filler as possible. Any remaining bytes necessary to push the
// header list size up to limit are added to h["Pad-Headers"].
func padHeaders(t *testing.T, h http.Header, limit uint64, filler string) {
	if limit > 0xffffffff {
		t.Fatalf("padHeaders: refusing to pad to more than 2^32-1 bytes. limit = %v", limit)
	}
	hf := hpack.HeaderField{Name: "Pad-Headers", Value: ""}
	minPadding := uint64(hf.Size())
	size := uint64(headerListSize(h))

	minlimit := size + minPadding
	if limit < minlimit {
		t.Fatalf("padHeaders: limit %v < %v", limit, minlimit)
	}

	// Use a fixed-width format for name so that fieldSize
	// remains constant.
	nameFmt := "Pad-Headers-%06d"
	hf = hpack.HeaderField{Name: fmt.Sprintf(nameFmt, 1), Value: filler}
	fieldSize := uint64(hf.Size())

	// Add as many complete filler values as possible, leaving
	// room for at least one empty "Pad-Headers" key.
	limit = limit - minPadding
	for i := 0; size+fieldSize < limit; i++ {
		name := fmt.Sprintf(nameFmt, i)
		h.Add(name, filler)
		size += fieldSize
	}

	// Add enough bytes to reach limit.
	remain := limit - size
	lastValue := strings.Repeat("*", int(remain))
	h.Add("Pad-Headers", lastValue)
}

func TestPadHeaders(t *testing.T) {
	check := func(h http.Header, limit uint32, fillerLen int) {
		if h == nil {
			h = make(http.Header)
		}
		filler := strings.Repeat("f", fillerLen)
		padHeaders(t, h, uint64(limit), filler)
		gotSize := headerListSize(h)
		if gotSize != limit {
			t.Errorf("Got size = %v; want %v", gotSize, limit)
		}
	}
	// Try all possible combinations for small fillerLen and limit.
	hf := hpack.HeaderField{Name: "Pad-Headers", Value: ""}
	minLimit := hf.Size()
	for limit := minLimit; limit <= 128; limit++ {
		for fillerLen := 0; uint32(fillerLen) <= limit; fillerLen++ {
			check(nil, limit, fillerLen)
		}
	}

	// Try a few tests with larger limits, plus cumulative
	// tests. Since these tests are cumulative, tests[i+1].limit
	// must be >= tests[i].limit + minLimit. See the comment on
	// padHeaders for more info on why the limit arg has this
	// restriction.
	tests := []struct {
		fillerLen int
		limit     uint32
	}{
		{
			fillerLen: 64,
			limit:     1024,
		},
		{
			fillerLen: 1024,
			limit:     1286,
		},
		{
			fillerLen: 256,
			limit:     2048,
		},
		{
			fillerLen: 1024,
			limit:     10 * 1024,
		},
		{
			fillerLen: 1023,
			limit:     11 * 1024,
		},
	}
	h := make(http.Header)
	for _, tc := range tests {
		check(nil, tc.limit, tc.fillerLen)
		check(h, tc.limit, tc.fillerLen)
	}
}

func TestTransportChecksRequestHeaderListSize(t *testing.T) {
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			// Consume body & force client to send
			// trailers before writing response.
			// ioutil.ReadAll returns non-nil err for
			// requests that attempt to send greater than
			// maxHeaderListSize bytes of trailers, since
			// those requests generate a stream reset.
			ioutil.ReadAll(r.Body)
			r.Body.Close()
		},
		func(ts *httptest.Server) {
			ts.Config.MaxHeaderBytes = 16 << 10
		},
		optOnlyServer,
		optQuiet,
	)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	checkRoundTrip := func(req *http.Request, wantErr error, desc string) {
		res, err := tr.RoundTrip(req)
		if err != wantErr {
			if res != nil {
				res.Body.Close()
			}
			t.Errorf("%v: RoundTrip err = %v; want %v", desc, err, wantErr)
			return
		}
		if err == nil {
			if res == nil {
				t.Errorf("%v: response nil; want non-nil.", desc)
				return
			}
			defer res.Body.Close()
			if res.StatusCode != http.StatusOK {
				t.Errorf("%v: response status = %v; want %v", desc, res.StatusCode, http.StatusOK)
			}
			return
		}
		if res != nil {
			t.Errorf("%v: RoundTrip err = %v but response non-nil", desc, err)
		}
	}
	headerListSizeForRequest := func(req *http.Request) (size uint64) {
		contentLen := actualContentLength(req)
		trailers, err := commaSeparatedTrailers(req)
		if err != nil {
			t.Fatalf("headerListSizeForRequest: %v", err)
		}
		cc := &ClientConn{peerMaxHeaderListSize: 0xffffffffffffffff}
		cc.henc = hpack.NewEncoder(&cc.hbuf)
		cc.mu.Lock()
		hdrs, err := cc.encodeHeaders(req, true, trailers, contentLen)
		cc.mu.Unlock()
		if err != nil {
			t.Fatalf("headerListSizeForRequest: %v", err)
		}
		hpackDec := hpack.NewDecoder(initialHeaderTableSize, func(hf hpack.HeaderField) {
			size += uint64(hf.Size())
		})
		if len(hdrs) > 0 {
			if _, err := hpackDec.Write(hdrs); err != nil {
				t.Fatalf("headerListSizeForRequest: %v", err)
			}
		}
		return size
	}
	// Create a new Request for each test, rather than reusing the
	// same Request, to avoid a race when modifying req.Headers.
	// See https://github.com/golang/go/issues/21316
	newRequest := func() *http.Request {
		// Body must be non-nil to enable writing trailers.
		body := strings.NewReader("hello")
		req, err := http.NewRequest("POST", st.ts.URL, body)
		if err != nil {
			t.Fatalf("newRequest: NewRequest: %v", err)
		}
		return req
	}

	// Make an arbitrary request to ensure we get the server's
	// settings frame and initialize peerMaxHeaderListSize.
	req := newRequest()
	checkRoundTrip(req, nil, "Initial request")

	// Get the ClientConn associated with the request and validate
	// peerMaxHeaderListSize.
	addr := authorityAddr(req.URL.Scheme, req.URL.Host)
	cc, err := tr.connPool().GetClientConn(req, addr)
	if err != nil {
		t.Fatalf("GetClientConn: %v", err)
	}
	cc.mu.Lock()
	peerSize := cc.peerMaxHeaderListSize
	cc.mu.Unlock()
	st.scMu.Lock()
	wantSize := uint64(st.sc.maxHeaderListSize())
	st.scMu.Unlock()
	if peerSize != wantSize {
		t.Errorf("peerMaxHeaderListSize = %v; want %v", peerSize, wantSize)
	}

	// Sanity check peerSize. (*serverConn) maxHeaderListSize adds
	// 320 bytes of padding.
	wantHeaderBytes := uint64(st.ts.Config.MaxHeaderBytes) + 320
	if peerSize != wantHeaderBytes {
		t.Errorf("peerMaxHeaderListSize = %v; want %v.", peerSize, wantHeaderBytes)
	}

	// Pad headers & trailers, but stay under peerSize.
	req = newRequest()
	req.Header = make(http.Header)
	req.Trailer = make(http.Header)
	filler := strings.Repeat("*", 1024)
	padHeaders(t, req.Trailer, peerSize, filler)
	// cc.encodeHeaders adds some default headers to the request,
	// so we need to leave room for those.
	defaultBytes := headerListSizeForRequest(req)
	padHeaders(t, req.Header, peerSize-defaultBytes, filler)
	checkRoundTrip(req, nil, "Headers & Trailers under limit")

	// Add enough header bytes to push us over peerSize.
	req = newRequest()
	req.Header = make(http.Header)
	padHeaders(t, req.Header, peerSize, filler)
	checkRoundTrip(req, errRequestHeaderListSize, "Headers over limit")

	// Push trailers over the limit.
	req = newRequest()
	req.Trailer = make(http.Header)
	padHeaders(t, req.Trailer, peerSize+1, filler)
	checkRoundTrip(req, errRequestHeaderListSize, "Trailers over limit")

	// Send headers with a single large value.
	req = newRequest()
	filler = strings.Repeat("*", int(peerSize))
	req.Header = make(http.Header)
	req.Header.Set("Big", filler)
	checkRoundTrip(req, errRequestHeaderListSize, "Single large header")

	// Send trailers with a single large value.
	req = newRequest()
	req.Trailer = make(http.Header)
	req.Trailer.Set("Big", filler)
	checkRoundTrip(req, errRequestHeaderListSize, "Single large trailer")
}

func TestTransportChecksResponseHeaderListSize(t *testing.T) {
	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != errResponseHeaderListSize {
			if res != nil {
				res.Body.Close()
			}
			size := int64(0)
			for k, vv := range res.Header {
				for _, v := range vv {
					size += int64(len(k)) + int64(len(v)) + 32
				}
			}
			return fmt.Errorf("RoundTrip Error = %v (and %d bytes of response headers); want errResponseHeaderListSize", err, size)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			switch f := f.(type) {
			case *HeadersFrame:
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
				large := strings.Repeat("a", 1<<10)
				for i := 0; i < 5042; i++ {
					enc.WriteField(hpack.HeaderField{Name: large, Value: large})
				}
				if size, want := buf.Len(), 6329; size != want {
					// Note: this number might change if
					// our hpack implementation
					// changes. That's fine. This is
					// just a sanity check that our
					// response can fit in a single
					// header block fragment frame.
					return fmt.Errorf("encoding over 10MB of duplicate keypairs took %d bytes; expected %d", size, want)
				}
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.StreamID,
					EndHeaders:    true,
					EndStream:     true,
					BlockFragment: buf.Bytes(),
				})
				return nil
			}
		}
	}
	ct.run()
}

func TestTransportCookieHeaderSplit(t *testing.T) {
	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		req.Header.Add("Cookie", "a=b;c=d;  e=f;")
		req.Header.Add("Cookie", "e=f;g=h; ")
		req.Header.Add("Cookie", "i=j")
		_, err := ct.tr.RoundTrip(req)
		return err
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			switch f := f.(type) {
			case *HeadersFrame:
				dec := hpack.NewDecoder(initialHeaderTableSize, nil)
				hfs, err := dec.DecodeFull(f.HeaderBlockFragment())
				if err != nil {
					return err
				}
				got := []string{}
				want := []string{"a=b", "c=d", "e=f", "e=f", "g=h", "i=j"}
				for _, hf := range hfs {
					if hf.Name == "cookie" {
						got = append(got, hf.Value)
					}
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("Cookies = %#v, want %#v", got, want)
				}

				var buf bytes.Buffer
				enc := hpack.NewEncoder(&buf)
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.StreamID,
					EndHeaders:    true,
					EndStream:     true,
					BlockFragment: buf.Bytes(),
				})
				return nil
			}
		}
	}
	ct.run()
}

// Test that the Transport returns a typed error from Response.Body.Read calls
// when the server sends an error. (here we use a panic, since that should generate
// a stream error, but others like cancel should be similar)
func TestTransportBodyReadErrorType(t *testing.T) {
	doPanic := make(chan bool, 1)
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			w.(http.Flusher).Flush() // force headers out
			<-doPanic
			panic("boom")
		},
		optOnlyServer,
		optQuiet,
	)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}

	res, err := c.Get(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	doPanic <- true
	buf := make([]byte, 100)
	n, err := res.Body.Read(buf)
	want := StreamError{StreamID: 0x1, Code: 0x2}
	if !reflect.DeepEqual(want, err) {
		t.Errorf("Read = %v, %#v; want error %#v", n, err, want)
	}
}

// golang.org/issue/13924
// This used to fail after many iterations, especially with -race:
// go test -v -run=TestTransportDoubleCloseOnWriteError -count=500 -race
func TestTransportDoubleCloseOnWriteError(t *testing.T) {
	var (
		mu   sync.Mutex
		conn net.Conn // to close if set
	)

	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()
			if conn != nil {
				conn.Close()
			}
		},
		optOnlyServer,
	)
	defer st.Close()

	tr := &Transport{
		TLSClientConfig: tlsConfigInsecure,
		DialTLS: func(network, addr string, cfg *tls.Config) (net.Conn, error) {
			tc, err := tls.Dial(network, addr, cfg)
			if err != nil {
				return nil, err
			}
			mu.Lock()
			defer mu.Unlock()
			conn = tc
			return tc, nil
		},
	}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}
	c.Get(st.ts.URL)
}

// Test that the http1 Transport.DisableKeepAlives option is respected
// and connections are closed as soon as idle.
// See golang.org/issue/14008
func TestTransportDisableKeepAlives(t *testing.T) {
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "hi")
		},
		optOnlyServer,
	)
	defer st.Close()

	connClosed := make(chan struct{}) // closed on tls.Conn.Close
	tr := &Transport{
		t1: &http.Transport{
			DisableKeepAlives: true,
		},
		TLSClientConfig: tlsConfigInsecure,
		DialTLS: func(network, addr string, cfg *tls.Config) (net.Conn, error) {
			tc, err := tls.Dial(network, addr, cfg)
			if err != nil {
				return nil, err
			}
			return &noteCloseConn{Conn: tc, closefn: func() { close(connClosed) }}, nil
		},
	}
	c := &http.Client{Transport: tr}
	res, err := c.Get(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ioutil.ReadAll(res.Body); err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	select {
	case <-connClosed:
	case <-time.After(1 * time.Second):
		t.Errorf("timeout")
	}

}

// Test concurrent requests with Transport.DisableKeepAlives. We can share connections,
// but when things are totally idle, it still needs to close.
func TestTransportDisableKeepAlives_Concurrency(t *testing.T) {
	const D = 25 * time.Millisecond
	st := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(D)
			io.WriteString(w, "hi")
		},
		optOnlyServer,
	)
	defer st.Close()

	var dials int32
	var conns sync.WaitGroup
	tr := &Transport{
		t1: &http.Transport{
			DisableKeepAlives: true,
		},
		TLSClientConfig: tlsConfigInsecure,
		DialTLS: func(network, addr string, cfg *tls.Config) (net.Conn, error) {
			tc, err := tls.Dial(network, addr, cfg)
			if err != nil {
				return nil, err
			}
			atomic.AddInt32(&dials, 1)
			conns.Add(1)
			return &noteCloseConn{Conn: tc, closefn: func() { conns.Done() }}, nil
		},
	}
	c := &http.Client{Transport: tr}
	var reqs sync.WaitGroup
	const N = 20
	for i := 0; i < N; i++ {
		reqs.Add(1)
		if i == N-1 {
			// For the final request, try to make all the
			// others close. This isn't verified in the
			// count, other than the Log statement, since
			// it's so timing dependent. This test is
			// really to make sure we don't interrupt a
			// valid request.
			time.Sleep(D * 2)
		}
		go func() {
			defer reqs.Done()
			res, err := c.Get(st.ts.URL)
			if err != nil {
				t.Error(err)
				return
			}
			if _, err := ioutil.ReadAll(res.Body); err != nil {
				t.Error(err)
				return
			}
			res.Body.Close()
		}()
	}
	reqs.Wait()
	conns.Wait()
	t.Logf("did %d dials, %d requests", atomic.LoadInt32(&dials), N)
}

type noteCloseConn struct {
	net.Conn
	onceClose sync.Once
	closefn   func()
}

func (c *noteCloseConn) Close() error {
	c.onceClose.Do(c.closefn)
	return c.Conn.Close()
}

func isTimeout(err error) bool {
	switch err := err.(type) {
	case nil:
		return false
	case *url.Error:
		return isTimeout(err.Err)
	case net.Error:
		return err.Timeout()
	}
	return false
}

// Test that the http1 Transport.ResponseHeaderTimeout option and cancel is sent.
func TestTransportResponseHeaderTimeout_NoBody(t *testing.T) {
	testTransportResponseHeaderTimeout(t, false)
}
func TestTransportResponseHeaderTimeout_Body(t *testing.T) {
	testTransportResponseHeaderTimeout(t, true)
}

func testTransportResponseHeaderTimeout(t *testing.T, body bool) {
	ct := newClientTester(t)
	ct.tr.t1 = &http.Transport{
		ResponseHeaderTimeout: 5 * time.Millisecond,
	}
	ct.client = func() error {
		c := &http.Client{Transport: ct.tr}
		var err error
		var n int64
		const bodySize = 4 << 20
		if body {
			_, err = c.Post("https://dummy.tld/", "text/foo", io.LimitReader(countingReader{&n}, bodySize))
		} else {
			_, err = c.Get("https://dummy.tld/")
		}
		if !isTimeout(err) {
			t.Errorf("client expected timeout error; got %#v", err)
		}
		if body && n != bodySize {
			t.Errorf("only read %d bytes of body; want %d", n, bodySize)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				t.Logf("ReadFrame: %v", err)
				return nil
			}
			switch f := f.(type) {
			case *DataFrame:
				dataLen := len(f.Data())
				if dataLen > 0 {
					if err := ct.fr.WriteWindowUpdate(0, uint32(dataLen)); err != nil {
						return err
					}
					if err := ct.fr.WriteWindowUpdate(f.StreamID, uint32(dataLen)); err != nil {
						return err
					}
				}
			case *RSTStreamFrame:
				if f.StreamID == 1 && f.ErrCode == ErrCodeCancel {
					return nil
				}
			}
		}
	}
	ct.run()
}

func TestTransportDisableCompression(t *testing.T) {
	const body = "sup"
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		want := http.Header{
			"User-Agent": []string{"Go-http-client/2.0"},
		}
		if !reflect.DeepEqual(r.Header, want) {
			t.Errorf("request headers = %v; want %v", r.Header, want)
		}
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{
		TLSClientConfig: tlsConfigInsecure,
		t1: &http.Transport{
			DisableCompression: true,
		},
	}
	defer tr.CloseIdleConnections()

	req, err := http.NewRequest("GET", st.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
}

// RFC 7540 section 8.1.2.2
func TestTransportRejectsConnHeaders(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		var got []string
		for k := range r.Header {
			got = append(got, k)
		}
		sort.Strings(got)
		w.Header().Set("Got-Header", strings.Join(got, ","))
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	tests := []struct {
		key   string
		value []string
		want  string
	}{
		{
			key:   "Upgrade",
			value: []string{"anything"},
			want:  "ERROR: http2: invalid Upgrade request header: [\"anything\"]",
		},
		{
			key:   "Connection",
			value: []string{"foo"},
			want:  "ERROR: http2: invalid Connection request header: [\"foo\"]",
		},
		{
			key:   "Connection",
			value: []string{"close"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Connection",
			value: []string{"CLoSe"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Connection",
			value: []string{"close", "something-else"},
			want:  "ERROR: http2: invalid Connection request header: [\"close\" \"something-else\"]",
		},
		{
			key:   "Connection",
			value: []string{"keep-alive"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Connection",
			value: []string{"Keep-ALIVE"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Proxy-Connection", // just deleted and ignored
			value: []string{"keep-alive"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{""},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"foo"},
			want:  "ERROR: http2: invalid Transfer-Encoding request header: [\"foo\"]",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"chunked"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"chunKed"}, // Kelvin sign
			want:  "ERROR: http2: invalid Transfer-Encoding request header: [\"chunKed\"]",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"chunked", "other"},
			want:  "ERROR: http2: invalid Transfer-Encoding request header: [\"chunked\" \"other\"]",
		},
		{
			key:   "Content-Length",
			value: []string{"123"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Keep-Alive",
			value: []string{"doop"},
			want:  "Accept-Encoding,User-Agent",
		},
	}

	for _, tt := range tests {
		req, _ := http.NewRequest("GET", st.ts.URL, nil)
		req.Header[tt.key] = tt.value
		res, err := tr.RoundTrip(req)
		var got string
		if err != nil {
			got = fmt.Sprintf("ERROR: %v", err)
		} else {
			got = res.Header.Get("Got-Header")
			res.Body.Close()
		}
		if got != tt.want {
			t.Errorf("For key %q, value %q, got = %q; want %q", tt.key, tt.value, got, tt.want)
		}
	}
}

// Reject content-length headers containing a sign.
// See https://golang.org/issue/39017
func TestTransportRejectsContentLengthWithSign(t *testing.T) {
	tests := []struct {
		name   string
		cl     []string
		wantCL string
	}{
		{
			name:   "proper content-length",
			cl:     []string{"3"},
			wantCL: "3",
		},
		{
			name:   "ignore cl with plus sign",
			cl:     []string{"+3"},
			wantCL: "",
		},
		{
			name:   "ignore cl with minus sign",
			cl:     []string{"-3"},
			wantCL: "",
		},
		{
			name:   "max int64, for safe uint64->int64 conversion",
			cl:     []string{"9223372036854775807"},
			wantCL: "9223372036854775807",
		},
		{
			name:   "overflows int64, so ignored",
			cl:     []string{"9223372036854775808"},
			wantCL: "",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Length", tt.cl[0])
			}, optOnlyServer)
			defer st.Close()
			tr := &Transport{TLSClientConfig: tlsConfigInsecure}
			defer tr.CloseIdleConnections()

			req, _ := http.NewRequest("HEAD", st.ts.URL, nil)
			res, err := tr.RoundTrip(req)

			var got string
			if err != nil {
				got = fmt.Sprintf("ERROR: %v", err)
			} else {
				got = res.Header.Get("Content-Length")
				res.Body.Close()
			}

			if got != tt.wantCL {
				t.Fatalf("Got: %q\nWant: %q", got, tt.wantCL)
			}
		})
	}
}

// golang.org/issue/14048
func TestTransportFailsOnInvalidHeaders(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		var got []string
		for k := range r.Header {
			got = append(got, k)
		}
		sort.Strings(got)
		w.Header().Set("Got-Header", strings.Join(got, ","))
	}, optOnlyServer)
	defer st.Close()

	tests := [...]struct {
		h       http.Header
		wantErr string
	}{
		0: {
			h:       http.Header{"with space": {"foo"}},
			wantErr: `invalid HTTP header name "with space"`,
		},
		1: {
			h:       http.Header{"name": {"Брэд"}},
			wantErr: "", // okay
		},
		2: {
			h:       http.Header{"имя": {"Brad"}},
			wantErr: `invalid HTTP header name "имя"`,
		},
		3: {
			h:       http.Header{"foo": {"foo\x01bar"}},
			wantErr: `invalid HTTP header value "foo\x01bar" for header "foo"`,
		},
	}

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	for i, tt := range tests {
		req, _ := http.NewRequest("GET", st.ts.URL, nil)
		req.Header = tt.h
		res, err := tr.RoundTrip(req)
		var bad bool
		if tt.wantErr == "" {
			if err != nil {
				bad = true
				t.Errorf("case %d: error = %v; want no error", i, err)
			}
		} else {
			if !strings.Contains(fmt.Sprint(err), tt.wantErr) {
				bad = true
				t.Errorf("case %d: error = %v; want error %q", i, err, tt.wantErr)
			}
		}
		if err == nil {
			if bad {
				t.Logf("case %d: server got headers %q", i, res.Header.Get("Got-Header"))
			}
			res.Body.Close()
		}
	}
}

// Tests that gzipReader doesn't crash on a second Read call following
// the first Read call's gzip.NewReader returning an error.
func TestGzipReader_DoubleReadCrash(t *testing.T) {
	gz := &gzipReader{
		body: ioutil.NopCloser(strings.NewReader("0123456789")),
	}
	var buf [1]byte
	n, err1 := gz.Read(buf[:])
	if n != 0 || !strings.Contains(fmt.Sprint(err1), "invalid header") {
		t.Fatalf("Read = %v, %v; want 0, invalid header", n, err1)
	}
	n, err2 := gz.Read(buf[:])
	if n != 0 || err2 != err1 {
		t.Fatalf("second Read = %v, %v; want 0, %v", n, err2, err1)
	}
}

func TestTransportNewTLSConfig(t *testing.T) {
	tests := [...]struct {
		conf *tls.Config
		host string
		want *tls.Config
	}{
		// Normal case.
		0: {
			conf: nil,
			host: "foo.com",
			want: &tls.Config{
				ServerName: "foo.com",
				NextProtos: []string{NextProtoTLS},
			},
		},

		// User-provided name (bar.com) takes precedence:
		1: {
			conf: &tls.Config{
				ServerName: "bar.com",
			},
			host: "foo.com",
			want: &tls.Config{
				ServerName: "bar.com",
				NextProtos: []string{NextProtoTLS},
			},
		},

		// NextProto is prepended:
		2: {
			conf: &tls.Config{
				NextProtos: []string{"foo", "bar"},
			},
			host: "example.com",
			want: &tls.Config{
				ServerName: "example.com",
				NextProtos: []string{NextProtoTLS, "foo", "bar"},
			},
		},

		// NextProto is not duplicated:
		3: {
			conf: &tls.Config{
				NextProtos: []string{"foo", "bar", NextProtoTLS},
			},
			host: "example.com",
			want: &tls.Config{
				ServerName: "example.com",
				NextProtos: []string{"foo", "bar", NextProtoTLS},
			},
		},
	}
	for i, tt := range tests {
		// Ignore the session ticket keys part, which ends up populating
		// unexported fields in the Config:
		if tt.conf != nil {
			tt.conf.SessionTicketsDisabled = true
		}

		tr := &Transport{TLSClientConfig: tt.conf}
		got := tr.newTLSConfig(tt.host)

		got.SessionTicketsDisabled = false

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%d. got %#v; want %#v", i, got, tt.want)
		}
	}
}

// The Google GFE responds to HEAD requests with a HEADERS frame
// without END_STREAM, followed by a 0-length DATA frame with
// END_STREAM. Make sure we don't get confused by that. (We did.)
func TestTransportReadHeadResponse(t *testing.T) {
	ct := newClientTester(t)
	clientDone := make(chan struct{})
	ct.client = func() error {
		defer close(clientDone)
		req, _ := http.NewRequest("HEAD", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		if res.ContentLength != 123 {
			return fmt.Errorf("Content-Length = %d; want 123", res.ContentLength)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("ReadAll: %v", err)
		}
		if len(slurp) > 0 {
			return fmt.Errorf("Unexpected non-empty ReadAll body: %q", slurp)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				t.Logf("ReadFrame: %v", err)
				return nil
			}
			hf, ok := f.(*HeadersFrame)
			if !ok {
				continue
			}
			var buf bytes.Buffer
			enc := hpack.NewEncoder(&buf)
			enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
			enc.WriteField(hpack.HeaderField{Name: "content-length", Value: "123"})
			ct.fr.WriteHeaders(HeadersFrameParam{
				StreamID:      hf.StreamID,
				EndHeaders:    true,
				EndStream:     false, // as the GFE does
				BlockFragment: buf.Bytes(),
			})
			ct.fr.WriteData(hf.StreamID, true, nil)

			<-clientDone
			return nil
		}
	}
	ct.run()
}

func TestTransportReadHeadResponseWithBody(t *testing.T) {
	// This test use not valid response format.
	// Discarding logger output to not spam tests output.
	log.SetOutput(ioutil.Discard)
	defer log.SetOutput(os.Stderr)

	response := "redirecting to /elsewhere"
	ct := newClientTester(t)
	clientDone := make(chan struct{})
	ct.client = func() error {
		defer close(clientDone)
		req, _ := http.NewRequest("HEAD", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		if res.ContentLength != int64(len(response)) {
			return fmt.Errorf("Content-Length = %d; want %d", res.ContentLength, len(response))
		}
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("ReadAll: %v", err)
		}
		if len(slurp) > 0 {
			return fmt.Errorf("Unexpected non-empty ReadAll body: %q", slurp)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				t.Logf("ReadFrame: %v", err)
				return nil
			}
			hf, ok := f.(*HeadersFrame)
			if !ok {
				continue
			}
			var buf bytes.Buffer
			enc := hpack.NewEncoder(&buf)
			enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
			enc.WriteField(hpack.HeaderField{Name: "content-length", Value: strconv.Itoa(len(response))})
			ct.fr.WriteHeaders(HeadersFrameParam{
				StreamID:      hf.StreamID,
				EndHeaders:    true,
				EndStream:     false,
				BlockFragment: buf.Bytes(),
			})
			ct.fr.WriteData(hf.StreamID, true, []byte(response))

			<-clientDone
			return nil
		}
	}
	ct.run()
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

// golang.org/issue/15425: test that a handler closing the request
// body doesn't terminate the stream to the peer. (It just stops
// readability from the handler's side, and eventually the client
// runs out of flow control tokens)
func TestTransportHandlerBodyClose(t *testing.T) {
	const bodySize = 10 << 20
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		r.Body.Close()
		io.Copy(w, io.LimitReader(neverEnding('A'), bodySize))
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	g0 := runtime.NumGoroutine()

	const numReq = 10
	for i := 0; i < numReq; i++ {
		req, err := http.NewRequest("POST", st.ts.URL, struct{ io.Reader }{io.LimitReader(neverEnding('A'), bodySize)})
		if err != nil {
			t.Fatal(err)
		}
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		n, err := io.Copy(ioutil.Discard, res.Body)
		res.Body.Close()
		if n != bodySize || err != nil {
			t.Fatalf("req#%d: Copy = %d, %v; want %d, nil", i, n, err, bodySize)
		}
	}
	tr.CloseIdleConnections()

	if !waitCondition(5*time.Second, 100*time.Millisecond, func() bool {
		gd := runtime.NumGoroutine() - g0
		return gd < numReq/2
	}) {
		t.Errorf("appeared to leak goroutines")
	}
}

// https://golang.org/issue/15930
func TestTransportFlowControl(t *testing.T) {
	const bufLen = 64 << 10
	var total int64 = 100 << 20 // 100MB
	if testing.Short() {
		total = 10 << 20
	}

	var wrote int64 // updated atomically
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		b := make([]byte, bufLen)
		for wrote < total {
			n, err := w.Write(b)
			atomic.AddInt64(&wrote, int64(n))
			if err != nil {
				t.Errorf("ResponseWriter.Write error: %v", err)
				break
			}
			w.(http.Flusher).Flush()
		}
	}, optOnlyServer)

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	req, err := http.NewRequest("GET", st.ts.URL, nil)
	if err != nil {
		t.Fatal("NewRequest error:", err)
	}
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal("RoundTrip error:", err)
	}
	defer resp.Body.Close()

	var read int64
	b := make([]byte, bufLen)
	for {
		n, err := resp.Body.Read(b)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal("Read error:", err)
		}
		read += int64(n)

		const max = transportDefaultStreamFlow
		if w := atomic.LoadInt64(&wrote); -max > read-w || read-w > max {
			t.Fatalf("Too much data inflight: server wrote %v bytes but client only received %v", w, read)
		}

		// Let the server get ahead of the client.
		time.Sleep(1 * time.Millisecond)
	}
}

// golang.org/issue/14627 -- if the server sends a GOAWAY frame, make
// the Transport remember it and return it back to users (via
// RoundTrip or request body reads) if needed (e.g. if the server
// proceeds to close the TCP connection before the client gets its
// response)
func TestTransportUsesGoAwayDebugError_RoundTrip(t *testing.T) {
	testTransportUsesGoAwayDebugError(t, false)
}

func TestTransportUsesGoAwayDebugError_Body(t *testing.T) {
	testTransportUsesGoAwayDebugError(t, true)
}

func testTransportUsesGoAwayDebugError(t *testing.T, failMidBody bool) {
	ct := newClientTester(t)
	clientDone := make(chan struct{})

	const goAwayErrCode = ErrCodeHTTP11Required // arbitrary
	const goAwayDebugData = "some debug data"

	ct.client = func() error {
		defer close(clientDone)
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if failMidBody {
			if err != nil {
				return fmt.Errorf("unexpected client RoundTrip error: %v", err)
			}
			_, err = io.Copy(ioutil.Discard, res.Body)
			res.Body.Close()
		}
		want := GoAwayError{
			LastStreamID: 5,
			ErrCode:      goAwayErrCode,
			DebugData:    goAwayDebugData,
		}
		if !reflect.DeepEqual(err, want) {
			t.Errorf("RoundTrip error = %T: %#v, want %T (%#v)", err, err, want, want)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				t.Logf("ReadFrame: %v", err)
				return nil
			}
			hf, ok := f.(*HeadersFrame)
			if !ok {
				continue
			}
			if failMidBody {
				var buf bytes.Buffer
				enc := hpack.NewEncoder(&buf)
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
				enc.WriteField(hpack.HeaderField{Name: "content-length", Value: "123"})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      hf.StreamID,
					EndHeaders:    true,
					EndStream:     false,
					BlockFragment: buf.Bytes(),
				})
			}
			// Write two GOAWAY frames, to test that the Transport takes
			// the interesting parts of both.
			ct.fr.WriteGoAway(5, ErrCodeNo, []byte(goAwayDebugData))
			ct.fr.WriteGoAway(5, goAwayErrCode, nil)
			ct.sc.(*net.TCPConn).CloseWrite()
			if runtime.GOOS == "plan9" {
				// CloseWrite not supported on Plan 9; Issue 17906
				ct.sc.(*net.TCPConn).Close()
			}
			<-clientDone
			return nil
		}
	}
	ct.run()
}

func testTransportReturnsUnusedFlowControl(t *testing.T, oneDataFrame bool) {
	ct := newClientTester(t)

	clientClosed := make(chan struct{})
	serverWroteFirstByte := make(chan struct{})

	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		<-serverWroteFirstByte

		if n, err := res.Body.Read(make([]byte, 1)); err != nil || n != 1 {
			return fmt.Errorf("body read = %v, %v; want 1, nil", n, err)
		}
		res.Body.Close() // leaving 4999 bytes unread
		close(clientClosed)

		return nil
	}
	ct.server = func() error {
		ct.greet()

		var hf *HeadersFrame
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return fmt.Errorf("ReadFrame while waiting for Headers: %v", err)
			}
			switch f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
				continue
			}
			var ok bool
			hf, ok = f.(*HeadersFrame)
			if !ok {
				return fmt.Errorf("Got %T; want HeadersFrame", f)
			}
			break
		}

		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		enc.WriteField(hpack.HeaderField{Name: "content-length", Value: "5000"})
		ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     false,
			BlockFragment: buf.Bytes(),
		})

		// Two cases:
		// - Send one DATA frame with 5000 bytes.
		// - Send two DATA frames with 1 and 4999 bytes each.
		//
		// In both cases, the client should consume one byte of data,
		// refund that byte, then refund the following 4999 bytes.
		//
		// In the second case, the server waits for the client connection to
		// close before seconding the second DATA frame. This tests the case
		// where the client receives a DATA frame after it has reset the stream.
		if oneDataFrame {
			ct.fr.WriteData(hf.StreamID, false /* don't end stream */, make([]byte, 5000))
			close(serverWroteFirstByte)
			<-clientClosed
		} else {
			ct.fr.WriteData(hf.StreamID, false /* don't end stream */, make([]byte, 1))
			close(serverWroteFirstByte)
			<-clientClosed
			ct.fr.WriteData(hf.StreamID, false /* don't end stream */, make([]byte, 4999))
		}

		waitingFor := "RSTStreamFrame"
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return fmt.Errorf("ReadFrame while waiting for %s: %v", waitingFor, err)
			}
			if _, ok := f.(*SettingsFrame); ok {
				continue
			}
			switch waitingFor {
			case "RSTStreamFrame":
				if rf, ok := f.(*RSTStreamFrame); !ok || rf.ErrCode != ErrCodeCancel {
					return fmt.Errorf("Expected a RSTStreamFrame with code cancel; got %v", summarizeFrame(f))
				}
				waitingFor = "WindowUpdateFrame"
			case "WindowUpdateFrame":
				if wuf, ok := f.(*WindowUpdateFrame); !ok || wuf.Increment != 4999 {
					return fmt.Errorf("Expected WindowUpdateFrame for 4999 bytes; got %v", summarizeFrame(f))
				}
				return nil
			}
		}
	}
	ct.run()
}

// See golang.org/issue/16481
func TestTransportReturnsUnusedFlowControlSingleWrite(t *testing.T) {
	testTransportReturnsUnusedFlowControl(t, true)
}

// See golang.org/issue/20469
func TestTransportReturnsUnusedFlowControlMultipleWrites(t *testing.T) {
	testTransportReturnsUnusedFlowControl(t, false)
}

// Issue 16612: adjust flow control on open streams when transport
// receives SETTINGS with INITIAL_WINDOW_SIZE from server.
func TestTransportAdjustsFlowControl(t *testing.T) {
	ct := newClientTester(t)
	clientDone := make(chan struct{})

	const bodySize = 1 << 20

	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		defer close(clientDone)

		req, _ := http.NewRequest("POST", "https://dummy.tld/", struct{ io.Reader }{io.LimitReader(neverEnding('A'), bodySize)})
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		res.Body.Close()
		return nil
	}
	ct.server = func() error {
		_, err := io.ReadFull(ct.sc, make([]byte, len(ClientPreface)))
		if err != nil {
			return fmt.Errorf("reading client preface: %v", err)
		}

		var gotBytes int64
		var sentSettings bool
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-clientDone:
					return nil
				default:
					return fmt.Errorf("ReadFrame while waiting for Headers: %v", err)
				}
			}
			switch f := f.(type) {
			case *DataFrame:
				gotBytes += int64(len(f.Data()))
				// After we've got half the client's
				// initial flow control window's worth
				// of request body data, give it just
				// enough flow control to finish.
				if gotBytes >= initialWindowSize/2 && !sentSettings {
					sentSettings = true

					ct.fr.WriteSettings(Setting{ID: SettingInitialWindowSize, Val: bodySize})
					ct.fr.WriteWindowUpdate(0, bodySize)
					ct.fr.WriteSettingsAck()
				}

				if f.StreamEnded() {
					var buf bytes.Buffer
					enc := hpack.NewEncoder(&buf)
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.StreamID,
						EndHeaders:    true,
						EndStream:     true,
						BlockFragment: buf.Bytes(),
					})
				}
			}
		}
	}
	ct.run()
}

// See golang.org/issue/16556
func TestTransportReturnsDataPaddingFlowControl(t *testing.T) {
	ct := newClientTester(t)

	unblockClient := make(chan bool, 1)

	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		defer res.Body.Close()
		<-unblockClient
		return nil
	}
	ct.server = func() error {
		ct.greet()

		var hf *HeadersFrame
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return fmt.Errorf("ReadFrame while waiting for Headers: %v", err)
			}
			switch f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
				continue
			}
			var ok bool
			hf, ok = f.(*HeadersFrame)
			if !ok {
				return fmt.Errorf("Got %T; want HeadersFrame", f)
			}
			break
		}

		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		enc.WriteField(hpack.HeaderField{Name: "content-length", Value: "5000"})
		ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     false,
			BlockFragment: buf.Bytes(),
		})
		pad := make([]byte, 5)
		ct.fr.WriteDataPadded(hf.StreamID, false, make([]byte, 5000), pad) // without ending stream

		f, err := ct.readNonSettingsFrame()
		if err != nil {
			return fmt.Errorf("ReadFrame while waiting for first WindowUpdateFrame: %v", err)
		}
		wantBack := uint32(len(pad)) + 1 // one byte for the length of the padding
		if wuf, ok := f.(*WindowUpdateFrame); !ok || wuf.Increment != wantBack || wuf.StreamID != 0 {
			return fmt.Errorf("Expected conn WindowUpdateFrame for %d bytes; got %v", wantBack, summarizeFrame(f))
		}

		f, err = ct.readNonSettingsFrame()
		if err != nil {
			return fmt.Errorf("ReadFrame while waiting for second WindowUpdateFrame: %v", err)
		}
		if wuf, ok := f.(*WindowUpdateFrame); !ok || wuf.Increment != wantBack || wuf.StreamID == 0 {
			return fmt.Errorf("Expected stream WindowUpdateFrame for %d bytes; got %v", wantBack, summarizeFrame(f))
		}
		unblockClient <- true
		return nil
	}
	ct.run()
}

// golang.org/issue/16572 -- RoundTrip shouldn't hang when it gets a
// StreamError as a result of the response HEADERS
func TestTransportReturnsErrorOnBadResponseHeaders(t *testing.T) {
	ct := newClientTester(t)

	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err == nil {
			res.Body.Close()
			return errors.New("unexpected successful GET")
		}
		want := StreamError{1, ErrCodeProtocol, headerFieldNameError("  content-type")}
		if !reflect.DeepEqual(want, err) {
			t.Errorf("RoundTrip error = %#v; want %#v", err, want)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()

		hf, err := ct.firstHeaders()
		if err != nil {
			return err
		}

		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		enc.WriteField(hpack.HeaderField{Name: "  content-type", Value: "bogus"}) // bogus spaces
		ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     false,
			BlockFragment: buf.Bytes(),
		})

		for {
			fr, err := ct.readFrame()
			if err != nil {
				return fmt.Errorf("error waiting for RST_STREAM from client: %v", err)
			}
			if _, ok := fr.(*SettingsFrame); ok {
				continue
			}
			if rst, ok := fr.(*RSTStreamFrame); !ok || rst.StreamID != 1 || rst.ErrCode != ErrCodeProtocol {
				t.Errorf("Frame = %v; want RST_STREAM for stream 1 with ErrCodeProtocol", summarizeFrame(fr))
			}
			break
		}

		return nil
	}
	ct.run()
}

// byteAndEOFReader returns is in an io.Reader which reads one byte
// (the underlying byte) and io.EOF at once in its Read call.
type byteAndEOFReader byte

func (b byteAndEOFReader) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		panic("unexpected useless call")
	}
	p[0] = byte(b)
	return 1, io.EOF
}

// Issue 16788: the Transport had a regression where it started
// sending a spurious DATA frame with a duplicate END_STREAM bit after
// the request body writer goroutine had already read an EOF from the
// Request.Body and included the END_STREAM on a data-carrying DATA
// frame.
//
// Notably, to trigger this, the requests need to use a Request.Body
// which returns (non-0, io.EOF) and also needs to set the ContentLength
// explicitly.
func TestTransportBodyDoubleEndStream(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		// Nothing.
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	for i := 0; i < 2; i++ {
		req, _ := http.NewRequest("POST", st.ts.URL, byteAndEOFReader('a'))
		req.ContentLength = 1
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatalf("failure on req %d: %v", i+1, err)
		}
		defer res.Body.Close()
	}
}

// golang.org/issue/16847, golang.org/issue/19103
func TestTransportRequestPathPseudo(t *testing.T) {
	type result struct {
		path string
		err  string
	}
	tests := []struct {
		req  *http.Request
		want result
	}{
		0: {
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Host: "foo.com",
					Path: "/foo",
				},
			},
			want: result{path: "/foo"},
		},
		// In Go 1.7, we accepted paths of "//foo".
		// In Go 1.8, we rejected it (issue 16847).
		// In Go 1.9, we accepted it again (issue 19103).
		1: {
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Host: "foo.com",
					Path: "//foo",
				},
			},
			want: result{path: "//foo"},
		},

		// Opaque with //$Matching_Hostname/path
		2: {
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Scheme: "https",
					Opaque: "//foo.com/path",
					Host:   "foo.com",
					Path:   "/ignored",
				},
			},
			want: result{path: "/path"},
		},

		// Opaque with some other Request.Host instead:
		3: {
			req: &http.Request{
				Method: "GET",
				Host:   "bar.com",
				URL: &url.URL{
					Scheme: "https",
					Opaque: "//bar.com/path",
					Host:   "foo.com",
					Path:   "/ignored",
				},
			},
			want: result{path: "/path"},
		},

		// Opaque without the leading "//":
		4: {
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Opaque: "/path",
					Host:   "foo.com",
					Path:   "/ignored",
				},
			},
			want: result{path: "/path"},
		},

		// Opaque we can't handle:
		5: {
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Scheme: "https",
					Opaque: "//unknown_host/path",
					Host:   "foo.com",
					Path:   "/ignored",
				},
			},
			want: result{err: `invalid request :path "https://unknown_host/path" from URL.Opaque = "//unknown_host/path"`},
		},

		// A CONNECT request:
		6: {
			req: &http.Request{
				Method: "CONNECT",
				URL: &url.URL{
					Host: "foo.com",
				},
			},
			want: result{},
		},
	}
	for i, tt := range tests {
		cc := &ClientConn{peerMaxHeaderListSize: 0xffffffffffffffff}
		cc.henc = hpack.NewEncoder(&cc.hbuf)
		cc.mu.Lock()
		hdrs, err := cc.encodeHeaders(tt.req, false, "", -1)
		cc.mu.Unlock()
		var got result
		hpackDec := hpack.NewDecoder(initialHeaderTableSize, func(f hpack.HeaderField) {
			if f.Name == ":path" {
				got.path = f.Value
			}
		})
		if err != nil {
			got.err = err.Error()
		} else if len(hdrs) > 0 {
			if _, err := hpackDec.Write(hdrs); err != nil {
				t.Errorf("%d. bogus hpack: %v", i, err)
				continue
			}
		}
		if got != tt.want {
			t.Errorf("%d. got %+v; want %+v", i, got, tt.want)
		}

	}

}

// golang.org/issue/17071 -- don't sniff the first byte of the request body
// before we've determined that the ClientConn is usable.
func TestRoundTripDoesntConsumeRequestBodyEarly(t *testing.T) {
	const body = "foo"
	req, _ := http.NewRequest("POST", "http://foo.com/", ioutil.NopCloser(strings.NewReader(body)))
	cc := &ClientConn{
		closed: true,
	}
	_, err := cc.RoundTrip(req)
	if err != errClientConnUnusable {
		t.Fatalf("RoundTrip = %v; want errClientConnUnusable", err)
	}
	slurp, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Errorf("ReadAll = %v", err)
	}
	if string(slurp) != body {
		t.Errorf("Body = %q; want %q", slurp, body)
	}
}

func TestClientConnPing(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {}, optOnlyServer)
	defer st.Close()
	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	ctx := context.Background()
	cc, err := tr.dialClientConn(ctx, st.ts.Listener.Addr().String(), false)
	if err != nil {
		t.Fatal(err)
	}
	if err = cc.Ping(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Issue 16974: if the server sent a DATA frame after the user
// canceled the Transport's Request, the Transport previously wrote to a
// closed pipe, got an error, and ended up closing the whole TCP
// connection.
func TestTransportCancelDataResponseRace(t *testing.T) {
	cancel := make(chan struct{})
	clientGotError := make(chan bool, 1)

	const msg = "Hello."
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/hello") {
			time.Sleep(50 * time.Millisecond)
			io.WriteString(w, msg)
			return
		}
		for i := 0; i < 50; i++ {
			io.WriteString(w, "Some data.")
			w.(http.Flusher).Flush()
			if i == 2 {
				close(cancel)
				<-clientGotError
			}
			time.Sleep(10 * time.Millisecond)
		}
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	c := &http.Client{Transport: tr}
	req, _ := http.NewRequest("GET", st.ts.URL, nil)
	req.Cancel = cancel
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = io.Copy(ioutil.Discard, res.Body); err == nil {
		t.Fatal("unexpected success")
	}
	clientGotError <- true

	res, err = c.Get(st.ts.URL + "/hello")
	if err != nil {
		t.Fatal(err)
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != msg {
		t.Errorf("Got = %q; want %q", slurp, msg)
	}
}

// Issue 21316: It should be safe to reuse an http.Request after the
// request has completed.
func TestTransportNoRaceOnRequestObjectAfterRequestComplete(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		io.WriteString(w, "body")
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	req, _ := http.NewRequest("GET", st.ts.URL, nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = io.Copy(ioutil.Discard, resp.Body); err != nil {
		t.Fatalf("error reading response body: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("error closing response body: %v", err)
	}

	// This access of req.Header should not race with code in the transport.
	req.Header = http.Header{}
}

func TestTransportCloseAfterLostPing(t *testing.T) {
	clientDone := make(chan struct{})
	ct := newClientTester(t)
	ct.tr.PingTimeout = 1 * time.Second
	ct.tr.ReadIdleTimeout = 1 * time.Second
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		defer close(clientDone)
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		_, err := ct.tr.RoundTrip(req)
		if err == nil || !strings.Contains(err.Error(), "client connection lost") {
			return fmt.Errorf("expected to get error about \"connection lost\", got %v", err)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		<-clientDone
		return nil
	}
	ct.run()
}

func TestTransportPingWhenReading(t *testing.T) {
	testCases := []struct {
		name              string
		readIdleTimeout   time.Duration
		deadline          time.Duration
		expectedPingCount int
	}{
		{
			name:              "two pings",
			readIdleTimeout:   100 * time.Millisecond,
			deadline:          time.Second,
			expectedPingCount: 2,
		},
		{
			name:              "zero ping",
			readIdleTimeout:   time.Second,
			deadline:          200 * time.Millisecond,
			expectedPingCount: 0,
		},
		{
			name:              "0 readIdleTimeout means no ping",
			readIdleTimeout:   0 * time.Millisecond,
			deadline:          500 * time.Millisecond,
			expectedPingCount: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc // capture range variable
		t.Run(tc.name, func(t *testing.T) {
			testTransportPingWhenReading(t, tc.readIdleTimeout, tc.deadline, tc.expectedPingCount)
		})
	}
}

func testTransportPingWhenReading(t *testing.T, readIdleTimeout, deadline time.Duration, expectedPingCount int) {
	var pingCount int
	ct := newClientTester(t)
	ct.tr.PingTimeout = 10 * time.Millisecond
	ct.tr.ReadIdleTimeout = readIdleTimeout

	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		req, _ := http.NewRequestWithContext(ctx, "GET", "https://dummy.tld/", nil)
		res, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("status code = %v; want %v", res.StatusCode, 200)
		}
		_, err = ioutil.ReadAll(res.Body)
		if expectedPingCount == 0 && errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil
		}

		cancel()
		return err
	}

	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		var streamID uint32
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-ctx.Done():
					// If the client's done, it
					// will have reported any
					// errors on its side.
					return nil
				default:
					return err
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				if !f.HeadersEnded() {
					return fmt.Errorf("headers should have END_HEADERS be ended: %v", f)
				}
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: strconv.Itoa(200)})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.StreamID,
					EndHeaders:    true,
					EndStream:     false,
					BlockFragment: buf.Bytes(),
				})
				streamID = f.StreamID
			case *PingFrame:
				pingCount++
				if pingCount == expectedPingCount {
					if err := ct.fr.WriteData(streamID, true, []byte("hello, this is last server data frame")); err != nil {
						return err
					}
				}
				if err := ct.fr.WritePing(true, f.Data); err != nil {
					return err
				}
			case *RSTStreamFrame:
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

func TestTransportRetryAfterGOAWAY(t *testing.T) {
	var dialer struct {
		sync.Mutex
		count int
	}
	ct1 := make(chan *clientTester)
	ct2 := make(chan *clientTester)

	ln := newLocalListener(t)
	defer ln.Close()

	tr := &Transport{
		TLSClientConfig: tlsConfigInsecure,
	}
	tr.DialTLS = func(network, addr string, cfg *tls.Config) (net.Conn, error) {
		dialer.Lock()
		defer dialer.Unlock()
		dialer.count++
		if dialer.count == 3 {
			return nil, errors.New("unexpected number of dials")
		}
		cc, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			return nil, fmt.Errorf("dial error: %v", err)
		}
		sc, err := ln.Accept()
		if err != nil {
			return nil, fmt.Errorf("accept error: %v", err)
		}
		ct := &clientTester{
			t:  t,
			tr: tr,
			cc: cc,
			sc: sc,
			fr: NewFramer(sc, sc),
		}
		switch dialer.count {
		case 1:
			ct1 <- ct
		case 2:
			ct2 <- ct
		}
		return cc, nil
	}

	errs := make(chan error, 3)

	// Client.
	go func() {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		res, err := tr.RoundTrip(req)
		if res != nil {
			res.Body.Close()
			if got := res.Header.Get("Foo"); got != "bar" {
				err = fmt.Errorf("foo header = %q; want bar", got)
			}
		}
		if err != nil {
			err = fmt.Errorf("RoundTrip: %v", err)
		}
		errs <- err
	}()

	connToClose := make(chan io.Closer, 2)

	// Server for the first request.
	go func() {
		ct := <-ct1

		connToClose <- ct.cc
		ct.greet()
		hf, err := ct.firstHeaders()
		if err != nil {
			errs <- fmt.Errorf("server1 failed reading HEADERS: %v", err)
			return
		}
		t.Logf("server1 got %v", hf)
		if err := ct.fr.WriteGoAway(0 /*max id*/, ErrCodeNo, nil); err != nil {
			errs <- fmt.Errorf("server1 failed writing GOAWAY: %v", err)
			return
		}
		errs <- nil
	}()

	// Server for the second request.
	go func() {
		ct := <-ct2

		connToClose <- ct.cc
		ct.greet()
		hf, err := ct.firstHeaders()
		if err != nil {
			errs <- fmt.Errorf("server2 failed reading HEADERS: %v", err)
			return
		}
		t.Logf("server2 got %v", hf)

		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		enc.WriteField(hpack.HeaderField{Name: "foo", Value: "bar"})
		err = ct.fr.WriteHeaders(HeadersFrameParam{
			StreamID:      hf.StreamID,
			EndHeaders:    true,
			EndStream:     false,
			BlockFragment: buf.Bytes(),
		})
		if err != nil {
			errs <- fmt.Errorf("server2 failed writing response HEADERS: %v", err)
		} else {
			errs <- nil
		}
	}()

	for k := 0; k < 3; k++ {
		err := <-errs
		if err != nil {
			t.Error(err)
		}
	}

	close(connToClose)
	for c := range connToClose {
		c.Close()
	}
}

func TestTransportRetryAfterRefusedStream(t *testing.T) {
	clientDone := make(chan struct{})
	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		defer close(clientDone)
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		resp, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != 204 {
			return fmt.Errorf("Status = %v; want 204", resp.StatusCode)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		nreq := 0

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-clientDone:
					// If the client's done, it
					// will have reported any
					// errors on its side.
					return nil
				default:
					return err
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				if !f.HeadersEnded() {
					return fmt.Errorf("headers should have END_HEADERS be ended: %v", f)
				}
				nreq++
				if nreq == 1 {
					ct.fr.WriteRSTStream(f.StreamID, ErrCodeRefusedStream)
				} else {
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "204"})
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.StreamID,
						EndHeaders:    true,
						EndStream:     true,
						BlockFragment: buf.Bytes(),
					})
				}
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

func TestTransportRetryHasLimit(t *testing.T) {
	// Skip in short mode because the total expected delay is 1s+2s+4s+8s+16s=29s.
	if testing.Short() {
		t.Skip("skipping long test in short mode")
	}
	clientDone := make(chan struct{})
	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		defer close(clientDone)
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		resp, err := ct.tr.RoundTrip(req)
		if err == nil {
			return fmt.Errorf("RoundTrip expected error, got response: %+v", resp)
		}
		t.Logf("expected error, got: %v", err)
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-clientDone:
					// If the client's done, it
					// will have reported any
					// errors on its side.
					return nil
				default:
					return err
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				if !f.HeadersEnded() {
					return fmt.Errorf("headers should have END_HEADERS be ended: %v", f)
				}
				ct.fr.WriteRSTStream(f.StreamID, ErrCodeRefusedStream)
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

func TestTransportResponseDataBeforeHeaders(t *testing.T) {
	// This test use not valid response format.
	// Discarding logger output to not spam tests output.
	log.SetOutput(ioutil.Discard)
	defer log.SetOutput(os.Stderr)

	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		req := httptest.NewRequest("GET", "https://dummy.tld/", nil)
		// First request is normal to ensure the check is per stream and not per connection.
		_, err := ct.tr.RoundTrip(req)
		if err != nil {
			return fmt.Errorf("RoundTrip expected no error, got: %v", err)
		}
		// Second request returns a DATA frame with no HEADERS.
		resp, err := ct.tr.RoundTrip(req)
		if err == nil {
			return fmt.Errorf("RoundTrip expected error, got response: %+v", resp)
		}
		if err, ok := err.(StreamError); !ok || err.Code != ErrCodeProtocol {
			return fmt.Errorf("expected stream PROTOCOL_ERROR, got: %v", err)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		for {
			f, err := ct.fr.ReadFrame()
			if err == io.EOF {
				return nil
			} else if err != nil {
				return err
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
				switch f.StreamID {
				case 1:
					// Send a valid response to first request.
					var buf bytes.Buffer
					enc := hpack.NewEncoder(&buf)
					enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
					ct.fr.WriteHeaders(HeadersFrameParam{
						StreamID:      f.StreamID,
						EndHeaders:    true,
						EndStream:     true,
						BlockFragment: buf.Bytes(),
					})
				case 3:
					ct.fr.WriteData(f.StreamID, true, []byte("payload"))
				}
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

// tests Transport.StrictMaxConcurrentStreams
func TestTransportRequestsStallAtServerLimit(t *testing.T) {
	const maxConcurrent = 2

	greet := make(chan struct{})      // server sends initial SETTINGS frame
	gotRequest := make(chan struct{}) // server received a request
	clientDone := make(chan struct{})

	// Collect errors from goroutines.
	var wg sync.WaitGroup
	errs := make(chan error, 100)
	defer func() {
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	}()

	// We will send maxConcurrent+2 requests. This checker goroutine waits for the
	// following stages:
	//   1. The first maxConcurrent requests are received by the server.
	//   2. The client will cancel the next request
	//   3. The server is unblocked so it can service the first maxConcurrent requests
	//   4. The client will send the final request
	wg.Add(1)
	unblockClient := make(chan struct{})
	clientRequestCancelled := make(chan struct{})
	unblockServer := make(chan struct{})
	go func() {
		defer wg.Done()
		// Stage 1.
		for k := 0; k < maxConcurrent; k++ {
			<-gotRequest
		}
		// Stage 2.
		close(unblockClient)
		<-clientRequestCancelled
		// Stage 3: give some time for the final RoundTrip call to be scheduled and
		// verify that the final request is not sent.
		time.Sleep(50 * time.Millisecond)
		select {
		case <-gotRequest:
			errs <- errors.New("last request did not stall")
			close(unblockServer)
			return
		default:
		}
		close(unblockServer)
		// Stage 4.
		<-gotRequest
	}()

	ct := newClientTester(t)
	ct.tr.StrictMaxConcurrentStreams = true
	ct.client = func() error {
		var wg sync.WaitGroup
		defer func() {
			wg.Wait()
			close(clientDone)
			ct.cc.(*net.TCPConn).CloseWrite()
			if runtime.GOOS == "plan9" {
				// CloseWrite not supported on Plan 9; Issue 17906
				ct.cc.(*net.TCPConn).Close()
			}
		}()
		for k := 0; k < maxConcurrent+2; k++ {
			wg.Add(1)
			go func(k int) {
				defer wg.Done()
				// Don't send the second request until after receiving SETTINGS from the server
				// to avoid a race where we use the default SettingMaxConcurrentStreams, which
				// is much larger than maxConcurrent. We have to send the first request before
				// waiting because the first request triggers the dial and greet.
				if k > 0 {
					<-greet
				}
				// Block until maxConcurrent requests are sent before sending any more.
				if k >= maxConcurrent {
					<-unblockClient
				}
				req, _ := http.NewRequest("GET", fmt.Sprintf("https://dummy.tld/%d", k), nil)
				if k == maxConcurrent {
					// This request will be canceled.
					cancel := make(chan struct{})
					req.Cancel = cancel
					close(cancel)
					_, err := ct.tr.RoundTrip(req)
					close(clientRequestCancelled)
					if err == nil {
						errs <- fmt.Errorf("RoundTrip(%d) should have failed due to cancel", k)
						return
					}
				} else {
					resp, err := ct.tr.RoundTrip(req)
					if err != nil {
						errs <- fmt.Errorf("RoundTrip(%d): %v", k, err)
						return
					}
					ioutil.ReadAll(resp.Body)
					resp.Body.Close()
					if resp.StatusCode != 204 {
						errs <- fmt.Errorf("Status = %v; want 204", resp.StatusCode)
						return
					}
				}
			}(k)
		}
		return nil
	}

	ct.server = func() error {
		var wg sync.WaitGroup
		defer wg.Wait()

		ct.greet(Setting{SettingMaxConcurrentStreams, maxConcurrent})

		// Server write loop.
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		writeResp := make(chan uint32, maxConcurrent+1)

		wg.Add(1)
		go func() {
			defer wg.Done()
			<-unblockServer
			for id := range writeResp {
				buf.Reset()
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "204"})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      id,
					EndHeaders:    true,
					EndStream:     true,
					BlockFragment: buf.Bytes(),
				})
			}
		}()

		// Server read loop.
		var nreq int
		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				select {
				case <-clientDone:
					// If the client's done, it will have reported any errors on its side.
					return nil
				default:
					return err
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame:
			case *SettingsFrame:
				// Wait for the client SETTINGS ack until ending the greet.
				close(greet)
			case *HeadersFrame:
				if !f.HeadersEnded() {
					return fmt.Errorf("headers should have END_HEADERS be ended: %v", f)
				}
				gotRequest <- struct{}{}
				nreq++
				writeResp <- f.StreamID
				if nreq == maxConcurrent+1 {
					close(writeResp)
				}
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}

	ct.run()
}

func TestAuthorityAddr(t *testing.T) {
	tests := []struct {
		scheme, authority string
		want              string
	}{
		{"http", "foo.com", "foo.com:80"},
		{"https", "foo.com", "foo.com:443"},
		{"https", "foo.com:1234", "foo.com:1234"},
		{"https", "1.2.3.4:1234", "1.2.3.4:1234"},
		{"https", "1.2.3.4", "1.2.3.4:443"},
		{"https", "[::1]:1234", "[::1]:1234"},
		{"https", "[::1]", "[::1]:443"},
	}
	for _, tt := range tests {
		got := authorityAddr(tt.scheme, tt.authority)
		if got != tt.want {
			t.Errorf("authorityAddr(%q, %q) = %q; want %q", tt.scheme, tt.authority, got, tt.want)
		}
	}
}

// Issue 20448: stop allocating for DATA frames' payload after
// Response.Body.Close is called.
func TestTransportAllocationsAfterResponseBodyClose(t *testing.T) {
	megabyteZero := make([]byte, 1<<20)

	writeErr := make(chan error, 1)

	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		var sum int64
		for i := 0; i < 100; i++ {
			n, err := w.Write(megabyteZero)
			sum += int64(n)
			if err != nil {
				writeErr <- err
				return
			}
		}
		t.Logf("wrote all %d bytes", sum)
		writeErr <- nil
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}
	res, err := c.Get(st.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	var buf [1]byte
	if _, err := res.Body.Read(buf[:]); err != nil {
		t.Error(err)
	}
	if err := res.Body.Close(); err != nil {
		t.Error(err)
	}

	trb, ok := res.Body.(transportResponseBody)
	if !ok {
		t.Fatalf("res.Body = %T; want transportResponseBody", res.Body)
	}
	if trb.cs.bufPipe.b != nil {
		t.Errorf("response body pipe is still open")
	}

	gotErr := <-writeErr
	if gotErr == nil {
		t.Errorf("Handler unexpectedly managed to write its entire response without getting an error")
	} else if gotErr != errStreamClosed {
		t.Errorf("Handler Write err = %v; want errStreamClosed", gotErr)
	}
}

// Issue 18891: make sure Request.Body == NoBody means no DATA frame
// is ever sent, even if empty.
func TestTransportNoBodyMeansNoDATA(t *testing.T) {
	ct := newClientTester(t)

	unblockClient := make(chan bool)

	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", http.NoBody)
		ct.tr.RoundTrip(req)
		<-unblockClient
		return nil
	}
	ct.server = func() error {
		defer close(unblockClient)
		defer ct.cc.(*net.TCPConn).Close()
		ct.greet()

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return fmt.Errorf("ReadFrame while waiting for Headers: %v", err)
			}
			switch f := f.(type) {
			default:
				return fmt.Errorf("Got %T; want HeadersFrame", f)
			case *WindowUpdateFrame, *SettingsFrame:
				continue
			case *HeadersFrame:
				if !f.StreamEnded() {
					return fmt.Errorf("got headers frame without END_STREAM")
				}
				return nil
			}
		}
	}
	ct.run()
}

func benchSimpleRoundTrip(b *testing.B, nReqHeaders, nResHeader int) {
	defer disableGoroutineTracking()()
	b.ReportAllocs()
	st := newServerTester(b,
		func(w http.ResponseWriter, r *http.Request) {
			for i := 0; i < nResHeader; i++ {
				name := fmt.Sprint("A-", i)
				w.Header().Set(name, "*")
			}
		},
		optOnlyServer,
		optQuiet,
	)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	req, err := http.NewRequest("GET", st.ts.URL, nil)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < nReqHeaders; i++ {
		name := fmt.Sprint("A-", i)
		req.Header.Set(name, "*")
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res, err := tr.RoundTrip(req)
		if err != nil {
			if res != nil {
				res.Body.Close()
			}
			b.Fatalf("RoundTrip err = %v; want nil", err)
		}
		res.Body.Close()
		if res.StatusCode != http.StatusOK {
			b.Fatalf("Response code = %v; want %v", res.StatusCode, http.StatusOK)
		}
	}
}

type infiniteReader struct{}

func (r infiniteReader) Read(b []byte) (int, error) {
	return len(b), nil
}

// Issue 20521: it is not an error to receive a response and end stream
// from the server without the body being consumed.
func TestTransportResponseAndResetWithoutConsumingBodyRace(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	// The request body needs to be big enough to trigger flow control.
	req, _ := http.NewRequest("PUT", st.ts.URL, infiniteReader{})
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Fatalf("Response code = %v; want %v", res.StatusCode, http.StatusOK)
	}
}

// Verify transport doesn't crash when receiving bogus response lacking a :status header.
// Issue 22880.
func TestTransportHandlesInvalidStatuslessResponse(t *testing.T) {
	ct := newClientTester(t)
	ct.client = func() error {
		req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
		_, err := ct.tr.RoundTrip(req)
		const substr = "malformed response from server: missing status pseudo header"
		if !strings.Contains(fmt.Sprint(err), substr) {
			return fmt.Errorf("RoundTrip error = %v; want substring %q", err, substr)
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}
			switch f := f.(type) {
			case *HeadersFrame:
				enc.WriteField(hpack.HeaderField{Name: "content-type", Value: "text/html"}) // no :status header
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.StreamID,
					EndHeaders:    true,
					EndStream:     false, // we'll send some DATA to try to crash the transport
					BlockFragment: buf.Bytes(),
				})
				ct.fr.WriteData(f.StreamID, true, []byte("payload"))
				return nil
			}
		}
	}
	ct.run()
}

func BenchmarkClientRequestHeaders(b *testing.B) {
	b.Run("   0 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 0, 0) })
	b.Run("  10 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 10, 0) })
	b.Run(" 100 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 100, 0) })
	b.Run("1000 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 1000, 0) })
}

func BenchmarkClientResponseHeaders(b *testing.B) {
	b.Run("   0 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 0, 0) })
	b.Run("  10 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 0, 10) })
	b.Run(" 100 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 0, 100) })
	b.Run("1000 Headers", func(b *testing.B) { benchSimpleRoundTrip(b, 0, 1000) })
}

func activeStreams(cc *ClientConn) int {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return len(cc.streams)
}

type closeMode int

const (
	closeAtHeaders closeMode = iota
	closeAtBody
	shutdown
	shutdownCancel
)

// See golang.org/issue/17292
func testClientConnClose(t *testing.T, closeMode closeMode) {
	clientDone := make(chan struct{})
	defer close(clientDone)
	handlerDone := make(chan struct{})
	closeDone := make(chan struct{})
	beforeHeader := func() {}
	bodyWrite := func(w http.ResponseWriter) {}
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		beforeHeader()
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		bodyWrite(w)
		select {
		case <-w.(http.CloseNotifier).CloseNotify():
			// client closed connection before completion
			if closeMode == shutdown || closeMode == shutdownCancel {
				t.Error("expected request to complete")
			}
		case <-clientDone:
			if closeMode == closeAtHeaders || closeMode == closeAtBody {
				t.Error("expected connection closed by client")
			}
		}
	}, optOnlyServer)
	defer st.Close()
	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	ctx := context.Background()
	cc, err := tr.dialClientConn(ctx, st.ts.Listener.Addr().String(), false)
	req, err := http.NewRequest("GET", st.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	if closeMode == closeAtHeaders {
		beforeHeader = func() {
			if err := cc.Close(); err != nil {
				t.Error(err)
			}
			close(closeDone)
		}
	}
	var sendBody chan struct{}
	if closeMode == closeAtBody {
		sendBody = make(chan struct{})
		bodyWrite = func(w http.ResponseWriter) {
			<-sendBody
			b := make([]byte, 32)
			w.Write(b)
			w.(http.Flusher).Flush()
			if err := cc.Close(); err != nil {
				t.Errorf("unexpected ClientConn close error: %v", err)
			}
			close(closeDone)
			w.Write(b)
			w.(http.Flusher).Flush()
		}
	}
	res, err := cc.RoundTrip(req)
	if res != nil {
		defer res.Body.Close()
	}
	if closeMode == closeAtHeaders {
		got := fmt.Sprint(err)
		want := "http2: client connection force closed via ClientConn.Close"
		if got != want {
			t.Fatalf("RoundTrip error = %v, want %v", got, want)
		}
	} else {
		if err != nil {
			t.Fatalf("RoundTrip: %v", err)
		}
		if got, want := activeStreams(cc), 1; got != want {
			t.Errorf("got %d active streams, want %d", got, want)
		}
	}
	switch closeMode {
	case shutdownCancel:
		if err = cc.Shutdown(canceledCtx); err != context.Canceled {
			t.Errorf("got %v, want %v", err, context.Canceled)
		}
		if cc.closing == false {
			t.Error("expected closing to be true")
		}
		if cc.CanTakeNewRequest() == true {
			t.Error("CanTakeNewRequest to return false")
		}
		if v, want := len(cc.streams), 1; v != want {
			t.Errorf("expected %d active streams, got %d", want, v)
		}
		clientDone <- struct{}{}
		<-handlerDone
	case shutdown:
		wait := make(chan struct{})
		shutdownEnterWaitStateHook = func() {
			close(wait)
			shutdownEnterWaitStateHook = func() {}
		}
		defer func() { shutdownEnterWaitStateHook = func() {} }()
		shutdown := make(chan struct{}, 1)
		go func() {
			if err = cc.Shutdown(context.Background()); err != nil {
				t.Error(err)
			}
			close(shutdown)
		}()
		// Let the shutdown to enter wait state
		<-wait
		cc.mu.Lock()
		if cc.closing == false {
			t.Error("expected closing to be true")
		}
		cc.mu.Unlock()
		if cc.CanTakeNewRequest() == true {
			t.Error("CanTakeNewRequest to return false")
		}
		if got, want := activeStreams(cc), 1; got != want {
			t.Errorf("got %d active streams, want %d", got, want)
		}
		// Let the active request finish
		clientDone <- struct{}{}
		// Wait for the shutdown to end
		select {
		case <-shutdown:
		case <-time.After(2 * time.Second):
			t.Fatal("expected server connection to close")
		}
	case closeAtHeaders, closeAtBody:
		if closeMode == closeAtBody {
			go close(sendBody)
			if _, err := io.Copy(ioutil.Discard, res.Body); err == nil {
				t.Error("expected a Copy error, got nil")
			}
		}
		<-closeDone
		if got, want := activeStreams(cc), 0; got != want {
			t.Errorf("got %d active streams, want %d", got, want)
		}
		// wait for server to get the connection close notice
		select {
		case <-handlerDone:
		case <-time.After(2 * time.Second):
			t.Fatal("expected server connection to close")
		}
	}
}

// The client closes the connection just after the server got the client's HEADERS
// frame, but before the server sends its HEADERS response back. The expected
// result is an error on RoundTrip explaining the client closed the connection.
func TestClientConnCloseAtHeaders(t *testing.T) {
	testClientConnClose(t, closeAtHeaders)
}

// The client closes the connection between two server's response DATA frames.
// The expected behavior is a response body io read error on the client.
func TestClientConnCloseAtBody(t *testing.T) {
	testClientConnClose(t, closeAtBody)
}

// The client sends a GOAWAY frame before the server finished processing a request.
// We expect the connection not to close until the request is completed.
func TestClientConnShutdown(t *testing.T) {
	testClientConnClose(t, shutdown)
}

// The client sends a GOAWAY frame before the server finishes processing a request,
// but cancels the passed context before the request is completed. The expected
// behavior is the client closing the connection after the context is canceled.
func TestClientConnShutdownCancel(t *testing.T) {
	testClientConnClose(t, shutdownCancel)
}

// Issue 25009: use Request.GetBody if present, even if it seems like
// we might not need it. Apparently something else can still read from
// the original request body. Data race? In any case, rewinding
// unconditionally on retry is a nicer model anyway and should
// simplify code in the future (after the Go 1.11 freeze)
func TestTransportUsesGetBodyWhenPresent(t *testing.T) {
	calls := 0
	someBody := func() io.ReadCloser {
		return struct{ io.ReadCloser }{ioutil.NopCloser(bytes.NewReader(nil))}
	}
	req := &http.Request{
		Body: someBody(),
		GetBody: func() (io.ReadCloser, error) {
			calls++
			return someBody(), nil
		},
	}

	afterBodyWrite := false // pretend we haven't read+written the body yet
	req2, err := shouldRetryRequest(req, errClientConnUnusable, afterBodyWrite)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("Calls = %d; want 1", calls)
	}
	if req2 == req {
		t.Error("req2 changed")
	}
	if req2 == nil {
		t.Fatal("req2 is nil")
	}
	if req2.Body == nil {
		t.Fatal("req2.Body is nil")
	}
	if req2.GetBody == nil {
		t.Fatal("req2.GetBody is nil")
	}
	if req2.Body == req.Body {
		t.Error("req2.Body unchanged")
	}
}

// Issue 22891: verify that the "https" altproto we register with net/http
// is a certain type: a struct with one field with our *http2.Transport in it.
func TestNoDialH2RoundTripperType(t *testing.T) {
	t1 := new(http.Transport)
	t2 := new(Transport)
	rt := noDialH2RoundTripper{t2}
	if err := registerHTTPSProtocol(t1, rt); err != nil {
		t.Fatal(err)
	}
	rv := reflect.ValueOf(rt)
	if rv.Type().Kind() != reflect.Struct {
		t.Fatalf("kind = %v; net/http expects struct", rv.Type().Kind())
	}
	if n := rv.Type().NumField(); n != 1 {
		t.Fatalf("fields = %d; net/http expects 1", n)
	}
	v := rv.Field(0)
	if _, ok := v.Interface().(*Transport); !ok {
		t.Fatalf("wrong kind %T; want *Transport", v.Interface())
	}
}

type errReader struct {
	body []byte
	err  error
}

func (r *errReader) Read(p []byte) (int, error) {
	if len(r.body) > 0 {
		n := copy(p, r.body)
		r.body = r.body[n:]
		return n, nil
	}
	return 0, r.err
}

func testTransportBodyReadError(t *testing.T, body []byte) {
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		// So far we've only seen this be flaky on Windows and Plan 9,
		// perhaps due to TCP behavior on shutdowns while
		// unread data is in flight. This test should be
		// fixed, but a skip is better than annoying people
		// for now.
		t.Skipf("skipping flaky test on %s; https://golang.org/issue/31260", runtime.GOOS)
	}
	clientDone := make(chan struct{})
	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		defer close(clientDone)

		checkNoStreams := func() error {
			cp, ok := ct.tr.connPool().(*clientConnPool)
			if !ok {
				return fmt.Errorf("conn pool is %T; want *clientConnPool", ct.tr.connPool())
			}
			cp.mu.Lock()
			defer cp.mu.Unlock()
			conns, ok := cp.conns["dummy.tld:443"]
			if !ok {
				return fmt.Errorf("missing connection")
			}
			if len(conns) != 1 {
				return fmt.Errorf("conn pool size: %v; expect 1", len(conns))
			}
			if activeStreams(conns[0]) != 0 {
				return fmt.Errorf("active streams count: %v; want 0", activeStreams(conns[0]))
			}
			return nil
		}
		bodyReadError := errors.New("body read error")
		body := &errReader{body, bodyReadError}
		req, err := http.NewRequest("PUT", "https://dummy.tld/", body)
		if err != nil {
			return err
		}
		_, err = ct.tr.RoundTrip(req)
		if err != bodyReadError {
			return fmt.Errorf("err = %v; want %v", err, bodyReadError)
		}
		if err = checkNoStreams(); err != nil {
			return err
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()
		var receivedBody []byte
		var resetCount int
		for {
			f, err := ct.fr.ReadFrame()
			t.Logf("server: ReadFrame = %v, %v", f, err)
			if err != nil {
				select {
				case <-clientDone:
					// If the client's done, it
					// will have reported any
					// errors on its side.
					if bytes.Compare(receivedBody, body) != 0 {
						return fmt.Errorf("body: %q; expected %q", receivedBody, body)
					}
					if resetCount != 1 {
						return fmt.Errorf("stream reset count: %v; expected: 1", resetCount)
					}
					return nil
				default:
					return err
				}
			}
			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
			case *DataFrame:
				receivedBody = append(receivedBody, f.Data()...)
			case *RSTStreamFrame:
				resetCount++
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

func TestTransportBodyReadError_Immediately(t *testing.T) { testTransportBodyReadError(t, nil) }
func TestTransportBodyReadError_Some(t *testing.T)        { testTransportBodyReadError(t, []byte("123")) }

// Issue 32254: verify that the client sends END_STREAM flag eagerly with the last
// (or in this test-case the only one) request body data frame, and does not send
// extra zero-len data frames.
func TestTransportBodyEagerEndStream(t *testing.T) {
	const reqBody = "some request body"
	const resBody = "some response body"

	ct := newClientTester(t)
	ct.client = func() error {
		defer ct.cc.(*net.TCPConn).CloseWrite()
		if runtime.GOOS == "plan9" {
			// CloseWrite not supported on Plan 9; Issue 17906
			defer ct.cc.(*net.TCPConn).Close()
		}
		body := strings.NewReader(reqBody)
		req, err := http.NewRequest("PUT", "https://dummy.tld/", body)
		if err != nil {
			return err
		}
		_, err = ct.tr.RoundTrip(req)
		if err != nil {
			return err
		}
		return nil
	}
	ct.server = func() error {
		ct.greet()

		for {
			f, err := ct.fr.ReadFrame()
			if err != nil {
				return err
			}

			switch f := f.(type) {
			case *WindowUpdateFrame, *SettingsFrame:
			case *HeadersFrame:
			case *DataFrame:
				if !f.StreamEnded() {
					ct.fr.WriteRSTStream(f.StreamID, ErrCodeRefusedStream)
					return fmt.Errorf("data frame without END_STREAM %v", f)
				}
				var buf bytes.Buffer
				enc := hpack.NewEncoder(&buf)
				enc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
				ct.fr.WriteHeaders(HeadersFrameParam{
					StreamID:      f.Header().StreamID,
					EndHeaders:    true,
					EndStream:     false,
					BlockFragment: buf.Bytes(),
				})
				ct.fr.WriteData(f.StreamID, true, []byte(resBody))
				return nil
			case *RSTStreamFrame:
			default:
				return fmt.Errorf("Unexpected client frame %v", f)
			}
		}
	}
	ct.run()
}

type chunkReader struct {
	chunks [][]byte
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(r.chunks) > 0 {
		n := copy(p, r.chunks[0])
		r.chunks = r.chunks[1:]
		return n, nil
	}
	panic("shouldn't read this many times")
}

// Issue 32254: if the request body is larger than the specified
// content length, the client should refuse to send the extra part
// and abort the stream.
//
// In _len3 case, the first Read() matches the expected content length
// but the second read returns more data.
//
// In _len2 case, the first Read() exceeds the expected content length.
func TestTransportBodyLargerThanSpecifiedContentLength_len3(t *testing.T) {
	body := &chunkReader{[][]byte{
		[]byte("123"),
		[]byte("456"),
	}}
	testTransportBodyLargerThanSpecifiedContentLength(t, body, 3)
}

func TestTransportBodyLargerThanSpecifiedContentLength_len2(t *testing.T) {
	body := &chunkReader{[][]byte{
		[]byte("123"),
	}}
	testTransportBodyLargerThanSpecifiedContentLength(t, body, 2)
}

func testTransportBodyLargerThanSpecifiedContentLength(t *testing.T, body *chunkReader, contentLen int64) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		r.Body.Read(make([]byte, 6))
	}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()

	req, _ := http.NewRequest("POST", st.ts.URL, body)
	req.ContentLength = contentLen
	_, err := tr.RoundTrip(req)
	if err != errReqBodyTooLong {
		t.Fatalf("expected %v, got %v", errReqBodyTooLong, err)
	}
}

func TestClientConnTooIdle(t *testing.T) {
	tests := []struct {
		cc   func() *ClientConn
		want bool
	}{
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 5 * time.Second, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			true,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 5 * time.Second, lastIdle: time.Time{}}
			},
			false,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 60 * time.Second, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			false,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 0, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			false,
		},
	}
	for i, tt := range tests {
		got := tt.cc().tooIdleLocked()
		if got != tt.want {
			t.Errorf("%d. got %v; want %v", i, got, tt.want)
		}
	}
}

type fakeConnErr struct {
	net.Conn
	writeErr error
	closed   bool
}

func (fce *fakeConnErr) Write(b []byte) (n int, err error) {
	return 0, fce.writeErr
}

func (fce *fakeConnErr) Close() error {
	fce.closed = true
	return nil
}

// issue 39337: close the connection on a failed write
func TestTransportNewClientConnCloseOnWriteError(t *testing.T) {
	tr := &Transport{}
	writeErr := errors.New("write error")
	fakeConn := &fakeConnErr{writeErr: writeErr}
	_, err := tr.NewClientConn(fakeConn)
	if err != writeErr {
		t.Fatalf("expected %v, got %v", writeErr, err)
	}
	if !fakeConn.closed {
		t.Error("expected closed conn")
	}
}

func TestTransportRoundtripCloseOnWriteError(t *testing.T) {
	req, err := http.NewRequest("GET", "https://dummy.tld/", nil)
	if err != nil {
		t.Fatal(err)
	}
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {}, optOnlyServer)
	defer st.Close()

	tr := &Transport{TLSClientConfig: tlsConfigInsecure}
	defer tr.CloseIdleConnections()
	ctx := context.Background()
	cc, err := tr.dialClientConn(ctx, st.ts.Listener.Addr().String(), false)
	if err != nil {
		t.Fatal(err)
	}

	writeErr := errors.New("write error")
	cc.wmu.Lock()
	cc.werr = writeErr
	cc.wmu.Unlock()

	_, err = cc.RoundTrip(req)
	if err != writeErr {
		t.Fatalf("expected %v, got %v", writeErr, err)
	}

	cc.mu.Lock()
	closed := cc.closed
	cc.mu.Unlock()
	if !closed {
		t.Fatal("expected closed")
	}
}

// Issue 31192: A failed request may be retried if the body has not been read
// already. If the request body has started to be sent, one must wait until it
// is completed.
func TestTransportBodyRewindRace(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Connection", "close")
		w.WriteHeader(http.StatusOK)
		return
	}, optOnlyServer)
	defer st.Close()

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		MaxConnsPerHost: 1,
	}
	err := ConfigureTransport(tr)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{
		Transport: tr,
	}

	const clients = 50

	var wg sync.WaitGroup
	wg.Add(clients)
	for i := 0; i < clients; i++ {
		req, err := http.NewRequest("POST", st.ts.URL, bytes.NewBufferString("abcdef"))
		if err != nil {
			t.Fatalf("unexpect new request error: %v", err)
		}

		go func() {
			defer wg.Done()
			res, err := client.Do(req)
			if err == nil {
				res.Body.Close()
			}
		}()
	}

	wg.Wait()
}

// Issue 42498: A request with a body will never be sent if the stream is
// reset prior to sending any data.
func TestTransportServerResetStreamAtHeaders(t *testing.T) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}, optOnlyServer)
	defer st.Close()

	tr := &http.Transport{
		TLSClientConfig:       tlsConfigInsecure,
		MaxConnsPerHost:       1,
		ExpectContinueTimeout: 10 * time.Second,
	}

	err := ConfigureTransport(tr)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{
		Transport: tr,
	}

	req, err := http.NewRequest("POST", st.ts.URL, errorReader{io.EOF})
	if err != nil {
		t.Fatalf("unexpect new request error: %v", err)
	}
	req.ContentLength = 0 // so transport is tempted to sniff it
	req.Header.Set("Expect", "100-continue")
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}
