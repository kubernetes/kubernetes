// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
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

func startH2cServer(t *testing.T) net.Listener {
	h2Server := &Server{}
	l := newLocalListener(t)
	go func() {
		conn, err := l.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		h2Server.ServeConn(conn, &ServeConnOpts{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "Hello, %v", r.URL.Path)
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
	if got, want := string(body), "Hello, /foobar"; got != want {
		t.Fatalf("response got %v, want %v", got, want)
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

	req, err := http.NewRequest("GET", st.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	t.Logf("Got res: %+v", res)
	if g, w := res.StatusCode, 200; g != w {
		t.Errorf("StatusCode = %v; want %v", g, w)
	}
	if g, w := res.Status, "200 OK"; g != w {
		t.Errorf("Status = %q; want %q", g, w)
	}
	wantHeader := http.Header{
		"Content-Length": []string{"3"},
		"Content-Type":   []string{"text/plain; charset=utf-8"},
		"Date":           []string{"XXX"}, // see cleanDate
	}
	cleanDate(res)
	if !reflect.DeepEqual(res.Header, wantHeader) {
		t.Errorf("res Header = %v; want %v", res.Header, wantHeader)
	}
	if res.Request != req {
		t.Errorf("Response.Request = %p; want %p", res.Request, req)
	}
	if res.TLS == nil {
		t.Error("Response.TLS = nil; want non-nil")
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Errorf("Body read: %v", err)
	} else if string(slurp) != body {
		t.Errorf("Body = %q; want %q", slurp, body)
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

	done := make(chan struct{})
	requestMade := make(chan struct{})
	go func() {
		defer close(done)
		tr := &Transport{TLSClientConfig: tlsConfigInsecure}
		req, err := http.NewRequest("GET", st.ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		close(requestMade)
		_, err = ioutil.ReadAll(res.Body)
		if err == nil {
			t.Error("expected error from res.Body.Read")
		}
	}()

	<-requestMade
	// Now force the serve loop to end, via closing the connection.
	st.closeConn()
	// deadlock? that's a bug.
	select {
	case <-done:
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

func TestTransportBody(t *testing.T) {
	bodyTests := []struct {
		body         string
		noContentLen bool
	}{
		{body: "some message"},
		{body: "some message", noContentLen: true},
		{body: ""},
		{body: "", noContentLen: true},
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
	if err == errTransportVersion {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%#v", *t1); !strings.Contains(got, `"h2"`) {
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

func (ct *clientTester) greet() {
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
	if err := ct.fr.WriteSettings(); err != nil {
		ct.t.Fatal(err)
	}
	if err := ct.fr.WriteSettingsAck(); err != nil {
		ct.t.Fatal(err)
	}
}

func (ct *clientTester) cleanup() {
	ct.tr.CloseIdleConnections()
}

func (ct *clientTester) run() {
	errc := make(chan error, 2)
	ct.start("client", errc, ct.client)
	ct.start("server", errc, ct.server)
	defer ct.cleanup()
	for i := 0; i < 2; i++ {
		if err := <-errc; err != nil {
			ct.t.Error(err)
			return
		}
	}
}

func (ct *clientTester) start(which string, errc chan<- error, fn func() error) {
	go func() {
		finished := false
		var err error
		defer func() {
			if !finished {
				err = fmt.Errorf("%s goroutine didn't finish.", which)
			} else if err != nil {
				err = fmt.Errorf("%s: %v", which, err)
			}
			errc <- err
		}()
		err = fn()
		finished = true
	}()
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
	ct := newClientTester(t)
	ct.client = func() error {
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
				return err
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
					return nil
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
		if err != wantErr {
			return fmt.Errorf("res.Body ReadAll error = %q, %#v; want %T of %#v", slurp, err, wantErr, wantErr)
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

// Test that the the Transport returns a typed error from Response.Body.Read calls
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
			want:  "ERROR: http2: invalid Upgrade request header",
		},
		{
			key:   "Connection",
			value: []string{"foo"},
			want:  "ERROR: http2: invalid Connection request header",
		},
		{
			key:   "Connection",
			value: []string{"close"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Connection",
			value: []string{"close", "something-else"},
			want:  "ERROR: http2: invalid Connection request header",
		},
		{
			key:   "Connection",
			value: []string{"keep-alive"},
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
			want:  "ERROR: http2: invalid Transfer-Encoding request header",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"chunked"},
			want:  "Accept-Encoding,User-Agent",
		},
		{
			key:   "Transfer-Encoding",
			value: []string{"chunked", "other"},
			want:  "ERROR: http2: invalid Transfer-Encoding request header",
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
		tr := &Transport{TLSClientConfig: tt.conf}
		got := tr.newTLSConfig(tt.host)
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

	gd := runtime.NumGoroutine() - g0
	if gd > numReq/2 {
		t.Errorf("appeared to leak goroutines")
	}

}

// https://golang.org/issue/15930
func TestTransportFlowControl(t *testing.T) {
	const (
		total  = 100 << 20 // 100MB
		bufLen = 1 << 16
	)

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
			t.Errorf("RoundTrip error = %T: %#v, want %T (%#T)", err, err, want, want)
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
			ct.sc.Close()
			<-clientDone
			return nil
		}
	}
	ct.run()
}
