// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

var serverAddr string
var once sync.Once

func echoServer(ws *Conn) {
	defer ws.Close()
	io.Copy(ws, ws)
}

type Count struct {
	S string
	N int
}

func countServer(ws *Conn) {
	defer ws.Close()
	for {
		var count Count
		err := JSON.Receive(ws, &count)
		if err != nil {
			return
		}
		count.N++
		count.S = strings.Repeat(count.S, count.N)
		err = JSON.Send(ws, count)
		if err != nil {
			return
		}
	}
}

type testCtrlAndDataHandler struct {
	hybiFrameHandler
}

func (h *testCtrlAndDataHandler) WritePing(b []byte) (int, error) {
	h.hybiFrameHandler.conn.wio.Lock()
	defer h.hybiFrameHandler.conn.wio.Unlock()
	w, err := h.hybiFrameHandler.conn.frameWriterFactory.NewFrameWriter(PingFrame)
	if err != nil {
		return 0, err
	}
	n, err := w.Write(b)
	w.Close()
	return n, err
}

func ctrlAndDataServer(ws *Conn) {
	defer ws.Close()
	h := &testCtrlAndDataHandler{hybiFrameHandler: hybiFrameHandler{conn: ws}}
	ws.frameHandler = h

	go func() {
		for i := 0; ; i++ {
			var b []byte
			if i%2 != 0 { // with or without payload
				b = []byte(fmt.Sprintf("#%d-CONTROL-FRAME-FROM-SERVER", i))
			}
			if _, err := h.WritePing(b); err != nil {
				break
			}
			if _, err := h.WritePong(b); err != nil { // unsolicited pong
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	b := make([]byte, 128)
	for {
		n, err := ws.Read(b)
		if err != nil {
			break
		}
		if _, err := ws.Write(b[:n]); err != nil {
			break
		}
	}
}

func subProtocolHandshake(config *Config, req *http.Request) error {
	for _, proto := range config.Protocol {
		if proto == "chat" {
			config.Protocol = []string{proto}
			return nil
		}
	}
	return ErrBadWebSocketProtocol
}

func subProtoServer(ws *Conn) {
	for _, proto := range ws.Config().Protocol {
		io.WriteString(ws, proto)
	}
}

func startServer() {
	http.Handle("/echo", Handler(echoServer))
	http.Handle("/count", Handler(countServer))
	http.Handle("/ctrldata", Handler(ctrlAndDataServer))
	subproto := Server{
		Handshake: subProtocolHandshake,
		Handler:   Handler(subProtoServer),
	}
	http.Handle("/subproto", subproto)
	server := httptest.NewServer(nil)
	serverAddr = server.Listener.Addr().String()
	log.Print("Test WebSocket server listening on ", serverAddr)
}

func newConfig(t *testing.T, path string) *Config {
	config, _ := NewConfig(fmt.Sprintf("ws://%s%s", serverAddr, path), "http://localhost")
	return config
}

func TestEcho(t *testing.T) {
	once.Do(startServer)

	// websocket.Dial()
	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}
	conn, err := NewClient(newConfig(t, "/echo"), client)
	if err != nil {
		t.Errorf("WebSocket handshake error: %v", err)
		return
	}

	msg := []byte("hello, world\n")
	if _, err := conn.Write(msg); err != nil {
		t.Errorf("Write: %v", err)
	}
	var actual_msg = make([]byte, 512)
	n, err := conn.Read(actual_msg)
	if err != nil {
		t.Errorf("Read: %v", err)
	}
	actual_msg = actual_msg[0:n]
	if !bytes.Equal(msg, actual_msg) {
		t.Errorf("Echo: expected %q got %q", msg, actual_msg)
	}
	conn.Close()
}

func TestAddr(t *testing.T) {
	once.Do(startServer)

	// websocket.Dial()
	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}
	conn, err := NewClient(newConfig(t, "/echo"), client)
	if err != nil {
		t.Errorf("WebSocket handshake error: %v", err)
		return
	}

	ra := conn.RemoteAddr().String()
	if !strings.HasPrefix(ra, "ws://") || !strings.HasSuffix(ra, "/echo") {
		t.Errorf("Bad remote addr: %v", ra)
	}
	la := conn.LocalAddr().String()
	if !strings.HasPrefix(la, "http://") {
		t.Errorf("Bad local addr: %v", la)
	}
	conn.Close()
}

func TestCount(t *testing.T) {
	once.Do(startServer)

	// websocket.Dial()
	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}
	conn, err := NewClient(newConfig(t, "/count"), client)
	if err != nil {
		t.Errorf("WebSocket handshake error: %v", err)
		return
	}

	var count Count
	count.S = "hello"
	if err := JSON.Send(conn, count); err != nil {
		t.Errorf("Write: %v", err)
	}
	if err := JSON.Receive(conn, &count); err != nil {
		t.Errorf("Read: %v", err)
	}
	if count.N != 1 {
		t.Errorf("count: expected %d got %d", 1, count.N)
	}
	if count.S != "hello" {
		t.Errorf("count: expected %q got %q", "hello", count.S)
	}
	if err := JSON.Send(conn, count); err != nil {
		t.Errorf("Write: %v", err)
	}
	if err := JSON.Receive(conn, &count); err != nil {
		t.Errorf("Read: %v", err)
	}
	if count.N != 2 {
		t.Errorf("count: expected %d got %d", 2, count.N)
	}
	if count.S != "hellohello" {
		t.Errorf("count: expected %q got %q", "hellohello", count.S)
	}
	conn.Close()
}

func TestWithQuery(t *testing.T) {
	once.Do(startServer)

	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	config := newConfig(t, "/echo")
	config.Location, err = url.ParseRequestURI(fmt.Sprintf("ws://%s/echo?q=v", serverAddr))
	if err != nil {
		t.Fatal("location url", err)
	}

	ws, err := NewClient(config, client)
	if err != nil {
		t.Errorf("WebSocket handshake: %v", err)
		return
	}
	ws.Close()
}

func testWithProtocol(t *testing.T, subproto []string) (string, error) {
	once.Do(startServer)

	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	config := newConfig(t, "/subproto")
	config.Protocol = subproto

	ws, err := NewClient(config, client)
	if err != nil {
		return "", err
	}
	msg := make([]byte, 16)
	n, err := ws.Read(msg)
	if err != nil {
		return "", err
	}
	ws.Close()
	return string(msg[:n]), nil
}

func TestWithProtocol(t *testing.T) {
	proto, err := testWithProtocol(t, []string{"chat"})
	if err != nil {
		t.Errorf("SubProto: unexpected error: %v", err)
	}
	if proto != "chat" {
		t.Errorf("SubProto: expected %q, got %q", "chat", proto)
	}
}

func TestWithTwoProtocol(t *testing.T) {
	proto, err := testWithProtocol(t, []string{"test", "chat"})
	if err != nil {
		t.Errorf("SubProto: unexpected error: %v", err)
	}
	if proto != "chat" {
		t.Errorf("SubProto: expected %q, got %q", "chat", proto)
	}
}

func TestWithBadProtocol(t *testing.T) {
	_, err := testWithProtocol(t, []string{"test"})
	if err != ErrBadStatus {
		t.Errorf("SubProto: expected %v, got %v", ErrBadStatus, err)
	}
}

func TestHTTP(t *testing.T) {
	once.Do(startServer)

	// If the client did not send a handshake that matches the protocol
	// specification, the server MUST return an HTTP response with an
	// appropriate error code (such as 400 Bad Request)
	resp, err := http.Get(fmt.Sprintf("http://%s/echo", serverAddr))
	if err != nil {
		t.Errorf("Get: error %#v", err)
		return
	}
	if resp == nil {
		t.Error("Get: resp is null")
		return
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Get: expected %q got %q", http.StatusBadRequest, resp.StatusCode)
	}
}

func TestTrailingSpaces(t *testing.T) {
	// http://code.google.com/p/go/issues/detail?id=955
	// The last runs of this create keys with trailing spaces that should not be
	// generated by the client.
	once.Do(startServer)
	config := newConfig(t, "/echo")
	for i := 0; i < 30; i++ {
		// body
		ws, err := DialConfig(config)
		if err != nil {
			t.Errorf("Dial #%d failed: %v", i, err)
			break
		}
		ws.Close()
	}
}

func TestDialConfigBadVersion(t *testing.T) {
	once.Do(startServer)
	config := newConfig(t, "/echo")
	config.Version = 1234

	_, err := DialConfig(config)

	if dialerr, ok := err.(*DialError); ok {
		if dialerr.Err != ErrBadProtocolVersion {
			t.Errorf("dial expected err %q but got %q", ErrBadProtocolVersion, dialerr.Err)
		}
	}
}

func TestSmallBuffer(t *testing.T) {
	// http://code.google.com/p/go/issues/detail?id=1145
	// Read should be able to handle reading a fragment of a frame.
	once.Do(startServer)

	// websocket.Dial()
	client, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}
	conn, err := NewClient(newConfig(t, "/echo"), client)
	if err != nil {
		t.Errorf("WebSocket handshake error: %v", err)
		return
	}

	msg := []byte("hello, world\n")
	if _, err := conn.Write(msg); err != nil {
		t.Errorf("Write: %v", err)
	}
	var small_msg = make([]byte, 8)
	n, err := conn.Read(small_msg)
	if err != nil {
		t.Errorf("Read: %v", err)
	}
	if !bytes.Equal(msg[:len(small_msg)], small_msg) {
		t.Errorf("Echo: expected %q got %q", msg[:len(small_msg)], small_msg)
	}
	var second_msg = make([]byte, len(msg))
	n, err = conn.Read(second_msg)
	if err != nil {
		t.Errorf("Read: %v", err)
	}
	second_msg = second_msg[0:n]
	if !bytes.Equal(msg[len(small_msg):], second_msg) {
		t.Errorf("Echo: expected %q got %q", msg[len(small_msg):], second_msg)
	}
	conn.Close()
}

var parseAuthorityTests = []struct {
	in  *url.URL
	out string
}{
	{
		&url.URL{
			Scheme: "ws",
			Host:   "www.google.com",
		},
		"www.google.com:80",
	},
	{
		&url.URL{
			Scheme: "wss",
			Host:   "www.google.com",
		},
		"www.google.com:443",
	},
	{
		&url.URL{
			Scheme: "ws",
			Host:   "www.google.com:80",
		},
		"www.google.com:80",
	},
	{
		&url.URL{
			Scheme: "wss",
			Host:   "www.google.com:443",
		},
		"www.google.com:443",
	},
	// some invalid ones for parseAuthority. parseAuthority doesn't
	// concern itself with the scheme unless it actually knows about it
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.google.com",
		},
		"www.google.com",
	},
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.google.com:80",
		},
		"www.google.com:80",
	},
	{
		&url.URL{
			Scheme: "asdf",
			Host:   "127.0.0.1",
		},
		"127.0.0.1",
	},
	{
		&url.URL{
			Scheme: "asdf",
			Host:   "www.google.com",
		},
		"www.google.com",
	},
}

func TestParseAuthority(t *testing.T) {
	for _, tt := range parseAuthorityTests {
		out := parseAuthority(tt.in)
		if out != tt.out {
			t.Errorf("got %v; want %v", out, tt.out)
		}
	}
}

type closerConn struct {
	net.Conn
	closed int // count of the number of times Close was called
}

func (c *closerConn) Close() error {
	c.closed++
	return c.Conn.Close()
}

func TestClose(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("see golang.org/issue/11454")
	}

	once.Do(startServer)

	conn, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	cc := closerConn{Conn: conn}

	client, err := NewClient(newConfig(t, "/echo"), &cc)
	if err != nil {
		t.Fatalf("WebSocket handshake: %v", err)
	}

	// set the deadline to ten minutes ago, which will have expired by the time
	// client.Close sends the close status frame.
	conn.SetDeadline(time.Now().Add(-10 * time.Minute))

	if err := client.Close(); err == nil {
		t.Errorf("ws.Close(): expected error, got %v", err)
	}
	if cc.closed < 1 {
		t.Fatalf("ws.Close(): expected underlying ws.rwc.Close to be called > 0 times, got: %v", cc.closed)
	}
}

var originTests = []struct {
	req    *http.Request
	origin *url.URL
}{
	{
		req: &http.Request{
			Header: http.Header{
				"Origin": []string{"http://www.example.com"},
			},
		},
		origin: &url.URL{
			Scheme: "http",
			Host:   "www.example.com",
		},
	},
	{
		req: &http.Request{},
	},
}

func TestOrigin(t *testing.T) {
	conf := newConfig(t, "/echo")
	conf.Version = ProtocolVersionHybi13
	for i, tt := range originTests {
		origin, err := Origin(conf, tt.req)
		if err != nil {
			t.Error(err)
			continue
		}
		if !reflect.DeepEqual(origin, tt.origin) {
			t.Errorf("#%d: got origin %v; want %v", i, origin, tt.origin)
			continue
		}
	}
}

func TestCtrlAndData(t *testing.T) {
	once.Do(startServer)

	c, err := net.Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewClient(newConfig(t, "/ctrldata"), c)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()

	h := &testCtrlAndDataHandler{hybiFrameHandler: hybiFrameHandler{conn: ws}}
	ws.frameHandler = h

	b := make([]byte, 128)
	for i := 0; i < 2; i++ {
		data := []byte(fmt.Sprintf("#%d-DATA-FRAME-FROM-CLIENT", i))
		if _, err := ws.Write(data); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var ctrl []byte
		if i%2 != 0 { // with or without payload
			ctrl = []byte(fmt.Sprintf("#%d-CONTROL-FRAME-FROM-CLIENT", i))
		}
		if _, err := h.WritePing(ctrl); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		n, err := ws.Read(b)
		if err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		if !bytes.Equal(b[:n], data) {
			t.Fatalf("#%d: got %v; want %v", i, b[:n], data)
		}
	}
}
