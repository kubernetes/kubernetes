// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!solaris,!js

package test

// direct-tcpip and direct-streamlocal functional tests

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"strings"
	"testing"
)

type dialTester interface {
	TestServerConn(t *testing.T, c net.Conn)
	TestClientConn(t *testing.T, c net.Conn)
}

func testDial(t *testing.T, n, listenAddr string, x dialTester) {
	server := newServer(t)
	defer server.Shutdown()
	sshConn := server.Dial(clientConfig())
	defer sshConn.Close()

	l, err := net.Listen(n, listenAddr)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer l.Close()

	testData := fmt.Sprintf("hello from %s, %s", n, listenAddr)
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				break
			}
			x.TestServerConn(t, c)

			io.WriteString(c, testData)
			c.Close()
		}
	}()

	conn, err := sshConn.Dial(n, l.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	x.TestClientConn(t, conn)
	defer conn.Close()
	b, err := ioutil.ReadAll(conn)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	t.Logf("got %q", string(b))
	if string(b) != testData {
		t.Fatalf("expected %q, got %q", testData, string(b))
	}
}

type tcpDialTester struct {
	listenAddr string
}

func (x *tcpDialTester) TestServerConn(t *testing.T, c net.Conn) {
	host := strings.Split(x.listenAddr, ":")[0]
	prefix := host + ":"
	if !strings.HasPrefix(c.LocalAddr().String(), prefix) {
		t.Fatalf("expected to start with %q, got %q", prefix, c.LocalAddr().String())
	}
	if !strings.HasPrefix(c.RemoteAddr().String(), prefix) {
		t.Fatalf("expected to start with %q, got %q", prefix, c.RemoteAddr().String())
	}
}

func (x *tcpDialTester) TestClientConn(t *testing.T, c net.Conn) {
	// we use zero addresses. see *Client.Dial.
	if c.LocalAddr().String() != "0.0.0.0:0" {
		t.Fatalf("expected \"0.0.0.0:0\", got %q", c.LocalAddr().String())
	}
	if c.RemoteAddr().String() != "0.0.0.0:0" {
		t.Fatalf("expected \"0.0.0.0:0\", got %q", c.RemoteAddr().String())
	}
}

func TestDialTCP(t *testing.T) {
	x := &tcpDialTester{
		listenAddr: "127.0.0.1:0",
	}
	testDial(t, "tcp", x.listenAddr, x)
}

type unixDialTester struct {
	listenAddr string
}

func (x *unixDialTester) TestServerConn(t *testing.T, c net.Conn) {
	if c.LocalAddr().String() != x.listenAddr {
		t.Fatalf("expected %q, got %q", x.listenAddr, c.LocalAddr().String())
	}
	if c.RemoteAddr().String() != "@" && c.RemoteAddr().String() != "" {
		t.Fatalf("expected \"@\" or \"\", got %q", c.RemoteAddr().String())
	}
}

func (x *unixDialTester) TestClientConn(t *testing.T, c net.Conn) {
	if c.RemoteAddr().String() != x.listenAddr {
		t.Fatalf("expected %q, got %q", x.listenAddr, c.RemoteAddr().String())
	}
	if c.LocalAddr().String() != "@" {
		t.Fatalf("expected \"@\", got %q", c.LocalAddr().String())
	}
}

func TestDialUnix(t *testing.T) {
	addr, cleanup := newTempSocket(t)
	defer cleanup()
	x := &unixDialTester{
		listenAddr: addr,
	}
	testDial(t, "unix", x.listenAddr, x)
}
