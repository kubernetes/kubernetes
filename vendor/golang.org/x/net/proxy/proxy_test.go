// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
)

type proxyFromEnvTest struct {
	allProxyEnv string
	noProxyEnv  string
	wantTypeOf  Dialer
}

func (t proxyFromEnvTest) String() string {
	var buf bytes.Buffer
	space := func() {
		if buf.Len() > 0 {
			buf.WriteByte(' ')
		}
	}
	if t.allProxyEnv != "" {
		fmt.Fprintf(&buf, "all_proxy=%q", t.allProxyEnv)
	}
	if t.noProxyEnv != "" {
		space()
		fmt.Fprintf(&buf, "no_proxy=%q", t.noProxyEnv)
	}
	return strings.TrimSpace(buf.String())
}

func TestFromEnvironment(t *testing.T) {
	ResetProxyEnv()

	type dummyDialer struct {
		direct
	}

	RegisterDialerType("irc", func(_ *url.URL, _ Dialer) (Dialer, error) {
		return dummyDialer{}, nil
	})

	proxyFromEnvTests := []proxyFromEnvTest{
		{allProxyEnv: "127.0.0.1:8080", noProxyEnv: "localhost, 127.0.0.1", wantTypeOf: direct{}},
		{allProxyEnv: "ftp://example.com:8000", noProxyEnv: "localhost, 127.0.0.1", wantTypeOf: direct{}},
		{allProxyEnv: "socks5://example.com:8080", noProxyEnv: "localhost, 127.0.0.1", wantTypeOf: &PerHost{}},
		{allProxyEnv: "irc://example.com:8000", wantTypeOf: dummyDialer{}},
		{noProxyEnv: "localhost, 127.0.0.1", wantTypeOf: direct{}},
		{wantTypeOf: direct{}},
	}

	for _, tt := range proxyFromEnvTests {
		os.Setenv("ALL_PROXY", tt.allProxyEnv)
		os.Setenv("NO_PROXY", tt.noProxyEnv)
		ResetCachedEnvironment()

		d := FromEnvironment()
		if got, want := fmt.Sprintf("%T", d), fmt.Sprintf("%T", tt.wantTypeOf); got != want {
			t.Errorf("%v: got type = %T, want %T", tt, d, tt.wantTypeOf)
		}
	}
}

func TestFromURL(t *testing.T) {
	endSystem, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer endSystem.Close()
	gateway, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer gateway.Close()

	var wg sync.WaitGroup
	wg.Add(1)
	go socks5Gateway(t, gateway, endSystem, socks5Domain, &wg)

	url, err := url.Parse("socks5://user:password@" + gateway.Addr().String())
	if err != nil {
		t.Fatalf("url.Parse failed: %v", err)
	}
	proxy, err := FromURL(url, Direct)
	if err != nil {
		t.Fatalf("FromURL failed: %v", err)
	}
	_, port, err := net.SplitHostPort(endSystem.Addr().String())
	if err != nil {
		t.Fatalf("net.SplitHostPort failed: %v", err)
	}
	if c, err := proxy.Dial("tcp", "localhost:"+port); err != nil {
		t.Fatalf("FromURL.Dial failed: %v", err)
	} else {
		c.Close()
	}

	wg.Wait()
}

func TestSOCKS5(t *testing.T) {
	endSystem, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer endSystem.Close()
	gateway, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer gateway.Close()

	var wg sync.WaitGroup
	wg.Add(1)
	go socks5Gateway(t, gateway, endSystem, socks5IP4, &wg)

	proxy, err := SOCKS5("tcp", gateway.Addr().String(), nil, Direct)
	if err != nil {
		t.Fatalf("SOCKS5 failed: %v", err)
	}
	if c, err := proxy.Dial("tcp", endSystem.Addr().String()); err != nil {
		t.Fatalf("SOCKS5.Dial failed: %v", err)
	} else {
		c.Close()
	}

	wg.Wait()
}

func socks5Gateway(t *testing.T, gateway, endSystem net.Listener, typ byte, wg *sync.WaitGroup) {
	defer wg.Done()

	c, err := gateway.Accept()
	if err != nil {
		t.Errorf("net.Listener.Accept failed: %v", err)
		return
	}
	defer c.Close()

	b := make([]byte, 32)
	var n int
	if typ == socks5Domain {
		n = 4
	} else {
		n = 3
	}
	if _, err := io.ReadFull(c, b[:n]); err != nil {
		t.Errorf("io.ReadFull failed: %v", err)
		return
	}
	if _, err := c.Write([]byte{socks5Version, socks5AuthNone}); err != nil {
		t.Errorf("net.Conn.Write failed: %v", err)
		return
	}
	if typ == socks5Domain {
		n = 16
	} else {
		n = 10
	}
	if _, err := io.ReadFull(c, b[:n]); err != nil {
		t.Errorf("io.ReadFull failed: %v", err)
		return
	}
	if b[0] != socks5Version || b[1] != socks5Connect || b[2] != 0x00 || b[3] != typ {
		t.Errorf("got an unexpected packet: %#02x %#02x %#02x %#02x", b[0], b[1], b[2], b[3])
		return
	}
	if typ == socks5Domain {
		copy(b[:5], []byte{socks5Version, 0x00, 0x00, socks5Domain, 9})
		b = append(b, []byte("localhost")...)
	} else {
		copy(b[:4], []byte{socks5Version, 0x00, 0x00, socks5IP4})
	}
	host, port, err := net.SplitHostPort(endSystem.Addr().String())
	if err != nil {
		t.Errorf("net.SplitHostPort failed: %v", err)
		return
	}
	b = append(b, []byte(net.ParseIP(host).To4())...)
	p, err := strconv.Atoi(port)
	if err != nil {
		t.Errorf("strconv.Atoi failed: %v", err)
		return
	}
	b = append(b, []byte{byte(p >> 8), byte(p)}...)
	if _, err := c.Write(b); err != nil {
		t.Errorf("net.Conn.Write failed: %v", err)
		return
	}
}

func ResetProxyEnv() {
	for _, env := range []*envOnce{allProxyEnv, noProxyEnv} {
		for _, v := range env.names {
			os.Setenv(v, "")
		}
	}
	ResetCachedEnvironment()
}

func ResetCachedEnvironment() {
	allProxyEnv.reset()
	noProxyEnv.reset()
}
