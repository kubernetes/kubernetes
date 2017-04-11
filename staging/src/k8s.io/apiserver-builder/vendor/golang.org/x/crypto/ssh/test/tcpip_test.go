// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package test

// direct-tcpip functional tests

import (
	"io"
	"net"
	"testing"
)

func TestDial(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	sshConn := server.Dial(clientConfig())
	defer sshConn.Close()

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer l.Close()

	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				break
			}

			io.WriteString(c, c.RemoteAddr().String())
			c.Close()
		}
	}()

	conn, err := sshConn.Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
}
