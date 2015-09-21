// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package test

import (
	"bytes"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"testing"
	"time"
)

func TestPortForward(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	sshListener, err := conn.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		sshConn, err := sshListener.Accept()
		if err != nil {
			t.Fatalf("listen.Accept failed: %v", err)
		}

		_, err = io.Copy(sshConn, sshConn)
		if err != nil && err != io.EOF {
			t.Fatalf("ssh client copy: %v", err)
		}
		sshConn.Close()
	}()

	forwardedAddr := sshListener.Addr().String()
	tcpConn, err := net.Dial("tcp", forwardedAddr)
	if err != nil {
		t.Fatalf("TCP dial failed: %v", err)
	}

	readChan := make(chan []byte)
	go func() {
		data, _ := ioutil.ReadAll(tcpConn)
		readChan <- data
	}()

	// Invent some data.
	data := make([]byte, 100*1000)
	for i := range data {
		data[i] = byte(i % 255)
	}

	var sent []byte
	for len(sent) < 1000*1000 {
		// Send random sized chunks
		m := rand.Intn(len(data))
		n, err := tcpConn.Write(data[:m])
		if err != nil {
			break
		}
		sent = append(sent, data[:n]...)
	}
	if err := tcpConn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Errorf("tcpConn.CloseWrite: %v", err)
	}

	read := <-readChan

	if len(sent) != len(read) {
		t.Fatalf("got %d bytes, want %d", len(read), len(sent))
	}
	if bytes.Compare(sent, read) != 0 {
		t.Fatalf("read back data does not match")
	}

	if err := sshListener.Close(); err != nil {
		t.Fatalf("sshListener.Close: %v", err)
	}

	// Check that the forward disappeared.
	tcpConn, err = net.Dial("tcp", forwardedAddr)
	if err == nil {
		tcpConn.Close()
		t.Errorf("still listening to %s after closing", forwardedAddr)
	}
}

func TestAcceptClose(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())

	sshListener, err := conn.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}

	quit := make(chan error, 1)
	go func() {
		for {
			c, err := sshListener.Accept()
			if err != nil {
				quit <- err
				break
			}
			c.Close()
		}
	}()
	sshListener.Close()

	select {
	case <-time.After(1 * time.Second):
		t.Errorf("timeout: listener did not close.")
	case err := <-quit:
		t.Logf("quit as expected (error %v)", err)
	}
}

// Check that listeners exit if the underlying client transport dies.
func TestPortForwardConnectionClose(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())

	sshListener, err := conn.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}

	quit := make(chan error, 1)
	go func() {
		for {
			c, err := sshListener.Accept()
			if err != nil {
				quit <- err
				break
			}
			c.Close()
		}
	}()

	// It would be even nicer if we closed the server side, but it
	// is more involved as the fd for that side is dup()ed.
	server.clientConn.Close()

	select {
	case <-time.After(1 * time.Second):
		t.Errorf("timeout: listener did not close.")
	case err := <-quit:
		t.Logf("quit as expected (error %v)", err)
	}
}
