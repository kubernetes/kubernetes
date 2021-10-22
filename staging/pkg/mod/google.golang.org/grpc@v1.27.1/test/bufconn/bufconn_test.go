/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package bufconn

import (
	"fmt"
	"io"
	"net"
	"reflect"
	"testing"
	"time"
)

func testRW(r io.Reader, w io.Writer) error {
	for i := 0; i < 20; i++ {
		d := make([]byte, i)
		for j := 0; j < i; j++ {
			d[j] = byte(i - j)
		}
		var rn int
		var rerr error
		b := make([]byte, i)
		done := make(chan struct{})
		go func() {
			for rn < len(b) && rerr == nil {
				var x int
				x, rerr = r.Read(b[rn:])
				rn += x
			}
			close(done)
		}()
		wn, werr := w.Write(d)
		if wn != i || werr != nil {
			return fmt.Errorf("%v: w.Write(%v) = %v, %v; want %v, nil", i, d, wn, werr, i)
		}
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
			return fmt.Errorf("%v: r.Read never returned", i)
		}
		if rn != i || rerr != nil {
			return fmt.Errorf("%v: r.Read = %v, %v; want %v, nil", i, rn, rerr, i)
		}
		if !reflect.DeepEqual(b, d) {
			return fmt.Errorf("%v: r.Read read %v; want %v", i, b, d)
		}
	}
	return nil
}

func TestPipe(t *testing.T) {
	p := newPipe(10)
	if err := testRW(p, p); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestPipeClose(t *testing.T) {
	p := newPipe(10)
	p.Close()
	if _, err := p.Write(nil); err != io.ErrClosedPipe {
		t.Fatalf("p.Write = _, %v; want _, %v", err, io.ErrClosedPipe)
	}
	if _, err := p.Read(nil); err != io.ErrClosedPipe {
		t.Fatalf("p.Read = _, %v; want _, %v", err, io.ErrClosedPipe)
	}
}

func TestConn(t *testing.T) {
	p1, p2 := newPipe(10), newPipe(10)
	c1, c2 := &conn{p1, p2}, &conn{p2, p1}

	if err := testRW(c1, c2); err != nil {
		t.Fatalf(err.Error())
	}
	if err := testRW(c2, c1); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestConnCloseWithData(t *testing.T) {
	lis := Listen(7)
	errChan := make(chan error, 1)
	var lisConn net.Conn
	go func() {
		var err error
		if lisConn, err = lis.Accept(); err != nil {
			errChan <- err
		}
		close(errChan)
	}()
	dialConn, err := lis.Dial()
	if err != nil {
		t.Fatalf("Dial error: %v", err)
	}
	if err := <-errChan; err != nil {
		t.Fatalf("Listen error: %v", err)
	}

	// Write some data on both sides of the connection.
	n, err := dialConn.Write([]byte("hello"))
	if n != 5 || err != nil {
		t.Fatalf("dialConn.Write([]byte{\"hello\"}) = %v, %v; want 5, <nil>", n, err)
	}
	n, err = lisConn.Write([]byte("hello"))
	if n != 5 || err != nil {
		t.Fatalf("lisConn.Write([]byte{\"hello\"}) = %v, %v; want 5, <nil>", n, err)
	}

	// Close dial-side; writes from either side should fail.
	dialConn.Close()
	if _, err := lisConn.Write([]byte("hello")); err != io.ErrClosedPipe {
		t.Fatalf("lisConn.Write() = _, <nil>; want _, <non-nil>")
	}
	if _, err := dialConn.Write([]byte("hello")); err != io.ErrClosedPipe {
		t.Fatalf("dialConn.Write() = _, <nil>; want _, <non-nil>")
	}

	// Read from both sides; reads on lisConn should work, but dialConn should
	// fail.
	buf := make([]byte, 6)
	if _, err := dialConn.Read(buf); err != io.ErrClosedPipe {
		t.Fatalf("dialConn.Read(buf) = %v, %v; want _, io.ErrClosedPipe", n, err)
	}
	n, err = lisConn.Read(buf)
	if n != 5 || err != nil {
		t.Fatalf("lisConn.Read(buf) = %v, %v; want 5, <nil>", n, err)
	}
}

func TestListener(t *testing.T) {
	l := Listen(7)
	var s net.Conn
	var serr error
	done := make(chan struct{})
	go func() {
		s, serr = l.Accept()
		close(done)
	}()
	c, cerr := l.Dial()
	<-done
	if cerr != nil || serr != nil {
		t.Fatalf("cerr = %v, serr = %v; want nil, nil", cerr, serr)
	}
	if err := testRW(c, s); err != nil {
		t.Fatalf(err.Error())
	}
	if err := testRW(s, c); err != nil {
		t.Fatalf(err.Error())
	}
}

func TestCloseWhileDialing(t *testing.T) {
	l := Listen(7)
	var c net.Conn
	var err error
	done := make(chan struct{})
	go func() {
		c, err = l.Dial()
		close(done)
	}()
	l.Close()
	<-done
	if c != nil || err != errClosed {
		t.Fatalf("c, err = %v, %v; want nil, %v", c, err, errClosed)
	}
}

func TestCloseWhileAccepting(t *testing.T) {
	l := Listen(7)
	var c net.Conn
	var err error
	done := make(chan struct{})
	go func() {
		c, err = l.Accept()
		close(done)
	}()
	l.Close()
	<-done
	if c != nil || err != errClosed {
		t.Fatalf("c, err = %v, %v; want nil, %v", c, err, errClosed)
	}
}

func TestDeadline(t *testing.T) {
	sig := make(chan error, 2)
	blockingWrite := func(conn net.Conn) {
		_, err := conn.Write([]byte("0123456789"))
		sig <- err
	}

	blockingRead := func(conn net.Conn) {
		_, err := conn.Read(make([]byte, 10))
		sig <- err
	}

	p1, p2 := newPipe(5), newPipe(5)
	c1, c2 := &conn{p1, p1}, &conn{p2, p2}
	defer c1.Close()
	defer c2.Close()

	// Test with deadline
	c1.SetWriteDeadline(time.Now())

	go blockingWrite(c1)
	select {
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("Write timeout timed out, c = %v", c1)
	case err := <-sig:
		if netErr, ok := err.(net.Error); ok {
			if !netErr.Timeout() {
				t.Fatalf("Write returned unexpected error, c = %v, err = %v", c1, netErr)
			}
		} else {
			t.Fatalf("Write returned unexpected error, c = %v, err = %v", c1, err)
		}
	}

	c2.SetReadDeadline(time.Now())

	go blockingRead(c2)
	select {
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("Read timeout timed out, c = %v", c2)
	case err := <-sig:
		if netErr, ok := err.(net.Error); ok {
			if !netErr.Timeout() {
				t.Fatalf("Read returned unexpected error, c = %v, err = %v", c2, netErr)
			}
		} else {
			t.Fatalf("Read returned unexpected error, c = %v, err = %v", c2, err)
		}
	}

	// Test timing out pending reads/writes
	c1.SetWriteDeadline(time.Time{})
	c2.SetReadDeadline(time.Time{})

	go blockingWrite(c1)
	select {
	case <-time.After(100 * time.Millisecond):
	case err := <-sig:
		t.Fatalf("Write returned before timeout, err = %v", err)
	}

	c1.SetWriteDeadline(time.Now())
	select {
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("Write timeout timed out, c = %v", c1)
	case err := <-sig:
		if netErr, ok := err.(net.Error); ok {
			if !netErr.Timeout() {
				t.Fatalf("Write returned unexpected error, c = %v, err = %v", c1, netErr)
			}
		} else {
			t.Fatalf("Write returned unexpected error, c = %v, err = %v", c1, err)
		}
	}

	go blockingRead(c2)
	select {
	case <-time.After(100 * time.Millisecond):
	case err := <-sig:
		t.Fatalf("Read returned before timeout, err = %v", err)
	}

	c2.SetReadDeadline(time.Now())
	select {
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("Read timeout timed out, c = %v", c2)
	case err := <-sig:
		if netErr, ok := err.(net.Error); ok {
			if !netErr.Timeout() {
				t.Fatalf("Read returned unexpected error, c = %v, err = %v", c2, netErr)
			}
		} else {
			t.Fatalf("Read returned unexpected error, c = %v, err = %v", c2, err)
		}
	}

	// Test non-blocking read/write
	c1, c2 = &conn{p1, p2}, &conn{p2, p1}

	c1.SetWriteDeadline(time.Now().Add(10 * time.Second))
	c2.SetReadDeadline(time.Now().Add(10 * time.Second))

	// Not blocking here
	go blockingWrite(c1)
	go blockingRead(c2)

	// Read response from both routines
	for i := 0; i < 2; i++ {
		select {
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("Read/Write timed out, c1 = %v, c2 = %v", c1, c2)
		case err := <-sig:
			if err != nil {
				t.Fatalf("Read/Write failed to complete, c1 = %v, c2 = %v, err = %v", c1, c2, err)
			}
		}
	}
}
