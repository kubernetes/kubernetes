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

package latency

import (
	"bytes"
	"fmt"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"
)

// bufConn is a net.Conn implemented by a bytes.Buffer (which is a ReadWriter).
type bufConn struct {
	*bytes.Buffer
}

func (bufConn) Close() error                       { panic("unimplemented") }
func (bufConn) LocalAddr() net.Addr                { panic("unimplemented") }
func (bufConn) RemoteAddr() net.Addr               { panic("unimplemented") }
func (bufConn) SetDeadline(t time.Time) error      { panic("unimplemneted") }
func (bufConn) SetReadDeadline(t time.Time) error  { panic("unimplemneted") }
func (bufConn) SetWriteDeadline(t time.Time) error { panic("unimplemneted") }

func restoreHooks() func() {
	s := sleep
	n := now
	return func() {
		sleep = s
		now = n
	}
}

func TestConn(t *testing.T) {
	defer restoreHooks()()

	// Constant time.
	now = func() time.Time { return time.Unix(123, 456) }

	// Capture sleep times for checking later.
	var sleepTimes []time.Duration
	sleep = func(t time.Duration) { sleepTimes = append(sleepTimes, t) }

	wantSleeps := func(want ...time.Duration) {
		if !reflect.DeepEqual(want, sleepTimes) {
			t.Fatalf("sleepTimes = %v; want %v", sleepTimes, want)
		}
		sleepTimes = nil
	}

	// Use a fairly high latency to cause a large BDP and avoid sleeps while
	// writing due to simulation of full buffers.
	latency := 1 * time.Second
	c, err := (&Network{Kbps: 1, Latency: latency, MTU: 5}).Conn(bufConn{&bytes.Buffer{}})
	if err != nil {
		t.Fatalf("Unexpected error creating connection: %v", err)
	}
	wantSleeps(latency) // Connection creation delay.

	// 1 kbps = 128 Bps.  Divides evenly by 1 second using nanos.
	byteLatency := time.Duration(time.Second / 128)

	write := func(b []byte) {
		n, err := c.Write(b)
		if n != len(b) || err != nil {
			t.Fatalf("c.Write(%v) = %v, %v; want %v, nil", b, n, err, len(b))
		}
	}

	write([]byte{1, 2, 3, 4, 5}) // One full packet
	pkt1Time := latency + byteLatency*5
	write([]byte{6}) // One partial packet
	pkt2Time := pkt1Time + byteLatency
	write([]byte{7, 8, 9, 10, 11, 12, 13}) // Two packets
	pkt3Time := pkt2Time + byteLatency*5
	pkt4Time := pkt3Time + byteLatency*2

	// No reads, so no sleeps yet.
	wantSleeps()

	read := func(n int, want []byte) {
		b := make([]byte, n)
		if rd, err := c.Read(b); err != nil || rd != len(want) {
			t.Fatalf("c.Read(<%v bytes>) = %v, %v; want %v, nil", n, rd, err, len(want))
		}
		if !reflect.DeepEqual(b[:len(want)], want) {
			t.Fatalf("read %v; want %v", b, want)
		}
	}

	read(1, []byte{1})
	wantSleeps(pkt1Time)
	read(1, []byte{2})
	wantSleeps()
	read(3, []byte{3, 4, 5})
	wantSleeps()
	read(2, []byte{6})
	wantSleeps(pkt2Time)
	read(2, []byte{7, 8})
	wantSleeps(pkt3Time)
	read(10, []byte{9, 10, 11})
	wantSleeps()
	read(10, []byte{12, 13})
	wantSleeps(pkt4Time)
}

func TestSync(t *testing.T) {
	defer restoreHooks()()

	// Infinitely fast CPU: time doesn't pass unless sleep is called.
	tn := time.Unix(123, 0)
	now = func() time.Time { return tn }
	sleep = func(d time.Duration) { tn = tn.Add(d) }

	// Simulate a 20ms latency network, then run sync across that and expect to
	// measure 20ms latency, or 10ms additional delay for a 30ms network.
	slowConn, err := (&Network{Kbps: 0, Latency: 20 * time.Millisecond, MTU: 5}).Conn(bufConn{&bytes.Buffer{}})
	if err != nil {
		t.Fatalf("Unexpected error creating connection: %v", err)
	}
	c, err := (&Network{Latency: 30 * time.Millisecond}).Conn(slowConn)
	if err != nil {
		t.Fatalf("Unexpected error creating connection: %v", err)
	}
	if c.(*conn).delay != 10*time.Millisecond {
		t.Fatalf("c.delay = %v; want 10ms", c.(*conn).delay)
	}
}

func TestSyncTooSlow(t *testing.T) {
	defer restoreHooks()()

	// Infinitely fast CPU: time doesn't pass unless sleep is called.
	tn := time.Unix(123, 0)
	now = func() time.Time { return tn }
	sleep = func(d time.Duration) { tn = tn.Add(d) }

	// Simulate a 10ms latency network, then attempt to simulate a 5ms latency
	// network and expect an error.
	slowConn, err := (&Network{Kbps: 0, Latency: 10 * time.Millisecond, MTU: 5}).Conn(bufConn{&bytes.Buffer{}})
	if err != nil {
		t.Fatalf("Unexpected error creating connection: %v", err)
	}

	errWant := "measured network latency (10ms) higher than desired latency (5ms)"
	if _, err := (&Network{Latency: 5 * time.Millisecond}).Conn(slowConn); err == nil || err.Error() != errWant {
		t.Fatalf("Conn() = _, %q; want _, %q", err, errWant)
	}
}

func TestListenerAndDialer(t *testing.T) {
	defer restoreHooks()()

	tn := time.Unix(123, 0)
	startTime := tn
	mu := &sync.Mutex{}
	now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return tn
	}

	// Use a fairly high latency to cause a large BDP and avoid sleeps while
	// writing due to simulation of full buffers.
	n := &Network{Kbps: 2, Latency: 1 * time.Second, MTU: 10}
	// 2 kbps = .25 kBps = 256 Bps
	byteLatency := func(n int) time.Duration {
		return time.Duration(n) * time.Second / 256
	}

	// Create a real listener and wrap it.
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Unexpected error creating listener: %v", err)
	}
	defer l.Close()
	l = n.Listener(l)

	var serverConn net.Conn
	var scErr error
	scDone := make(chan struct{})
	go func() {
		serverConn, scErr = l.Accept()
		close(scDone)
	}()

	// Create a dialer and use it.
	clientConn, err := n.TimeoutDialer(net.DialTimeout)("tcp", l.Addr().String(), 2*time.Second)
	if err != nil {
		t.Fatalf("Unexpected error dialing: %v", err)
	}
	defer clientConn.Close()

	// Block until server's Conn is available.
	<-scDone
	if scErr != nil {
		t.Fatalf("Unexpected error listening: %v", scErr)
	}
	defer serverConn.Close()

	// sleep (only) advances tn.   Done after connections established so sync detects zero delay.
	sleep = func(d time.Duration) {
		mu.Lock()
		defer mu.Unlock()
		if d > 0 {
			tn = tn.Add(d)
		}
	}

	seq := func(a, b int) []byte {
		buf := make([]byte, b-a)
		for i := 0; i < b-a; i++ {
			buf[i] = byte(i + a)
		}
		return buf
	}

	pkt1 := seq(0, 10)
	pkt2 := seq(10, 30)
	pkt3 := seq(30, 35)

	write := func(c net.Conn, b []byte) {
		n, err := c.Write(b)
		if n != len(b) || err != nil {
			t.Fatalf("c.Write(%v) = %v, %v; want %v, nil", b, n, err, len(b))
		}
	}

	write(serverConn, pkt1)
	write(serverConn, pkt2)
	write(serverConn, pkt3)
	write(clientConn, pkt3)
	write(clientConn, pkt1)
	write(clientConn, pkt2)

	if tn != startTime {
		t.Fatalf("unexpected sleep in write; tn = %v; want %v", tn, startTime)
	}

	read := func(c net.Conn, n int, want []byte, timeWant time.Time) {
		b := make([]byte, n)
		if rd, err := c.Read(b); err != nil || rd != len(want) {
			t.Fatalf("c.Read(<%v bytes>) = %v, %v; want %v, nil (read: %v)", n, rd, err, len(want), b[:rd])
		}
		if !reflect.DeepEqual(b[:len(want)], want) {
			t.Fatalf("read %v; want %v", b, want)
		}
		if !tn.Equal(timeWant) {
			t.Errorf("tn after read(%v) = %v; want %v", want, tn, timeWant)
		}
	}

	read(clientConn, len(pkt1)+1, pkt1, startTime.Add(n.Latency+byteLatency(len(pkt1))))
	read(serverConn, len(pkt3)+1, pkt3, tn) // tn was advanced by the above read; pkt3 is shorter than pkt1

	read(clientConn, len(pkt2), pkt2[:10], startTime.Add(n.Latency+byteLatency(len(pkt1)+10)))
	read(clientConn, len(pkt2), pkt2[10:], startTime.Add(n.Latency+byteLatency(len(pkt1)+len(pkt2))))
	read(clientConn, len(pkt3), pkt3, startTime.Add(n.Latency+byteLatency(len(pkt1)+len(pkt2)+len(pkt3))))

	read(serverConn, len(pkt1), pkt1, tn) // tn already past the arrival time due to prior reads
	read(serverConn, len(pkt2), pkt2[:10], tn)
	read(serverConn, len(pkt2), pkt2[10:], tn)

	// Sleep awhile and make sure the read happens disregarding previous writes
	// (lastSendEnd handling).
	sleep(10 * time.Second)
	write(clientConn, pkt1)
	read(serverConn, len(pkt1), pkt1, tn.Add(n.Latency+byteLatency(len(pkt1))))

	// Send, sleep longer than the network delay, then make sure the read happens
	// instantly.
	write(serverConn, pkt1)
	sleep(10 * time.Second)
	read(clientConn, len(pkt1), pkt1, tn)
}

func TestBufferBloat(t *testing.T) {
	defer restoreHooks()()

	// Infinitely fast CPU: time doesn't pass unless sleep is called.
	tn := time.Unix(123, 0)
	now = func() time.Time { return tn }
	// Capture sleep times for checking later.
	var sleepTimes []time.Duration
	sleep = func(d time.Duration) {
		sleepTimes = append(sleepTimes, d)
		tn = tn.Add(d)
	}

	wantSleeps := func(want ...time.Duration) error {
		if !reflect.DeepEqual(want, sleepTimes) {
			return fmt.Errorf("sleepTimes = %v; want %v", sleepTimes, want)
		}
		sleepTimes = nil
		return nil
	}

	n := &Network{Kbps: 8 /* 1KBps */, Latency: time.Second, MTU: 8}
	bdpBytes := (n.Kbps * 1024 / 8) * int(n.Latency/time.Second) // 1024
	c, err := n.Conn(bufConn{&bytes.Buffer{}})
	if err != nil {
		t.Fatalf("Unexpected error creating connection: %v", err)
	}
	wantSleeps(n.Latency) // Connection creation delay.

	write := func(n int, sleeps ...time.Duration) {
		if wt, err := c.Write(make([]byte, n)); err != nil || wt != n {
			t.Fatalf("c.Write(<%v bytes>) = %v, %v; want %v, nil", n, wt, err, n)
		}
		if err := wantSleeps(sleeps...); err != nil {
			t.Fatalf("After writing %v bytes: %v", n, err)
		}
	}

	read := func(n int, sleeps ...time.Duration) {
		if rd, err := c.Read(make([]byte, n)); err != nil || rd != n {
			t.Fatalf("c.Read(_) = %v, %v; want %v, nil", rd, err, n)
		}
		if err := wantSleeps(sleeps...); err != nil {
			t.Fatalf("After reading %v bytes: %v", n, err)
		}
	}

	write(8) // No reads and buffer not full, so no sleeps yet.
	read(8, time.Second+n.pktTime(8))

	write(bdpBytes)            // Fill the buffer.
	write(1)                   // We can send one extra packet even when the buffer is full.
	write(n.MTU, n.pktTime(1)) // Make sure we sleep to clear the previous write.
	write(1, n.pktTime(n.MTU))
	write(n.MTU+1, n.pktTime(1), n.pktTime(n.MTU))

	tn = tn.Add(10 * time.Second) // Wait long enough for the buffer to clear.
	write(bdpBytes)               // No sleeps required.
}
