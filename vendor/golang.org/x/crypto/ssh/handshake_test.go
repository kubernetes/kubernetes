// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

type testChecker struct {
	calls []string
}

func (t *testChecker) Check(dialAddr string, addr net.Addr, key PublicKey) error {
	if dialAddr == "bad" {
		return fmt.Errorf("dialAddr is bad")
	}

	if tcpAddr, ok := addr.(*net.TCPAddr); !ok || tcpAddr == nil {
		return fmt.Errorf("testChecker: got %T want *net.TCPAddr", addr)
	}

	t.calls = append(t.calls, fmt.Sprintf("%s %v %s %x", dialAddr, addr, key.Type(), key.Marshal()))

	return nil
}

// netPipe is analogous to net.Pipe, but it uses a real net.Conn, and
// therefore is buffered (net.Pipe deadlocks if both sides start with
// a write.)
func netPipe() (net.Conn, net.Conn, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		listener, err = net.Listen("tcp", "[::1]:0")
		if err != nil {
			return nil, nil, err
		}
	}
	defer listener.Close()
	c1, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		return nil, nil, err
	}

	c2, err := listener.Accept()
	if err != nil {
		c1.Close()
		return nil, nil, err
	}

	return c1, c2, nil
}

// noiseTransport inserts ignore messages to check that the read loop
// and the key exchange filters out these messages.
type noiseTransport struct {
	keyingTransport
}

func (t *noiseTransport) writePacket(p []byte) error {
	ignore := []byte{msgIgnore}
	if err := t.keyingTransport.writePacket(ignore); err != nil {
		return err
	}
	debug := []byte{msgDebug, 1, 2, 3}
	if err := t.keyingTransport.writePacket(debug); err != nil {
		return err
	}

	return t.keyingTransport.writePacket(p)
}

func addNoiseTransport(t keyingTransport) keyingTransport {
	return &noiseTransport{t}
}

// handshakePair creates two handshakeTransports connected with each
// other. If the noise argument is true, both transports will try to
// confuse the other side by sending ignore and debug messages.
func handshakePair(clientConf *ClientConfig, addr string, noise bool) (client *handshakeTransport, server *handshakeTransport, err error) {
	a, b, err := netPipe()
	if err != nil {
		return nil, nil, err
	}

	var trC, trS keyingTransport

	trC = newTransport(a, rand.Reader, true)
	trS = newTransport(b, rand.Reader, false)
	if noise {
		trC = addNoiseTransport(trC)
		trS = addNoiseTransport(trS)
	}
	clientConf.SetDefaults()

	v := []byte("version")
	client = newClientTransport(trC, v, v, clientConf, addr, a.RemoteAddr())

	serverConf := &ServerConfig{}
	serverConf.AddHostKey(testSigners["ecdsa"])
	serverConf.AddHostKey(testSigners["rsa"])
	serverConf.SetDefaults()
	server = newServerTransport(trS, v, v, serverConf)

	if err := server.waitSession(); err != nil {
		return nil, nil, fmt.Errorf("server.waitSession: %v", err)
	}
	if err := client.waitSession(); err != nil {
		return nil, nil, fmt.Errorf("client.waitSession: %v", err)
	}

	return client, server, nil
}

func TestHandshakeBasic(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("see golang.org/issue/7237")
	}

	checker := &syncChecker{
		waitCall: make(chan int, 10),
		called:   make(chan int, 10),
	}

	checker.waitCall <- 1
	trC, trS, err := handshakePair(&ClientConfig{HostKeyCallback: checker.Check}, "addr", false)
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}

	defer trC.Close()
	defer trS.Close()

	// Let first kex complete normally.
	<-checker.called

	clientDone := make(chan int, 0)
	gotHalf := make(chan int, 0)
	const N = 20

	go func() {
		defer close(clientDone)
		// Client writes a bunch of stuff, and does a key
		// change in the middle. This should not confuse the
		// handshake in progress. We do this twice, so we test
		// that the packet buffer is reset correctly.
		for i := 0; i < N; i++ {
			p := []byte{msgRequestSuccess, byte(i)}
			if err := trC.writePacket(p); err != nil {
				t.Fatalf("sendPacket: %v", err)
			}
			if (i % 10) == 5 {
				<-gotHalf
				// halfway through, we request a key change.
				trC.requestKeyExchange()

				// Wait until we can be sure the key
				// change has really started before we
				// write more.
				<-checker.called
			}
			if (i % 10) == 7 {
				// write some packets until the kex
				// completes, to test buffering of
				// packets.
				checker.waitCall <- 1
			}
		}
	}()

	// Server checks that client messages come in cleanly
	i := 0
	err = nil
	for ; i < N; i++ {
		var p []byte
		p, err = trS.readPacket()
		if err != nil {
			break
		}
		if (i % 10) == 5 {
			gotHalf <- 1
		}

		want := []byte{msgRequestSuccess, byte(i)}
		if bytes.Compare(p, want) != 0 {
			t.Errorf("message %d: got %v, want %v", i, p, want)
		}
	}
	<-clientDone
	if err != nil && err != io.EOF {
		t.Fatalf("server error: %v", err)
	}
	if i != N {
		t.Errorf("received %d messages, want 10.", i)
	}

	close(checker.called)
	if _, ok := <-checker.called; ok {
		// If all went well, we registered exactly 2 key changes: one
		// that establishes the session, and one that we requested
		// additionally.
		t.Fatalf("got another host key checks after 2 handshakes")
	}
}

func TestForceFirstKex(t *testing.T) {
	// like handshakePair, but must access the keyingTransport.
	checker := &testChecker{}
	clientConf := &ClientConfig{HostKeyCallback: checker.Check}
	a, b, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}

	var trC, trS keyingTransport

	trC = newTransport(a, rand.Reader, true)

	// This is the disallowed packet:
	trC.writePacket(Marshal(&serviceRequestMsg{serviceUserAuth}))

	// Rest of the setup.
	trS = newTransport(b, rand.Reader, false)
	clientConf.SetDefaults()

	v := []byte("version")
	client := newClientTransport(trC, v, v, clientConf, "addr", a.RemoteAddr())

	serverConf := &ServerConfig{}
	serverConf.AddHostKey(testSigners["ecdsa"])
	serverConf.AddHostKey(testSigners["rsa"])
	serverConf.SetDefaults()
	server := newServerTransport(trS, v, v, serverConf)

	defer client.Close()
	defer server.Close()

	// We setup the initial key exchange, but the remote side
	// tries to send serviceRequestMsg in cleartext, which is
	// disallowed.

	if err := server.waitSession(); err == nil {
		t.Errorf("server first kex init should reject unexpected packet")
	}
}

func TestHandshakeAutoRekeyWrite(t *testing.T) {
	checker := &syncChecker{
		called:   make(chan int, 10),
		waitCall: nil,
	}
	clientConf := &ClientConfig{HostKeyCallback: checker.Check}
	clientConf.RekeyThreshold = 500
	trC, trS, err := handshakePair(clientConf, "addr", false)
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}
	defer trC.Close()
	defer trS.Close()

	input := make([]byte, 251)
	input[0] = msgRequestSuccess

	done := make(chan int, 1)
	const numPacket = 5
	go func() {
		defer close(done)
		j := 0
		for ; j < numPacket; j++ {
			if p, err := trS.readPacket(); err != nil {
				break
			} else if !bytes.Equal(input, p) {
				t.Errorf("got packet type %d, want %d", p[0], input[0])
			}
		}

		if j != numPacket {
			t.Errorf("got %d, want 5 messages", j)
		}
	}()

	<-checker.called

	for i := 0; i < numPacket; i++ {
		p := make([]byte, len(input))
		copy(p, input)
		if err := trC.writePacket(p); err != nil {
			t.Errorf("writePacket: %v", err)
		}
		if i == 2 {
			// Make sure the kex is in progress.
			<-checker.called
		}

	}
	<-done
}

type syncChecker struct {
	waitCall chan int
	called   chan int
}

func (c *syncChecker) Check(dialAddr string, addr net.Addr, key PublicKey) error {
	c.called <- 1
	if c.waitCall != nil {
		<-c.waitCall
	}
	return nil
}

func TestHandshakeAutoRekeyRead(t *testing.T) {
	sync := &syncChecker{
		called:   make(chan int, 2),
		waitCall: nil,
	}
	clientConf := &ClientConfig{
		HostKeyCallback: sync.Check,
	}
	clientConf.RekeyThreshold = 500

	trC, trS, err := handshakePair(clientConf, "addr", false)
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}
	defer trC.Close()
	defer trS.Close()

	packet := make([]byte, 501)
	packet[0] = msgRequestSuccess
	if err := trS.writePacket(packet); err != nil {
		t.Fatalf("writePacket: %v", err)
	}

	// While we read out the packet, a key change will be
	// initiated.
	done := make(chan int, 1)
	go func() {
		defer close(done)
		if _, err := trC.readPacket(); err != nil {
			t.Fatalf("readPacket(client): %v", err)
		}

	}()

	<-done
	<-sync.called
}

// errorKeyingTransport generates errors after a given number of
// read/write operations.
type errorKeyingTransport struct {
	packetConn
	readLeft, writeLeft int
}

func (n *errorKeyingTransport) prepareKeyChange(*algorithms, *kexResult) error {
	return nil
}

func (n *errorKeyingTransport) getSessionID() []byte {
	return nil
}

func (n *errorKeyingTransport) writePacket(packet []byte) error {
	if n.writeLeft == 0 {
		n.Close()
		return errors.New("barf")
	}

	n.writeLeft--
	return n.packetConn.writePacket(packet)
}

func (n *errorKeyingTransport) readPacket() ([]byte, error) {
	if n.readLeft == 0 {
		n.Close()
		return nil, errors.New("barf")
	}

	n.readLeft--
	return n.packetConn.readPacket()
}

func TestHandshakeErrorHandlingRead(t *testing.T) {
	for i := 0; i < 20; i++ {
		testHandshakeErrorHandlingN(t, i, -1, false)
	}
}

func TestHandshakeErrorHandlingWrite(t *testing.T) {
	for i := 0; i < 20; i++ {
		testHandshakeErrorHandlingN(t, -1, i, false)
	}
}

func TestHandshakeErrorHandlingReadCoupled(t *testing.T) {
	for i := 0; i < 20; i++ {
		testHandshakeErrorHandlingN(t, i, -1, true)
	}
}

func TestHandshakeErrorHandlingWriteCoupled(t *testing.T) {
	for i := 0; i < 20; i++ {
		testHandshakeErrorHandlingN(t, -1, i, true)
	}
}

// testHandshakeErrorHandlingN runs handshakes, injecting errors. If
// handshakeTransport deadlocks, the go runtime will detect it and
// panic.
func testHandshakeErrorHandlingN(t *testing.T, readLimit, writeLimit int, coupled bool) {
	msg := Marshal(&serviceRequestMsg{strings.Repeat("x", int(minRekeyThreshold)/4)})

	a, b := memPipe()
	defer a.Close()
	defer b.Close()

	key := testSigners["ecdsa"]
	serverConf := Config{RekeyThreshold: minRekeyThreshold}
	serverConf.SetDefaults()
	serverConn := newHandshakeTransport(&errorKeyingTransport{a, readLimit, writeLimit}, &serverConf, []byte{'a'}, []byte{'b'})
	serverConn.hostKeys = []Signer{key}
	go serverConn.readLoop()
	go serverConn.kexLoop()

	clientConf := Config{RekeyThreshold: 10 * minRekeyThreshold}
	clientConf.SetDefaults()
	clientConn := newHandshakeTransport(&errorKeyingTransport{b, -1, -1}, &clientConf, []byte{'a'}, []byte{'b'})
	clientConn.hostKeyAlgorithms = []string{key.PublicKey().Type()}
	clientConn.hostKeyCallback = InsecureIgnoreHostKey()
	go clientConn.readLoop()
	go clientConn.kexLoop()

	var wg sync.WaitGroup

	for _, hs := range []packetConn{serverConn, clientConn} {
		if !coupled {
			wg.Add(2)
			go func(c packetConn) {
				for i := 0; ; i++ {
					str := fmt.Sprintf("%08x", i) + strings.Repeat("x", int(minRekeyThreshold)/4-8)
					err := c.writePacket(Marshal(&serviceRequestMsg{str}))
					if err != nil {
						break
					}
				}
				wg.Done()
				c.Close()
			}(hs)
			go func(c packetConn) {
				for {
					_, err := c.readPacket()
					if err != nil {
						break
					}
				}
				wg.Done()
			}(hs)
		} else {
			wg.Add(1)
			go func(c packetConn) {
				for {
					_, err := c.readPacket()
					if err != nil {
						break
					}
					if err := c.writePacket(msg); err != nil {
						break
					}

				}
				wg.Done()
			}(hs)
		}
	}
	wg.Wait()
}

func TestDisconnect(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("see golang.org/issue/7237")
	}
	checker := &testChecker{}
	trC, trS, err := handshakePair(&ClientConfig{HostKeyCallback: checker.Check}, "addr", false)
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}

	defer trC.Close()
	defer trS.Close()

	trC.writePacket([]byte{msgRequestSuccess, 0, 0})
	errMsg := &disconnectMsg{
		Reason:  42,
		Message: "such is life",
	}
	trC.writePacket(Marshal(errMsg))
	trC.writePacket([]byte{msgRequestSuccess, 0, 0})

	packet, err := trS.readPacket()
	if err != nil {
		t.Fatalf("readPacket 1: %v", err)
	}
	if packet[0] != msgRequestSuccess {
		t.Errorf("got packet %v, want packet type %d", packet, msgRequestSuccess)
	}

	_, err = trS.readPacket()
	if err == nil {
		t.Errorf("readPacket 2 succeeded")
	} else if !reflect.DeepEqual(err, errMsg) {
		t.Errorf("got error %#v, want %#v", err, errMsg)
	}

	_, err = trS.readPacket()
	if err == nil {
		t.Errorf("readPacket 3 succeeded")
	}
}

func TestHandshakeRekeyDefault(t *testing.T) {
	clientConf := &ClientConfig{
		Config: Config{
			Ciphers: []string{"aes128-ctr"},
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	trC, trS, err := handshakePair(clientConf, "addr", false)
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}
	defer trC.Close()
	defer trS.Close()

	trC.writePacket([]byte{msgRequestSuccess, 0, 0})
	trC.Close()

	rgb := (1024 + trC.readBytesLeft) >> 30
	wgb := (1024 + trC.writeBytesLeft) >> 30

	if rgb != 64 {
		t.Errorf("got rekey after %dG read, want 64G", rgb)
	}
	if wgb != 64 {
		t.Errorf("got rekey after %dG write, want 64G", wgb)
	}
}
