// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"net"
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
		return nil, nil, err
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

func handshakePair(clientConf *ClientConfig, addr string) (client *handshakeTransport, server *handshakeTransport, err error) {
	a, b, err := netPipe()
	if err != nil {
		return nil, nil, err
	}

	trC := newTransport(a, rand.Reader, true)
	trS := newTransport(b, rand.Reader, false)
	clientConf.SetDefaults()

	v := []byte("version")
	client = newClientTransport(trC, v, v, clientConf, addr, a.RemoteAddr())

	serverConf := &ServerConfig{}
	serverConf.AddHostKey(testSigners["ecdsa"])
	serverConf.SetDefaults()
	server = newServerTransport(trS, v, v, serverConf)

	return client, server, nil
}

func TestHandshakeBasic(t *testing.T) {
	checker := &testChecker{}
	trC, trS, err := handshakePair(&ClientConfig{HostKeyCallback: checker.Check}, "addr")
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}

	defer trC.Close()
	defer trS.Close()

	go func() {
		// Client writes a bunch of stuff, and does a key
		// change in the middle. This should not confuse the
		// handshake in progress
		for i := 0; i < 10; i++ {
			p := []byte{msgRequestSuccess, byte(i)}
			if err := trC.writePacket(p); err != nil {
				t.Fatalf("sendPacket: %v", err)
			}
			if i == 5 {
				// halfway through, we request a key change.
				_, _, err := trC.sendKexInit()
				if err != nil {
					t.Fatalf("sendKexInit: %v", err)
				}
			}
		}
		trC.Close()
	}()

	// Server checks that client messages come in cleanly
	i := 0
	for {
		p, err := trS.readPacket()
		if err != nil {
			break
		}
		if p[0] == msgNewKeys {
			continue
		}
		want := []byte{msgRequestSuccess, byte(i)}
		if bytes.Compare(p, want) != 0 {
			t.Errorf("message %d: got %q, want %q", i, p, want)
		}
		i++
	}
	if i != 10 {
		t.Errorf("received %d messages, want 10.", i)
	}

	// If all went well, we registered exactly 1 key change.
	if len(checker.calls) != 1 {
		t.Fatalf("got %d host key checks, want 1", len(checker.calls))
	}

	pub := testSigners["ecdsa"].PublicKey()
	want := fmt.Sprintf("%s %v %s %x", "addr", trC.remoteAddr, pub.Type(), pub.Marshal())
	if want != checker.calls[0] {
		t.Errorf("got %q want %q for host key check", checker.calls[0], want)
	}
}

func TestHandshakeError(t *testing.T) {
	checker := &testChecker{}
	trC, trS, err := handshakePair(&ClientConfig{HostKeyCallback: checker.Check}, "bad")
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}
	defer trC.Close()
	defer trS.Close()

	// send a packet
	packet := []byte{msgRequestSuccess, 42}
	if err := trC.writePacket(packet); err != nil {
		t.Errorf("writePacket: %v", err)
	}

	// Now request a key change.
	_, _, err = trC.sendKexInit()
	if err != nil {
		t.Errorf("sendKexInit: %v", err)
	}

	// the key change will fail, and afterwards we can't write.
	if err := trC.writePacket([]byte{msgRequestSuccess, 43}); err == nil {
		t.Errorf("writePacket after botched rekey succeeded.")
	}

	readback, err := trS.readPacket()
	if err != nil {
		t.Fatalf("server closed too soon: %v", err)
	}
	if bytes.Compare(readback, packet) != 0 {
		t.Errorf("got %q want %q", readback, packet)
	}
	readback, err = trS.readPacket()
	if err == nil {
		t.Errorf("got a message %q after failed key change", readback)
	}
}

func TestHandshakeTwice(t *testing.T) {
	checker := &testChecker{}
	trC, trS, err := handshakePair(&ClientConfig{HostKeyCallback: checker.Check}, "addr")
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}

	defer trC.Close()
	defer trS.Close()

	// send a packet
	packet := make([]byte, 5)
	packet[0] = msgRequestSuccess
	if err := trC.writePacket(packet); err != nil {
		t.Errorf("writePacket: %v", err)
	}

	// Now request a key change.
	_, _, err = trC.sendKexInit()
	if err != nil {
		t.Errorf("sendKexInit: %v", err)
	}

	// Send another packet. Use a fresh one, since writePacket destroys.
	packet = make([]byte, 5)
	packet[0] = msgRequestSuccess
	if err := trC.writePacket(packet); err != nil {
		t.Errorf("writePacket: %v", err)
	}

	// 2nd key change.
	_, _, err = trC.sendKexInit()
	if err != nil {
		t.Errorf("sendKexInit: %v", err)
	}

	packet = make([]byte, 5)
	packet[0] = msgRequestSuccess
	if err := trC.writePacket(packet); err != nil {
		t.Errorf("writePacket: %v", err)
	}

	packet = make([]byte, 5)
	packet[0] = msgRequestSuccess
	for i := 0; i < 5; i++ {
		msg, err := trS.readPacket()
		if err != nil {
			t.Fatalf("server closed too soon: %v", err)
		}
		if msg[0] == msgNewKeys {
			continue
		}

		if bytes.Compare(msg, packet) != 0 {
			t.Errorf("packet %d: got %q want %q", i, msg, packet)
		}
	}
	if len(checker.calls) != 2 {
		t.Errorf("got %d key changes, want 2", len(checker.calls))
	}
}

func TestHandshakeAutoRekeyWrite(t *testing.T) {
	checker := &testChecker{}
	clientConf := &ClientConfig{HostKeyCallback: checker.Check}
	clientConf.RekeyThreshold = 500
	trC, trS, err := handshakePair(clientConf, "addr")
	if err != nil {
		t.Fatalf("handshakePair: %v", err)
	}
	defer trC.Close()
	defer trS.Close()

	for i := 0; i < 5; i++ {
		packet := make([]byte, 251)
		packet[0] = msgRequestSuccess
		if err := trC.writePacket(packet); err != nil {
			t.Errorf("writePacket: %v", err)
		}
	}

	j := 0
	for ; j < 5; j++ {
		_, err := trS.readPacket()
		if err != nil {
			break
		}
	}

	if j != 5 {
		t.Errorf("got %d, want 5 messages", j)
	}

	if len(checker.calls) != 2 {
		t.Errorf("got %d key changes, wanted 2", len(checker.calls))
	}
}

type syncChecker struct {
	called chan int
}

func (t *syncChecker) Check(dialAddr string, addr net.Addr, key PublicKey) error {
	t.called <- 1
	return nil
}

func TestHandshakeAutoRekeyRead(t *testing.T) {
	sync := &syncChecker{make(chan int, 2)}
	clientConf := &ClientConfig{
		HostKeyCallback: sync.Check,
	}
	clientConf.RekeyThreshold = 500

	trC, trS, err := handshakePair(clientConf, "addr")
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
	if _, err := trC.readPacket(); err != nil {
		t.Fatalf("readPacket(client): %v", err)
	}

	<-sync.called
}
