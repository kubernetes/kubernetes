// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"io"
	"io/ioutil"
	"sync"
	"testing"
)

func muxPair() (*mux, *mux) {
	a, b := memPipe()

	s := newMux(a)
	c := newMux(b)

	return s, c
}

// Returns both ends of a channel, and the mux for the the 2nd
// channel.
func channelPair(t *testing.T) (*channel, *channel, *mux) {
	c, s := muxPair()

	res := make(chan *channel, 1)
	go func() {
		newCh, ok := <-s.incomingChannels
		if !ok {
			t.Fatalf("No incoming channel")
		}
		if newCh.ChannelType() != "chan" {
			t.Fatalf("got type %q want chan", newCh.ChannelType())
		}
		ch, _, err := newCh.Accept()
		if err != nil {
			t.Fatalf("Accept %v", err)
		}
		res <- ch.(*channel)
	}()

	ch, err := c.openChannel("chan", nil)
	if err != nil {
		t.Fatalf("OpenChannel: %v", err)
	}

	return <-res, ch, c
}

// Test that stderr and stdout can be addressed from different
// goroutines. This is intended for use with the race detector.
func TestMuxChannelExtendedThreadSafety(t *testing.T) {
	writer, reader, mux := channelPair(t)
	defer writer.Close()
	defer reader.Close()
	defer mux.Close()

	var wr, rd sync.WaitGroup
	magic := "hello world"

	wr.Add(2)
	go func() {
		io.WriteString(writer, magic)
		wr.Done()
	}()
	go func() {
		io.WriteString(writer.Stderr(), magic)
		wr.Done()
	}()

	rd.Add(2)
	go func() {
		c, err := ioutil.ReadAll(reader)
		if string(c) != magic {
			t.Fatalf("stdout read got %q, want %q (error %s)", c, magic, err)
		}
		rd.Done()
	}()
	go func() {
		c, err := ioutil.ReadAll(reader.Stderr())
		if string(c) != magic {
			t.Fatalf("stderr read got %q, want %q (error %s)", c, magic, err)
		}
		rd.Done()
	}()

	wr.Wait()
	writer.CloseWrite()
	rd.Wait()
}

func TestMuxReadWrite(t *testing.T) {
	s, c, mux := channelPair(t)
	defer s.Close()
	defer c.Close()
	defer mux.Close()

	magic := "hello world"
	magicExt := "hello stderr"
	go func() {
		_, err := s.Write([]byte(magic))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
		_, err = s.Extended(1).Write([]byte(magicExt))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
		err = s.Close()
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	var buf [1024]byte
	n, err := c.Read(buf[:])
	if err != nil {
		t.Fatalf("server Read: %v", err)
	}
	got := string(buf[:n])
	if got != magic {
		t.Fatalf("server: got %q want %q", got, magic)
	}

	n, err = c.Extended(1).Read(buf[:])
	if err != nil {
		t.Fatalf("server Read: %v", err)
	}

	got = string(buf[:n])
	if got != magicExt {
		t.Fatalf("server: got %q want %q", got, magic)
	}
}

func TestMuxChannelOverflow(t *testing.T) {
	reader, writer, mux := channelPair(t)
	defer reader.Close()
	defer writer.Close()
	defer mux.Close()

	wDone := make(chan int, 1)
	go func() {
		if _, err := writer.Write(make([]byte, channelWindowSize)); err != nil {
			t.Errorf("could not fill window: %v", err)
		}
		writer.Write(make([]byte, 1))
		wDone <- 1
	}()
	writer.remoteWin.waitWriterBlocked()

	// Send 1 byte.
	packet := make([]byte, 1+4+4+1)
	packet[0] = msgChannelData
	marshalUint32(packet[1:], writer.remoteId)
	marshalUint32(packet[5:], uint32(1))
	packet[9] = 42

	if err := writer.mux.conn.writePacket(packet); err != nil {
		t.Errorf("could not send packet")
	}
	if _, err := reader.SendRequest("hello", true, nil); err == nil {
		t.Errorf("SendRequest succeeded.")
	}
	<-wDone
}

func TestMuxChannelCloseWriteUnblock(t *testing.T) {
	reader, writer, mux := channelPair(t)
	defer reader.Close()
	defer writer.Close()
	defer mux.Close()

	wDone := make(chan int, 1)
	go func() {
		if _, err := writer.Write(make([]byte, channelWindowSize)); err != nil {
			t.Errorf("could not fill window: %v", err)
		}
		if _, err := writer.Write(make([]byte, 1)); err != io.EOF {
			t.Errorf("got %v, want EOF for unblock write", err)
		}
		wDone <- 1
	}()

	writer.remoteWin.waitWriterBlocked()
	reader.Close()
	<-wDone
}

func TestMuxConnectionCloseWriteUnblock(t *testing.T) {
	reader, writer, mux := channelPair(t)
	defer reader.Close()
	defer writer.Close()
	defer mux.Close()

	wDone := make(chan int, 1)
	go func() {
		if _, err := writer.Write(make([]byte, channelWindowSize)); err != nil {
			t.Errorf("could not fill window: %v", err)
		}
		if _, err := writer.Write(make([]byte, 1)); err != io.EOF {
			t.Errorf("got %v, want EOF for unblock write", err)
		}
		wDone <- 1
	}()

	writer.remoteWin.waitWriterBlocked()
	mux.Close()
	<-wDone
}

func TestMuxReject(t *testing.T) {
	client, server := muxPair()
	defer server.Close()
	defer client.Close()

	go func() {
		ch, ok := <-server.incomingChannels
		if !ok {
			t.Fatalf("Accept")
		}
		if ch.ChannelType() != "ch" || string(ch.ExtraData()) != "extra" {
			t.Fatalf("unexpected channel: %q, %q", ch.ChannelType(), ch.ExtraData())
		}
		ch.Reject(RejectionReason(42), "message")
	}()

	ch, err := client.openChannel("ch", []byte("extra"))
	if ch != nil {
		t.Fatal("openChannel not rejected")
	}

	ocf, ok := err.(*OpenChannelError)
	if !ok {
		t.Errorf("got %#v want *OpenChannelError", err)
	} else if ocf.Reason != 42 || ocf.Message != "message" {
		t.Errorf("got %#v, want {Reason: 42, Message: %q}", ocf, "message")
	}

	want := "ssh: rejected: unknown reason 42 (message)"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestMuxChannelRequest(t *testing.T) {
	client, server, mux := channelPair(t)
	defer server.Close()
	defer client.Close()
	defer mux.Close()

	var received int
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		for r := range server.incomingRequests {
			received++
			r.Reply(r.Type == "yes", nil)
		}
		wg.Done()
	}()
	_, err := client.SendRequest("yes", false, nil)
	if err != nil {
		t.Fatalf("SendRequest: %v", err)
	}
	ok, err := client.SendRequest("yes", true, nil)
	if err != nil {
		t.Fatalf("SendRequest: %v", err)
	}

	if !ok {
		t.Errorf("SendRequest(yes): %v", ok)

	}

	ok, err = client.SendRequest("no", true, nil)
	if err != nil {
		t.Fatalf("SendRequest: %v", err)
	}
	if ok {
		t.Errorf("SendRequest(no): %v", ok)

	}

	client.Close()
	wg.Wait()

	if received != 3 {
		t.Errorf("got %d requests, want %d", received, 3)
	}
}

func TestMuxGlobalRequest(t *testing.T) {
	clientMux, serverMux := muxPair()
	defer serverMux.Close()
	defer clientMux.Close()

	var seen bool
	go func() {
		for r := range serverMux.incomingRequests {
			seen = seen || r.Type == "peek"
			if r.WantReply {
				err := r.Reply(r.Type == "yes",
					append([]byte(r.Type), r.Payload...))
				if err != nil {
					t.Errorf("AckRequest: %v", err)
				}
			}
		}
	}()

	_, _, err := clientMux.SendRequest("peek", false, nil)
	if err != nil {
		t.Errorf("SendRequest: %v", err)
	}

	ok, data, err := clientMux.SendRequest("yes", true, []byte("a"))
	if !ok || string(data) != "yesa" || err != nil {
		t.Errorf("SendRequest(\"yes\", true, \"a\"): %v %v %v",
			ok, data, err)
	}
	if ok, data, err := clientMux.SendRequest("yes", true, []byte("a")); !ok || string(data) != "yesa" || err != nil {
		t.Errorf("SendRequest(\"yes\", true, \"a\"): %v %v %v",
			ok, data, err)
	}

	if ok, data, err := clientMux.SendRequest("no", true, []byte("a")); ok || string(data) != "noa" || err != nil {
		t.Errorf("SendRequest(\"no\", true, \"a\"): %v %v %v",
			ok, data, err)
	}

	if !seen {
		t.Errorf("never saw 'peek' request")
	}
}

func TestMuxGlobalRequestUnblock(t *testing.T) {
	clientMux, serverMux := muxPair()
	defer serverMux.Close()
	defer clientMux.Close()

	result := make(chan error, 1)
	go func() {
		_, _, err := clientMux.SendRequest("hello", true, nil)
		result <- err
	}()

	<-serverMux.incomingRequests
	serverMux.conn.Close()
	err := <-result

	if err != io.EOF {
		t.Errorf("want EOF, got %v", io.EOF)
	}
}

func TestMuxChannelRequestUnblock(t *testing.T) {
	a, b, connB := channelPair(t)
	defer a.Close()
	defer b.Close()
	defer connB.Close()

	result := make(chan error, 1)
	go func() {
		_, err := a.SendRequest("hello", true, nil)
		result <- err
	}()

	<-b.incomingRequests
	connB.conn.Close()
	err := <-result

	if err != io.EOF {
		t.Errorf("want EOF, got %v", err)
	}
}

func TestMuxCloseChannel(t *testing.T) {
	r, w, mux := channelPair(t)
	defer mux.Close()
	defer r.Close()
	defer w.Close()

	result := make(chan error, 1)
	go func() {
		var b [1024]byte
		_, err := r.Read(b[:])
		result <- err
	}()
	if err := w.Close(); err != nil {
		t.Errorf("w.Close: %v", err)
	}

	if _, err := w.Write([]byte("hello")); err != io.EOF {
		t.Errorf("got err %v, want io.EOF after Close", err)
	}

	if err := <-result; err != io.EOF {
		t.Errorf("got %v (%T), want io.EOF", err, err)
	}
}

func TestMuxCloseWriteChannel(t *testing.T) {
	r, w, mux := channelPair(t)
	defer mux.Close()

	result := make(chan error, 1)
	go func() {
		var b [1024]byte
		_, err := r.Read(b[:])
		result <- err
	}()
	if err := w.CloseWrite(); err != nil {
		t.Errorf("w.CloseWrite: %v", err)
	}

	if _, err := w.Write([]byte("hello")); err != io.EOF {
		t.Errorf("got err %v, want io.EOF after CloseWrite", err)
	}

	if err := <-result; err != io.EOF {
		t.Errorf("got %v (%T), want io.EOF", err, err)
	}
}

func TestMuxInvalidRecord(t *testing.T) {
	a, b := muxPair()
	defer a.Close()
	defer b.Close()

	packet := make([]byte, 1+4+4+1)
	packet[0] = msgChannelData
	marshalUint32(packet[1:], 29348723 /* invalid channel id */)
	marshalUint32(packet[5:], 1)
	packet[9] = 42

	a.conn.writePacket(packet)
	go a.SendRequest("hello", false, nil)
	// 'a' wrote an invalid packet, so 'b' has exited.
	req, ok := <-b.incomingRequests
	if ok {
		t.Errorf("got request %#v after receiving invalid packet", req)
	}
}

func TestZeroWindowAdjust(t *testing.T) {
	a, b, mux := channelPair(t)
	defer a.Close()
	defer b.Close()
	defer mux.Close()

	go func() {
		io.WriteString(a, "hello")
		// bogus adjust.
		a.sendMessage(windowAdjustMsg{})
		io.WriteString(a, "world")
		a.Close()
	}()

	want := "helloworld"
	c, _ := ioutil.ReadAll(b)
	if string(c) != want {
		t.Errorf("got %q want %q", c, want)
	}
}

func TestMuxMaxPacketSize(t *testing.T) {
	a, b, mux := channelPair(t)
	defer a.Close()
	defer b.Close()
	defer mux.Close()

	large := make([]byte, a.maxRemotePayload+1)
	packet := make([]byte, 1+4+4+1+len(large))
	packet[0] = msgChannelData
	marshalUint32(packet[1:], a.remoteId)
	marshalUint32(packet[5:], uint32(len(large)))
	packet[9] = 42

	if err := a.mux.conn.writePacket(packet); err != nil {
		t.Errorf("could not send packet")
	}

	go a.SendRequest("hello", false, nil)

	_, ok := <-b.incomingRequests
	if ok {
		t.Errorf("connection still alive after receiving large packet.")
	}
}

// Don't ship code with debug=true.
func TestDebug(t *testing.T) {
	if debugMux {
		t.Error("mux debug switched on")
	}
	if debugHandshake {
		t.Error("handshake debug switched on")
	}
	if debugTransport {
		t.Error("transport debug switched on")
	}
}
