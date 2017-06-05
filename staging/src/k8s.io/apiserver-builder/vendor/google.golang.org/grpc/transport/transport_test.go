/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package transport

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"google.golang.org/grpc/codes"
)

type server struct {
	lis        net.Listener
	port       string
	startedErr chan error // error (or nil) with server start value
	mu         sync.Mutex
	conns      map[ServerTransport]bool
}

var (
	expectedRequest       = []byte("ping")
	expectedResponse      = []byte("pong")
	expectedRequestLarge  = make([]byte, initialWindowSize*2)
	expectedResponseLarge = make([]byte, initialWindowSize*2)
)

type testStreamHandler struct {
	t ServerTransport
}

type hType int

const (
	normal hType = iota
	suspended
	misbehaved
	encodingRequiredStatus
)

func (h *testStreamHandler) handleStream(t *testing.T, s *Stream) {
	req := expectedRequest
	resp := expectedResponse
	if s.Method() == "foo.Large" {
		req = expectedRequestLarge
		resp = expectedResponseLarge
	}
	p := make([]byte, len(req))
	_, err := io.ReadFull(s, p)
	if err != nil {
		return
	}
	if !bytes.Equal(p, req) {
		t.Fatalf("handleStream got %v, want %v", p, req)
	}
	// send a response back to the client.
	h.t.Write(s, resp, &Options{})
	// send the trailer to end the stream.
	h.t.WriteStatus(s, codes.OK, "")
}

// handleStreamSuspension blocks until s.ctx is canceled.
func (h *testStreamHandler) handleStreamSuspension(s *Stream) {
	go func() {
		<-s.ctx.Done()
	}()
}

func (h *testStreamHandler) handleStreamMisbehave(t *testing.T, s *Stream) {
	conn, ok := s.ServerTransport().(*http2Server)
	if !ok {
		t.Fatalf("Failed to convert %v to *http2Server", s.ServerTransport())
	}
	var sent int
	p := make([]byte, http2MaxFrameLen)
	for sent < initialWindowSize {
		<-conn.writableChan
		n := initialWindowSize - sent
		// The last message may be smaller than http2MaxFrameLen
		if n <= http2MaxFrameLen {
			if s.Method() == "foo.Connection" {
				// Violate connection level flow control window of client but do not
				// violate any stream level windows.
				p = make([]byte, n)
			} else {
				// Violate stream level flow control window of client.
				p = make([]byte, n+1)
			}
		}
		if err := conn.framer.writeData(true, s.id, false, p); err != nil {
			conn.writableChan <- 0
			break
		}
		conn.writableChan <- 0
		sent += len(p)
	}
}

func (h *testStreamHandler) handleStreamEncodingRequiredStatus(t *testing.T, s *Stream) {
	// raw newline is not accepted by http2 framer so it must be encoded.
	h.t.WriteStatus(s, encodingTestStatusCode, encodingTestStatusDesc)
}

// start starts server. Other goroutines should block on s.readyChan for further operations.
func (s *server) start(t *testing.T, port int, maxStreams uint32, ht hType) {
	var err error
	if port == 0 {
		s.lis, err = net.Listen("tcp", "localhost:0")
	} else {
		s.lis, err = net.Listen("tcp", "localhost:"+strconv.Itoa(port))
	}
	if err != nil {
		s.startedErr <- fmt.Errorf("failed to listen: %v", err)
		return
	}
	_, p, err := net.SplitHostPort(s.lis.Addr().String())
	if err != nil {
		s.startedErr <- fmt.Errorf("failed to parse listener address: %v", err)
		return
	}
	s.port = p
	s.conns = make(map[ServerTransport]bool)
	s.startedErr <- nil
	for {
		conn, err := s.lis.Accept()
		if err != nil {
			return
		}
		transport, err := NewServerTransport("http2", conn, maxStreams, nil)
		if err != nil {
			return
		}
		s.mu.Lock()
		if s.conns == nil {
			s.mu.Unlock()
			transport.Close()
			return
		}
		s.conns[transport] = true
		s.mu.Unlock()
		h := &testStreamHandler{transport}
		switch ht {
		case suspended:
			go transport.HandleStreams(h.handleStreamSuspension)
		case misbehaved:
			go transport.HandleStreams(func(s *Stream) {
				go h.handleStreamMisbehave(t, s)
			})
		case encodingRequiredStatus:
			go transport.HandleStreams(func(s *Stream) {
				go h.handleStreamEncodingRequiredStatus(t, s)
			})
		default:
			go transport.HandleStreams(func(s *Stream) {
				go h.handleStream(t, s)
			})
		}
	}
}

func (s *server) wait(t *testing.T, timeout time.Duration) {
	select {
	case err := <-s.startedErr:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(timeout):
		t.Fatalf("Timed out after %v waiting for server to be ready", timeout)
	}
}

func (s *server) stop() {
	s.lis.Close()
	s.mu.Lock()
	for c := range s.conns {
		c.Close()
	}
	s.conns = nil
	s.mu.Unlock()
}

func setUp(t *testing.T, port int, maxStreams uint32, ht hType) (*server, ClientTransport) {
	server := &server{startedErr: make(chan error, 1)}
	go server.start(t, port, maxStreams, ht)
	server.wait(t, 2*time.Second)
	addr := "localhost:" + server.port
	var (
		ct      ClientTransport
		connErr error
	)
	ct, connErr = NewClientTransport(context.Background(), addr, ConnectOptions{})
	if connErr != nil {
		t.Fatalf("failed to create transport: %v", connErr)
	}
	return server, ct
}

func TestClientSendAndReceive(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, normal)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Small",
	}
	s1, err1 := ct.NewStream(context.Background(), callHdr)
	if err1 != nil {
		t.Fatalf("failed to open stream: %v", err1)
	}
	if s1.id != 1 {
		t.Fatalf("wrong stream id: %d", s1.id)
	}
	s2, err2 := ct.NewStream(context.Background(), callHdr)
	if err2 != nil {
		t.Fatalf("failed to open stream: %v", err2)
	}
	if s2.id != 3 {
		t.Fatalf("wrong stream id: %d", s2.id)
	}
	opts := Options{
		Last:  true,
		Delay: false,
	}
	if err := ct.Write(s1, expectedRequest, &opts); err != nil && err != io.EOF {
		t.Fatalf("failed to send data: %v", err)
	}
	p := make([]byte, len(expectedResponse))
	_, recvErr := io.ReadFull(s1, p)
	if recvErr != nil || !bytes.Equal(p, expectedResponse) {
		t.Fatalf("Error: %v, want <nil>; Result: %v, want %v", recvErr, p, expectedResponse)
	}
	_, recvErr = io.ReadFull(s1, p)
	if recvErr != io.EOF {
		t.Fatalf("Error: %v; want <EOF>", recvErr)
	}
	ct.Close()
	server.stop()
}

func TestClientErrorNotify(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, normal)
	go server.stop()
	// ct.reader should detect the error and activate ct.Error().
	<-ct.Error()
	ct.Close()
}

func performOneRPC(ct ClientTransport) {
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Small",
	}
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		return
	}
	opts := Options{
		Last:  true,
		Delay: false,
	}
	if err := ct.Write(s, expectedRequest, &opts); err == nil || err == io.EOF {
		time.Sleep(5 * time.Millisecond)
		// The following s.Recv()'s could error out because the
		// underlying transport is gone.
		//
		// Read response
		p := make([]byte, len(expectedResponse))
		io.ReadFull(s, p)
		// Read io.EOF
		io.ReadFull(s, p)
	}
}

func TestClientMix(t *testing.T) {
	s, ct := setUp(t, 0, math.MaxUint32, normal)
	go func(s *server) {
		time.Sleep(5 * time.Second)
		s.stop()
	}(s)
	go func(ct ClientTransport) {
		<-ct.Error()
		ct.Close()
	}(ct)
	for i := 0; i < 1000; i++ {
		time.Sleep(10 * time.Millisecond)
		go performOneRPC(ct)
	}
}

func TestLargeMessage(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, normal)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Large",
	}
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := ct.NewStream(context.Background(), callHdr)
			if err != nil {
				t.Errorf("%v.NewStream(_, _) = _, %v, want _, <nil>", ct, err)
			}
			if err := ct.Write(s, expectedRequestLarge, &Options{Last: true, Delay: false}); err != nil && err != io.EOF {
				t.Errorf("%v.Write(_, _, _) = %v, want  <nil>", ct, err)
			}
			p := make([]byte, len(expectedResponseLarge))
			if _, err := io.ReadFull(s, p); err != nil || !bytes.Equal(p, expectedResponseLarge) {
				t.Errorf("io.ReadFull(_, %v) = _, %v, want %v, <nil>", err, p, expectedResponse)
			}
			if _, err = io.ReadFull(s, p); err != io.EOF {
				t.Errorf("Failed to complete the stream %v; want <EOF>", err)
			}
		}()
	}
	wg.Wait()
	ct.Close()
	server.stop()
}

func TestGracefulClose(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, normal)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Small",
	}
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		t.Fatalf("%v.NewStream(_, _) = _, %v, want _, <nil>", ct, err)
	}
	if err = ct.GracefulClose(); err != nil {
		t.Fatalf("%v.GracefulClose() = %v, want <nil>", ct, err)
	}
	var wg sync.WaitGroup
	// Expect the failure for all the follow-up streams because ct has been closed gracefully.
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := ct.NewStream(context.Background(), callHdr); err != ErrStreamDrain {
				t.Errorf("%v.NewStream(_, _) = _, %v, want _, %v", ct, err, ErrStreamDrain)
			}
		}()
	}
	opts := Options{
		Last:  true,
		Delay: false,
	}
	// The stream which was created before graceful close can still proceed.
	if err := ct.Write(s, expectedRequest, &opts); err != nil && err != io.EOF {
		t.Fatalf("%v.Write(_, _, _) = %v, want  <nil>", ct, err)
	}
	p := make([]byte, len(expectedResponse))
	if _, err := io.ReadFull(s, p); err != nil || !bytes.Equal(p, expectedResponse) {
		t.Fatalf("io.ReadFull(_, %v) = _, %v, want %v, <nil>", err, p, expectedResponse)
	}
	if _, err = io.ReadFull(s, p); err != io.EOF {
		t.Fatalf("Failed to complete the stream %v; want <EOF>", err)
	}
	wg.Wait()
	ct.Close()
	server.stop()
}

func TestLargeMessageSuspension(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, suspended)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Large",
	}
	// Set a long enough timeout for writing a large message out.
	ctx, _ := context.WithTimeout(context.Background(), time.Second)
	s, err := ct.NewStream(ctx, callHdr)
	if err != nil {
		t.Fatalf("failed to open stream: %v", err)
	}
	// Write should not be done successfully due to flow control.
	err = ct.Write(s, expectedRequestLarge, &Options{Last: true, Delay: false})
	expectedErr := StreamErrorf(codes.DeadlineExceeded, "%v", context.DeadlineExceeded)
	if err != expectedErr {
		t.Fatalf("Write got %v, want %v", err, expectedErr)
	}
	ct.Close()
	server.stop()
}

func TestMaxStreams(t *testing.T) {
	server, ct := setUp(t, 0, 1, suspended)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Large",
	}
	// Have a pending stream which takes all streams quota.
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}
	cc, ok := ct.(*http2Client)
	if !ok {
		t.Fatalf("Failed to convert %v to *http2Client", ct)
	}
	done := make(chan struct{})
	ch := make(chan int)
	ready := make(chan struct{})
	go func() {
		for {
			select {
			case <-time.After(5 * time.Millisecond):
				select {
				case ch <- 0:
				case <-ready:
					return
				}
			case <-time.After(5 * time.Second):
				close(done)
				return
			case <-ready:
				return
			}
		}
	}()
	for {
		select {
		case <-ch:
		case <-done:
			t.Fatalf("Client has not received the max stream setting in 5 seconds.")
		}
		cc.mu.Lock()
		// cc.streamsQuota should be initialized once receiving the 1st setting frame from
		// the server.
		if cc.streamsQuota != nil {
			cc.mu.Unlock()
			select {
			case <-cc.streamsQuota.acquire():
				t.Fatalf("streamsQuota.acquire() becomes readable mistakenly.")
			default:
				if cc.streamsQuota.quota != 0 {
					t.Fatalf("streamsQuota.quota got non-zero quota mistakenly.")
				}
			}
			break
		}
		cc.mu.Unlock()
	}
	close(ready)
	// Close the pending stream so that the streams quota becomes available for the next new stream.
	ct.CloseStream(s, nil)
	select {
	case i := <-cc.streamsQuota.acquire():
		if i != 1 {
			t.Fatalf("streamsQuota.acquire() got %d quota, want 1.", i)
		}
		cc.streamsQuota.add(i)
	default:
		t.Fatalf("streamsQuota.acquire() is not readable.")
	}
	if _, err := ct.NewStream(context.Background(), callHdr); err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}
	ct.Close()
	server.stop()
}

func TestServerContextCanceledOnClosedConnection(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, suspended)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo",
	}
	var sc *http2Server
	// Wait until the server transport is setup.
	for {
		server.mu.Lock()
		if len(server.conns) == 0 {
			server.mu.Unlock()
			time.Sleep(time.Millisecond)
			continue
		}
		for k := range server.conns {
			var ok bool
			sc, ok = k.(*http2Server)
			if !ok {
				t.Fatalf("Failed to convert %v to *http2Server", k)
			}
		}
		server.mu.Unlock()
		break
	}
	cc, ok := ct.(*http2Client)
	if !ok {
		t.Fatalf("Failed to convert %v to *http2Client", ct)
	}
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}
	// Make sure the headers frame is flushed out.
	<-cc.writableChan
	if err = cc.framer.writeData(true, s.id, false, make([]byte, http2MaxFrameLen)); err != nil {
		t.Fatalf("Failed to write data: %v", err)
	}
	cc.writableChan <- 0
	// Loop until the server side stream is created.
	var ss *Stream
	for {
		time.Sleep(time.Second)
		sc.mu.Lock()
		if len(sc.activeStreams) == 0 {
			sc.mu.Unlock()
			continue
		}
		ss = sc.activeStreams[s.id]
		sc.mu.Unlock()
		break
	}
	cc.Close()
	select {
	case <-ss.Context().Done():
		if ss.Context().Err() != context.Canceled {
			t.Fatalf("ss.Context().Err() got %v, want %v", ss.Context().Err(), context.Canceled)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("Failed to cancel the context of the sever side stream.")
	}
	server.stop()
}

func TestServerWithMisbehavedClient(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, suspended)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo",
	}
	var sc *http2Server
	// Wait until the server transport is setup.
	for {
		server.mu.Lock()
		if len(server.conns) == 0 {
			server.mu.Unlock()
			time.Sleep(time.Millisecond)
			continue
		}
		for k := range server.conns {
			var ok bool
			sc, ok = k.(*http2Server)
			if !ok {
				t.Fatalf("Failed to convert %v to *http2Server", k)
			}
		}
		server.mu.Unlock()
		break
	}
	cc, ok := ct.(*http2Client)
	if !ok {
		t.Fatalf("Failed to convert %v to *http2Client", ct)
	}
	// Test server behavior for violation of stream flow control window size restriction.
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}
	var sent int
	// Drain the stream flow control window
	<-cc.writableChan
	if err = cc.framer.writeData(true, s.id, false, make([]byte, http2MaxFrameLen)); err != nil {
		t.Fatalf("Failed to write data: %v", err)
	}
	cc.writableChan <- 0
	sent += http2MaxFrameLen
	// Wait until the server creates the corresponding stream and receive some data.
	var ss *Stream
	for {
		time.Sleep(time.Millisecond)
		sc.mu.Lock()
		if len(sc.activeStreams) == 0 {
			sc.mu.Unlock()
			continue
		}
		ss = sc.activeStreams[s.id]
		sc.mu.Unlock()
		ss.fc.mu.Lock()
		if ss.fc.pendingData > 0 {
			ss.fc.mu.Unlock()
			break
		}
		ss.fc.mu.Unlock()
	}
	if ss.fc.pendingData != http2MaxFrameLen || ss.fc.pendingUpdate != 0 || sc.fc.pendingData != http2MaxFrameLen || sc.fc.pendingUpdate != 0 {
		t.Fatalf("Server mistakenly updates inbound flow control params: got %d, %d, %d, %d; want %d, %d, %d, %d", ss.fc.pendingData, ss.fc.pendingUpdate, sc.fc.pendingData, sc.fc.pendingUpdate, http2MaxFrameLen, 0, http2MaxFrameLen, 0)
	}
	// Keep sending until the server inbound window is drained for that stream.
	for sent <= initialWindowSize {
		<-cc.writableChan
		if err = cc.framer.writeData(true, s.id, false, make([]byte, 1)); err != nil {
			t.Fatalf("Failed to write data: %v", err)
		}
		cc.writableChan <- 0
		sent++
	}
	// Server sent a resetStream for s already.
	code := http2ErrConvTab[http2.ErrCodeFlowControl]
	if _, err := io.ReadFull(s, make([]byte, 1)); err != io.EOF || s.statusCode != code {
		t.Fatalf("%v got err %v with statusCode %d, want err <EOF> with statusCode %d", s, err, s.statusCode, code)
	}

	if ss.fc.pendingData != 0 || ss.fc.pendingUpdate != 0 || sc.fc.pendingData != 0 || sc.fc.pendingUpdate <= initialWindowSize {
		t.Fatalf("Server mistakenly resets inbound flow control params: got %d, %d, %d, %d; want 0, 0, 0, >%d", ss.fc.pendingData, ss.fc.pendingUpdate, sc.fc.pendingData, sc.fc.pendingUpdate, initialWindowSize)
	}
	ct.CloseStream(s, nil)
	// Test server behavior for violation of connection flow control window size restriction.
	//
	// Keep creating new streams until the connection window is drained on the server and
	// the server tears down the connection.
	for {
		s, err := ct.NewStream(context.Background(), callHdr)
		if err != nil {
			// The server tears down the connection.
			break
		}
		<-cc.writableChan
		cc.framer.writeData(true, s.id, true, make([]byte, http2MaxFrameLen))
		cc.writableChan <- 0
	}
	ct.Close()
	server.stop()
}

func TestClientWithMisbehavedServer(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, misbehaved)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Stream",
	}
	conn, ok := ct.(*http2Client)
	if !ok {
		t.Fatalf("Failed to convert %v to *http2Client", ct)
	}
	// Test the logic for the violation of stream flow control window size restriction.
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}
	d := make([]byte, 1)
	if err := ct.Write(s, d, &Options{Last: true, Delay: false}); err != nil && err != io.EOF {
		t.Fatalf("Failed to write: %v", err)
	}
	// Read without window update.
	for {
		p := make([]byte, http2MaxFrameLen)
		if _, err = s.dec.Read(p); err != nil {
			break
		}
	}
	if s.fc.pendingData <= initialWindowSize || s.fc.pendingUpdate != 0 || conn.fc.pendingData <= initialWindowSize || conn.fc.pendingUpdate != 0 {
		t.Fatalf("Client mistakenly updates inbound flow control params: got %d, %d, %d, %d; want >%d, %d, >%d, %d", s.fc.pendingData, s.fc.pendingUpdate, conn.fc.pendingData, conn.fc.pendingUpdate, initialWindowSize, 0, initialWindowSize, 0)
	}
	if err != io.EOF || s.statusCode != codes.Internal {
		t.Fatalf("Got err %v and the status code %d, want <EOF> and the code %d", err, s.statusCode, codes.Internal)
	}
	conn.CloseStream(s, err)
	if s.fc.pendingData != 0 || s.fc.pendingUpdate != 0 || conn.fc.pendingData != 0 || conn.fc.pendingUpdate <= initialWindowSize {
		t.Fatalf("Client mistakenly resets inbound flow control params: got %d, %d, %d, %d; want 0, 0, 0, >%d", s.fc.pendingData, s.fc.pendingUpdate, conn.fc.pendingData, conn.fc.pendingUpdate, initialWindowSize)
	}
	// Test the logic for the violation of the connection flow control window size restriction.
	//
	// Generate enough streams to drain the connection window. Make the server flood the traffic
	// to violate flow control window size of the connection.
	callHdr.Method = "foo.Connection"
	for i := 0; i < int(initialConnWindowSize/initialWindowSize+10); i++ {
		s, err := ct.NewStream(context.Background(), callHdr)
		if err != nil {
			break
		}
		if err := ct.Write(s, d, &Options{Last: true, Delay: false}); err != nil {
			break
		}
	}
	// http2Client.errChan is closed due to connection flow control window size violation.
	<-conn.Error()
	ct.Close()
	server.stop()
}

var (
	encodingTestStatusCode = codes.Internal
	encodingTestStatusDesc = "\n"
)

func TestEncodingRequiredStatus(t *testing.T) {
	server, ct := setUp(t, 0, math.MaxUint32, encodingRequiredStatus)
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo",
	}
	s, err := ct.NewStream(context.Background(), callHdr)
	if err != nil {
		return
	}
	opts := Options{
		Last:  true,
		Delay: false,
	}
	if err := ct.Write(s, expectedRequest, &opts); err != nil && err != io.EOF {
		t.Fatalf("Failed to write the request: %v", err)
	}
	p := make([]byte, http2MaxFrameLen)
	if _, err := s.dec.Read(p); err != io.EOF {
		t.Fatalf("Read got error %v, want %v", err, io.EOF)
	}
	if s.StatusCode() != encodingTestStatusCode || s.StatusDesc() != encodingTestStatusDesc {
		t.Fatalf("stream with status code %d, status desc %v, want %d, %v", s.StatusCode(), s.StatusDesc(), encodingTestStatusCode, encodingTestStatusDesc)
	}
	ct.Close()
	server.stop()
}

func TestStreamContext(t *testing.T) {
	expectedStream := &Stream{}
	ctx := newContextWithStream(context.Background(), expectedStream)
	s, ok := StreamFromContext(ctx)
	if !ok || expectedStream != s {
		t.Fatalf("GetStreamFromContext(%v) = %v, %t, want: %v, true", ctx, s, ok, expectedStream)
	}
}

func TestIsReservedHeader(t *testing.T) {
	tests := []struct {
		h    string
		want bool
	}{
		{"", false}, // but should be rejected earlier
		{"foo", false},
		{"content-type", true},
		{"grpc-message-type", true},
		{"grpc-encoding", true},
		{"grpc-message", true},
		{"grpc-status", true},
		{"grpc-timeout", true},
		{"te", true},
	}
	for _, tt := range tests {
		got := isReservedHeader(tt.h)
		if got != tt.want {
			t.Errorf("isReservedHeader(%q) = %v; want %v", tt.h, got, tt.want)
		}
	}
}
