package spdystream

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/docker/spdystream/spdy"
)

func TestSpdyStreams(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	authenticated = true
	stream, streamErr := spdyConn.CreateStream(http.Header{}, nil, false)
	if streamErr != nil {
		t.Fatalf("Error creating stream: %s", streamErr)
	}

	waitErr := stream.Wait()
	if waitErr != nil {
		t.Fatalf("Error waiting for stream: %s", waitErr)
	}

	message := []byte("hello")
	writeErr := stream.WriteData(message, false)
	if writeErr != nil {
		t.Fatalf("Error writing data")
	}

	buf := make([]byte, 10)
	n, readErr := stream.Read(buf)
	if readErr != nil {
		t.Fatalf("Error reading data from stream: %s", readErr)
	}
	if n != 5 {
		t.Fatalf("Unexpected number of bytes read:\nActual: %d\nExpected: 5", n)
	}
	if bytes.Compare(buf[:n], message) != 0 {
		t.Fatalf("Did not receive expected message:\nActual: %s\nExpectd: %s", buf, message)
	}

	headers := http.Header{
		"TestKey": []string{"TestVal"},
	}
	sendErr := stream.SendHeader(headers, false)
	if sendErr != nil {
		t.Fatalf("Error sending headers: %s", sendErr)
	}
	receiveHeaders, receiveErr := stream.ReceiveHeader()
	if receiveErr != nil {
		t.Fatalf("Error receiving headers: %s", receiveErr)
	}
	if len(receiveHeaders) != 1 {
		t.Fatalf("Unexpected number of headers:\nActual: %d\nExpecting:%d", len(receiveHeaders), 1)
	}
	testVal := receiveHeaders.Get("TestKey")
	if testVal != "TestVal" {
		t.Fatalf("Wrong test value:\nActual: %q\nExpecting: %q", testVal, "TestVal")
	}

	writeErr = stream.WriteData(message, true)
	if writeErr != nil {
		t.Fatalf("Error writing data")
	}

	smallBuf := make([]byte, 3)
	n, readErr = stream.Read(smallBuf)
	if readErr != nil {
		t.Fatalf("Error reading data from stream: %s", readErr)
	}
	if n != 3 {
		t.Fatalf("Unexpected number of bytes read:\nActual: %d\nExpected: 3", n)
	}
	if bytes.Compare(smallBuf[:n], []byte("hel")) != 0 {
		t.Fatalf("Did not receive expected message:\nActual: %s\nExpectd: %s", smallBuf[:n], message)
	}
	n, readErr = stream.Read(smallBuf)
	if readErr != nil {
		t.Fatalf("Error reading data from stream: %s", readErr)
	}
	if n != 2 {
		t.Fatalf("Unexpected number of bytes read:\nActual: %d\nExpected: 2", n)
	}
	if bytes.Compare(smallBuf[:n], []byte("lo")) != 0 {
		t.Fatalf("Did not receive expected message:\nActual: %s\nExpected: lo", smallBuf[:n])
	}

	n, readErr = stream.Read(buf)
	if readErr != io.EOF {
		t.Fatalf("Expected EOF reading from finished stream, read %d bytes", n)
	}

	// Closing again should return error since stream is already closed
	streamCloseErr := stream.Close()
	if streamCloseErr == nil {
		t.Fatalf("No error closing finished stream")
	}
	if streamCloseErr != ErrWriteClosedStream {
		t.Fatalf("Unexpected error closing stream: %s", streamCloseErr)
	}

	streamResetErr := stream.Reset()
	if streamResetErr != nil {
		t.Fatalf("Error reseting stream: %s", streamResetErr)
	}

	authenticated = false
	badStream, badStreamErr := spdyConn.CreateStream(http.Header{}, nil, false)
	if badStreamErr != nil {
		t.Fatalf("Error creating stream: %s", badStreamErr)
	}

	waitErr = badStream.Wait()
	if waitErr == nil {
		t.Fatalf("Did not receive error creating stream")
	}
	if waitErr != ErrReset {
		t.Fatalf("Unexpected error creating stream: %s", waitErr)
	}
	streamCloseErr = badStream.Close()
	if streamCloseErr == nil {
		t.Fatalf("No error closing bad stream")
	}

	spdyCloseErr := spdyConn.Close()
	if spdyCloseErr != nil {
		t.Fatalf("Error closing spdy connection: %s", spdyCloseErr)
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestPing(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	pingTime, pingErr := spdyConn.Ping()
	if pingErr != nil {
		t.Fatalf("Error pinging server: %s", pingErr)
	}
	if pingTime == time.Duration(0) {
		t.Fatalf("Expecting non-zero ping time")
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestHalfClose(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	authenticated = true
	stream, streamErr := spdyConn.CreateStream(http.Header{}, nil, false)
	if streamErr != nil {
		t.Fatalf("Error creating stream: %s", streamErr)
	}

	waitErr := stream.Wait()
	if waitErr != nil {
		t.Fatalf("Error waiting for stream: %s", waitErr)
	}

	message := []byte("hello and will read after close")
	writeErr := stream.WriteData(message, false)
	if writeErr != nil {
		t.Fatalf("Error writing data")
	}

	streamCloseErr := stream.Close()
	if streamCloseErr != nil {
		t.Fatalf("Error closing stream: %s", streamCloseErr)
	}

	buf := make([]byte, 40)
	n, readErr := stream.Read(buf)
	if readErr != nil {
		t.Fatalf("Error reading data from stream: %s", readErr)
	}
	if n != 31 {
		t.Fatalf("Unexpected number of bytes read:\nActual: %d\nExpected: 5", n)
	}
	if bytes.Compare(buf[:n], message) != 0 {
		t.Fatalf("Did not receive expected message:\nActual: %s\nExpectd: %s", buf, message)
	}

	spdyCloseErr := spdyConn.Close()
	if spdyCloseErr != nil {
		t.Fatalf("Error closing spdy connection: %s", spdyCloseErr)
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestUnexpectedRemoteConnectionClosed(t *testing.T) {
	tt := []struct {
		closeReceiver bool
		closeSender   bool
	}{
		{closeReceiver: true, closeSender: false},
		{closeReceiver: false, closeSender: true},
		{closeReceiver: false, closeSender: false},
	}
	for tix, tc := range tt {
		listener, listenErr := net.Listen("tcp", "localhost:0")
		if listenErr != nil {
			t.Fatalf("Error listening: %v", listenErr)
		}

		var serverConn net.Conn
		var connErr error
		go func() {
			serverConn, connErr = listener.Accept()
			if connErr != nil {
				t.Fatalf("Error accepting: %v", connErr)
			}

			serverSpdyConn, _ := NewConnection(serverConn, true)
			go serverSpdyConn.Serve(func(stream *Stream) {
				stream.SendReply(http.Header{}, tc.closeSender)
			})
		}()

		conn, dialErr := net.Dial("tcp", listener.Addr().String())
		if dialErr != nil {
			t.Fatalf("Error dialing server: %s", dialErr)
		}

		spdyConn, spdyErr := NewConnection(conn, false)
		if spdyErr != nil {
			t.Fatalf("Error creating spdy connection: %s", spdyErr)
		}
		go spdyConn.Serve(NoOpStreamHandler)

		authenticated = true
		stream, streamErr := spdyConn.CreateStream(http.Header{}, nil, false)
		if streamErr != nil {
			t.Fatalf("Error creating stream: %s", streamErr)
		}

		waitErr := stream.Wait()
		if waitErr != nil {
			t.Fatalf("Error waiting for stream: %s", waitErr)
		}

		if tc.closeReceiver {
			// make stream half closed, receive only
			stream.Close()
		}

		streamch := make(chan error, 1)
		go func() {
			b := make([]byte, 1)
			_, err := stream.Read(b)
			streamch <- err
		}()

		closeErr := serverConn.Close()
		if closeErr != nil {
			t.Fatalf("Error shutting down server: %s", closeErr)
		}

		select {
		case e := <-streamch:
			if e == nil || e != io.EOF {
				t.Fatalf("(%d) Expected to get an EOF stream error", tix)
			}
		}

		closeErr = conn.Close()
		if closeErr != nil {
			t.Fatalf("Error closing client connection: %s", closeErr)
		}

		listenErr = listener.Close()
		if listenErr != nil {
			t.Fatalf("Error closing listener: %s", listenErr)
		}
	}
}

func TestCloseNotification(t *testing.T) {
	listener, listenErr := net.Listen("tcp", "localhost:0")
	if listenErr != nil {
		t.Fatalf("Error listening: %v", listenErr)
	}
	listen := listener.Addr().String()

	serverConnChan := make(chan net.Conn)
	go func() {
		serverConn, err := listener.Accept()
		if err != nil {
			t.Fatalf("Error accepting: %v", err)
		}

		serverSpdyConn, err := NewConnection(serverConn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go serverSpdyConn.Serve(NoOpStreamHandler)
		<-serverSpdyConn.CloseChan()
		serverConnChan <- serverConn
	}()

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	// close client conn
	err := conn.Close()
	if err != nil {
		t.Fatalf("Error closing client connection: %v", err)
	}

	var serverConn net.Conn
	select {
	case serverConn = <-serverConnChan:
	}

	err = serverConn.Close()
	if err != nil {
		t.Fatalf("Error closing serverConn: %v", err)
	}

	listenErr = listener.Close()
	if listenErr != nil {
		t.Fatalf("Error closing listener: %s", listenErr)
	}
}

func TestIdleShutdownRace(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	authenticated = true
	stream, err := spdyConn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("Error creating stream: %v", err)
	}

	spdyConn.SetIdleTimeout(5 * time.Millisecond)
	go func() {
		time.Sleep(5 * time.Millisecond)
		stream.Reset()
	}()

	select {
	case <-spdyConn.CloseChan():
	case <-time.After(20 * time.Millisecond):
		t.Fatal("Timed out waiting for idle connection closure")
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestIdleNoTimeoutSet(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	select {
	case <-spdyConn.CloseChan():
		t.Fatal("Unexpected connection closure")
	case <-time.After(10 * time.Millisecond):
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestIdleClearTimeout(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	spdyConn.SetIdleTimeout(10 * time.Millisecond)
	spdyConn.SetIdleTimeout(0)
	select {
	case <-spdyConn.CloseChan():
		t.Fatal("Unexpected connection closure")
	case <-time.After(20 * time.Millisecond):
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestIdleNoData(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	spdyConn.SetIdleTimeout(10 * time.Millisecond)
	<-spdyConn.CloseChan()

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestIdleWithData(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	spdyConn.SetIdleTimeout(25 * time.Millisecond)

	authenticated = true
	stream, err := spdyConn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("Error creating stream: %v", err)
	}

	writeCh := make(chan struct{})

	go func() {
		b := []byte{1, 2, 3, 4, 5}
		for i := 0; i < 10; i++ {
			_, err = stream.Write(b)
			if err != nil {
				t.Fatalf("Error writing to stream: %v", err)
			}
			time.Sleep(10 * time.Millisecond)
		}
		close(writeCh)
	}()

	writesFinished := false

Loop:
	for {
		select {
		case <-writeCh:
			writesFinished = true
		case <-spdyConn.CloseChan():
			if !writesFinished {
				t.Fatal("Connection closed before all writes finished")
			}
			break Loop
		}
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestIdleRace(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	spdyConn.SetIdleTimeout(10 * time.Millisecond)

	authenticated = true

	for i := 0; i < 10; i++ {
		_, err := spdyConn.CreateStream(http.Header{}, nil, false)
		if err != nil {
			t.Fatalf("Error creating stream: %v", err)
		}
	}

	<-spdyConn.CloseChan()

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestHalfClosedIdleTimeout(t *testing.T) {
	listener, listenErr := net.Listen("tcp", "localhost:0")
	if listenErr != nil {
		t.Fatalf("Error listening: %v", listenErr)
	}
	listen := listener.Addr().String()

	go func() {
		serverConn, err := listener.Accept()
		if err != nil {
			t.Fatalf("Error accepting: %v", err)
		}

		serverSpdyConn, err := NewConnection(serverConn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go serverSpdyConn.Serve(func(s *Stream) {
			s.SendReply(http.Header{}, true)
		})
		serverSpdyConn.SetIdleTimeout(10 * time.Millisecond)
	}()

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	stream, err := spdyConn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("Error creating stream: %v", err)
	}

	time.Sleep(20 * time.Millisecond)

	stream.Reset()

	err = spdyConn.Close()
	if err != nil {
		t.Fatalf("Error closing client spdy conn: %v", err)
	}
}

func TestStreamReset(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	authenticated = true
	stream, streamErr := spdyConn.CreateStream(http.Header{}, nil, false)
	if streamErr != nil {
		t.Fatalf("Error creating stream: %s", streamErr)
	}

	buf := []byte("dskjahfkdusahfkdsahfkdsafdkas")
	for i := 0; i < 10; i++ {
		if _, err := stream.Write(buf); err != nil {
			t.Fatalf("Error writing to stream: %s", err)
		}
	}
	for i := 0; i < 10; i++ {
		if _, err := stream.Read(buf); err != nil {
			t.Fatalf("Error reading from stream: %s", err)
		}
	}

	// fmt.Printf("Resetting...\n")
	if err := stream.Reset(); err != nil {
		t.Fatalf("Error reseting stream: %s", err)
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

func TestStreamResetWithDataRemaining(t *testing.T) {
	var wg sync.WaitGroup
	server, listen, serverErr := runServer(&wg)
	if serverErr != nil {
		t.Fatalf("Error initializing server: %s", serverErr)
	}

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	authenticated = true
	stream, streamErr := spdyConn.CreateStream(http.Header{}, nil, false)
	if streamErr != nil {
		t.Fatalf("Error creating stream: %s", streamErr)
	}

	buf := []byte("dskjahfkdusahfkdsahfkdsafdkas")
	for i := 0; i < 10; i++ {
		if _, err := stream.Write(buf); err != nil {
			t.Fatalf("Error writing to stream: %s", err)
		}
	}

	// read a bit to make sure a goroutine gets to <-dataChan
	if _, err := stream.Read(buf); err != nil {
		t.Fatalf("Error reading from stream: %s", err)
	}

	// fmt.Printf("Resetting...\n")
	if err := stream.Reset(); err != nil {
		t.Fatalf("Error reseting stream: %s", err)
	}

	closeErr := server.Close()
	if closeErr != nil {
		t.Fatalf("Error shutting down server: %s", closeErr)
	}
	wg.Wait()
}

type roundTripper struct {
	conn net.Conn
}

func (s *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	r := *req
	req = &r

	conn, err := net.Dial("tcp", req.URL.Host)
	if err != nil {
		return nil, err
	}

	err = req.Write(conn)
	if err != nil {
		return nil, err
	}

	resp, err := http.ReadResponse(bufio.NewReader(conn), req)
	if err != nil {
		return nil, err
	}

	s.conn = conn

	return resp, nil
}

// see https://github.com/GoogleCloudPlatform/kubernetes/issues/4882
func TestFramingAfterRemoteConnectionClosed(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamCh := make(chan *Stream)

		w.WriteHeader(http.StatusSwitchingProtocols)

		netconn, _, _ := w.(http.Hijacker).Hijack()
		conn, _ := NewConnection(netconn, true)
		go conn.Serve(func(s *Stream) {
			s.SendReply(http.Header{}, false)
			streamCh <- s
		})

		stream := <-streamCh
		io.Copy(stream, stream)

		closeChan := make(chan struct{})
		go func() {
			stream.Reset()
			conn.Close()
			close(closeChan)
		}()

		<-closeChan
	}))

	server.Start()
	defer server.Close()

	req, err := http.NewRequest("GET", server.URL, nil)
	if err != nil {
		t.Fatalf("Error creating request: %s", err)
	}

	rt := &roundTripper{}
	client := &http.Client{Transport: rt}

	_, err = client.Do(req)
	if err != nil {
		t.Fatalf("unexpected error from client.Do: %s", err)
	}

	conn, err := NewConnection(rt.conn, false)
	go conn.Serve(NoOpStreamHandler)

	stream, err := conn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("error creating client stream: %s", err)
	}

	n, err := stream.Write([]byte("hello"))
	if err != nil {
		t.Fatalf("error writing to stream: %s", err)
	}
	if n != 5 {
		t.Fatalf("Expected to write 5 bytes, but actually wrote %d", n)
	}

	b := make([]byte, 5)
	n, err = stream.Read(b)
	if err != nil {
		t.Fatalf("error reading from stream: %s", err)
	}
	if n != 5 {
		t.Fatalf("Expected to read 5 bytes, but actually read %d", n)
	}
	if e, a := "hello", string(b[0:n]); e != a {
		t.Fatalf("expected '%s', got '%s'", e, a)
	}

	stream.Reset()
	conn.Close()
}

func TestGoAwayRace(t *testing.T) {
	var done sync.WaitGroup
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error listening: %v", err)
	}
	listen := listener.Addr().String()

	processDataFrame := make(chan struct{})
	serverClosed := make(chan struct{})

	done.Add(1)
	go func() {
		defer done.Done()
		serverConn, err := listener.Accept()
		if err != nil {
			t.Fatalf("Error accepting: %v", err)
		}

		serverSpdyConn, err := NewConnection(serverConn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go func() {
			<-serverSpdyConn.CloseChan()
			close(serverClosed)
		}()

		// force the data frame handler to sleep before delivering the frame
		serverSpdyConn.dataFrameHandler = func(frame *spdy.DataFrame) error {
			<-processDataFrame
			return serverSpdyConn.handleDataFrame(frame)
		}

		streamCh := make(chan *Stream)
		go serverSpdyConn.Serve(func(s *Stream) {
			s.SendReply(http.Header{}, false)
			streamCh <- s
		})

		stream, ok := <-streamCh
		if !ok {
			t.Fatalf("didn't get a stream")
		}
		stream.Close()
		data, err := ioutil.ReadAll(stream)
		if err != nil {
			t.Error(err)
		}
		if e, a := "hello1hello2hello3hello4hello5", string(data); e != a {
			t.Errorf("Expected %q, got %q", e, a)
		}
	}()

	dialConn, err := net.Dial("tcp", listen)
	if err != nil {
		t.Fatalf("Error dialing server: %s", err)
	}
	conn, err := NewConnection(dialConn, false)
	if err != nil {
		t.Fatalf("Error creating client connectin: %v", err)
	}
	go conn.Serve(NoOpStreamHandler)

	stream, err := conn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("error creating client stream: %s", err)
	}
	if err := stream.Wait(); err != nil {
		t.Fatalf("error waiting for stream creation: %v", err)
	}

	fmt.Fprint(stream, "hello1")
	fmt.Fprint(stream, "hello2")
	fmt.Fprint(stream, "hello3")
	fmt.Fprint(stream, "hello4")
	fmt.Fprint(stream, "hello5")

	stream.Close()
	conn.Close()

	// wait for the server to get the go away frame
	<-serverClosed

	// allow the data frames to be delivered to the server's stream
	close(processDataFrame)

	done.Wait()
}

func TestSetIdleTimeoutAfterRemoteConnectionClosed(t *testing.T) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error listening: %v", err)
	}
	listen := listener.Addr().String()

	serverConns := make(chan *Connection, 1)
	go func() {
		conn, connErr := listener.Accept()
		if connErr != nil {
			t.Fatal(connErr)
		}
		serverSpdyConn, err := NewConnection(conn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go serverSpdyConn.Serve(NoOpStreamHandler)
		serverConns <- serverSpdyConn
	}()

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	if err := spdyConn.Close(); err != nil {
		t.Fatal(err)
	}

	serverConn := <-serverConns
	defer serverConn.Close()
	<-serverConn.closeChan

	serverConn.SetIdleTimeout(10 * time.Second)
}

func TestClientConnectionStopsServingAfterGoAway(t *testing.T) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error listening: %v", err)
	}
	listen := listener.Addr().String()

	serverConns := make(chan *Connection, 1)
	go func() {
		conn, connErr := listener.Accept()
		if connErr != nil {
			t.Fatal(connErr)
		}
		serverSpdyConn, err := NewConnection(conn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go serverSpdyConn.Serve(NoOpStreamHandler)
		serverConns <- serverSpdyConn
	}()

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	stream, err := spdyConn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("Error creating stream: %v", err)
	}
	if err := stream.WaitTimeout(30 * time.Second); err != nil {
		t.Fatalf("Timed out waiting for stream: %v", err)
	}

	readChan := make(chan struct{})
	go func() {
		_, err := ioutil.ReadAll(stream)
		if err != nil {
			t.Fatalf("Error reading stream: %v", err)
		}
		close(readChan)
	}()

	serverConn := <-serverConns
	serverConn.Close()

	// make sure the client conn breaks out of the main loop in Serve()
	<-spdyConn.closeChan
	// make sure the remote channels are closed and the stream read is unblocked
	<-readChan
}

func TestStreamReadUnblocksAfterCloseThenReset(t *testing.T) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error listening: %v", err)
	}
	listen := listener.Addr().String()

	serverConns := make(chan *Connection, 1)
	go func() {
		conn, connErr := listener.Accept()
		if connErr != nil {
			t.Fatal(connErr)
		}
		serverSpdyConn, err := NewConnection(conn, true)
		if err != nil {
			t.Fatalf("Error creating server connection: %v", err)
		}
		go serverSpdyConn.Serve(NoOpStreamHandler)
		serverConns <- serverSpdyConn
	}()

	conn, dialErr := net.Dial("tcp", listen)
	if dialErr != nil {
		t.Fatalf("Error dialing server: %s", dialErr)
	}

	spdyConn, spdyErr := NewConnection(conn, false)
	if spdyErr != nil {
		t.Fatalf("Error creating spdy connection: %s", spdyErr)
	}
	go spdyConn.Serve(NoOpStreamHandler)

	stream, err := spdyConn.CreateStream(http.Header{}, nil, false)
	if err != nil {
		t.Fatalf("Error creating stream: %v", err)
	}
	if err := stream.WaitTimeout(30 * time.Second); err != nil {
		t.Fatalf("Timed out waiting for stream: %v", err)
	}

	readChan := make(chan struct{})
	go func() {
		_, err := ioutil.ReadAll(stream)
		if err != nil {
			t.Fatalf("Error reading stream: %v", err)
		}
		close(readChan)
	}()

	serverConn := <-serverConns
	defer serverConn.Close()

	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Reset(); err != nil {
		t.Fatal(err)
	}

	// make sure close followed by reset unblocks stream.Read()
	select {
	case <-readChan:
	case <-time.After(10 * time.Second):
		t.Fatal("Timed out waiting for stream read to unblock")
	}
}

var authenticated bool

func authStreamHandler(stream *Stream) {
	if !authenticated {
		stream.Refuse()
	}
	MirrorStreamHandler(stream)
}

func runServer(wg *sync.WaitGroup) (io.Closer, string, error) {
	listener, listenErr := net.Listen("tcp", "localhost:0")
	if listenErr != nil {
		return nil, "", listenErr
	}
	wg.Add(1)
	go func() {
		for {
			conn, connErr := listener.Accept()
			if connErr != nil {
				break
			}

			spdyConn, _ := NewConnection(conn, true)
			go spdyConn.Serve(authStreamHandler)

		}
		wg.Done()
	}()
	return listener, listener.Addr().String(), nil
}
