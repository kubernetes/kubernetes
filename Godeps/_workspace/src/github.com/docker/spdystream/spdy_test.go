package spdystream

import (
	"bytes"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"
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
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("(%d) Timeout waiting for stream closure", tix)
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
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Timed out waiting for connection closed notification")
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

	expired := time.NewTimer(200 * time.Millisecond)

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
		case <-expired.C:
			t.Fatal("Timed out waiting for idle connection closure")
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
