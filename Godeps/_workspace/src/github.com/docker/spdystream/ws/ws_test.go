package ws

import (
	"bytes"
	"github.com/docker/spdystream"
	"github.com/gorilla/websocket"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

var serverSpdyConn *spdystream.Connection

// Connect to the Websocket endpoint at ws://localhost
// using SPDY over Websockets framing.
func ExampleConn() {
	wsconn, _, _ := websocket.DefaultDialer.Dial("ws://localhost/", http.Header{"Origin": {"http://localhost/"}})
	conn, _ := spdystream.NewConnection(NewConnection(wsconn), false)
	go conn.Serve(spdystream.NoOpStreamHandler, spdystream.NoAuthHandler)
	stream, _ := conn.CreateStream(http.Header{}, nil, false)
	stream.Wait()
}

func serveWs(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		if _, ok := err.(websocket.HandshakeError); !ok {
			log.Println(err)
		}
		return
	}

	wrap := NewConnection(ws)
	spdyConn, err := spdystream.NewConnection(wrap, true)
	if err != nil {
		log.Fatal(err)
		return
	}
	serverSpdyConn = spdyConn
	go spdyConn.Serve(spdystream.MirrorStreamHandler, authStreamHandler)
}

func TestSpdyStreamOverWs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(serveWs))
	defer server.Close()
	defer func() {
		if serverSpdyConn != nil {
			serverSpdyConn.Close()
		}
	}()

	wsconn, _, err := websocket.DefaultDialer.Dial(strings.Replace(server.URL, "http://", "ws://", 1), http.Header{"Origin": {server.URL}})
	if err != nil {
		t.Fatal(err)
	}

	wrap := NewConnection(wsconn)
	spdyConn, err := spdystream.NewConnection(wrap, false)
	if err != nil {
		defer wsconn.Close()
		t.Fatal(err)
	}
	defer spdyConn.Close()
	authenticated = true
	go spdyConn.Serve(spdystream.NoOpStreamHandler, spdystream.RejectAuthHandler)

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

	streamCloseErr := stream.Close()
	if streamCloseErr != nil {
		t.Fatalf("Error closing stream: %s", streamCloseErr)
	}

	// Closing again should return nil
	streamCloseErr = stream.Close()
	if streamCloseErr != nil {
		t.Fatalf("Error closing stream: %s", streamCloseErr)
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
	if waitErr != spdystream.ErrReset {
		t.Fatalf("Unexpected error creating stream: %s", waitErr)
	}

	spdyCloseErr := spdyConn.Close()
	if spdyCloseErr != nil {
		t.Fatalf("Error closing spdy connection: %s", spdyCloseErr)
	}
}

var authenticated bool

func authStreamHandler(header http.Header, slot uint8, parent uint32) bool {
	return authenticated
}
