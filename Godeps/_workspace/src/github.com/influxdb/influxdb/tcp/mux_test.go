package tcp_test

import (
	"bytes"
	"io"
	"io/ioutil"
	"log"
	"net"
	"strings"
	"sync"
	"testing"
	"testing/quick"
	"time"

	"github.com/influxdb/influxdb/tcp"
)

// Ensure the muxer can split a listener's connections across multiple listeners.
func TestMux(t *testing.T) {
	if err := quick.Check(func(n uint8, msg []byte) bool {
		if testing.Verbose() {
			if len(msg) == 0 {
				log.Printf("n=%d, <no message>", n)
			} else {
				log.Printf("n=%d, hdr=%d, len=%d", n, msg[0], len(msg))
			}
		}

		var wg sync.WaitGroup

		// Open single listener on random port.
		tcpListener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer tcpListener.Close()

		// Setup muxer & listeners.
		mux := tcp.NewMux()
		mux.Timeout = 200 * time.Millisecond
		if !testing.Verbose() {
			mux.Logger = log.New(ioutil.Discard, "", 0)
		}
		for i := uint8(0); i < n; i++ {
			ln := mux.Listen(byte(i))

			wg.Add(1)
			go func(i uint8, ln net.Listener) {
				defer wg.Done()

				// Wait for a connection for this listener.
				conn, err := ln.Accept()
				if conn != nil {
					defer conn.Close()
				}

				// If there is no message or the header byte
				// doesn't match then expect close.
				if len(msg) == 0 || msg[0] != byte(i) {
					if err == nil || err.Error() != "network connection closed" {
						t.Fatalf("unexpected error: %s", err)
					}
					return
				}

				// If the header byte matches this listener
				// then expect a connection and read the message.
				var buf bytes.Buffer
				if _, err := io.CopyN(&buf, conn, int64(len(msg)-1)); err != nil {
					t.Fatal(err)
				} else if !bytes.Equal(msg[1:], buf.Bytes()) {
					t.Fatalf("message mismatch:\n\nexp=%x\n\ngot=%x\n\n", msg[1:], buf.Bytes())
				}

				// Write response.
				if _, err := conn.Write([]byte("OK")); err != nil {
					t.Fatal(err)
				}
			}(i, ln)
		}

		// Begin serving from the listener.
		go mux.Serve(tcpListener)

		// Write message to TCP listener and read OK response.
		conn, err := net.Dial("tcp", tcpListener.Addr().String())
		if err != nil {
			t.Fatal(err)
		} else if _, err = conn.Write(msg); err != nil {
			t.Fatal(err)
		}

		// Read the response into the buffer.
		var resp [2]byte
		_, err = io.ReadFull(conn, resp[:])

		// If the message header is less than n then expect a response.
		// Otherwise we should get an EOF because the mux closed.
		if len(msg) > 0 && uint8(msg[0]) < n {
			if string(resp[:]) != `OK` {
				t.Fatalf("unexpected response: %s", resp[:])
			}
		} else {
			if err == nil || (err != io.EOF && !(strings.Contains(err.Error(), "connection reset by peer") ||
				strings.Contains(err.Error(), "closed by the remote host"))) {
				t.Fatalf("unexpected error: %s", err)
			}
		}

		// Close connection.
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}

		// Close original TCP listener and wait for all goroutines to close.
		tcpListener.Close()
		wg.Wait()

		return true
	}, nil); err != nil {
		t.Error(err)
	}
}

// Ensure two handlers cannot be registered for the same header byte.
func TestMux_Listen_ErrAlreadyRegistered(t *testing.T) {
	defer func() {
		if r := recover(); r != `listener already registered under header byte: 5` {
			t.Fatalf("unexpected recover: %#v", r)
		}
	}()

	// Register two listeners with the same header byte.
	mux := tcp.NewMux()
	mux.Listen(5)
	mux.Listen(5)
}
