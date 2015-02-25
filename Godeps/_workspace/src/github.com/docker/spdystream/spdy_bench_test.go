package spdystream

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
)

func configureServer() (io.Closer, string, *sync.WaitGroup) {
	authenticated = true
	wg := &sync.WaitGroup{}
	server, listen, serverErr := runServer(wg)

	if serverErr != nil {
		panic(serverErr)
	}

	return server, listen, wg
}

func BenchmarkDial10000(b *testing.B) {
	server, addr, wg := configureServer()

	defer func() {
		server.Close()
		wg.Wait()
	}()

	for i := 0; i < b.N; i++ {
		conn, dialErr := net.Dial("tcp", addr)
		if dialErr != nil {
			panic(fmt.Sprintf("Error dialing server: %s", dialErr))
		}
		conn.Close()
	}
}

func BenchmarkDialWithSPDYStream10000(b *testing.B) {
	server, addr, wg := configureServer()

	defer func() {
		server.Close()
		wg.Wait()
	}()

	for i := 0; i < b.N; i++ {
		conn, dialErr := net.Dial("tcp", addr)
		if dialErr != nil {
			b.Fatalf("Error dialing server: %s", dialErr)
		}

		spdyConn, spdyErr := NewConnection(conn, false)
		if spdyErr != nil {
			b.Fatalf("Error creating spdy connection: %s", spdyErr)
		}
		go spdyConn.Serve(NoOpStreamHandler)

		closeErr := spdyConn.Close()
		if closeErr != nil {
			b.Fatalf("Error closing connection: %s, closeErr")
		}
	}
}

func benchmarkStreamWithDataAndSize(size uint64, b *testing.B) {
	server, addr, wg := configureServer()

	defer func() {
		server.Close()
		wg.Wait()
	}()

	for i := 0; i < b.N; i++ {
		conn, dialErr := net.Dial("tcp", addr)
		if dialErr != nil {
			b.Fatalf("Error dialing server: %s", dialErr)
		}

		spdyConn, spdyErr := NewConnection(conn, false)
		if spdyErr != nil {
			b.Fatalf("Error creating spdy connection: %s", spdyErr)
		}

		go spdyConn.Serve(MirrorStreamHandler)

		stream, err := spdyConn.CreateStream(http.Header{}, nil, false)

		writer := make([]byte, size)

		stream.Write(writer)

		if err != nil {
			panic(err)
		}

		reader := make([]byte, size)
		stream.Read(reader)

		stream.Close()

		closeErr := spdyConn.Close()
		if closeErr != nil {
			b.Fatalf("Error closing connection: %s, closeErr")
		}
	}
}

func BenchmarkStreamWith1Byte10000(b *testing.B)     { benchmarkStreamWithDataAndSize(1, b) }
func BenchmarkStreamWith1KiloByte10000(b *testing.B) { benchmarkStreamWithDataAndSize(1024, b) }
func BenchmarkStreamWith1Megabyte10000(b *testing.B) { benchmarkStreamWithDataAndSize(1024*1024, b) }
