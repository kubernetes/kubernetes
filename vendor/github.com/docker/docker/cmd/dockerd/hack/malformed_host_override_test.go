// +build !windows

package hack

import (
	"bytes"
	"io"
	"net"
	"strings"
	"testing"
)

type bufConn struct {
	net.Conn
	buf *bytes.Buffer
}

func (bc *bufConn) Read(b []byte) (int, error) {
	return bc.buf.Read(b)
}

func TestHeaderOverrideHack(t *testing.T) {
	tests := [][2][]byte{
		{
			[]byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\r\n\r\n"),
			[]byte("GET /foo\nHost: \r\nConnection: close\r\nUser-Agent: Docker\r\n\r\n"),
		},
		{
			[]byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\nFoo: Bar\r\n"),
			[]byte("GET /foo\nHost: \r\nConnection: close\r\nUser-Agent: Docker\nFoo: Bar\r\n"),
		},
		{
			[]byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\r\n\r\ntest something!"),
			[]byte("GET /foo\nHost: \r\nConnection: close\r\nUser-Agent: Docker\r\n\r\ntest something!"),
		},
		{
			[]byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\r\n\r\ntest something! " + strings.Repeat("test", 15000)),
			[]byte("GET /foo\nHost: \r\nConnection: close\r\nUser-Agent: Docker\r\n\r\ntest something! " + strings.Repeat("test", 15000)),
		},
		{
			[]byte("GET /foo\nFoo: Bar\nHost: /var/run/docker.sock\nUser-Agent: Docker\r\n\r\n"),
			[]byte("GET /foo\nFoo: Bar\nHost: /var/run/docker.sock\nUser-Agent: Docker\r\n\r\n"),
		},
	}

	// Test for https://github.com/docker/docker/issues/23045
	h0 := "GET /foo\nUser-Agent: Docker\r\n\r\n"
	h0 = h0 + strings.Repeat("a", 4096-len(h0)-1) + "\n"
	tests = append(tests, [2][]byte{[]byte(h0), []byte(h0)})

	for _, pair := range tests {
		read := make([]byte, 4096)
		client := &bufConn{
			buf: bytes.NewBuffer(pair[0]),
		}
		l := MalformedHostHeaderOverrideConn{client, true}

		n, err := l.Read(read)
		if err != nil && err != io.EOF {
			t.Fatalf("read: %d - %d, err: %v\n%s", n, len(pair[0]), err, string(read[:n]))
		}
		if !bytes.Equal(read[:n], pair[1][:n]) {
			t.Fatalf("\n%s\n%s\n", read[:n], pair[1][:n])
		}
	}
}

func BenchmarkWithHack(b *testing.B) {
	client, srv := net.Pipe()
	done := make(chan struct{})
	req := []byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\n")
	read := make([]byte, 4096)
	b.SetBytes(int64(len(req) * 30))

	l := MalformedHostHeaderOverrideConn{client, true}
	go func() {
		for {
			if _, err := srv.Write(req); err != nil {
				srv.Close()
				break
			}
			l.first = true // make sure each subsequent run uses the hack parsing
		}
		close(done)
	}()

	for i := 0; i < b.N; i++ {
		for i := 0; i < 30; i++ {
			if n, err := l.Read(read); err != nil && err != io.EOF {
				b.Fatalf("read: %d - %d, err: %v\n%s", n, len(req), err, string(read[:n]))
			}
		}
	}
	l.Close()
	<-done
}

func BenchmarkNoHack(b *testing.B) {
	client, srv := net.Pipe()
	done := make(chan struct{})
	req := []byte("GET /foo\nHost: /var/run/docker.sock\nUser-Agent: Docker\n")
	read := make([]byte, 4096)
	b.SetBytes(int64(len(req) * 30))

	go func() {
		for {
			if _, err := srv.Write(req); err != nil {
				srv.Close()
				break
			}
		}
		close(done)
	}()

	for i := 0; i < b.N; i++ {
		for i := 0; i < 30; i++ {
			if _, err := client.Read(read); err != nil && err != io.EOF {
				b.Fatal(err)
			}
		}
	}
	client.Close()
	<-done
}
