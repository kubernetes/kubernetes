package httpd_test

import (
	"io"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/influxdata/influxdb/services/httpd"
)

type fakeListener struct {
	AcceptFn func() (net.Conn, error)
}

func (l *fakeListener) Accept() (net.Conn, error) {
	if l.AcceptFn != nil {
		return l.AcceptFn()
	}
	return &fakeConn{}, nil
}

func (*fakeListener) Close() error   { return nil }
func (*fakeListener) Addr() net.Addr { return nil }

type fakeConn struct {
	closed bool
}

func (*fakeConn) Read([]byte) (int, error)    { return 0, io.EOF }
func (*fakeConn) Write(b []byte) (int, error) { return len(b), nil }
func (c *fakeConn) Close() error {
	c.closed = true
	return nil
}
func (*fakeConn) LocalAddr() net.Addr              { return nil }
func (*fakeConn) RemoteAddr() net.Addr             { return nil }
func (*fakeConn) SetDeadline(time.Time) error      { return nil }
func (*fakeConn) SetReadDeadline(time.Time) error  { return nil }
func (*fakeConn) SetWriteDeadline(time.Time) error { return nil }

func TestLimitListener(t *testing.T) {
	conns := make(chan net.Conn, 2)
	l := httpd.LimitListener(&fakeListener{
		AcceptFn: func() (net.Conn, error) {
			select {
			case c := <-conns:
				if c != nil {
					return c, nil
				}
			default:
			}
			return nil, io.EOF
		},
	}, 1)
	c1, c2 := &fakeConn{}, &fakeConn{}
	conns <- c1
	conns <- c2

	var c net.Conn
	var err error
	if c, err = l.Accept(); err != nil {
		t.Fatalf("expected accept to succeed: %s", err)
	}

	if _, err = l.Accept(); err != io.EOF {
		t.Fatalf("expected eof, got %s", err)
	} else if !c2.closed {
		t.Fatalf("expected connection to be automatically closed")
	}
	c.Close()

	conns <- &fakeConn{}
	if _, err = l.Accept(); err != nil {
		t.Fatalf("expeced accept to succeed: %s", err)
	}
}

func BenchmarkLimitListener(b *testing.B) {
	var wg sync.WaitGroup
	wg.Add(b.N)

	l := httpd.LimitListener(&fakeListener{}, b.N)
	for i := 0; i < b.N; i++ {
		go func() {
			c, err := l.Accept()
			if err != nil {
				b.Fatal(err)
			}
			c.Close()
			wg.Done()
		}()
	}
	wg.Wait()
}
