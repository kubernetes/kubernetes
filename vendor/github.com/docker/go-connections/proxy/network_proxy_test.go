package proxy

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

var testBuf = []byte("Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo")
var testBufSize = len(testBuf)

type EchoServer interface {
	Run()
	Close()
	LocalAddr() net.Addr
}

type TCPEchoServer struct {
	listener net.Listener
	testCtx  *testing.T
}

type UDPEchoServer struct {
	conn    net.PacketConn
	testCtx *testing.T
}

func NewEchoServer(t *testing.T, proto, address string) EchoServer {
	var server EchoServer
	if strings.HasPrefix(proto, "tcp") {
		listener, err := net.Listen(proto, address)
		if err != nil {
			t.Fatal(err)
		}
		server = &TCPEchoServer{listener: listener, testCtx: t}
	} else {
		socket, err := net.ListenPacket(proto, address)
		if err != nil {
			t.Fatal(err)
		}
		server = &UDPEchoServer{conn: socket, testCtx: t}
	}
	return server
}

func (server *TCPEchoServer) Run() {
	go func() {
		for {
			client, err := server.listener.Accept()
			if err != nil {
				return
			}
			go func(client net.Conn) {
				if _, err := io.Copy(client, client); err != nil {
					server.testCtx.Logf("can't echo to the client: %v\n", err.Error())
				}
				client.Close()
			}(client)
		}
	}()
}

func (server *TCPEchoServer) LocalAddr() net.Addr { return server.listener.Addr() }
func (server *TCPEchoServer) Close()              { server.listener.Close() }

func (server *UDPEchoServer) Run() {
	go func() {
		readBuf := make([]byte, 1024)
		for {
			read, from, err := server.conn.ReadFrom(readBuf)
			if err != nil {
				return
			}
			for i := 0; i != read; {
				written, err := server.conn.WriteTo(readBuf[i:read], from)
				if err != nil {
					break
				}
				i += written
			}
		}
	}()
}

func (server *UDPEchoServer) LocalAddr() net.Addr { return server.conn.LocalAddr() }
func (server *UDPEchoServer) Close()              { server.conn.Close() }

func testProxyAt(t *testing.T, proto string, proxy Proxy, addr string) {
	defer proxy.Close()
	go proxy.Run()
	client, err := net.Dial(proto, addr)
	if err != nil {
		t.Fatalf("Can't connect to the proxy: %v", err)
	}
	defer client.Close()
	client.SetDeadline(time.Now().Add(10 * time.Second))
	if _, err = client.Write(testBuf); err != nil {
		t.Fatal(err)
	}
	recvBuf := make([]byte, testBufSize)
	if _, err = client.Read(recvBuf); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(testBuf, recvBuf) {
		t.Fatal(fmt.Errorf("Expected [%v] but got [%v]", testBuf, recvBuf))
	}
}

func testProxy(t *testing.T, proto string, proxy Proxy) {
	testProxyAt(t, proto, proxy, proxy.FrontendAddr().String())
}

func TestTCP4Proxy(t *testing.T) {
	backend := NewEchoServer(t, "tcp", "127.0.0.1:0")
	defer backend.Close()
	backend.Run()
	frontendAddr := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	proxy, err := NewProxy(frontendAddr, backend.LocalAddr())
	if err != nil {
		t.Fatal(err)
	}
	testProxy(t, "tcp", proxy)
}

func TestTCP6Proxy(t *testing.T) {
	backend := NewEchoServer(t, "tcp", "[::1]:0")
	defer backend.Close()
	backend.Run()
	frontendAddr := &net.TCPAddr{IP: net.IPv6loopback, Port: 0}
	proxy, err := NewProxy(frontendAddr, backend.LocalAddr())
	if err != nil {
		t.Fatal(err)
	}
	testProxy(t, "tcp", proxy)
}

func TestTCPDualStackProxy(t *testing.T) {
	// If I understand `godoc -src net favoriteAddrFamily` (used by the
	// net.Listen* functions) correctly this should work, but it doesn't.
	t.Skip("No support for dual stack yet")
	backend := NewEchoServer(t, "tcp", "[::1]:0")
	defer backend.Close()
	backend.Run()
	frontendAddr := &net.TCPAddr{IP: net.IPv6loopback, Port: 0}
	proxy, err := NewProxy(frontendAddr, backend.LocalAddr())
	if err != nil {
		t.Fatal(err)
	}
	ipv4ProxyAddr := &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: proxy.FrontendAddr().(*net.TCPAddr).Port,
	}
	testProxyAt(t, "tcp", proxy, ipv4ProxyAddr.String())
}

func TestUDP4Proxy(t *testing.T) {
	backend := NewEchoServer(t, "udp", "127.0.0.1:0")
	defer backend.Close()
	backend.Run()
	frontendAddr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	proxy, err := NewProxy(frontendAddr, backend.LocalAddr())
	if err != nil {
		t.Fatal(err)
	}
	testProxy(t, "udp", proxy)
}

func TestUDP6Proxy(t *testing.T) {
	backend := NewEchoServer(t, "udp", "[::1]:0")
	defer backend.Close()
	backend.Run()
	frontendAddr := &net.UDPAddr{IP: net.IPv6loopback, Port: 0}
	proxy, err := NewProxy(frontendAddr, backend.LocalAddr())
	if err != nil {
		t.Fatal(err)
	}
	testProxy(t, "udp", proxy)
}

func TestUDPWriteError(t *testing.T) {
	frontendAddr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	// Hopefully, this port will be free: */
	backendAddr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 25587}
	proxy, err := NewProxy(frontendAddr, backendAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer proxy.Close()
	go proxy.Run()
	client, err := net.Dial("udp", "127.0.0.1:25587")
	if err != nil {
		t.Fatalf("Can't connect to the proxy: %v", err)
	}
	defer client.Close()
	// Make sure the proxy doesn't stop when there is no actual backend:
	client.Write(testBuf)
	client.Write(testBuf)
	backend := NewEchoServer(t, "udp", "127.0.0.1:25587")
	defer backend.Close()
	backend.Run()
	client.SetDeadline(time.Now().Add(10 * time.Second))
	if _, err = client.Write(testBuf); err != nil {
		t.Fatal(err)
	}
	recvBuf := make([]byte, testBufSize)
	if _, err = client.Read(recvBuf); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(testBuf, recvBuf) {
		t.Fatal(fmt.Errorf("Expected [%v] but got [%v]", testBuf, recvBuf))
	}
}
