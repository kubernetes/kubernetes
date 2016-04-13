package proxy

import (
	"fmt"
	"net"
)

type Proxy interface {
	// Start forwarding traffic back and forth the front and back-end
	// addresses.
	Run()
	// Stop forwarding traffic and close both ends of the Proxy.
	Close()
	// Return the address on which the proxy is listening.
	FrontendAddr() net.Addr
	// Return the proxied address.
	BackendAddr() net.Addr
}

func NewProxy(frontendAddr, backendAddr net.Addr) (Proxy, error) {
	switch frontendAddr.(type) {
	case *net.UDPAddr:
		return NewUDPProxy(frontendAddr.(*net.UDPAddr), backendAddr.(*net.UDPAddr))
	case *net.TCPAddr:
		return NewTCPProxy(frontendAddr.(*net.TCPAddr), backendAddr.(*net.TCPAddr))
	default:
		panic(fmt.Errorf("Unsupported protocol"))
	}
}
