package main

import (
	"net"
)

// StubProxy is a proxy that is a stub (does nothing).
type StubProxy struct {
	frontendAddr net.Addr
	backendAddr  net.Addr
}

// Run does nothing.
func (p *StubProxy) Run() {}

// Close does nothing.
func (p *StubProxy) Close() {}

// FrontendAddr returns the frontend address.
func (p *StubProxy) FrontendAddr() net.Addr { return p.frontendAddr }

// BackendAddr returns the backend address.
func (p *StubProxy) BackendAddr() net.Addr { return p.backendAddr }

// NewStubProxy creates a new StubProxy
func NewStubProxy(frontendAddr, backendAddr net.Addr) (Proxy, error) {
	return &StubProxy{
		frontendAddr: frontendAddr,
		backendAddr:  backendAddr,
	}, nil
}
