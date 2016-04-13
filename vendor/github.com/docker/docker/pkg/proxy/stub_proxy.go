package proxy

import (
	"net"
)

type StubProxy struct {
	frontendAddr net.Addr
	backendAddr  net.Addr
}

func (p *StubProxy) Run()                   {}
func (p *StubProxy) Close()                 {}
func (p *StubProxy) FrontendAddr() net.Addr { return p.frontendAddr }
func (p *StubProxy) BackendAddr() net.Addr  { return p.backendAddr }

func NewStubProxy(frontendAddr, backendAddr net.Addr) (Proxy, error) {
	return &StubProxy{
		frontendAddr: frontendAddr,
		backendAddr:  backendAddr,
	}, nil
}
