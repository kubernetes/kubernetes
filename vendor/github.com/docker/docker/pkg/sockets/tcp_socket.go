package sockets

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"

	"github.com/docker/docker/pkg/listenbuffer"
)

func NewTcpSocket(addr string, tlsConfig *tls.Config, activate <-chan struct{}) (net.Listener, error) {
	l, err := listenbuffer.NewListenBuffer("tcp", addr, activate)
	if err != nil {
		return nil, err
	}
	if tlsConfig != nil {
		tlsConfig.NextProtos = []string{"http/1.1"}
		l = tls.NewListener(l, tlsConfig)
	}
	return l, nil
}

func ConfigureTCPTransport(tr *http.Transport, proto, addr string) {
	// Why 32? See https://github.com/docker/docker/pull/8035.
	timeout := 32 * time.Second
	if proto == "unix" {
		// No need for compression in local communications.
		tr.DisableCompression = true
		tr.Dial = func(_, _ string) (net.Conn, error) {
			return net.DialTimeout(proto, addr, timeout)
		}
	} else {
		tr.Proxy = http.ProxyFromEnvironment
		tr.Dial = (&net.Dialer{Timeout: timeout}).Dial
	}
}
