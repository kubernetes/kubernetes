package cleanhttp

import (
	"net"
	"net/http"
	"runtime"
	"time"
)

// DefaultTransport returns a new http.Transport with the same default values
// as http.DefaultTransport
func DefaultTransport() *http.Transport {
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
	}
	SetTransportFinalizer(transport)
	return transport
}

// DefaultClient returns a new http.Client with the same default values as
// http.Client, but with a non-shared Transport
func DefaultClient() *http.Client {
	return &http.Client{
		Transport: DefaultTransport(),
	}
}

// SetTransportFinalizer sets a finalizer on the transport to ensure that
// idle connections are closed prior to garbage collection; otherwise
// these may leak
func SetTransportFinalizer(transport *http.Transport) {
	runtime.SetFinalizer(&transport, func(t **http.Transport) {
		(*t).CloseIdleConnections()
	})
}
