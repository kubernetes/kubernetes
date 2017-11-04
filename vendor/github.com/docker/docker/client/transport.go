package client

import (
	"crypto/tls"
	"net/http"
)

// transportFunc allows us to inject a mock transport for testing. We define it
// here so we can detect the tlsconfig and return nil for only this type.
type transportFunc func(*http.Request) (*http.Response, error)

func (tf transportFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return tf(req)
}

// resolveTLSConfig attempts to resolve the TLS configuration from the
// RoundTripper.
func resolveTLSConfig(transport http.RoundTripper) *tls.Config {
	switch tr := transport.(type) {
	case *http.Transport:
		return tr.TLSClientConfig
	default:
		return nil
	}
}
