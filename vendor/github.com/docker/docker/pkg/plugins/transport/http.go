package transport

import (
	"io"
	"net/http"
)

// httpTransport holds an http.RoundTripper
// and information about the scheme and address the transport
// sends request to.
type httpTransport struct {
	http.RoundTripper
	scheme string
	addr   string
}

// NewHTTPTransport creates a new httpTransport.
func NewHTTPTransport(r http.RoundTripper, scheme, addr string) Transport {
	return httpTransport{
		RoundTripper: r,
		scheme:       scheme,
		addr:         addr,
	}
}

// NewRequest creates a new http.Request and sets the URL
// scheme and address with the transport's fields.
func (t httpTransport) NewRequest(path string, data io.Reader) (*http.Request, error) {
	req, err := newHTTPRequest(path, data)
	if err != nil {
		return nil, err
	}
	req.URL.Scheme = t.scheme
	req.URL.Host = t.addr
	return req, nil
}
