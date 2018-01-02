package transport

import (
	"io"
	"net/http"
	"strings"
)

// VersionMimetype is the Content-Type the engine sends to plugins.
const VersionMimetype = "application/vnd.docker.plugins.v1.2+json"

// RequestFactory defines an interface that
// transports can implement to create new requests.
type RequestFactory interface {
	NewRequest(path string, data io.Reader) (*http.Request, error)
}

// Transport defines an interface that plugin transports
// must implement.
type Transport interface {
	http.RoundTripper
	RequestFactory
}

// newHTTPRequest creates a new request with a path and a body.
func newHTTPRequest(path string, data io.Reader) (*http.Request, error) {
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	req, err := http.NewRequest("POST", path, data)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Accept", VersionMimetype)
	return req, nil
}
