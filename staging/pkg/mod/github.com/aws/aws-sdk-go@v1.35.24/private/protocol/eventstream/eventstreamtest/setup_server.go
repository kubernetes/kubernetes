// +build !go1.10

package eventstreamtest

import (
	"net/http"
	"net/http/httptest"
)

// /x/net/http2 is only available for the latest two versions of Go. Any Go
// version older than that cannot use the utility to configure the http2
// server.
func setupServer(server *httptest.Server, useH2 bool) *http.Client {
	server.Start()

	return nil
}
