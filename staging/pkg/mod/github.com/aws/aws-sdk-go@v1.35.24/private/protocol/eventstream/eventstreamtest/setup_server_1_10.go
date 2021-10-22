// +build go1.10

package eventstreamtest

import (
	"crypto/tls"
	"net/http"
	"net/http/httptest"

	"golang.org/x/net/http2"
)

// /x/net/http2 is only available for the latest two versions of Go. Any Go
// version older than that cannot use the utility to configure the http2
// server.
func setupServer(server *httptest.Server, useH2 bool) *http.Client {
	server.Config.TLSConfig = &tls.Config{
		InsecureSkipVerify: true,
	}

	clientTrans := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}

	if useH2 {
		http2.ConfigureServer(server.Config, nil)
		http2.ConfigureTransport(clientTrans)
		server.Config.TLSConfig.NextProtos = []string{http2.NextProtoTLS}
		clientTrans.TLSClientConfig.NextProtos = []string{http2.NextProtoTLS}
	}
	server.TLS = server.Config.TLSConfig

	server.StartTLS()

	return &http.Client{
		Transport: clientTrans,
	}
}
