//go:build !go1.13 && go1.7
// +build !go1.13,go1.7

package session

import (
	"net"
	"net/http"
	"time"
)

// Transport that should be used when a custom CA bundle is specified with the
// SDK.
func getCustomTransport() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).DialContext,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
}
