package netutil

import (
	"net/url"
	"strings"
)

// FROM: http://golang.org/src/net/http/client.go
// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }

// FROM: http://golang.org/src/net/http/transport.go
var portMap = map[string]string{
	"http":  "80",
	"https": "443",
}

// FROM: http://golang.org/src/net/http/transport.go
// canonicalAddr returns url.Host but always with a ":port" suffix
func CanonicalAddr(url *url.URL) string {
	addr := url.Host
	if !hasPort(addr) {
		return addr + ":" + portMap[url.Scheme]
	}
	return addr
}
