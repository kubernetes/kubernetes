// +build !go1.8

package aws

import (
	"net/url"
	"strings"
)

// URLHostname will extract the Hostname without port from the URL value.
//
// Copy of Go 1.8's net/url#URL.Hostname functionality.
func URLHostname(url *url.URL) string {
	return stripPort(url.Host)

}

// stripPort is copy of Go 1.8 url#URL.Hostname functionality.
// https://golang.org/src/net/url/url.go
func stripPort(hostport string) string {
	colon := strings.IndexByte(hostport, ':')
	if colon == -1 {
		return hostport
	}
	if i := strings.IndexByte(hostport, ']'); i != -1 {
		return strings.TrimPrefix(hostport[:i], "[")
	}
	return hostport[:colon]
}
