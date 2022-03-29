//go:build go1.8
// +build go1.8

package aws

import "net/url"

// URLHostname will extract the Hostname without port from the URL value.
//
// Wrapper of net/url#URL.Hostname for backwards Go version compatibility.
func URLHostname(url *url.URL) string {
	return url.Hostname()
}
