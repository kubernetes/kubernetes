// +build go1.8

package tlsconfig

import "crypto/tls"

// Clone returns a clone of tls.Config. This function is provided for
// compatibility for go1.7 that doesn't include this method in stdlib.
func Clone(c *tls.Config) *tls.Config {
	return c.Clone()
}
