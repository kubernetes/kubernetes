// +build !go1.7

package tlsconfig

import (
	"crypto/x509"
)

// SystemCertPool returns an new empty cert pool,
// accessing system cert pool is supported in go 1.7
func SystemCertPool() (*x509.CertPool, error) {
	return x509.NewCertPool(), nil
}
