// +build !darwin

package rootcerts

import "crypto/x509"

// LoadSystemCAs does nothing on non-Darwin systems. We return nil so that
// default behavior of standard TLS config libraries is triggered, which is to
// load system certs.
func LoadSystemCAs() (*x509.CertPool, error) {
	return nil, nil
}
