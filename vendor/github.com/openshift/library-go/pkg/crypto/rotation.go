package crypto

import (
	"crypto/x509"
	"time"
)

// FilterExpiredCerts checks are all certificates in the bundle valid, i.e. they have not expired.
// The function returns new bundle with only valid certificates or error if no valid certificate is found.
func FilterExpiredCerts(certs ...*x509.Certificate) []*x509.Certificate {
	currentTime := time.Now()
	var validCerts []*x509.Certificate
	for _, c := range certs {
		if c.NotAfter.After(currentTime) {
			validCerts = append(validCerts, c)
		}
	}

	return validCerts
}
