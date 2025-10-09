package x509util

import (
	"crypto/x509"
)

// NewCertPool returns a new CertPool with the given X.509 certificates
func NewCertPool(certs []*x509.Certificate) *x509.CertPool {
	pool := x509.NewCertPool()
	for _, cert := range certs {
		pool.AddCert(cert)
	}
	return pool
}

// CopyX509Authorities copies a slice of X.509 certificates to a new slice.
func CopyX509Authorities(x509Authorities []*x509.Certificate) []*x509.Certificate {
	copiedX509Authorities := make([]*x509.Certificate, len(x509Authorities))
	copy(copiedX509Authorities, x509Authorities)

	return copiedX509Authorities
}

// CertsEqual returns true if the slices of X.509 certificates are equal.
func CertsEqual(a, b []*x509.Certificate) bool {
	if len(a) != len(b) {
		return false
	}

	for i, cert := range a {
		if !cert.Equal(b[i]) {
			return false
		}
	}

	return true
}

func RawCertsFromCerts(certs []*x509.Certificate) [][]byte {
	rawCerts := make([][]byte, 0, len(certs))
	for _, cert := range certs {
		rawCerts = append(rawCerts, cert.Raw)
	}
	return rawCerts
}

func ConcatRawCertsFromCerts(certs []*x509.Certificate) []byte {
	var rawCerts []byte
	for _, cert := range certs {
		rawCerts = append(rawCerts, cert.Raw...)
	}
	return rawCerts
}
