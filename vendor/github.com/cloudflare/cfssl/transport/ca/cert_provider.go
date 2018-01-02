// Package ca provides the CertificateAuthority interface for the
// transport package, which provides an interface to get a CSR signed
// by some certificate authority.
package ca

// A CertificateAuthority is capable of signing certificates given
// certificate signing requests.
type CertificateAuthority interface {
	// SignCSR submits a PKCS #10 certificate signing request to a
	// CA for signing.
	SignCSR(csrPEM []byte) (cert []byte, err error)

	// CACertificate returns the certificate authority's
	// certificate.
	CACertificate() (cert []byte, err error)
}
