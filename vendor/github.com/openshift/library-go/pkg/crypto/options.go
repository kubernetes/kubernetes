package crypto

import (
	"crypto/x509/pkix"
	"time"
)

// CertificateOptions holds optional configuration collected from functional options.
type CertificateOptions struct {
	lifetime     time.Duration
	subject      *pkix.Name
	extensionFns []CertificateExtensionFunc
	signer       *CA
}

// CertificateOption is a functional option for certificate creation.
type CertificateOption func(*CertificateOptions)

// WithLifetime sets the certificate lifetime duration.
func WithLifetime(d time.Duration) CertificateOption {
	return func(o *CertificateOptions) {
		o.lifetime = d
	}
}

// WithSubject overrides the certificate subject. For signing certificates,
// this overrides the default subject derived from the name parameter.
func WithSubject(s pkix.Name) CertificateOption {
	return func(o *CertificateOptions) {
		o.subject = &s
	}
}

// WithSigner specifies a CA to sign the certificate. When used with
// NewSigningCertificate, this creates an intermediate CA signed by the
// given CA instead of a self-signed root CA.
func WithSigner(ca *CA) CertificateOption {
	return func(o *CertificateOptions) {
		o.signer = ca
	}
}

// WithExtensions adds certificate extension functions that are called
// to modify the certificate template before signing.
func WithExtensions(fns ...CertificateExtensionFunc) CertificateOption {
	return func(o *CertificateOptions) {
		o.extensionFns = append(o.extensionFns, fns...)
	}
}
