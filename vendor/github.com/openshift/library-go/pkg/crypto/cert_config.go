package crypto

import (
	"crypto"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
)

// KeyPairGenerator generates a cryptographic key pair.
type KeyPairGenerator interface {
	GenerateKeyPair() (crypto.PublicKey, crypto.PrivateKey, error)
}

// NewSigningCertificate creates a CA certificate.
// By default it creates a self-signed root CA. Use WithSigner to create an
// intermediate CA signed by a parent CA.
// The name parameter is used as the CommonName unless overridden with WithSubject.
// Optional: WithSigner, WithSubject, WithLifetime (defaults to DefaultCACertificateLifetimeDuration).
func NewSigningCertificate(name string, keyGen KeyPairGenerator, opts ...CertificateOption) (*TLSCertificateConfig, error) {
	o := &CertificateOptions{
		lifetime: DefaultCACertificateLifetimeDuration,
	}
	for _, opt := range opts {
		opt(o)
	}

	subject := pkix.Name{CommonName: name}
	if o.subject != nil {
		subject = *o.subject
	}

	publicKey, privateKey, err := keyGen.GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("failed to generate key pair: %w", err)
	}
	subjectKeyId, err := SubjectKeyIDFromPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute subject key ID: %w", err)
	}

	if o.signer != nil {
		// Intermediate CA signed by the provided signer.
		authorityKeyId := o.signer.Config.Certs[0].SubjectKeyId
		template := newSigningCertificateTemplateForDuration(subject, o.lifetime, time.Now, authorityKeyId, subjectKeyId)
		template.SignatureAlgorithm = 0
		template.KeyUsage = KeyUsageForPublicKey(publicKey) | x509.KeyUsageCertSign

		cert, err := o.signer.SignCertificate(template, publicKey)
		if err != nil {
			return nil, fmt.Errorf("failed to sign certificate: %w", err)
		}

		return &TLSCertificateConfig{
			Certs: append([]*x509.Certificate{cert}, o.signer.Config.Certs...),
			Key:   privateKey,
		}, nil
	}

	// Self-signed root CA. AuthorityKeyId and SubjectKeyId match.
	template := newSigningCertificateTemplateForDuration(subject, o.lifetime, time.Now, subjectKeyId, subjectKeyId)
	template.SignatureAlgorithm = 0
	template.KeyUsage = KeyUsageForPublicKey(publicKey) | x509.KeyUsageCertSign

	cert, err := signCertificate(template, publicKey, template, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %w", err)
	}

	return &TLSCertificateConfig{
		Certs: []*x509.Certificate{cert},
		Key:   privateKey,
	}, nil
}

// NewServerCertificate creates a server/serving certificate signed by this CA.
// Optional: WithLifetime (defaults to DefaultCertificateLifetimeDuration), WithExtensions.
func (ca *CA) NewServerCertificate(hostnames sets.Set[string], keyGen KeyPairGenerator, opts ...CertificateOption) (*TLSCertificateConfig, error) {
	o := &CertificateOptions{
		lifetime: DefaultCertificateLifetimeDuration,
	}
	for _, opt := range opts {
		opt(o)
	}

	publicKey, privateKey, err := keyGen.GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("failed to generate key pair: %w", err)
	}
	subjectKeyId, err := SubjectKeyIDFromPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute subject key ID: %w", err)
	}

	sortedHostnames := sets.List(hostnames)
	authorityKeyId := ca.Config.Certs[0].SubjectKeyId
	template := newServerCertificateTemplateForDuration(
		pkix.Name{CommonName: sortedHostnames[0]},
		sortedHostnames,
		o.lifetime,
		time.Now,
		authorityKeyId,
		subjectKeyId,
	)
	// Let x509.CreateCertificate auto-detect the signature algorithm from the CA's key.
	template.SignatureAlgorithm = 0
	template.KeyUsage = KeyUsageForPublicKey(publicKey)

	for _, fn := range o.extensionFns {
		if err := fn(template); err != nil {
			return nil, fmt.Errorf("failed to apply certificate extension: %w", err)
		}
	}

	cert, err := ca.SignCertificate(template, publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %w", err)
	}

	return &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{cert}, ca.Config.Certs...),
		Key:   privateKey,
	}, nil
}

// NewClientCertificate creates a client certificate signed by this CA.
// Optional: WithLifetime (defaults to DefaultCertificateLifetimeDuration).
func (ca *CA) NewClientCertificate(u user.Info, keyGen KeyPairGenerator, opts ...CertificateOption) (*TLSCertificateConfig, error) {
	o := &CertificateOptions{
		lifetime: DefaultCertificateLifetimeDuration,
	}
	for _, opt := range opts {
		opt(o)
	}

	publicKey, privateKey, err := keyGen.GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("failed to generate key pair: %w", err)
	}
	subjectKeyId, err := SubjectKeyIDFromPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute subject key ID: %w", err)
	}

	authorityKeyId := ca.Config.Certs[0].SubjectKeyId
	template := NewClientCertificateTemplateForDuration(UserToSubject(u), o.lifetime, time.Now)
	template.AuthorityKeyId = authorityKeyId
	template.SubjectKeyId = subjectKeyId
	// Let x509.CreateCertificate auto-detect the signature algorithm from the CA's key.
	template.SignatureAlgorithm = 0
	template.KeyUsage = KeyUsageForPublicKey(publicKey)

	cert, err := ca.SignCertificate(template, publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %w", err)
	}

	return &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{cert}, ca.Config.Certs...),
		Key:   privateKey,
	}, nil
}

// NewPeerCertificate creates a peer certificate (both server and client auth)
// signed by this CA.
// Optional: WithLifetime (defaults to DefaultCertificateLifetimeDuration), WithExtensions.
func (ca *CA) NewPeerCertificate(hostnames sets.Set[string], u user.Info, keyGen KeyPairGenerator, opts ...CertificateOption) (*TLSCertificateConfig, error) {
	o := &CertificateOptions{
		lifetime: DefaultCertificateLifetimeDuration,
	}
	for _, opt := range opts {
		opt(o)
	}

	publicKey, privateKey, err := keyGen.GenerateKeyPair()
	if err != nil {
		return nil, fmt.Errorf("failed to generate key pair: %w", err)
	}
	subjectKeyId, err := SubjectKeyIDFromPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute subject key ID: %w", err)
	}

	sortedHostnames := sets.List(hostnames)
	authorityKeyId := ca.Config.Certs[0].SubjectKeyId

	// Start from a server certificate template for the hostnames.
	template := newServerCertificateTemplateForDuration(
		pkix.Name{CommonName: sortedHostnames[0]},
		sortedHostnames,
		o.lifetime,
		time.Now,
		authorityKeyId,
		subjectKeyId,
	)
	// Let x509.CreateCertificate auto-detect the signature algorithm from the CA's key.
	template.SignatureAlgorithm = 0
	template.KeyUsage = KeyUsageForPublicKey(publicKey)

	// Set subject from user info for client authentication.
	template.Subject = UserToSubject(u)

	// Enable both server and client authentication.
	template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth}

	for _, fn := range o.extensionFns {
		if err := fn(template); err != nil {
			return nil, fmt.Errorf("failed to apply certificate extension: %w", err)
		}
	}

	cert, err := ca.SignCertificate(template, publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %w", err)
	}

	return &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{cert}, ca.Config.Certs...),
		Key:   privateKey,
	}, nil
}
