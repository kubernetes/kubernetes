// Package tlsconfig provides primitives to retrieve secure-enough TLS configurations for both clients and servers.
//
// As a reminder from https://golang.org/pkg/crypto/tls/#Config:
//	A Config structure is used to configure a TLS client or server. After one has been passed to a TLS function it must not be modified.
//	A Config may be reused; the tls package will also not modify it.
package tlsconfig

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/Sirupsen/logrus"
)

// Options represents the information needed to create client and server TLS configurations.
type Options struct {
	CAFile string

	// If either CertFile or KeyFile is empty, Client() will not load them
	// preventing the client from authenticating to the server.
	// However, Server() requires them and will error out if they are empty.
	CertFile string
	KeyFile  string

	// client-only option
	InsecureSkipVerify bool
	// server-only option
	ClientAuth tls.ClientAuthType
}

// Extra (server-side) accepted CBC cipher suites - will phase out in the future
var acceptedCBCCiphers = []uint16{
	tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
	tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
	tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
	tls.TLS_RSA_WITH_AES_256_CBC_SHA,
	tls.TLS_RSA_WITH_AES_128_CBC_SHA,
}

// Client TLS cipher suites (dropping CBC ciphers for client preferred suite set)
var clientCipherSuites = []uint16{
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
	tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
}

// For use by code which already has a crypto/tls options struct but wants to
// use a commonly accepted set of TLS cipher suites, with known weak algorithms removed
var DefaultServerAcceptedCiphers = append(clientCipherSuites, acceptedCBCCiphers...)

// ServerDefault is a secure-enough TLS configuration for the server TLS configuration.
var ServerDefault = tls.Config{
	// Avoid fallback to SSL protocols < TLS1.0
	MinVersion:               tls.VersionTLS10,
	PreferServerCipherSuites: true,
	CipherSuites:             DefaultServerAcceptedCiphers,
}

// ClientDefault is a secure-enough TLS configuration for the client TLS configuration.
var ClientDefault = tls.Config{
	// Prefer TLS1.2 as the client minimum
	MinVersion:   tls.VersionTLS12,
	CipherSuites: clientCipherSuites,
}

// certPool returns an X.509 certificate pool from `caFile`, the certificate file.
func certPool(caFile string) (*x509.CertPool, error) {
	// If we should verify the server, we need to load a trusted ca
	certPool := x509.NewCertPool()
	pem, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("Could not read CA certificate %q: %v", caFile, err)
	}
	if !certPool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("failed to append certificates from PEM file: %q", caFile)
	}
	s := certPool.Subjects()
	subjects := make([]string, len(s))
	for i, subject := range s {
		subjects[i] = string(subject)
	}
	logrus.Debugf("Trusting certs with subjects: %v", subjects)
	return certPool, nil
}

// Client returns a TLS configuration meant to be used by a client.
func Client(options Options) (*tls.Config, error) {
	tlsConfig := ClientDefault
	tlsConfig.InsecureSkipVerify = options.InsecureSkipVerify
	if !options.InsecureSkipVerify {
		CAs, err := certPool(options.CAFile)
		if err != nil {
			return nil, err
		}
		tlsConfig.RootCAs = CAs
	}

	if options.CertFile != "" && options.KeyFile != "" {
		tlsCert, err := tls.LoadX509KeyPair(options.CertFile, options.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("Could not load X509 key pair: %v. Make sure the key is not encrypted", err)
		}
		tlsConfig.Certificates = []tls.Certificate{tlsCert}
	}

	return &tlsConfig, nil
}

// Server returns a TLS configuration meant to be used by a server.
func Server(options Options) (*tls.Config, error) {
	tlsConfig := ServerDefault
	tlsConfig.ClientAuth = options.ClientAuth
	tlsCert, err := tls.LoadX509KeyPair(options.CertFile, options.KeyFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("Could not load X509 key pair (cert: %q, key: %q): %v", options.CertFile, options.KeyFile, err)
		}
		return nil, fmt.Errorf("Error reading X509 key pair (cert: %q, key: %q): %v. Make sure the key is not encrypted.", options.CertFile, options.KeyFile, err)
	}
	tlsConfig.Certificates = []tls.Certificate{tlsCert}
	if options.ClientAuth >= tls.VerifyClientCertIfGiven {
		CAs, err := certPool(options.CAFile)
		if err != nil {
			return nil, err
		}
		tlsConfig.ClientCAs = CAs
	}
	return &tlsConfig, nil
}
