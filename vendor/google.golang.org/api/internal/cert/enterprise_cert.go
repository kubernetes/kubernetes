// Copyright 2022 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cert contains certificate tools for Google API clients.
// This package is intended to be used with crypto/tls.Config.GetClientCertificate.
//
// The certificates can be used to satisfy Google's Endpoint Validation.
// See https://cloud.google.com/endpoint-verification/docs/overview
//
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.
package cert

import (
	"crypto/tls"
	"errors"

	"github.com/googleapis/enterprise-certificate-proxy/client"
)

type ecpSource struct {
	key *client.Key
}

// NewEnterpriseCertificateProxySource creates a certificate source
// using the Enterprise Certificate Proxy client, which delegates
// certifcate related operations to an OS-specific "signer binary"
// that communicates with the native keystore (ex. keychain on MacOS).
//
// The configFilePath points to a config file containing relevant parameters
// such as the certificate issuer and the location of the signer binary.
// If configFilePath is empty, the client will attempt to load the config from
// a well-known gcloud location.
func NewEnterpriseCertificateProxySource(configFilePath string) (Source, error) {
	key, err := client.Cred(configFilePath)
	if err != nil {
		if errors.Is(err, client.ErrCredUnavailable) {
			return nil, errSourceUnavailable
		}
		return nil, err
	}

	return (&ecpSource{
		key: key,
	}).getClientCertificate, nil
}

func (s *ecpSource) getClientCertificate(info *tls.CertificateRequestInfo) (*tls.Certificate, error) {
	var cert tls.Certificate
	cert.PrivateKey = s.key
	cert.Certificate = s.key.CertificateChain()
	return &cert, nil
}
