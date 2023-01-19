// Copyright 2020 Google LLC.
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
	"sync"
)

// defaultCertData holds all the variables pertaining to
// the default certficate source created by DefaultSource.
//
// A singleton model is used to allow the source to be reused
// by the transport layer.
type defaultCertData struct {
	once   sync.Once
	source Source
	err    error
}

var (
	defaultCert defaultCertData
)

// Source is a function that can be passed into crypto/tls.Config.GetClientCertificate.
type Source func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// errSourceUnavailable is a sentinel error to indicate certificate source is unavailable.
var errSourceUnavailable = errors.New("certificate source is unavailable")

// DefaultSource returns a certificate source using the preferred EnterpriseCertificateProxySource.
// If EnterpriseCertificateProxySource is not available, fall back to the legacy SecureConnectSource.
//
// If neither source is available (due to missing configurations), a nil Source and a nil Error are
// returned to indicate that a default certificate source is unavailable.
func DefaultSource() (Source, error) {
	defaultCert.once.Do(func() {
		defaultCert.source, defaultCert.err = NewEnterpriseCertificateProxySource("")
		if errors.Is(defaultCert.err, errSourceUnavailable) {
			defaultCert.source, defaultCert.err = NewSecureConnectSource("")
			if errors.Is(defaultCert.err, errSourceUnavailable) {
				defaultCert.source, defaultCert.err = nil, nil
			}
		}
	})
	return defaultCert.source, defaultCert.err
}
