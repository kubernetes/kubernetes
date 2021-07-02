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
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"sync"
	"time"
)

const (
	metadataPath = ".secureConnect"
	metadataFile = "context_aware_metadata.json"
)

// defaultCertData holds all the variables pertaining to
// the default certficate source created by DefaultSource.
type defaultCertData struct {
	once            sync.Once
	source          Source
	err             error
	cachedCertMutex sync.Mutex
	cachedCert      *tls.Certificate
}

var (
	defaultCert defaultCertData
)

// Source is a function that can be passed into crypto/tls.Config.GetClientCertificate.
type Source func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// DefaultSource returns a certificate source that execs the command specified
// in the file at ~/.secureConnect/context_aware_metadata.json
//
// If that file does not exist, a nil source is returned.
func DefaultSource() (Source, error) {
	defaultCert.once.Do(func() {
		defaultCert.source, defaultCert.err = newSecureConnectSource()
	})
	return defaultCert.source, defaultCert.err
}

type secureConnectSource struct {
	metadata secureConnectMetadata
}

type secureConnectMetadata struct {
	Cmd []string `json:"cert_provider_command"`
}

// newSecureConnectSource creates a secureConnectSource by reading the well-known file.
func newSecureConnectSource() (Source, error) {
	user, err := user.Current()
	if err != nil {
		// Ignore.
		return nil, nil
	}
	filename := filepath.Join(user.HomeDir, metadataPath, metadataFile)
	file, err := ioutil.ReadFile(filename)
	if os.IsNotExist(err) {
		// Ignore.
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var metadata secureConnectMetadata
	if err := json.Unmarshal(file, &metadata); err != nil {
		return nil, fmt.Errorf("cert: could not parse JSON in %q: %v", filename, err)
	}
	if err := validateMetadata(metadata); err != nil {
		return nil, fmt.Errorf("cert: invalid config in %q: %v", filename, err)
	}
	return (&secureConnectSource{
		metadata: metadata,
	}).getClientCertificate, nil
}

func validateMetadata(metadata secureConnectMetadata) error {
	if len(metadata.Cmd) == 0 {
		return errors.New("empty cert_provider_command")
	}
	return nil
}

func (s *secureConnectSource) getClientCertificate(info *tls.CertificateRequestInfo) (*tls.Certificate, error) {
	defaultCert.cachedCertMutex.Lock()
	defer defaultCert.cachedCertMutex.Unlock()
	if defaultCert.cachedCert != nil && !isCertificateExpired(defaultCert.cachedCert) {
		return defaultCert.cachedCert, nil
	}
	// Expand OS environment variables in the cert provider command such as "$HOME".
	for i := 0; i < len(s.metadata.Cmd); i++ {
		s.metadata.Cmd[i] = os.ExpandEnv(s.metadata.Cmd[i])
	}
	command := s.metadata.Cmd
	data, err := exec.Command(command[0], command[1:]...).Output()
	if err != nil {
		// TODO(cbro): read stderr for error message? Might contain sensitive info.
		return nil, err
	}
	cert, err := tls.X509KeyPair(data, data)
	if err != nil {
		return nil, err
	}
	defaultCert.cachedCert = &cert
	return &cert, nil
}

// isCertificateExpired returns true if the given cert is expired or invalid.
func isCertificateExpired(cert *tls.Certificate) bool {
	if len(cert.Certificate) == 0 {
		return true
	}
	parsed, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return true
	}
	return time.Now().After(parsed.NotAfter)
}
