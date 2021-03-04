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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"sync"
)

const (
	metadataPath = ".secureConnect"
	metadataFile = "context_aware_metadata.json"
)

var (
	defaultSourceOnce sync.Once
	defaultSource     Source
	defaultSourceErr  error
)

// Source is a function that can be passed into crypto/tls.Config.GetClientCertificate.
type Source func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// DefaultSource returns a certificate source that execs the command specified
// in the file at ~/.secureConnect/context_aware_metadata.json
//
// If that file does not exist, a nil source is returned.
func DefaultSource() (Source, error) {
	defaultSourceOnce.Do(func() {
		defaultSource, defaultSourceErr = newSecureConnectSource()
	})
	return defaultSource, defaultSourceErr
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
	// TODO(cbro): consider caching valid certificates rather than exec'ing every time.
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
	return &cert, nil
}
