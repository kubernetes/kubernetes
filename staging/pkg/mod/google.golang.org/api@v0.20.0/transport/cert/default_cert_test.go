// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cert

import (
	"testing"
)

func TestGetClientCertificateSuccess(t *testing.T) {
	source := secureConnectSource{metadata: secureConnectMetadata{Cmd: []string{"cat", "testdata/testcert.pem"}}}
	cert, err := source.getClientCertificate(nil)
	if err != nil {
		t.Error(err)
	}
	if cert.Certificate == nil {
		t.Error("want non-nil cert, got nil")
	}
	if cert.PrivateKey == nil {
		t.Error("want non-nil PrivateKey, got nil")
	}
}

func TestGetClientCertificateFailure(t *testing.T) {
	source := secureConnectSource{metadata: secureConnectMetadata{Cmd: []string{"cat"}}}
	_, err := source.getClientCertificate(nil)
	if err == nil {
		t.Error("Expecting error.")
	}
	if got, want := err.Error(), "tls: failed to find any PEM data in certificate input"; got != want {
		t.Errorf("getClientCertificate, want %v err, got %v", want, got)
	}
}

func TestValidateMetadataSuccess(t *testing.T) {
	metadata := secureConnectMetadata{Cmd: []string{"cat", "testdata/testcert.pem"}}
	err := validateMetadata(metadata)
	if err != nil {
		t.Error(err)
	}
}

func TestValidateMetadataFailure(t *testing.T) {
	metadata := secureConnectMetadata{Cmd: []string{}}
	err := validateMetadata(metadata)
	if err == nil {
		t.Error("validateMetadata: want non-nil err, got nil")
	}
	if want, got := "empty cert_provider_command", err.Error(); want != got {
		t.Errorf("validateMetadata: want %v err, got %v", want, got)
	}
}
