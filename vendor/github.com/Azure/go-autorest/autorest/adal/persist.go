package adal

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"golang.org/x/crypto/pkcs12"
)

var (
	// ErrMissingCertificate is returned when no local certificate is found in the provided PFX data.
	ErrMissingCertificate = errors.New("adal: certificate missing")

	// ErrMissingPrivateKey is returned when no private key is found in the provided PFX data.
	ErrMissingPrivateKey = errors.New("adal: private key missing")
)

// LoadToken restores a Token object from a file located at 'path'.
func LoadToken(path string) (*Token, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file (%s) while loading token: %v", path, err)
	}
	defer file.Close()

	var token Token

	dec := json.NewDecoder(file)
	if err = dec.Decode(&token); err != nil {
		return nil, fmt.Errorf("failed to decode contents of file (%s) into Token representation: %v", path, err)
	}
	return &token, nil
}

// SaveToken persists an oauth token at the given location on disk.
// It moves the new file into place so it can safely be used to replace an existing file
// that maybe accessed by multiple processes.
func SaveToken(path string, mode os.FileMode, token Token) error {
	dir := filepath.Dir(path)
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		return fmt.Errorf("failed to create directory (%s) to store token in: %v", dir, err)
	}

	newFile, err := ioutil.TempFile(dir, "token")
	if err != nil {
		return fmt.Errorf("failed to create the temp file to write the token: %v", err)
	}
	tempPath := newFile.Name()

	if err := json.NewEncoder(newFile).Encode(token); err != nil {
		return fmt.Errorf("failed to encode token to file (%s) while saving token: %v", tempPath, err)
	}
	if err := newFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file %s: %v", tempPath, err)
	}

	// Atomic replace to avoid multi-writer file corruptions
	if err := os.Rename(tempPath, path); err != nil {
		return fmt.Errorf("failed to move temporary token to desired output location. src=%s dst=%s: %v", tempPath, path, err)
	}
	if err := os.Chmod(path, mode); err != nil {
		return fmt.Errorf("failed to chmod the token file %s: %v", path, err)
	}
	return nil
}

// DecodePfxCertificateData extracts the x509 certificate and RSA private key from the provided PFX data.
// The PFX data must contain a private key along with a certificate whose public key matches that of the
// private key or an error is returned.
// If the private key is not password protected pass the empty string for password.
func DecodePfxCertificateData(pfxData []byte, password string) (*x509.Certificate, *rsa.PrivateKey, error) {
	blocks, err := pkcs12.ToPEM(pfxData, password)
	if err != nil {
		return nil, nil, err
	}
	// first extract the private key
	var priv *rsa.PrivateKey
	for _, block := range blocks {
		if block.Type == "PRIVATE KEY" {
			priv, err = x509.ParsePKCS1PrivateKey(block.Bytes)
			if err != nil {
				return nil, nil, err
			}
			break
		}
	}
	if priv == nil {
		return nil, nil, ErrMissingPrivateKey
	}
	// now find the certificate with the matching public key of our private key
	var cert *x509.Certificate
	for _, block := range blocks {
		if block.Type == "CERTIFICATE" {
			pcert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, nil, err
			}
			certKey, ok := pcert.PublicKey.(*rsa.PublicKey)
			if !ok {
				// keep looking
				continue
			}
			if priv.E == certKey.E && priv.N.Cmp(certKey.N) == 0 {
				// found a match
				cert = pcert
				break
			}
		}
	}
	if cert == nil {
		return nil, nil, ErrMissingCertificate
	}
	return cert, priv, nil
}
