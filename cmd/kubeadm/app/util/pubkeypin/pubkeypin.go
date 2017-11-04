/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package pubkeypin provides primitives for x509 public key pinning in the
// style of RFC7469.
package pubkeypin

import (
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	// formatSHA256 is the prefix for pins that are full-length SHA-256 hashes encoded in base 16 (hex)
	formatSHA256 = "sha256"
)

// Set is a set of pinned x509 public keys.
type Set struct {
	sha256Hashes map[string]bool
}

// NewSet returns a new, empty PubKeyPinSet
func NewSet() *Set {
	return &Set{make(map[string]bool)}
}

// Allow adds an allowed public key hash to the Set
func (s *Set) Allow(pubKeyHashes ...string) error {
	for _, pubKeyHash := range pubKeyHashes {
		parts := strings.Split(pubKeyHash, ":")
		if len(parts) != 2 {
			return fmt.Errorf("invalid public key hash, expected \"format:value\"")
		}
		format, value := parts[0], parts[1]

		switch strings.ToLower(format) {
		case "sha256":
			return s.allowSHA256(value)
		default:
			return fmt.Errorf("unknown hash format %q", format)
		}
	}
	return nil
}

// Check if a certificate matches one of the public keys in the set
func (s *Set) Check(certificate *x509.Certificate) error {
	if s.checkSHA256(certificate) {
		return nil
	}
	return fmt.Errorf("public key %s not pinned", Hash(certificate))
}

// Empty returns true if the Set contains no pinned public keys.
func (s *Set) Empty() bool {
	return len(s.sha256Hashes) == 0
}

// Hash calculates the SHA-256 hash of the Subject Public Key Information (SPKI)
// object in an x509 certificate (in DER encoding). It returns the full hash as a
// hex encoded string (suitable for passing to Set.Allow).
func Hash(certificate *x509.Certificate) string {
	spkiHash := sha256.Sum256(certificate.RawSubjectPublicKeyInfo)
	return formatSHA256 + ":" + strings.ToLower(hex.EncodeToString(spkiHash[:]))
}

// allowSHA256 validates a "sha256" format hash and adds a canonical version of it into the Set
func (s *Set) allowSHA256(hash string) error {
	// validate that the hash is the right length to be a full SHA-256 hash
	hashLength := hex.DecodedLen(len(hash))
	if hashLength != sha256.Size {
		return fmt.Errorf("expected a %d byte SHA-256 hash, found %d bytes", sha256.Size, hashLength)
	}

	// validate that the hash is valid hex
	_, err := hex.DecodeString(hash)
	if err != nil {
		return err
	}

	// in the end, just store the original hex string in memory (in lowercase)
	s.sha256Hashes[strings.ToLower(hash)] = true
	return nil
}

// checkSHA256 returns true if the certificate's "sha256" hash is pinned in the Set
func (s *Set) checkSHA256(certificate *x509.Certificate) bool {
	actualHash := sha256.Sum256(certificate.RawSubjectPublicKeyInfo)
	actualHashHex := strings.ToLower(hex.EncodeToString(actualHash[:]))
	return s.sha256Hashes[actualHashHex]
}
